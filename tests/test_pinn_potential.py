import math

import numpy as np
import pytest
import torch

from apno.constants import GAMMA
from apno.geometry.ogrid import FAR_RADIUS, Cylinder, Section, map_xy, stretch_ratio
from apno.models.pinn import Pinn, PinnCase, PinnNet, to_dataset_q
from apno.models.pinn_potential import (
    PotentialNet, PotentialPinn, bernoulli_density, state_from_potential, vortex_angle, wall_weight,
)
from apno.models.train_pinn import (
    PinnConfig, _points, evaluate_on_grid, exclude_trailing_edge, losses, make_pinn, train,
)
from apno.physics.potential import residual
from apno.score.forces import wall_forces

SEC = Section(0.12, 0.03)
RATIO = stretch_ratio(FAR_RADIUS)


@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only; a module-level set_default_dtype leaks into every test."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pinn(geom=SEC, mach=0.5, alpha=3.0, gamma=0.0, psi_zero=False, circulation=True):
    torch.manual_seed(0)
    net = PotentialNet(hidden=16, depth=2, circulation=circulation)
    if psi_zero:  # the last layer's weights and bias zero: psi = 0 everywhere
        last = net.mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
    with torch.no_grad():
        net.gamma.fill_(gamma)
    return PotentialPinn(PinnCase(mach, alpha, geom), net)


def _pts(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.01 + 0.98 * torch.rand(n, generator=g), 1e-3 + (1 - 1e-3) * torch.rand(n, generator=g)


def test_free_stream_has_zero_residual_on_the_curved_grid():
    p = _pinn(psi_zero=True)
    xi, eta = _pts()
    r = residual(xi, eta, p.state, SEC, FAR_RADIUS, RATIO)
    assert r.abs().max() < 1e-9
    _, u, v, _ = p.state(xi, eta)
    torch.testing.assert_close(u, torch.full_like(u, p.ca), rtol=0, atol=1e-12)
    torch.testing.assert_close(v, torch.full_like(v, p.sa), rtol=0, atol=1e-12)


def _cylinder_potential(gamma=0.0):
    """Incompressible flow past the radius-0.5 cylinder about (0.5, 0), plus gamma * xi, which on
    the cylinder's O-grid (radial xi-lines) is exactly the clockwise point vortex."""
    a = 0.5

    def phi(xi, eta, x, y):
        X = x - 0.5
        return X + a * a * X / (X * X + y * y) + gamma * xi

    return phi


def test_analytic_incompressible_cylinder_potential_satisfies_the_residual_at_low_mach():
    geom, mach = Cylinder(), 0.01
    g = torch.Generator().manual_seed(1)
    xi, eta = torch.rand(20000, generator=g), 1 - torch.rand(20000, generator=g)
    phi = _cylinder_potential()

    def state(xi, eta):
        return state_from_potential(phi, xi, eta, geom, FAR_RADIUS, RATIO, mach)

    r = residual(xi, eta, state, geom, FAR_RADIUS, RATIO)
    assert float((r * r).mean().detach()) < 1e-6
    # the velocity through the metric terms is the analytic one: 2 at the shoulder
    _, u, v, _ = state(torch.tensor([0.25]), torch.tensor([0.0]))
    assert float(torch.hypot(u, v).detach()) == pytest.approx(2.0, abs=1e-12)


def _grad_theta(xi, eta, geom=SEC):
    """grad(Theta_v) through the map's metric terms, as the PINN's own state computes it."""
    def phi(xi_, eta_, x, y):
        return vortex_angle(x, y)

    _, u, v, _ = state_from_potential(phi, xi, eta, geom, FAR_RADIUS, RATIO, 0.5)
    return u.detach(), v.detach()


def test_vortex_angle_has_a_unit_jump_across_the_wake_cut_and_a_continuous_gradient():
    eta = torch.linspace(0.02, 1.0, 50)
    eps = 1e-7
    ua, va = _grad_theta(torch.full_like(eta, 1.0 - eps), eta)
    ub, vb = _grad_theta(torch.full_like(eta, eps), eta)
    scale = torch.hypot(ub, vb)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert ((ua - ub).abs() / scale).max() < 1e-4 and ((va - vb).abs() / scale).max() < 1e-4
    for e in (0.0, 0.3, 1.0):  # the jump, on the wall and off it
        xi = torch.tensor([1.0 - eps, eps])
        x, y = map_xy(xi, torch.full_like(xi, e), SEC, FAR_RADIUS, RATIO, xp=torch)
        th = vortex_angle(x, y)
        assert float(th[0] - th[1]) == pytest.approx(1.0, abs=1e-5)


def test_vortex_angle_gradient_is_regular_at_the_trailing_edge():
    """Theta = xi had |grad| ~ 1/sqrt(r) there (the map's x_xi = y_xi = 0); Theta_v is the vortex
    about the quarter chord, 1 / (2 pi |x - (0.25, 0)|) = 1 / (1.5 pi) at the TE, on both sides."""
    want = 1.0 / (2 * math.pi * 0.75)
    for xi in (1e-4, 1e-3, 1.0 - 1e-3, 1.0 - 1e-4):
        for eta in (0.0, 1e-4, 1e-3):
            u, v = _grad_theta(torch.tensor([xi]), torch.tensor([eta]))
            assert float(torch.hypot(u, v)) == pytest.approx(want, rel=0.05)
    # clockwise about the quarter chord: just behind the TE the vortex flows in -y
    u, v = _grad_theta(torch.tensor([1e-3]), torch.tensor([1e-3]))
    assert float(v) < 0 and abs(float(u)) < 0.1 * abs(float(v))


def test_vortex_angle_tends_to_the_vortex_velocity_at_the_outer_circle():
    xi = torch.linspace(0, 1, 129)[:-1]
    u, v = _grad_theta(xi, torch.ones_like(xi))
    x, y = map_xy(xi, torch.ones_like(xi), SEC, FAR_RADIUS, RATIO, xp=torch)
    want = 1.0 / (2 * math.pi * torch.hypot(x - 0.25, y))
    assert ((torch.hypot(u, v) - want).abs() / want).max() < 1e-6
    # clockwise: at the top of the circle (xi = 0.75, polar angle +90 deg) the vortex flows in +x
    k = 96
    assert float(u[k]) > 0 and abs(float(v[k])) < 1e-2 * float(u[k])
    # and far from the body Theta_v is the wake-cut angle xi
    torch.testing.assert_close(vortex_angle(x, y)[1:], xi[1:], rtol=0, atol=1e-3)


def test_positive_gamma_is_clockwise_circulation_and_positive_lift():
    """Convention: Gamma is the CLOCKWISE circulation (the counterclockwise line integral of the
    velocity is -Gamma), so Kutta-Joukowski gives L' = rho U Gamma > 0, cl = 2 Gamma / (U c).
    The line integral is the model's own term (a vortex at the quarter chord); the lift is the
    same convention's vortex moved to the cylinder's centre, where the wall is a streamline and
    the pressure integral is exactly Kutta-Joukowski."""
    geom, mach, gamma = Cylinder(), 0.01, 0.2
    p = _pinn(geom=geom, mach=mach, alpha=0.0, gamma=gamma, psi_zero=True)
    doublet = _cylinder_potential()

    def phi(xi, eta, x, y):  # the model's own free-stream and circulation terms plus the doublet
        return p.potential(xi, eta, x, y) + doublet(xi, eta, x, y) - (x - 0.5)

    def centred(xi, eta, x, y):
        return doublet(xi, eta, x, y) + gamma * vortex_angle(x, y, x0=0.5)

    n = 512
    xi = torch.arange(n) / n
    eta = torch.full_like(xi, 0.5)
    _, u, v, _ = state_from_potential(phi, xi, eta, geom, FAR_RADIUS, RATIO, mach)
    x, y = map_xy(xi, eta, geom, FAR_RADIUS, RATIO, xp=torch)
    xn, yn = torch.roll(x, -1), torch.roll(y, -1)  # xi increases clockwise: reverse for ccw
    ccw = -(0.5 * (u + torch.roll(u, -1)) * (xn - x) + 0.5 * (v + torch.roll(v, -1)) * (yn - y))
    assert float(ccw.sum().detach()) == pytest.approx(-gamma, rel=1e-3)
    xw = torch.arange(n) / n
    X, Y = map_xy(xw.numpy(), np.zeros(n), geom, FAR_RADIUS, RATIO)
    for f, want in ((centred, 2 * gamma), (phi, None)):
        rho, u, v, pr = state_from_potential(f, xw, torch.zeros_like(xw), geom, FAR_RADIUS,
                                             RATIO, mach)
        q = to_dataset_q(rho, u, v, pr, mach).detach().numpy()
        cp = (q[:, 3] - 1.0) / (0.5 * GAMMA * mach * mach)
        cl, cd, _ = wall_forces(X, Y, cp, 0.0)
        if want is None:
            assert cl > 0  # the model's own off-centre vortex: still positive lift
        else:
            assert cl == pytest.approx(want, rel=1e-2) and abs(cd) < 1e-3


def test_outer_dirichlet_holds_exactly():
    assert wall_weight(torch.tensor(1.0)) == 0.0 and wall_weight(torch.tensor(0.0)) == 1.0
    p = _pinn(gamma=0.3)
    xi = torch.rand(64)
    one = torch.ones_like(xi)
    x, y = map_xy(xi, one, SEC, FAR_RADIUS, RATIO, xp=torch)
    want = x * p.ca + y * p.sa + 0.3 * vortex_angle(x, y)
    torch.testing.assert_close(p.potential(xi, one, x, y), want, rtol=0, atol=0)


def test_state_satisfies_the_isentropic_relations_exactly():
    p = _pinn(gamma=0.1)
    xi, eta = _pts(200, 3)
    rho, u, v, pr = p.state(xi, eta)
    M = p.case.mach
    want = (1 + 0.5 * (GAMMA - 1) * M * M * (1 - (u * u + v * v))) ** (1 / (GAMMA - 1))
    torch.testing.assert_close(rho, want, rtol=1e-14, atol=0)
    q = to_dataset_q(rho, u, v, pr, M)
    torch.testing.assert_close(q[:, 3], rho**GAMMA, rtol=1e-14, atol=0)  # p / p_inf
    assert float(bernoulli_density(torch.tensor(100.0), M)) > 0  # clamped, never NaN


def test_trailing_edge_node_is_finite():
    p = _pinn(gamma=0.2)
    zero = torch.zeros(1)
    rho, u, v, pr = (t.detach() for t in p.state(zero, zero))  # the map is degenerate (J = 0)
    assert torch.isfinite(torch.stack([rho, u, v, pr])).all()
    assert float(u) == 0.0 and float(v) == 0.0


def test_equal_speed_kutta_term_is_zero_for_a_symmetric_state_and_not_otherwise():
    sym = _pinn(geom=Section(0.12, 0.0), alpha=0.0, gamma=0.0)
    with torch.no_grad():  # psi even in xi: no dependence on sin(2 pi xi), a symmetric flow
        sym.net.mlp[0].weight[:, 1] = 0.0
    like = torch.zeros(1)
    assert PinnConfig().kutta_delta == 0.055
    _, u, v, _ = sym.state(torch.tensor([0.055]), torch.zeros(1))
    assert abs(float(torch.hypot(u, v).detach()) - 1.0) > 1e-3  # not merely the uniform free stream
    assert abs(float(sym.kutta_residual(0.055, like).detach())) < 1e-12
    for p in (_pinn(geom=Section(0.12, 0.0), alpha=0.0, gamma=0.1, psi_zero=True),
              _pinn(geom=Section(0.12, 0.0), alpha=0.0),  # psi with a sin(2 pi xi) part
              _pinn(alpha=3.0)):  # cambered, at incidence
        assert abs(float(p.kutta_residual(0.055, like).detach())) > 1e-4
    # it is the difference of |u|^2 at the two wall points xi = delta and 1 - delta
    p = _pinn(gamma=0.1)
    xk = torch.tensor([0.055, 1 - 0.055])
    _, u, v, _ = p.state(xk, torch.zeros_like(xk))
    q2 = (u * u + v * v).detach()
    assert float(p.kutta_residual(0.055, like).detach()) == pytest.approx(float(q2[0] - q2[1]))
    cfg = _tiny()
    L = losses(p, _points(cfg, torch.Generator().manual_seed(0), torch.float64), cfg, RATIO)
    assert float(L["kutta"].detach()) == pytest.approx(float(q2[0] - q2[1]) ** 2)


def test_trailing_edge_exclusion_drops_exactly_the_points_within_the_radius():
    cfg = _tiny(n_interior=20000)
    pts = _points(cfg, torch.Generator().manual_seed(0), torch.float64)
    assert PinnConfig().te_exclusion == 0.02
    x, y = map_xy(pts["xi"], pts["eta"], SEC, FAR_RADIUS, RATIO, xp=torch)
    near = torch.hypot(x - 1.0, y) < 0.02
    assert int(near.sum()) > 0
    kept, dropped = exclude_trailing_edge(pts, SEC, FAR_RADIUS, RATIO, 0.02)
    assert dropped == int(near.sum())
    torch.testing.assert_close(kept["xi"], pts["xi"][~near], rtol=0, atol=0)
    torch.testing.assert_close(kept["eta"], pts["eta"][~near], rtol=0, atol=0)
    for k in ("xi_wall", "xi_outer"):  # the wall and Kutta terms are unaffected
        assert kept[k] is pts[k]
    same, none = exclude_trailing_edge(pts, Cylinder(), FAR_RADIUS, RATIO, 0.02)
    assert none == 0 and same["xi"] is pts["xi"]


def test_training_drops_the_trailing_edge_points_and_records_how_many():
    cfg = _tiny(n_interior=4000, resample_every=10)
    _, rec = train(PinnCase(0.3, 2.0, SEC), cfg, log=lambda *a: None)
    assert [d["step"] for d in rec["te_dropped"]] == [0, 10, 20]
    assert all(d["dropped"] > 0 for d in rec["te_dropped"])
    _, cyl = train(PinnCase(0.1, 0.0, Cylinder()), _tiny(kutta=False, zero_circulation=True),
                   log=lambda *a: None)
    assert all(d["dropped"] == 0 for d in cyl["te_dropped"])


def test_config_default_is_potential_and_euler_still_builds_todays_pinn():
    case = PinnCase(0.3, 2.0, SEC)
    assert PinnConfig().formulation == "potential"
    euler = make_pinn(case, PinnConfig(hidden=8, depth=2, formulation="euler"))
    assert type(euler) is Pinn and type(euler.net) is PinnNet
    pot = make_pinn(case, PinnConfig(hidden=8, depth=2))
    assert type(pot) is PotentialPinn and pot.formulation == "potential"
    assert any(q is pot.net.gamma for q in pot.parameters())
    fixed = make_pinn(case, PinnConfig(hidden=8, depth=2, zero_circulation=True))
    assert not any(q is fixed.net.gamma for q in fixed.parameters())
    assert float(fixed.net.gamma) == 0.0
    with pytest.raises(ValueError):
        make_pinn(case, PinnConfig(formulation="navier-stokes"))


def _tiny(**kw):
    base = dict(hidden=8, depth=2, n_interior=128, resample_every=50, n_outer=32, n_wall=32,
                adam_steps=30, lbfgs_iters=2, device="cpu")
    return PinnConfig(**{**base, **kw})


def _losses_on(device, cfg, case=PinnCase(0.3, 2.0, SEC)):
    torch.manual_seed(0)
    pinn = make_pinn(case, cfg, device, torch.float32)
    pts = _points(cfg, torch.Generator().manual_seed(0), torch.float32, device)
    return losses(pinn, pts, cfg, stretch_ratio(case.radius))


CONFIGS = (_tiny(), _tiny(kutta=False, zero_circulation=True), _tiny(viscosity=1e-3),
           _tiny(kutta=False))


def test_potential_losses_have_their_terms_and_follow_the_cpu():
    for cfg in CONFIGS:
        L = {k: v.detach() for k, v in _losses_on("cpu", cfg).items()}
        assert set(L) == {"residual", "wall", "kutta", "circulation", "total"}
        assert all(v.device.type == "cpu" and torch.isfinite(v) for v in L.values())
        assert float(L["circulation"]) == 0.0
        assert (float(L["kutta"]) == 0.0) == (not cfg.kutta)


def test_potential_training_runs_records_its_formulation_and_evaluates():
    pinn, rec = train(PinnCase(0.3, 2.0, SEC), _tiny(), log=lambda *a: None)
    assert rec["formulation"] == "potential" and rec["weights"] == {"wall": 10.0, "kutta": 10.0}
    assert "w_wall" in rec["history"][0] and np.isfinite(rec["final"]["total"])
    assert next(pinn.net.parameters()).dtype == torch.float64
    assert pinn.net.gamma.dtype == torch.float64
    q = evaluate_on_grid(pinn, (32, 9))
    assert q.shape == (32, 9, 4) and np.isfinite(q).all()
    _, euler = train(PinnCase(0.3, 2.0, SEC), _tiny(formulation="euler"), log=lambda *a: None)
    assert euler["formulation"] == "euler" and euler["weights"] == {"outer": 10.0, "kutta": 10.0}


def test_grad_balanced_potential_variant_updates_the_wall_weight():
    _, rec = train(PinnCase(0.3, 2.0, SEC), _tiny(grad_balance=True, balance_every=10),
                   log=lambda *a: None)
    assert rec["weights"]["wall"] != 10.0 and np.isfinite(rec["weights"]["wall"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU (MPS)")
def test_potential_state_and_boundary_terms_on_mps_match_the_cpu():
    """The device-specific code (the J = 0 mask, the Kutta term, the Gamma parameter) on
    the GPU against the CPU. The residual is not compared here: its first MPS call compiles ~1 s of
    kernels on a GPU another stage is using, and its operator is `euler.divergence`, which the Euler
    PINN's MPS test in test_train_pinn.py already exercises on the GPU."""
    case = PinnCase(0.3, 2.0, SEC)
    xi = torch.tensor([0.0, 0.1, 0.3, 0.6, 0.9], dtype=torch.float32)
    eta = torch.tensor([0.0, 0.0, 0.2, 0.5, 1.0], dtype=torch.float32)
    out = {}
    for device in ("cpu", "mps"):
        torch.manual_seed(0)
        pinn = make_pinn(case, PinnConfig(hidden=4, depth=1), device, torch.float32)
        with torch.no_grad():
            pinn.net.gamma.fill_(0.1)
        x, e = xi.to(device), eta.to(device)
        terms = [*pinn.state(x, e), pinn.wall_normal_velocity(x),
                 pinn.kutta_residual(0.055, x)]
        assert all(t.device.type == device for t in terms)
        out[device] = [t.detach().cpu() for t in terms]
    for a, b in zip(out["cpu"], out["mps"]):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)
