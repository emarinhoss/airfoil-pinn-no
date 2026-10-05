import math

import numpy as np
import pytest
import torch

from apno.constants import GAMMA
from apno.geometry.ogrid import FAR_RADIUS, Cylinder, Section, map_xy, stretch_ratio
from apno.models.pinn import Pinn, PinnCase, PinnNet, to_dataset_q
from apno.models.pinn_potential import (
    PotentialNet, PotentialPinn, bernoulli_density, state_from_potential, wall_weight,
)
from apno.models.train_pinn import PinnConfig, _points, evaluate_on_grid, losses, make_pinn, train
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


def test_circulation_term_has_a_continuous_gradient_across_the_wake_cut_and_a_unit_jump():
    p = _pinn(gamma=1.0, psi_zero=True, alpha=0.0)
    eta = torch.linspace(0.02, 1.0, 50)
    eps = 1e-7

    def grad_theta(xi):
        def phi(xi_, eta_, x, y):
            return p.theta(xi_)

        _, u, v, _ = state_from_potential(phi, xi, eta, SEC, FAR_RADIUS, RATIO, 0.5)
        return u, v

    ua, va = grad_theta(torch.full_like(eta, 1.0 - eps))
    ub, vb = grad_theta(torch.full_like(eta, eps))
    scale = torch.hypot(ub, vb)
    assert ((ua - ub).abs() / scale).max() < 1e-4 and ((va - vb).abs() / scale).max() < 1e-4
    jump = p.theta(torch.tensor(1.0 - eps)) - p.theta(torch.tensor(0.0))
    assert float(jump) == pytest.approx(1.0, abs=1e-6)


def test_circulation_term_tends_to_the_vortex_velocity_at_the_outer_circle():
    xi = torch.linspace(0, 1, 129)[:-1]
    eta = torch.ones_like(xi)

    def phi(xi_, eta_, x, y):
        return 0.7 * xi_

    _, u, v, _ = state_from_potential(phi, xi, eta, SEC, FAR_RADIUS, RATIO, 0.5)
    x, y = map_xy(xi, eta, SEC, FAR_RADIUS, RATIO, xp=torch)
    r = torch.hypot(x - 0.5, y)
    want = 0.7 / (2 * math.pi * r)
    assert ((torch.hypot(u, v) - want).abs() / want).max() < 0.05
    # clockwise: at the top of the circle (xi = 0.75, polar angle +90 deg) the vortex flows in +x
    k = 96
    u, v = u.detach(), v.detach()
    assert float(u[k]) > 0 and abs(float(v[k])) < 1e-3 * float(u[k])


def test_positive_gamma_is_clockwise_circulation_and_positive_lift():
    """Convention: Gamma is the CLOCKWISE circulation (the counterclockwise line integral of the
    velocity is -Gamma), so Kutta-Joukowski gives L' = rho U Gamma > 0, cl = 2 Gamma / (U c)."""
    geom, mach, gamma = Cylinder(), 0.01, 0.2
    p = _pinn(geom=geom, mach=mach, alpha=0.0, gamma=gamma, psi_zero=True)
    doublet = _cylinder_potential()

    def phi(xi, eta, x, y):  # the model's own free-stream and circulation terms plus the doublet
        return p.potential(xi, eta, x, y) + doublet(xi, eta, x, y) - (x - 0.5)

    n = 512
    xi = torch.arange(n) / n
    eta = torch.full_like(xi, 0.5)
    _, u, v, _ = state_from_potential(phi, xi, eta, geom, FAR_RADIUS, RATIO, mach)
    x, y = map_xy(xi, eta, geom, FAR_RADIUS, RATIO, xp=torch)
    xn, yn = torch.roll(x, -1), torch.roll(y, -1)  # xi increases clockwise: reverse for ccw
    ccw = -(0.5 * (u + torch.roll(u, -1)) * (xn - x) + 0.5 * (v + torch.roll(v, -1)) * (yn - y))
    assert float(ccw.sum().detach()) == pytest.approx(-gamma, rel=1e-3)
    xw = torch.arange(n) / n
    rho, u, v, pr = state_from_potential(phi, xw, torch.zeros_like(xw), geom, FAR_RADIUS, RATIO,
                                         mach)
    q = to_dataset_q(rho, u, v, pr, mach).detach().numpy()
    cp = (q[:, 3] - 1.0) / (0.5 * GAMMA * mach * mach)
    X, Y = map_xy(xw.numpy(), np.zeros(n), geom, FAR_RADIUS, RATIO)
    cl, cd, _ = wall_forces(X, Y, cp, 0.0)
    assert cl == pytest.approx(2 * gamma, rel=1e-2) and abs(cd) < 1e-3


def test_outer_dirichlet_holds_exactly():
    assert wall_weight(torch.tensor(1.0)) == 0.0 and wall_weight(torch.tensor(0.0)) == 1.0
    p = _pinn(gamma=0.3)
    xi = torch.rand(64)
    one = torch.ones_like(xi)
    x, y = map_xy(xi, one, SEC, FAR_RADIUS, RATIO, xp=torch)
    want = x * p.ca + y * p.sa + 0.3 * xi
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


def test_trailing_edge_node_is_finite_and_the_kutta_slope_is_dphi_dxi():
    p = _pinn(gamma=0.2)
    zero = torch.zeros(1)
    rho, u, v, pr = (t.detach() for t in p.state(zero, zero))  # the map is degenerate (J = 0)
    assert torch.isfinite(torch.stack([rho, u, v, pr])).all()
    assert float(u) == 0.0 and float(v) == 0.0  # the stagnation value Kutta imposes
    xi = zero.clone().requires_grad_(True)
    x, y = map_xy(xi, zero, SEC, FAR_RADIUS, RATIO, xp=torch)
    want = torch.autograd.grad(p.potential(xi, zero, x, y).sum(), xi)[0]
    torch.testing.assert_close(p.trailing_edge_slope(zero), want)


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
    return PinnConfig(hidden=8, depth=2, n_interior=128, resample_every=50, n_outer=32, n_wall=32,
                      adam_steps=30, lbfgs_iters=2, device="cpu", **kw)


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
    """The device-specific code (the J = 0 mask, the trailing-edge slope, the Gamma parameter) on
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
        terms = [*pinn.state(x, e), pinn.wall_normal_velocity(x), pinn.trailing_edge_slope(x)]
        assert all(t.device.type == device for t in terms)
        out[device] = [t.detach().cpu() for t in terms]
    for a, b in zip(out["cpu"], out["mps"]):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)
