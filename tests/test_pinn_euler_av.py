import dataclasses
import json
import math
import re

import numpy as np
import pytest
import sympy as sp
import torch

import apno.models.pinn_euler_av as mod
from apno.geometry.ogrid import Section, map_xy, stretch_ratio
from apno.models.pinn import PinnCase, wall_normal
from apno.models.pinn_euler_av import (
    AvConfig, EulerAvNet, EulerAvPinn, config_from_dict, freestream, from_dataset_q, losses,
    to_dataset_q,
)
from apno.physics.euler import metrics
from apno.physics.viscosity import abs_jacobian

GEOM = Section(0.12, 0.03)
CASE = PinnCase(0.72, 3.0, GEOM)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pts(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.01 + 0.98 * torch.rand(n, generator=g), 1e-3 + (1 - 1e-3) * torch.rand(n, generator=g)


def test_freestream_and_dataset_units_round_trip():
    prim = tuple(torch.tensor(x) for x in freestream(0.72, 3.0))
    q = to_dataset_q(*prim, 0.72)
    a = math.radians(3.0)
    torch.testing.assert_close(q, torch.tensor([1.0, math.cos(a), math.sin(a), 1.0]))
    for got, want in zip(from_dataset_q(q, 0.72), prim):
        torch.testing.assert_close(got, want)


def test_network_size_and_shapes():
    net = EulerAvNet(AvConfig())
    assert sum(p.numel() for p in net.parameters()) == 100_109
    xi, eta = _pts(7)
    assert net(xi, eta).shape == (7, 4)
    assert EulerAvNet(AvConfig(closure="enthalpy"))(xi, eta).shape == (7, 3)
    assert EulerAvNet(AvConfig(encoding="periodic"))(xi, eta).shape == (7, 4)


def test_the_embedding_is_fixed_by_the_seed():
    a, b, c = (EulerAvNet(AvConfig(seed=s)).embed.phi for s in (0, 0, 1))
    assert torch.equal(a, b) and not torch.equal(a, c) and a.shape == (128, 2)


def test_config_round_trips_through_json():
    cfg = AvConfig(sensor="B", te_exclusion=0.02)
    assert config_from_dict(json.loads(json.dumps(dataclasses.asdict(cfg)))) == cfg


def test_unknown_options_are_refused():
    with pytest.raises(ValueError):
        EulerAvPinn(CASE, AvConfig(encoding="raw"))
    with pytest.raises(ValueError):
        EulerAvPinn(CASE, AvConfig(closure="potential"))


# Every string option and its vocabulary (design §2, §4): a typo must not silently run another arm.
VOCABULARY = {
    "encoding": ("fourier", "periodic"),
    "closure": ("primitive", "enthalpy"),
    "weighting": ("volume", "uniform"),
    "collocation": ("nodes", "random"),
    "viscosity": ("matrix", "scalar", "none"),
    "sensor": ("A", "B"),
    "optimizer": ("lbfgs", "adam_lbfgs"),
}


@pytest.mark.parametrize("field", sorted(VOCABULARY))
def test_a_typo_in_any_string_option_is_refused_naming_the_field(field):
    named = re.escape(f"AvConfig.{field} = 'typo'")
    with pytest.raises(ValueError, match=named):
        AvConfig(**{field: "typo"})
    with pytest.raises(ValueError, match=named):  # the way the S5 stage builds its arms
        dataclasses.replace(AvConfig(), **{field: "typo"})
    with pytest.raises(ValueError, match=named):
        EulerAvPinn(CASE, AvConfig(**{field: "typo"}))


@pytest.mark.parametrize("field", sorted(VOCABULARY))
def test_every_listed_value_of_a_string_option_is_accepted(field):
    tiny = AvConfig(hidden=4, depth=1, n_freq=2)
    for value in VOCABULARY[field]:
        EulerAvPinn(CASE, dataclasses.replace(tiny, **{field: value}))


def _const(values):
    def state(xi, eta):
        z = 0 * xi + 0 * eta
        return tuple(z + v for v in values)
    return state


@pytest.mark.parametrize("viscosity", ["none", "matrix", "scalar"])
def test_uniform_flow_has_zero_residual_on_the_curved_grid(viscosity, monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity=viscosity))
    monkeypatch.setattr(pinn, "primitive", _const(pinn.w_inf))
    xi, eta = _pts()
    r = pinn.residual(xi, eta)
    assert r.shape == (300, 4) and r.abs().max() < 1e-9


def _manufactured_fields():
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 10) * sp.sin(x) * sp.cos(y)
    u = 1 + sp.Rational(1, 10) * sp.cos(x) * sp.sin(2 * y)
    v = sp.Rational(1, 10) * sp.sin(y) * sp.cos(x)
    p = 3 + sp.Rational(1, 10) * sp.cos(x + y)
    e = p / sp.Rational(2, 5) + rho * (u**2 + v**2) / 2  # rho E; gamma - 1 = 0.4
    return (x, y), (rho, u, v, p, e)


def _manufactured():
    (x, y), (rho, u, v, p, e) = _manufactured_fields()
    F = [rho * u, rho * u**2 + p, rho * u * v, u * (e + p)]
    G = [rho * v, rho * u * v, rho * v**2 + p, v * (e + p)]
    div = [sp.lambdify((x, y), sp.diff(F[k], x) + sp.diff(G[k], y)) for k in range(4)]
    lap_rho = sp.lambdify((x, y), sp.diff(rho, x, 2) + sp.diff(rho, y, 2))
    lam = sp.lambdify((x, y), sp.sqrt(sp.Rational(7, 5) * p / rho) + sp.sqrt(u**2 + v**2))
    return div, lap_rho, lam


def _second_derivatives():
    """(W_xx, W_yy) of the manufactured W = (rho, rho u, rho v, rho E): functions of numpy (x, y)
    returning (n, 4)."""
    (x, y), (rho, u, v, p, e) = _manufactured_fields()
    w = (rho, rho * u, rho * v, e)

    def along(s):
        fs = [sp.lambdify((x, y), sp.diff(c, s, 2)) for c in w]
        return lambda xv, yv: np.stack([np.broadcast_to(f(xv, yv), xv.shape) for f in fs], -1)

    return along(x), along(y)


def _manufactured_state(xi, eta):
    x, y, *_ = metrics(xi, eta, GEOM)
    rho = 1 + 0.1 * torch.sin(x) * torch.cos(y)
    u = 1 + 0.1 * torch.cos(x) * torch.sin(2 * y)
    v = 0.1 * torch.sin(y) * torch.cos(x)
    p = 3 + 0.1 * torch.cos(x + y)
    return rho, u, v, p


def test_manufactured_solution_through_all_four_equations(monkeypatch):
    div, _, _ = _manufactured()
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    monkeypatch.setattr(pinn, "primitive", _manufactured_state)
    xi, eta = _pts(200, 1)
    eta = eta * 0.05  # keep x, y within a few chords so the fields stay O(1)
    r = pinn.residual(xi, eta).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    want = np.stack([div[k](x, y) for k in range(4)], 1)
    np.testing.assert_allclose(r, want, rtol=1e-8, atol=1e-8)


def test_scalar_viscosity_through_the_map_matches_sympy(monkeypatch):
    div, lap_rho, lam = _manufactured()
    nu = 1e-2
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="scalar", nu=nu))
    monkeypatch.setattr(pinn, "primitive", _manufactured_state)
    monkeypatch.setattr(mod, "sensor", lambda u, *a: torch.ones_like(u))
    xi, eta = _pts(150, 2)
    eta = eta * 0.05
    r = pinn.residual(xi, eta).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    np.testing.assert_allclose(r[:, 0], div[0](x, y) - nu * lam(x, y) * lap_rho(x, y),
                               rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(r[:, 1:3], np.stack([div[1](x, y), div[2](x, y)], 1),
                               rtol=1e-8, atol=1e-8)
    w_xx, w_yy = _second_derivatives()
    lap_energy = w_xx(x, y)[:, 3] + w_yy(x, y)[:, 3]  # Laplacian of rho E, not of p or E
    np.testing.assert_allclose(r[:, 3], div[3](x, y) - nu * lam(x, y) * lap_energy,
                               rtol=1e-7, atol=1e-8)


def test_matrix_viscosity_through_the_map_matches_sympy(monkeypatch):
    """D = nu s I_{rho,rhoE} (|A| W_xx + |B| W_yy), |A| and |B| of the x and y flux Jacobians."""
    div, _, _ = _manufactured()
    w_xx, w_yy = _second_derivatives()
    nu = 1e-2
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="matrix", nu=nu))
    monkeypatch.setattr(pinn, "primitive", _manufactured_state)
    monkeypatch.setattr(mod, "sensor", lambda u, *a: torch.ones_like(u))
    xi, eta = _pts(150, 3)
    eta = eta * 0.05
    r = pinn.residual(xi, eta).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    state = [q.detach() for q in _manufactured_state(xi, eta)]
    abs_a = abs_jacobian(*state, 1.0, 0.0).numpy()  # checked against eig in test_viscosity.py
    abs_b = abs_jacobian(*state, 0.0, 1.0).numpy()
    select = np.array([1.0, 0.0, 0.0, 1.0])  # I_{rho, rho E}

    def d(mu_x, mu_y):
        return nu * select * (np.einsum("nij,nj->ni", mu_x, w_xx(x, y))
                              + np.einsum("nij,nj->ni", mu_y, w_yy(x, y)))

    div_f = np.stack([div[k](x, y) for k in range(4)], 1)
    np.testing.assert_allclose(r, div_f - d(abs_a, abs_b), rtol=1e-7, atol=1e-8)
    # |A| and |B| differ here by far more than that tolerance, on both viscous rows
    swap = np.abs(d(abs_b, abs_a) - d(abs_a, abs_b))
    assert swap[:, 0].max() > 1e-5 and swap[:, 3].max() > 1e-5


def test_far_field_and_periodic_terms_vanish_at_free_stream(monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    monkeypatch.setattr(pinn, "primitive", _const(pinn.w_inf))
    xi, eta = _pts(10)
    bnd = {"xi_ring": torch.arange(64) / 64, "eta_lr": torch.arange(25) / 24}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(10)}, bnd, pinn.cfg)
    assert L["far"] < 1e-28 and L["periodic"] < 1e-28 and L["residual"] < 1e-16


def test_the_wall_term_vanishes_for_tangent_flow(monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))

    def tangent(xi, eta):
        nx, ny = wall_normal(xi, GEOM)
        return torch.ones_like(xi), -ny, nx, torch.ones_like(xi)

    monkeypatch.setattr(pinn, "primitive", tangent)
    xi, eta = _pts(10)
    bnd = {"xi_ring": torch.arange(64) / 64, "eta_lr": torch.arange(25) / 24}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(10)}, bnd, pinn.cfg)
    assert L["wall"] < 1e-28


RING, ETA_LR = np.arange(64) / 64, np.arange(25) / 24  # float64 whatever torch's default


def _boundary_losses(monkeypatch, field):
    """losses() with primitive replaced by field(xi, eta), on the 64-point ring and 25 eta."""
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    monkeypatch.setattr(pinn, "primitive", field)
    xi, eta = _pts(10)
    return losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(10)},
                  {"xi_ring": torch.from_numpy(RING), "eta_lr": torch.from_numpy(ETA_LR)}, pinn.cfg)


def test_the_far_field_term_reads_the_outer_row_along_the_ring(monkeypatch):
    # departure from free stream (k + 1)(sin(2 pi xi) eta^2 / 10 + (1 - eta) / 5): on the outer row
    # it varies along the ring and differs per component; on the wall row it is a constant
    w_inf = freestream(CASE.mach, CASE.alpha_deg)

    def field(xi, eta):
        ring = torch.sin(2 * math.pi * xi)
        return tuple(w + (k + 1) * (0.1 * ring * eta ** 2 + 0.2 * (1 - eta))
                     for k, w in enumerate(w_inf))

    L = _boundary_losses(monkeypatch, field)
    ring = np.sin(2 * np.pi * RING)
    want = sum(((k + 1) * 0.1 * ring) ** 2 for k in range(4)).mean()
    torch.testing.assert_close(L["far"], torch.tensor(want), rtol=1e-12, atol=1e-14)


def test_the_wall_term_reads_the_wall_row_along_the_ring(monkeypatch):
    # normal velocity 0.3 cos(2 pi xi) + eta, tangential 1 + eta; rho and p are not the velocity
    def field(xi, eta):
        nx, ny = wall_normal(xi, GEOM)
        un, ut = 0.3 * torch.cos(2 * math.pi * xi) + eta, 1 + eta
        return 2 + eta + 0 * xi, un * nx - ut * ny, un * ny + ut * nx, 3 - eta + 0 * xi

    L = _boundary_losses(monkeypatch, field)
    want = ((0.3 * np.cos(2 * np.pi * RING)) ** 2).mean()
    torch.testing.assert_close(L["wall"], torch.tensor(want), rtol=1e-12, atol=1e-14)


def test_the_periodic_term_pairs_xi_0_with_xi_1_at_each_eta(monkeypatch):
    # the jump from xi = 0 to xi = 1 is (k + 1)(1 + eta) / 10; the cos(2 pi xi) part cancels only
    # between xi = 0 and xi = 1, and the eta^2 part only between equal eta
    def field(xi, eta):
        return tuple(1 + 0.5 * k + 0.2 * torch.cos(2 * math.pi * xi)
                     + (k + 1) * (0.1 * xi * (1 + eta) + 0.05 * eta ** 2) for k in range(4))

    L = _boundary_losses(monkeypatch, field)
    want = sum(((k + 1) * 0.1 * (1 + ETA_LR)) ** 2 for k in range(4)).mean()
    torch.testing.assert_close(L["periodic"], torch.tensor(want), rtol=1e-12, atol=1e-14)


def test_the_total_weights_the_residual_by_lambda_res():
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    xi, eta = _pts(20)
    bnd = {"xi_ring": torch.arange(16) / 16, "eta_lr": torch.arange(9) / 8}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(20)}, bnd, pinn.cfg)
    want = 2e4 * L["residual"] + L["far"] + L["wall"] + L["periodic"]
    torch.testing.assert_close(L["total"], want)


def test_the_duplicate_column_is_the_same_physical_points():
    eta = torch.linspace(0, 1, 25)
    ratio = stretch_ratio(100.0)
    x0, y0 = map_xy(torch.zeros(25), eta, GEOM, 100.0, ratio, xp=torch)
    x1, y1 = map_xy(torch.ones(25), eta, GEOM, 100.0, ratio, xp=torch)
    torch.testing.assert_close(x0, x1, atol=1e-12, rtol=0)
    torch.testing.assert_close(y0, y1, atol=1e-12, rtol=0)


def test_the_enthalpy_closure_is_tangent_on_the_wall_and_has_three_rows():
    pinn = EulerAvPinn(CASE, AvConfig(closure="enthalpy", viscosity="none"))
    xr = torch.arange(64) / 64
    _, u, v, _ = pinn.primitive(xr, torch.zeros(64))
    nx, ny = wall_normal(xr, GEOM)
    assert (u * nx + v * ny).abs().max() < 1e-12
    xi, eta = _pts(20)
    assert pinn.residual(xi, eta).shape == (20, 3)


def test_the_pinn_carries_no_flow_field_data():
    pinn = EulerAvPinn(CASE, AvConfig())
    assert set(vars(pinn)) == {"case", "cfg", "net", "ratio", "w_inf", "k", "viscous_everywhere"}
    assert set(vars(CASE)) == {"mach", "alpha_deg", "geometry", "radius"}
