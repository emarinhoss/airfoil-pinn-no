import dataclasses
import json
import math

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


def _manufactured():
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 10) * sp.sin(x) * sp.cos(y)
    u = 1 + sp.Rational(1, 10) * sp.cos(x) * sp.sin(2 * y)
    v = sp.Rational(1, 10) * sp.sin(y) * sp.cos(x)
    p = 3 + sp.Rational(1, 10) * sp.cos(x + y)
    e = p / sp.Rational(2, 5) + rho * (u**2 + v**2) / 2  # gamma - 1 = 0.4
    F = [rho * u, rho * u**2 + p, rho * u * v, u * (e + p)]
    G = [rho * v, rho * u * v, rho * v**2 + p, v * (e + p)]
    div = [sp.lambdify((x, y), sp.diff(F[k], x) + sp.diff(G[k], y)) for k in range(4)]
    lap_rho = sp.lambdify((x, y), sp.diff(rho, x, 2) + sp.diff(rho, y, 2))
    lam = sp.lambdify((x, y), sp.sqrt(sp.Rational(7, 5) * p / rho) + sp.sqrt(u**2 + v**2))
    return div, lap_rho, lam


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
