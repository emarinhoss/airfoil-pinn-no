import dataclasses

import numpy as np
import pytest
import torch

from apno.constants import GAMMA
from apno.geometry.ogrid import Cylinder, Section
from apno.models.pinn import Pinn, PinnCase, PinnNet, to_dataset_q, wall_normal


@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only; a module-level set_default_dtype leaks into every test."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pinn(geom=Section(0.12, 0.02)):
    torch.manual_seed(0)
    return Pinn(PinnCase(0.6, 3.0, geom), PinnNet(hidden=16, depth=2))


def test_no_flow_through_the_wall_by_construction():
    p = _pinn()
    xi = torch.linspace(0, 1, 257)[:-1]
    _, u, v, _ = p.state(xi, torch.zeros_like(xi))
    nx, ny = wall_normal(xi, p.case.geometry)
    assert (u * nx + v * ny).abs().max() < 1e-12


def test_total_enthalpy_constant_by_construction():
    p = _pinn()
    xi, eta = torch.rand(100), torch.rand(100)
    rho, u, v, pr = p.state(xi, eta)
    h = GAMMA / (GAMMA - 1) * pr / rho + 0.5 * (u * u + v * v)
    torch.testing.assert_close(h, torch.full_like(h, p.h_inf), rtol=1e-12, atol=1e-12)


def test_normals_point_outward_on_the_cylinder():
    xi = torch.linspace(0, 1, 65)[:-1]
    nx, ny = wall_normal(xi, Cylinder())
    th = -2 * np.pi * xi
    torch.testing.assert_close(nx, torch.cos(th), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ny, torch.sin(th), rtol=1e-12, atol=1e-12)


def test_section_normal_finite_at_trailing_edge_and_matches_finite_differences():
    sec = Section(0.12, 0.03)
    xi = torch.cat([torch.zeros(1), torch.linspace(0.02, 0.98, 49)])
    nx, ny = wall_normal(xi, sec)
    assert torch.isfinite(nx).all() and torch.isfinite(ny).all()
    assert nx[0] > 0  # the trailing-edge normal points aft, out of the section
    h = 1e-6
    xa, ya = sec.wall(xi[1:] - h, xp=torch)
    xb, yb = sec.wall(xi[1:] + h, xp=torch)
    tx, ty = xb - xa, yb - ya
    n = torch.sqrt(tx * tx + ty * ty)
    torch.testing.assert_close(nx[1:], -ty / n, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ny[1:], tx / n, rtol=1e-6, atol=1e-6)


def test_case_carries_no_flow_data():
    assert {f.name for f in dataclasses.fields(PinnCase)} == {"mach", "alpha_deg", "geometry",
                                                              "radius"}


def test_dataset_units():
    M = 0.5
    q = to_dataset_q(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0),
                     torch.tensor(1 / (GAMMA * M * M)), M)
    torch.testing.assert_close(q, torch.tensor([1.0, 1.0, 0.0, 1.0]))
