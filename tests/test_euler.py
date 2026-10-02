import numpy as np
import pytest
import sympy as sp
import torch

from apno.geometry.ogrid import Section
from apno.physics.euler import metrics, residual

GEOM = Section(0.12, 0.03)


@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only; a module-level set_default_dtype leaks into every test."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pts(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.01 + 0.98 * torch.rand(n, generator=g), 1e-3 + (1 - 1e-3) * torch.rand(n, generator=g)


def test_free_stream_preserved_on_the_curved_grid():
    xi, eta = _pts()

    def state(xi, eta):
        z = 0 * xi + 0 * eta
        return 1 + z, 0.9 + z, 0.1 + z, 2.5 + z

    r = residual(xi, eta, state, GEOM)
    assert r.abs().max() < 1e-9


def _manufactured():
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 10) * sp.sin(x) * sp.cos(y)
    u = 1 + sp.Rational(1, 10) * sp.cos(x) * sp.sin(2 * y)
    v = sp.Rational(1, 10) * sp.sin(y) * sp.cos(x)
    p = 3 + sp.Rational(1, 10) * sp.cos(x + y)
    F = [rho * u, rho * u**2 + p, rho * u * v]
    G = [rho * v, rho * u * v, rho * v**2 + p]
    div = [sp.lambdify((x, y), sp.diff(F[k], x) + sp.diff(G[k], y)) for k in range(3)]
    cons = (rho, rho * u, rho * v)
    lap = [sp.lambdify((x, y), sp.diff(q, x, 2) + sp.diff(q, y, 2)) for q in cons]
    fields = [sp.lambdify((x, y), f, "math") for f in (rho, u, v, p)]
    return div, lap, fields


def test_manufactured_solution():
    div, lap, _ = _manufactured()
    xi, eta = _pts(200, 1)
    eta = eta * 0.05  # keep x, y within a few chords so the fields stay O(1)

    def state(xi, eta):
        x, y, *_ = metrics(xi, eta, GEOM)
        rho = 1 + 0.1 * torch.sin(x) * torch.cos(y)
        u = 1 + 0.1 * torch.cos(x) * torch.sin(2 * y)
        v = 0.1 * torch.sin(y) * torch.cos(x)
        p = 3 + 0.1 * torch.cos(x + y)
        return rho, u, v, p

    nu = 1e-2
    r = residual(xi, eta, state, GEOM, viscosity=nu).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    want = np.stack([div[k](x, y) - nu * lap[k](x, y) for k in range(3)], 1)
    np.testing.assert_allclose(r, want, rtol=1e-8, atol=1e-8)
