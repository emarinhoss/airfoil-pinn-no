import numpy as np
import pytest
import torch

from apno.physics.viscosity import (
    READING_A, READING_B, abs_jacobian, flux_jacobian_parts, sensor, shock_sensor,
    stagnation_sensor, viscous_term,
)

G = 1.4


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _states(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    rho = 0.5 + torch.rand(n, generator=g)
    p = 0.5 + torch.rand(n, generator=g)
    u = 2 * torch.rand(n, generator=g) - 1
    v = 2 * torch.rand(n, generator=g) - 1
    return rho, u, v, p


def _flux(w, kx, ky):
    r, m, n, e = w
    u, v = m / r, n / r
    p = (G - 1) * (e - 0.5 * r * (u * u + v * v))
    q = u * kx + v * ky
    return torch.stack([r * q, m * q + p * kx, n * q + p * ky, (e + p) * q])


@pytest.mark.parametrize("kx,ky", [(1.0, 0.0), (0.0, 1.0)])
def test_flux_jacobian_matches_autograd(kx, ky):
    rho, u, v, p = _states()
    m, q, _ = flux_jacobian_parts(rho, u, v, p, kx, ky)
    for k in range(len(rho)):
        e = p[k] / (G - 1) + 0.5 * rho[k] * (u[k] ** 2 + v[k] ** 2)
        w = torch.stack([rho[k], rho[k] * u[k], rho[k] * v[k], e])
        want = torch.autograd.functional.jacobian(lambda x: _flux(x, kx, ky), w)
        torch.testing.assert_close(q[k] * torch.eye(4) + m[k], want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("eps", [0.0, 0.25])
@pytest.mark.parametrize("kx,ky", [(1.0, 0.0), (0.0, 1.0)])
def test_abs_jacobian_equals_eigendecomposition(kx, ky, eps):
    rho, u, v, p = _states()
    got = abs_jacobian(rho, u, v, p, kx, ky, eps=eps).numpy()
    m, q, _ = flux_jacobian_parts(rho, u, v, p, kx, ky)
    for k in range(len(rho)):
        lam, t = np.linalg.eig((q[k] * torch.eye(4) + m[k]).numpy())
        want = (t @ np.diag(np.maximum(np.abs(lam), eps)) @ np.linalg.inv(t)).real
        np.testing.assert_allclose(got[k], want, rtol=1e-9, atol=1e-11)


def test_limiter_floors_every_eigenvalue():
    rho, u, v, p = _states()
    u = torch.zeros_like(u)  # q = 0 along x: the double eigenvalue u is 0 without the floor
    lam = np.linalg.eigvals(abs_jacobian(rho, u, v, p, 1.0, 0.0, eps=0.25).numpy())
    assert np.all(lam.real >= 0.25 - 1e-9)


def test_shock_sensor_positive_only_for_pressure_rising_along_the_flow():
    u, v = torch.ones(3), torch.zeros(3)
    s = shock_sensor(u, v, torch.tensor([2.0, -2.0, 0.0]), torch.zeros(3), READING_A)
    assert s[0] > 0.9 and s[1] == 0 and s[2] == 0


def test_stagnation_sensor_switches_off_at_large_momentum_gradients():
    s = stagnation_sensor(torch.tensor([0.0, 10.0]), torch.tensor([0.0, 10.0]), READING_A)
    assert s[0] > 0.99 and s[1] < 1e-6


def test_sensor_is_zero_on_uniform_flow_and_carries_no_gradient():
    u = torch.full((5,), 0.8, requires_grad=True)
    z = torch.zeros(5)
    s = sensor(u, z, z, z, z, z, z, z, READING_A)
    assert torch.all(s == 0) and not s.requires_grad


def test_readings_are_the_two_reads_of_table_vii():
    assert (READING_A.k1_shock, READING_A.k0_shock) == (4.0, 0.03)
    assert (READING_B.k1_shock, READING_B.k0_shock) == (0.03, 4.0)
    assert (READING_A.k0_stag, READING_A.k1_stag) == (3.0, 5.0)


@pytest.mark.parametrize("form", ["matrix", "scalar"])
def test_viscous_term_acts_on_density_and_energy_rows_only(form):
    rho, u, v, p = _states(8)
    g = torch.Generator().manual_seed(3)
    wxx, wyy = torch.randn(8, 4, generator=g), torch.randn(8, 4, generator=g)
    d = viscous_term(wxx, wyy, rho, u, v, p, torch.ones(8), 1.9e-3, form)
    assert torch.all(d[:, 1:3] == 0) and torch.any(d[:, 0] != 0) and torch.any(d[:, 3] != 0)
    assert torch.all(viscous_term(wxx, wyy, rho, u, v, p, torch.zeros(8), 1.9e-3, form) == 0)


def test_scalar_form_is_nu_s_spectral_radius_laplacian():
    rho, u, v, p = _states(6)
    s = torch.full((6,), 0.5)
    lam = torch.sqrt(1.4 * p / rho) + torch.sqrt(u * u + v * v)
    d = viscous_term(torch.ones(6, 4), 2 * torch.ones(6, 4), rho, u, v, p, s, 1e-3, "scalar")
    torch.testing.assert_close(d[:, 0], 1e-3 * 0.5 * lam * 3.0)


def test_unknown_form_is_refused():
    rho, u, v, p = _states(2)
    with pytest.raises(ValueError, match="unknown viscosity form"):
        viscous_term(torch.ones(2, 4), torch.ones(2, 4), rho, u, v, p, torch.ones(2), 1e-3, "x")


def test_non_physical_states_give_finite_viscosity():
    rho, p = torch.tensor([0.0, -0.5, 1.0]), torch.tensor([-1.0, 0.0, 1e-9])
    z = torch.zeros(3)
    d = viscous_term(torch.ones(3, 4), torch.ones(3, 4), rho, z, z, p, torch.ones(3), 1e-3)
    assert torch.isfinite(d).all()


def test_non_physical_states_give_a_dissipative_matrix():
    # Early-training states: a quarter have rho <= 0 and p <= 0. |A| must stay the absolute value of
    # the floored state's Jacobian (every eigenvalue >= eps), never sign-flipped (anti-dissipative).
    g = torch.Generator().manual_seed(5)
    rho, u, v, p = (0.1 * torch.randn(500, generator=g) for _ in range(4))
    for kx, ky in [(1.0, 0.0), (0.0, 1.0)]:
        lam = np.linalg.eigvals(abs_jacobian(rho, u, v, p, kx, ky).numpy())
        assert np.all(lam.real >= 0.25 - 1e-6)


def test_empty_batch_gives_empty_term():
    z = torch.zeros(0)
    d = viscous_term(torch.zeros(0, 4), torch.zeros(0, 4), z, z, z, z, z, 1e-3, "matrix")
    assert d.shape == (0, 4)
