import numpy as np

from apno.constants import GAMMA
from apno.models.targets import Standardiser, features, from_targets, to_targets


def _q(n=3, ni=8, nj=5, seed=0):
    rng = np.random.default_rng(seed)
    return 1 + 0.1 * rng.standard_normal((n, ni, nj, 4)), np.column_stack(
        [rng.uniform(0.3, 0.8, n), rng.uniform(-2, 6, n), rng.uniform(.08, .16, n),
         rng.uniform(0, .04, n)])


def test_round_trip_exact():
    q, inp = _q()
    np.testing.assert_allclose(from_targets(to_targets(q, inp), inp), q, atol=1e-13)


def test_scaled_density_matches_cp_at_low_mach():
    M = 0.05
    rho = 1 + 1e-4 * np.random.default_rng(1).standard_normal((1, 4, 3))
    p = rho**GAMMA  # isentropic small perturbation
    q = np.stack([rho, np.ones_like(rho), np.zeros_like(rho), p], -1)
    y = to_targets(q, np.array([[M, 0.0, 0.12, 0.0]]))
    np.testing.assert_allclose(y[:, 3], y[:, 2], rtol=2e-3)


def test_standardiser():
    q, inp = _q(n=20)
    y = to_targets(q, inp)
    s = Standardiser().fit(y)
    z = s.transform(y)
    np.testing.assert_allclose(z.mean(axis=(0, 2, 3)), 0, atol=1e-12)
    np.testing.assert_allclose(Standardiser.from_state(s.state()).inverse(z), y, atol=1e-12)


def test_features_shape_and_ring():
    xy = np.random.default_rng(2).uniform(1, 2, (2, 8, 5, 2))
    f = features(xy, np.array([[0.5, 2.0, 0.12, 0.0], [0.7, -1.0, 0.1, 0.02]]))
    assert f.shape == (2, 6, 8, 5) and f.dtype == np.float32
    np.testing.assert_allclose(f[0, 2, 0], np.linspace(0, 1, 5), atol=1e-7)
    assert np.allclose(f[1, 3], 0.7) and np.allclose(f[1, 4], np.cos(np.radians(-1.0)))
