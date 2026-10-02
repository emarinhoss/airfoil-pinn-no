import numpy as np

from apno.models.pod import PodModel


def _family(inp, ni=6, nj=4):
    # fields linear in the inputs: thin-plate splines reproduce degree-1 polynomials exactly
    base = np.random.default_rng(0).standard_normal((4, 4, ni, nj))
    return np.einsum("nk,kcij->ncij", inp, base)


def test_reproduces_training_with_all_modes():
    inp = np.random.default_rng(1).uniform(size=(30, 4))
    y = _family(inp) + 0.01 * np.random.default_rng(2).standard_normal((30, 4, 6, 4))
    m = PodModel(n_modes=29, smoothing=0.0).fit(inp, y)
    np.testing.assert_allclose(m.predict(inp), y, atol=1e-8)


def test_interpolates_a_linear_family():
    rng = np.random.default_rng(3)
    inp, new = rng.uniform(size=(40, 4)), rng.uniform(size=(5, 4))
    m = PodModel(n_modes=4, smoothing=0.0).fit(inp, _family(inp))
    np.testing.assert_allclose(m.predict(new), _family(new), atol=1e-8)
