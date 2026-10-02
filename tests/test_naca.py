import numpy as np
import pytest
import torch

from apno.geometry.naca import section


def test_trailing_edge_closed_and_leading_edge_at_origin():
    for t, m in ((0.12, 0.0), (0.08, 0.04)):
        x, y = section(np.array([0.0, 0.5]), t, m)
        assert x[0] == pytest.approx(1.0, abs=1e-14) and y[0] == pytest.approx(0.0, abs=1e-14)
        assert x[1] == pytest.approx(0.0, abs=1e-14) and y[1] == pytest.approx(0.0, abs=1e-14)


def test_symmetric_section_is_mirror_symmetric():
    xi = np.linspace(0.01, 0.49, 50)
    x1, y1 = section(xi, 0.12, 0.0)
    x2, y2 = section(1.0 - xi, 0.12, 0.0)
    np.testing.assert_allclose(x1, x2, atol=1e-14)
    np.testing.assert_allclose(y1, -y2, atol=1e-14)
    assert np.all(y1 < 0)  # lower surface first


def test_thickness_is_t():
    xi = np.linspace(0.5, 1.0, 20001)[:-1]
    _, y = section(xi, 0.12, 0.0)
    assert 2 * y.max() == pytest.approx(0.12, rel=5e-3)


def test_pair_midpoint_is_the_camber_line():
    # points at xi and 1 - xi share s**2, so they straddle the same camber-line point
    s = np.sqrt(0.4)
    xi = np.array([np.arccos(s) / np.pi])
    xl, yl = section(xi, 0.10, 0.04)
    xu, yu = section(1.0 - xi, 0.10, 0.04)
    assert 0.5 * (xl[0] + xu[0]) == pytest.approx(0.4, abs=1e-12)
    assert 0.5 * (yl[0] + yu[0]) == pytest.approx(0.04, abs=1e-12)


def test_numpy_and_torch_agree():
    xi = np.linspace(0.0, 1.0, 101)[:-1]
    xn, yn = section(xi, 0.14, 0.03)
    xt, yt = section(torch.tensor(xi, dtype=torch.float64), 0.14, 0.03, xp=torch)
    np.testing.assert_allclose(xn, xt.numpy(), atol=1e-14)
    np.testing.assert_allclose(yn, yt.numpy(), atol=1e-14)
