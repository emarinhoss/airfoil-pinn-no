import numpy as np
import pytest
import torch

from apno.geometry.naca import section
from apno.geometry.ogrid import (
    COARSE, FAR_RADIUS, FINE, TRAIN, Cylinder, Section, cell_areas, grid, map_xy, stretch,
    stretch_ratio, write_su2,
)

CORNERS = [Section(t, m) for t in (0.08, 0.16) for m in (0.0, 0.04)]


@pytest.mark.parametrize("sec", CORNERS, ids=lambda s: f"t{s.t}-m{s.m}")
@pytest.mark.parametrize("shape", [FINE, TRAIN, COARSE])
def test_every_cell_positive_at_box_corners(sec, shape):
    X, Y = grid(sec, shape)
    assert cell_areas(X, Y).min() > 0


@pytest.mark.parametrize("radius", [50.0, 200.0])
def test_positive_at_other_radii(radius):
    for sec in CORNERS:
        X, Y = grid(sec, TRAIN, radius)
        assert cell_areas(X, Y).min() > 0


def test_grids_nest_exactly():
    sec = Section(0.12, 0.02)
    Xf, Yf = grid(sec, FINE)
    Xt, Yt = grid(sec, TRAIN)
    Xc, Yc = grid(sec, COARSE)
    assert np.array_equal(Xf[::2, ::2], Xt) and np.array_equal(Yf[::2, ::2], Yt)
    assert np.array_equal(Xf[::4, ::4], Xc) and np.array_equal(Yf[::4, ::4], Yc)


def test_wall_on_section_and_outer_circle():
    sec = Section(0.10, 0.03)
    X, Y = grid(sec, TRAIN)
    xw, yw = section(np.arange(256) / 256, 0.10, 0.03)
    np.testing.assert_array_equal(X[:, 0], xw)
    np.testing.assert_array_equal(Y[:, 0], yw)
    np.testing.assert_allclose(np.hypot(X[:, -1] - 0.5, Y[:, -1]), FAR_RADIUS, rtol=1e-13)


def test_training_grid_first_spacing():
    r = stretch_ratio()
    assert stretch(1 / 96, r) * FAR_RADIUS == pytest.approx(2.0e-3, rel=1e-9)


def test_symmetric_section_gives_mirror_grid():
    X, Y = grid(Section(0.12, 0.0), TRAIN)
    mirror = (-np.arange(256)) % 256
    np.testing.assert_allclose(X[mirror], X, atol=1e-12)
    np.testing.assert_allclose(Y[mirror], -Y, atol=1e-12)


def test_torch_map_equals_numpy_map():
    sec = Section(0.12, 0.04)
    xi = np.random.default_rng(0).random(200)
    eta = np.random.default_rng(1).random(200)
    xn, yn = map_xy(xi, eta, sec)
    xt, yt = map_xy(torch.tensor(xi), torch.tensor(eta), sec, xp=torch)
    np.testing.assert_allclose(xn, xt.numpy(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(yn, yt.numpy(), rtol=1e-13, atol=1e-13)


def test_cylinder_grid_is_polar():
    X, Y = grid(Cylinder(), COARSE)
    np.testing.assert_allclose(np.hypot(X[:, 0] - 0.5, Y[:, 0]), 0.5, rtol=1e-14)
    assert cell_areas(X, Y).min() > 0


def test_write_su2_layout(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    write_su2(tmp_path / "mesh.su2", X, Y)
    lines = (tmp_path / "mesh.su2").read_text().splitlines()
    assert lines[0] == "NDIME= 2" and lines[1] == f"NELEM= {128 * 48}"
    k = lines.index(f"NPOIN= {128 * 49}")
    i, j = 5, 3
    x, y, nid = lines[k + 1 + j * 128 + i].split()
    assert int(nid) == j * 128 + i
    assert float(x) == pytest.approx(X[i, j], abs=1e-14)
    assert float(y) == pytest.approx(Y[i, j], abs=1e-14)
    assert "MARKER_TAG= airfoil" in lines and "MARKER_TAG= farfield" in lines


def test_write_su2_refuses_folded_grid(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    X[:, 1], X[:, 2] = X[:, 2].copy(), X[:, 1].copy()
    Y[:, 1], Y[:, 2] = Y[:, 2].copy(), Y[:, 1].copy()
    with pytest.raises(ValueError, match="non-positive"):
        write_su2(tmp_path / "bad.su2", X, Y)


def test_write_su2_refuses_nan_grid(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-positive"):
        write_su2(tmp_path / "nan.su2", X, Y)
    assert not (tmp_path / "nan.su2").exists()
