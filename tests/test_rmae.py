import numpy as np
import pytest

from apno.geometry.ogrid import Section, grid
from apno.score.rmae import lattice, on_lattice, rmae, rmae_cp_mach, triangulation

X, Y = grid(Section(0.12, 0.0), (64, 25))  # a small grid keeps these fast


def _field():
    return np.sin(X) + 0.5 * np.cos(2 * Y)


def test_identical_fields_have_zero_error():
    f = _field()
    assert rmae(f, f, X, Y, n=101) == 0.0


def test_a_constant_offset_gives_the_offset_over_the_span():
    f = _field()
    ft = on_lattice(f, triangulation(X, Y), lattice(X, Y, 101))
    span = np.nanmax(ft) - np.nanmin(ft)
    assert rmae(f + 0.01, f, X, Y, n=101) == pytest.approx(0.01 / span, rel=1e-9)


def test_lattice_points_inside_the_section_are_dropped():
    pts = lattice(X, Y, 101)
    assert len(pts) < 101 * 101
    on_chord = (np.abs(pts[:, 1]) < 1e-9) & (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95)
    assert not on_chord.any()


def test_a_non_finite_prediction_fails_the_metric():
    f = _field()
    g = f.copy()
    g[10, :] = np.nan  # a whole radial line: the lattice certainly meets it
    assert rmae(g, f, X, Y, n=101) == float("inf")


def test_cp_and_mach_from_dataset_q():
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1.0
    q[..., 1] = 1.0 + 0.05 * np.cos(Y)
    q[..., 3] = 1.0 + 0.1 * np.sin(X)
    assert rmae_cp_mach(q, q, X, Y, 0.7, n=51) == {"cp": 0.0, "mach": 0.0}
    p = q.copy()
    p[..., 3] += 0.01
    assert rmae_cp_mach(p, q, X, Y, 0.7, n=51)["cp"] > 0
