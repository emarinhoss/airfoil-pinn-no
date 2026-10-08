import numpy as np
import pytest

from apno.geometry.ogrid import TRAIN, Section, grid
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


def _q():
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1.0
    q[..., 1] = 1.0 + 0.05 * np.cos(Y)
    q[..., 3] = 1.0 + 0.1 * np.sin(X)
    return q


def _ring(ni=16, nj=6):
    """A polar O-grid about mid-chord, ordered like the section's: clockwise in i, outward in j."""
    th = -2.0 * np.pi * np.arange(ni) / ni
    r = 0.3 * 10.0 ** (np.arange(nj) / (nj - 1))
    return 0.5 + r * np.cos(th)[:, None], r * np.sin(th)[:, None]


def _signed_areas(tri):
    """Read before any interpolation: matplotlib reorients inverted triangles in place."""
    x, y, t = tri.x, tri.y, tri.triangles
    return 0.5 * ((x[t[:, 1]] - x[t[:, 0]]) * (y[t[:, 2]] - y[t[:, 0]])
                  - (x[t[:, 2]] - x[t[:, 0]]) * (y[t[:, 1]] - y[t[:, 0]]))


def _quad_nodes(i, j, ni=16, nj=6):
    return {i * nj + j, ((i + 1) % ni) * nj + j, ((i + 1) % ni) * nj + j + 1, i * nj + j + 1}


def test_a_non_convex_quad_is_split_along_its_other_diagonal():
    Xr, Yr = _ring()
    a, b, c, d = (4, 2), (5, 2), (5, 3), (4, 3)
    Xr[b] = (Xr[a] + Xr[c] + Xr[d]) / 3  # b inside triangle a c d: the a-c split inverts a b c
    Yr[b] = (Yr[a] + Yr[c] + Yr[d]) / 3
    tri = triangulation(Xr, Yr)
    assert (_signed_areas(tri) > 0).all()
    assert len(tri.triangles) == 2 * 16 * 5
    ia, ib, ic, id_ = (i * 6 + j for i, j in (a, b, c, d))
    got = {frozenset(t) for t in tri.triangles.tolist() if set(t) <= {ia, ib, ic, id_}}
    assert got == {frozenset((ia, ib, id_)), frozenset((ib, ic, id_))}


def test_a_folded_quad_and_its_neighbours_across_the_fold_are_left_out():
    Xr, Yr = _ring()
    # wall node 1 moved across radial line 0, so radial segments 0 and 1 cross at (0.872, 0) inside
    # quad (0, 0), as at the trailing edge of a thin, strongly cambered section
    Xr[1, 0], Yr[1, 0] = 0.85, 0.06
    tri = triangulation(Xr, Yr)
    assert (_signed_areas(tri) > 0).all()
    assert len(tri.triangles) == 2 * 16 * 5 - 2 * 3
    for quad in ((15, 0), (0, 0), (1, 0)):
        assert not any(set(t) <= _quad_nodes(*quad) for t in tri.triangles.tolist())
    beyond_the_crossing = np.array([[0.93, -0.06]])  # in quad (0, 0) only, before it is left out
    assert np.isnan(on_lattice(Xr + Yr, tri, beyond_the_crossing)).all()  # excluded


@pytest.mark.parametrize("t, m, left_out", [(0.12, 0.0, 0), (0.08, 0.04, 3)])  # 0012, a corner
def test_on_the_256x97_grid_only_a_folded_trailing_edge_is_left_out(t, m, left_out):
    Xg, Yg = grid(Section(t, m), TRAIN)
    tri = triangulation(Xg, Yg)
    assert (_signed_areas(tri) > 0).all()
    assert len(tri.triangles) == 2 * 256 * 96 - 2 * left_out
    f = np.sin(Xg) + 0.5 * np.cos(2 * Yg)
    assert rmae(f, f, Xg, Yg, n=101) == 0.0  # the triangulation is valid: points can be located


def test_a_non_finite_truth_is_refused():
    f = _field()
    t = f.copy()
    t[3, 4], t[20, 7] = np.nan, np.inf
    with pytest.raises(ValueError, match="non-finite truth: 2 of"):
        rmae(f, t, X, Y, n=51)
    q = _q()
    tq = q.copy()
    tq[5, 6, 1] = np.nan  # Cp is finite there, Mach is not
    with pytest.raises(ValueError, match="non-finite truth: 1 of"):
        rmae_cp_mach(q, tq, X, Y, 0.7, n=51)


@pytest.mark.parametrize("rho, p", [(-1.0, -1.0), (0.0, 1.0), (1.0, 0.0), (1.0, -0.5)])
def test_a_non_physical_prediction_scores_inf_for_cp_and_mach(rho, p):
    q = _q()
    bad = q.copy()
    bad[10, :, 0], bad[10, :, 3] = rho, p  # a whole radial line: the lattice certainly meets it
    assert rmae_cp_mach(bad, q, X, Y, 0.7, n=51) == {"cp": float("inf"), "mach": float("inf")}


def test_a_non_physical_node_of_no_cell_holding_lattice_points_does_not_count():
    q = _q()
    bad = q.copy()
    assert np.hypot(X[:, -2] - 0.5, Y[:, -2]).min() > 10  # the last ring of cells is off the box
    bad[:, -1, 0], bad[:, -1, 3] = -1.0, -1.0
    assert rmae_cp_mach(bad, q, X, Y, 0.7, n=51) == {"cp": 0.0, "mach": 0.0}
