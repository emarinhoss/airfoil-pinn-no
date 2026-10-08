"""The S5 gate metric (S5 design §3.3): relative mean absolute error on a uniform lattice, as in
Wassing, Langer & Bekemeyer (2025), Eq. 28.

Their box (-1, 1)^2 about the mid-chord of a section centred at the origin becomes x in [-0.5, 1.5],
y in [-1, 1] for this study's chord [0, 1]. Lattice points inside the section are dropped; both
fields are interpolated linearly on the grid's own cells, each quad (i, j)-(i+1, j)-(i+1, j+1)-
(i, j+1), periodic in i, split into two triangles, so nothing is interpolated across the body.
RMAE(f) = mean |f_hat - f| / (max f - min f), with the max and min of the truth over the lattice.
The truth must be finite at every node (a ValueError otherwise). A prediction that is non-finite
anywhere the lattice reads it scores inf; so, for both Cp and Mach, does one with rho <= 0 or p <= 0
at any node of a cell that holds lattice points.

Each quad is split along its diagonal (i, j)-(i+1, j+1) when both triangles are positively
oriented, else along (i+1, j)-(i, j+1) when both of those are (a non-convex quad). A quad that has
neither is folded (its radial edges cross), so its neighbours (i-1, j) and (i+1, j), which hold
those edges, overlap each other: all three are left out, their lattice points excluded like those
inside the section. With the folded quad alone left out, matplotlib's point location still refuses
the triangulation ("Triangulation is invalid"), as it refuses the fixed split.
Measured at 256x97 on 2026-10-07: every quad is convex for NACA 0012 (the validation cases), the six
step-3 cases and test_in-0022 (also on their 128x49 subgrids), and at the box corners (t, m) =
(0.08, 0), (0.16, 0), (0.16, 0.04). At the corner (0.08, 0.04) the trailing-edge quad (0, 0) is
folded (the lower-surface wall node i = 1 lies above the trailing edge) and quad (1, 0) is
non-convex, its first diagonal valid. Of the 1,104 dataset cases, 6 fold there (5 train, 1 val;
t <= 0.088, m >= 0.038); none needs the second diagonal.
"""

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.tri import LinearTriInterpolator, Triangulation

from apno.physics.flow import local_mach, pressure_coefficient

BOX = ((-0.5, 1.5), (-1.0, 1.0))
N_LATTICE = 401


def _quads(ni, nj):
    """Node ids (a, b, c, d) of quad (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1), periodic in i; the quads
    are numbered i (nj - 1) + j."""
    i, j = np.meshgrid(np.arange(ni), np.arange(nj - 1), indexing="ij")
    a = (i * nj + j).ravel()
    b = (((i + 1) % ni) * nj + j).ravel()
    return a, b, b + 1, a + 1


def _cells(X, Y):
    """(triangles (k, 3), the quad of each triangle (k,)), split as the module docstring says."""
    ni, nj = X.shape
    x, y = np.ravel(X), np.ravel(Y)

    def positive(p, q, r):
        return (x[q] - x[p]) * (y[r] - y[p]) - (x[r] - x[p]) * (y[q] - y[p]) > 0

    a, b, c, d = _quads(ni, nj)
    first = positive(a, b, c) & positive(a, c, d)
    second = ~first & positive(a, b, d) & positive(b, c, d)
    i, j = np.divmod(np.flatnonzero(~(first | second)), nj - 1)
    keep = np.ones_like(first)
    for di in (-1, 0, 1):  # a folded quad and its neighbours across its crossing radial edges
        keep[((i + di) % ni) * (nj - 1) + j] = False
    first, second = first & keep, second & keep
    k = np.arange(len(a))
    split = (((a, b, c), first), ((a, c, d), first), ((a, b, d), second), ((b, c, d), second))
    return (np.concatenate([np.stack(t, -1)[m] for t, m in split]),
            np.concatenate([k[m] for _, m in split]))


def triangulation(X, Y):
    return Triangulation(np.ravel(X), np.ravel(Y), _cells(X, Y)[0])


def lattice(X, Y, n=N_LATTICE, box=BOX):
    """(m, 2) points of the uniform n x n lattice over box that lie outside the section."""
    xs, ys = np.meshgrid(np.linspace(*box[0], n), np.linspace(*box[1], n), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    inside = MplPath(np.stack([X[:, 0], Y[:, 0]], 1)).contains_points(pts)
    return pts[~inside]


def on_lattice(field, tri, pts):
    """field (ni, nj) interpolated linearly at pts; NaN where a point is outside every cell."""
    v = LinearTriInterpolator(tri, np.asarray(field, dtype=float).ravel())(pts[:, 0], pts[:, 1])
    return np.ma.filled(v.astype(float), np.nan)


def _require_finite_truth(true):
    t = np.asarray(true, dtype=float)
    bad = ~np.isfinite(t.reshape(t.shape[0], t.shape[1], -1)).all(-1)
    if bad.any():
        raise ValueError(f"non-finite truth: {int(bad.sum())} of {bad.size} nodes; "
                         "the truth must be finite at every node")


def _rmae(pred, true, tri, pts):
    _require_finite_truth(true)
    fp, ft = on_lattice(pred, tri, pts), on_lattice(true, tri, pts)
    ok = np.isfinite(ft)  # the truth is finite, so False only at points outside every cell
    if not np.all(np.isfinite(fp[ok])):
        return float("inf")
    span = ft[ok].max() - ft[ok].min()
    return float(np.mean(np.abs(fp[ok] - ft[ok])) / span)


def rmae(pred, true, X, Y, n=N_LATTICE):
    return _rmae(pred, true, triangulation(X, Y), lattice(X, Y, n))


def rmae_cp_mach(pred_q, true_q, X, Y, mach, n=N_LATTICE):
    """{"cp": RMAE of Cp, "mach": RMAE of local Mach}; q (ni, nj, 4) in the dataset's units."""
    _require_finite_truth(true_q)
    nodes, quad = _cells(X, Y)
    tri, pts = Triangulation(np.ravel(X), np.ravel(Y), nodes), lattice(X, Y, n)
    k = tri.get_trifinder()(pts[:, 0], pts[:, 1])
    held = np.stack(_quads(*X.shape), -1)[np.unique(quad[k[k >= 0]])]  # cells holding points
    rho, p = (np.ravel(pred_q[..., s])[held] for s in (0, 3))
    if not np.all((rho > 0) & (p > 0)):
        return {"cp": float("inf"), "mach": float("inf")}
    cp = [pressure_coefficient(q[..., 3], mach) for q in (pred_q, true_q)]
    # past the check, a non-physical node (NaN or inf Mach) lies only in cells no point reads
    with np.errstate(invalid="ignore", divide="ignore"):
        m = [local_mach(q[..., 0], q[..., 1], q[..., 2], q[..., 3], mach) for q in (pred_q, true_q)]
    return {"cp": _rmae(cp[0], cp[1], tri, pts), "mach": _rmae(m[0], m[1], tri, pts)}
