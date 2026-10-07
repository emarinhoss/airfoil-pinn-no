"""The S5 gate metric (S5 design §3.3): relative mean absolute error on a uniform lattice, as in
Wassing, Langer & Bekemeyer (2025), Eq. 28.

Their box (-1, 1)^2 about the mid-chord of a section centred at the origin becomes x in [-0.5, 1.5],
y in [-1, 1] for this study's chord [0, 1]. Lattice points inside the section are dropped; both
fields are interpolated linearly on the grid's own cells, each quad (i, j)-(i+1, j)-(i+1, j+1)-
(i, j+1), periodic in i, split into two triangles, so nothing is interpolated across the body.
RMAE(f) = mean |f_hat - f| / (max f - min f), with the max and min of the truth over the lattice;
a prediction that is non-finite anywhere the truth is finite scores inf.
"""

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.tri import LinearTriInterpolator, Triangulation

from apno.physics.flow import local_mach, pressure_coefficient

BOX = ((-0.5, 1.5), (-1.0, 1.0))
N_LATTICE = 401


def triangulation(X, Y):
    ni, nj = X.shape
    i, j = np.meshgrid(np.arange(ni), np.arange(nj - 1), indexing="ij")
    a = i * nj + j
    b = ((i + 1) % ni) * nj + j
    tri = np.concatenate([np.stack([a, b, b + 1], -1).reshape(-1, 3),
                          np.stack([a, b + 1, a + 1], -1).reshape(-1, 3)])
    return Triangulation(X.ravel(), Y.ravel(), tri)


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


def rmae(pred, true, X, Y, n=N_LATTICE):
    tri, pts = triangulation(X, Y), lattice(X, Y, n)
    fp, ft = on_lattice(pred, tri, pts), on_lattice(true, tri, pts)
    ok = np.isfinite(ft)
    if not np.all(np.isfinite(fp[ok])):
        return float("inf")
    span = ft[ok].max() - ft[ok].min()
    return float(np.mean(np.abs(fp[ok] - ft[ok])) / span)


def rmae_cp_mach(pred_q, true_q, X, Y, mach, n=N_LATTICE):
    """{"cp": RMAE of Cp, "mach": RMAE of local Mach}; q (ni, nj, 4) in the dataset's units."""
    cp = [pressure_coefficient(q[..., 3], mach) for q in (pred_q, true_q)]
    with np.errstate(invalid="ignore"):  # a non-physical prediction gives NaN Mach: scored inf
        m = [local_mach(q[..., 0], q[..., 1], q[..., 2], q[..., 3], mach) for q in (pred_q, true_q)]
    return {"cp": rmae(cp[0], cp[1], X, Y, n), "mach": rmae(m[0], m[1], X, Y, n)}
