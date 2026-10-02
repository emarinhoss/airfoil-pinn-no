"""The O-grid: one continuous map (xi, eta) -> (x, y), sampled into three nested grids.

The wall (eta = 0) is the section; the outer boundary (eta = 1) is a circle of radius R about
mid-chord. The grid line of constant xi joins the wall point at xi to the circle point at polar
angle -2*pi*xi. This is NOT the spike's construction, which put the far point at the wall point's
angle about mid-chord: that folds the mesh (9,792 inverted cells for t = 0.08, m = 0.04 at 256x97),
measured on 2026-10-01. The stretching is defined on the fine grid, so the coarser grids are exact
subsets of it.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apno.geometry.naca import section, section_s

FINE = (512, 193)
TRAIN = (256, 97)
COARSE = (128, 49)
FAR_RADIUS = 100.0
WALL_SPACING = 2.0e-3  # first radial spacing of the 256x97 grid, chords
N_FINE = FINE[1] - 1


def level_name(shape):
    return f"{shape[0]}x{shape[1]}"


@dataclass(frozen=True)
class Section:
    t: float
    m: float
    p: float = 0.4

    def wall(self, xi, xp=np):
        return section(xi, self.t, self.m, self.p, xp)

    def wall_s(self, s, xp=np):
        return section_s(s, self.t, self.m, self.p, xp)


@dataclass(frozen=True)
class Cylinder:
    """A circle about mid-chord, ordered like the section (clockwise, xi = 0 at (0.5 + r, 0))."""

    radius: float = 0.5

    def wall(self, xi, xp=np):
        th = -2.0 * np.pi * xi
        return 0.5 + self.radius * xp.cos(th), self.radius * xp.sin(th)


def stretch_ratio(radius=FAR_RADIUS, wall_spacing=WALL_SPACING, n_fine=N_FINE):
    """Geometric ratio r of the fine grid, such that its first TWO intervals total wall_spacing."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        first_two = radius * (r * r - 1.0) / (r**n_fine - 1.0)
        lo, hi = (lo, r) if first_two < wall_spacing else (r, hi)
    return 0.5 * (lo + hi)


def stretch(eta, ratio, n_fine=N_FINE, xp=np):
    return (xp.exp(n_fine * eta * math.log(ratio)) - 1.0) / (ratio**n_fine - 1.0)


def map_xy(xi, eta, geom, radius=FAR_RADIUS, ratio=None, xp=np):
    ratio = stretch_ratio(radius) if ratio is None else ratio
    xw, yw = geom.wall(xi, xp)
    th = -2.0 * np.pi * xi
    xf, yf = 0.5 + radius * xp.cos(th), radius * xp.sin(th)
    s = stretch(eta, ratio, xp=xp)
    return xw + s * (xf - xw), yw + s * (yf - yw)


def grid(geom, shape=TRAIN, radius=FAR_RADIUS):
    ni, nj = shape
    xi = np.arange(ni) / ni
    eta = np.arange(nj) / (nj - 1)
    return map_xy(xi[:, None], eta[None, :], geom, radius)


def cell_areas(X, Y):
    """Signed area of quad (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1), periodic in i."""
    x0, y0 = X[:, :-1], Y[:, :-1]
    x1, y1 = np.roll(X, -1, 0)[:, :-1], np.roll(Y, -1, 0)[:, :-1]
    x2, y2 = np.roll(X, -1, 0)[:, 1:], np.roll(Y, -1, 0)[:, 1:]
    x3, y3 = X[:, 1:], Y[:, 1:]
    return 0.5 * (
        (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3)
    )


def write_su2(path, X, Y):
    """Write the grid in SU2 native format. Node id = j*NI + i; markers 'airfoil', 'farfield'."""
    if cell_areas(X, Y).min() <= 0:
        raise ValueError("grid has non-positive cells; refusing to write it")
    ni, nj = X.shape

    def nid(i, j):
        return j * ni + (i % ni)

    out = ["NDIME= 2", f"NELEM= {ni * (nj - 1)}"]
    e = 0
    for j in range(nj - 1):
        for i in range(ni):
            out.append(f"9 {nid(i, j)} {nid(i + 1, j)} {nid(i + 1, j + 1)} {nid(i, j + 1)} {e}")
            e += 1
    out.append(f"NPOIN= {ni * nj}")
    for j in range(nj):
        for i in range(ni):
            out.append(f"{X[i, j]:.17e} {Y[i, j]:.17e} {nid(i, j)}")
    out.append("NMARK= 2")
    for tag, j in (("airfoil", 0), ("farfield", nj - 1)):
        out += [f"MARKER_TAG= {tag}", f"MARKER_ELEMS= {ni}"]
        out += [f"3 {nid(i, j)} {nid(i + 1, j)}" for i in range(ni)]
    Path(path).write_text("\n".join(out) + "\n")
