"""The one force integrator (spec 8.1), used for SU2, POD, FNO and PINN alike.

Vertex-lumped: each wall node carries half of each adjacent edge's outward normal. On the
2026-10-01 spike this reproduces SU2's CL, CD and CMz to 5e-11; the edge-midpoint trapezoid matches
CL and CD but misses CMz by 1.6e-5. Moments about (x_ref, y_ref), sign as SU2's CMz.
"""

import numpy as np


def vertex_normals(xw, yw):
    P = np.stack([xw, yw], 1)
    e = np.roll(P, -1, 0) - P
    twice_area = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    n_e = np.stack([-e[:, 1], e[:, 0]], 1)  # left normal: outward for a clockwise polygon
    if twice_area > 0:
        n_e = -n_e
    return 0.5 * (n_e + np.roll(n_e, 1, 0))


def wall_forces(xw, yw, cp, alpha_deg, x_ref=0.25, y_ref=0.0):
    f = -np.asarray(cp)[:, None] * vertex_normals(xw, yw)
    a = np.radians(alpha_deg)
    fx, fy = f[:, 0].sum(), f[:, 1].sum()
    cd = fx * np.cos(a) + fy * np.sin(a)
    cl = -fx * np.sin(a) + fy * np.cos(a)
    cm = np.sum((np.asarray(xw) - x_ref) * f[:, 1] - (np.asarray(yw) - y_ref) * f[:, 0])
    return float(cl), float(cd), float(cm)
