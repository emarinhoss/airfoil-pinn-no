"""A NACA 4-digit section as one continuous closed curve of xi in [0, 1).

xi = 0 is the trailing edge. xi in [0, 0.5] runs along the lower surface to the leading edge and
xi in [0.5, 1) back along the upper surface (the spike's ordering, clockwise). With s = cos(pi xi),
the chord station is x = s**2 and sqrt(x) = |s|, so the curve has no square-root singularity at
the leading edge. The trailing edge is closed (x**4 coefficient -0.1036). Works on NumPy arrays and
torch tensors alike, so the mesh generator and the PINN share one definition of the geometry.
"""

import numpy as np

A = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)


def section(xi, t, m, p=0.4, xp=np):
    return section_s(xp.cos(np.pi * xi), t, m, p, xp)


def section_s(s, t, m, p=0.4, xp=np):
    """The same curve as a function of s = cos(pi xi) in [-1, 1]. d/dxi vanishes at the trailing
    edge (ds/dxi = -pi sin(pi xi) = 0 there); d/ds does not, so normals are taken in s."""
    x = s * s
    poly = A[1] + A[2] * x + A[3] * x * x + A[4] * x * x * x
    ys = -5.0 * t * (A[0] * s + s * xp.abs(s) * poly)  # signed half-thickness, negative = lower
    if m == 0:
        return x, ys
    fore = x < p
    yc = xp.where(
        fore,
        m / p**2 * (2 * p * x - x * x),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x),
    )
    dyc = xp.where(fore, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    k = 1.0 / xp.sqrt(1.0 + dyc * dyc)
    return x - ys * dyc * k, yc + ys * k
