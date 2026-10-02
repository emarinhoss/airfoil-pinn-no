"""What the models predict and what they are given (spec 6.1-6.2).

Targets, each of order 1 across the Mach range: u - cos(alpha), v - sin(alpha), Cp, and the scaled
density perturbation (rho - 1)/(M^2/2), which tends to Cp at low Mach. Features: radially compressed
node position about mid-chord, the ring index j/(NJ-1), M, cos(alpha), sin(alpha).
"""

import numpy as np

from apno.constants import GAMMA


def _ma(inputs):
    M = np.asarray(inputs)[:, 0][:, None, None]
    a = np.radians(np.asarray(inputs)[:, 1])[:, None, None]
    return M, a


def to_targets(q, inputs):
    M, a = _ma(inputs)
    rho, u, v, p = (q[..., k] for k in range(4))
    y = [u - np.cos(a), v - np.sin(a), (p - 1) / (0.5 * GAMMA * M * M), (rho - 1) / (0.5 * M * M)]
    return np.stack(y, 1)


def from_targets(y, inputs):
    M, a = _ma(inputs)
    u = y[:, 0] + np.cos(a)
    v = y[:, 1] + np.sin(a)
    p = 1 + y[:, 2] * 0.5 * GAMMA * M * M
    rho = 1 + y[:, 3] * 0.5 * M * M
    return np.stack([rho, u, v, p], -1)


class Standardiser:
    def fit(self, y):
        self.mean = y.mean(axis=(0, 2, 3), keepdims=True)
        self.std = y.std(axis=(0, 2, 3), keepdims=True)
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

    def state(self):
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.mean = np.array(st["mean"]).reshape(1, -1, 1, 1)
        s.std = np.array(st["std"]).reshape(1, -1, 1, 1)
        return s


def features(xy, inputs):
    n, ni, nj, _ = xy.shape
    dx, dy = xy[..., 0] - 0.5, xy[..., 1]
    r = np.hypot(dx, dy)  # > 0: mid-chord is inside every section
    k = np.log1p(r) / r
    ring = np.broadcast_to(np.linspace(0.0, 1.0, nj), (n, ni, nj))
    M, a = _ma(inputs)
    const = [np.broadcast_to(c, (n, ni, nj)) for c in (M, np.cos(a), np.sin(a))]
    return np.stack([dx * k, dy * k, ring, *const], 1).astype(np.float32)
