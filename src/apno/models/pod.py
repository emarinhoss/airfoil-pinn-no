"""The POD baseline (spec 6.5): a linear basis of the standardised training fields, with each mode's
weight interpolated over the four inputs by a thin-plate-spline RBF. Possible only because every
case is the same array shape. Inputs are scaled to the unit box first.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from apno.data.sampling import BOX

_LO = np.array([BOX[k][0] for k in BOX])
_SPAN = np.array([BOX[k][1] - BOX[k][0] for k in BOX])


class PodModel:
    def __init__(self, n_modes, smoothing=0.0):
        self.n_modes, self.smoothing = n_modes, smoothing

    def fit(self, inputs, y):
        n = y.shape[0]
        self.shape = y.shape[1:]
        Y = y.reshape(n, -1)
        self.mean = Y.mean(0)
        Yc = Y - self.mean
        lam, V = np.linalg.eigh(Yc @ Yc.T)  # snapshot method: n x n, not D x D
        lam, V = lam[::-1], V[:, ::-1]
        k = min(self.n_modes, int(np.sum(lam > 1e-12 * lam[0])))
        self.modes = (Yc.T @ V[:, :k]) / np.sqrt(lam[:k])  # (D, k), orthonormal columns
        coeffs = Yc @ self.modes
        self.rbf = RBFInterpolator((inputs - _LO) / _SPAN, coeffs, kernel="thin_plate_spline",
                                   smoothing=self.smoothing)
        return self

    def predict(self, inputs):
        c = self.rbf((np.asarray(inputs) - _LO) / _SPAN)
        return (c @ self.modes.T + self.mean).reshape(len(inputs), *self.shape)
