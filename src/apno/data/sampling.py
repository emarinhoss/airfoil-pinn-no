"""The design, fixed before any model is trained (spec 5.3). Scrambled Sobol, recorded seeds."""

import numpy as np
from scipy.stats import qmc

from apno.geometry.ogrid import FINE, TRAIN
from apno.su2.run import CaseSpec

BOX = {"mach": (0.30, 0.80), "alpha_deg": (-2.0, 6.0), "t": (0.08, 0.16), "m": (0.0, 0.04)}
EXTRAP_MACH = (0.80, 0.85)
SEEDS = {"pool": 20261001, "test_in": 20261002, "test_ext": 20261003}
PREFIXES = (64, 128, 256, 512, 896)
_LO = np.array([BOX[k][0] for k in BOX])
_HI = np.array([BOX[k][1] for k in BOX])


def _sobol(n, seed):
    return qmc.Sobol(d=4, scramble=True, seed=seed).random(n)


def design():
    pool = _LO + _sobol(1024, SEEDS["pool"]) * (_HI - _LO)
    test_in = _LO + _sobol(64, SEEDS["test_in"]) * (_HI - _LO)
    lo, hi = _LO.copy(), _HI.copy()
    lo[0], hi[0] = EXTRAP_MACH
    test_ext = lo + _sobol(16, SEEDS["test_ext"]) * (hi - lo)
    return {"train": pool[:896], "val": pool[896:], "test_in": test_in, "test_ext": test_ext}


def case_specs(split, shape):
    return [CaseSpec(f"{split}-{k:04d}", *map(float, row), shape)
            for k, row in enumerate(design()[split])]


def campaign_specs(which):
    if which == "train":
        return [s for split in ("train", "val", "test_in", "test_ext")
                for s in case_specs(split, TRAIN)]
    if which == "fine":
        return [s for split in ("test_in", "test_ext") for s in case_specs(split, FINE)]
    raise ValueError(which)
