"""Campaign settings. Starting values from spec 4.4; stage S0 confirms or revises them, then they
are FROZEN before S1. Changing a value after S1 starts requires a new dated entry in
docs/findings.md and re-running every case it affects.
"""

from dataclasses import replace

from apno.geometry.ogrid import FINE
from apno.su2.config import Su2Settings

BASE = Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-7)
FINE_ITER_CAP = 20000
CONVERGENCE = {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}
WORKERS = 6


def settings_for(spec):
    return replace(BASE, iter_cap=FINE_ITER_CAP) if spec.shape == FINE else BASE
