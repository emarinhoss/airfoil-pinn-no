"""Campaign settings, FROZEN by the 2026-10-01 stage S0 findings entry in docs/findings.md.
Starting values were spec 4.4's; S0 revised cauchy_eps (1e-7 -> 1e-9) and WORKERS (6 -> 8).
Changing a value after S1 starts requires a new dated entry in docs/findings.md and re-running
every case it affects.

The freeze was one edit: the S0-confirmed values into BASE and FROZEN = True, with its findings
entry, and the pinned values in tests/test_settings.py updated in the same commit. S1 refuses to
start on unfrozen settings, and on settings whose SU2 stop cannot satisfy our convergence rule.
"""

import math
from dataclasses import replace

from apno.geometry.ogrid import FINE
from apno.su2.config import Su2Settings

BASE = Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-9)
FINE_ITER_CAP = 20000
CONVERGENCE = {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}
WORKERS = 8
FROZEN = True

# |CL| ceiling over the input box: thin-airfoil lift with Prandtl-Glauert at its corner (alpha
# 6 deg, m 0.04 so alpha_0 = -4.15 deg, M 0.80), 2 pi (0.1047 + 0.0725) / 0.6 = 1.856. Euler lift
# at M 0.8 is below Prandtl-Glauert; S1's summary records the measured maximum beside this bound.
CL_BOUND = 1.86


def settings_for(spec):
    return replace(BASE, iter_cap=FINE_ITER_CAP) if spec.shape == FINE else BASE


def consistency_errors(base=None, convergence=None, cl_bound=None):
    """Why SU2's Cauchy stop could end a run before our rule can pass, or [] if it cannot.

    SU2 stops when the mean relative change of CL and CD over cauchy_elems iterations is below
    cauchy_eps, so at the stop CL may still vary by up to cauchy_eps * cauchy_elems * |CL| over
    that window; our rule needs at most force_tol over `window` iterations. Defaults are this
    module's current BASE, CONVERGENCE and CL_BOUND.
    """
    base = BASE if base is None else base
    convergence = CONVERGENCE if convergence is None else convergence
    cl_bound = CL_BOUND if cl_bound is None else cl_bound
    out = []
    if base.cauchy_elems != convergence["window"]:
        out.append(f"cauchy_elems {base.cauchy_elems} != convergence window "
                   f"{convergence['window']}")
    worst = base.cauchy_eps * base.cauchy_elems * cl_bound
    if not worst < convergence["force_tol"]:
        out.append(f"cauchy_eps * cauchy_elems * CL_BOUND = {worst:.2e} is not below force_tol "
                   f"{convergence['force_tol']:.0e}: SU2 may stop before CL is steady enough")
    if not (math.isfinite(base.cauchy_eps) and base.cauchy_eps > 0):
        out.append(f"cauchy_eps {base.cauchy_eps} is not a positive number")
    return out


def check_frozen():
    """Raise unless S0's freeze has been applied and the frozen settings are self-consistent."""
    if not FROZEN:
        raise RuntimeError("apno.settings is not FROZEN: apply the S0 freeze (BASE, FROZEN = True, "
                           "a dated docs/findings.md entry, tests/test_settings.py) before S1")
    errors = consistency_errors()
    if errors:
        raise RuntimeError("frozen settings are inconsistent: " + "; ".join(errors))
