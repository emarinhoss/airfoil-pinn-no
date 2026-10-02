"""Convergence is judged by our rule, not by SU2's flag (spec 4.4).

Converged iff the density residual dropped by at least `residual_drop` orders from the first
iteration AND CL and CD each varied by at most `force_tol` over the last `window` iterations. The
spike's coarse case shows why: SU2 said "not converged" (residual stuck 6.1 orders down, against an
absolute -11 target) while CL and CD had been steady to 9e-7 for 500 iterations.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("Inner_Iter", "rms[Rho]", "CL", "CD")


@dataclass(frozen=True)
class Verdict:
    converged: bool
    reason: str
    residual_drop: float
    cl_range: float
    cd_range: float
    iterations: int

    def as_dict(self):
        return asdict(self)


def read_history(path):
    """Columns by name, or None if there is no whole data row. A truncated or garbled row ends the
    history there: SU2 killed partway through a buffered write leaves a ragged last row, and that
    must become a non-converged verdict with a reason, never an exception in the batch driver.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))
    if len(rows) < 2:
        return None
    head = [h.strip().strip('"').strip() for h in rows[0]]
    body = []
    for r in rows[1:]:
        if not r:
            continue
        try:
            vals = [float(v) for v in r]
        except ValueError:
            break  # a row cut mid-write (SU2 killed): the history ends at the last whole row
        if len(vals) != len(head):
            break
        body.append(vals)
    if not body:
        return None
    body = np.array(body, dtype=float)
    return {h: body[:, k] for k, h in enumerate(head)}


def judge(history, residual_drop=5.0, window=500, force_tol=1e-6):
    nan = float("nan")
    if not history:
        return Verdict(False, "no history", nan, nan, nan, 0)
    missing = [c for c in COLUMNS if c not in history]
    if missing:
        return Verdict(False, f"history missing column {missing}", nan, nan, nan, 0)
    it, res, cl, cd = (history[c] for c in COLUMNS)
    n_iter = int(it[-1])
    if not all(np.isfinite(a).all() for a in (res, cl, cd)):
        return Verdict(False, "non-finite values in history", nan, nan, nan, n_iter)
    if it[-1] - it[0] < window:
        return Verdict(False, f"only {n_iter} iterations, fewer than the {window}-iteration window",
                       nan, nan, nan, n_iter)
    drop = float(res[0] - res[-1])
    sel = it >= it[-1] - window
    clr, cdr = float(np.ptp(cl[sel])), float(np.ptp(cd[sel]))
    reasons = []
    if drop < residual_drop:
        reasons.append(f"residual dropped {drop:.2f} orders < {residual_drop}")
    if clr > force_tol:
        reasons.append(f"CL varied {clr:.2e} > {force_tol:.0e} over the last {window}")
    if cdr > force_tol:
        reasons.append(f"CD varied {cdr:.2e} > {force_tol:.0e} over the last {window}")
    return Verdict(not reasons, "; ".join(reasons) or "converged", drop, clr, cdr, n_iter)
