"""Stage S5's cases, truth and run scores, shared by scripts/s5_pinn.py and scripts/s5_score.py
(S5 design §3.3, §4, §5).

A run is scored at the 256x97 nodes against the truth (SU2 at 512x193 sampled every second node)
by the gate metric (RMAE of Cp and of local Mach, `apno.score.rmae`), the S4 scores
(`score_case`) and the collapse check: its largest wall speed below half the truth's, the
fluid-at-rest signature. The cylinder (A6a) is scored by the S3a criterion against potential flow,
whose wall speed peaks at 2 U_inf. Every score dict carries status, passed and collapsed.
"""

from pathlib import Path

import h5py
import numpy as np

from apno.data.build import load_case
from apno.data.sampling import case_specs, s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN, Cylinder, Section, grid
from apno.models.pinn import PinnCase
from apno.models.train_pinn import circle_cp_error
from apno.score.forces import wall_forces
from apno.score.metrics import score_case
from apno.score.rmae import rmae_cp_mach

GATE_RMAE = 0.01
STEP3_IN = ("test_in-0004", "test_in-0031", "test_in-0040")  # the supercritical showcase
STEP3_EXT = ("test_ext-0005", "test_ext-0009", "test_ext-0006")  # S1's rule, p = 0.1, 0.5, 0.9
CYLINDER = "cylinder"
ABLATION_AIRFOIL = "test_in-0022"
SU2_ROOT = Path("runs/su2")
H5 = {"truth": Path("data/hdf5/512x193.h5"), "floor": Path("data/hdf5/256x97.h5")}


def all_specs():
    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    specs.update({s.case_id: s for s in s5_validation_specs(TRAIN)})
    return specs


def pinn_case(case_id, specs):
    if case_id == CYLINDER:
        return PinnCase(0.1, 0.0, Cylinder())
    s = specs[case_id]
    return PinnCase(s.mach, s.alpha_deg, Section(s.t, s.m))


def truth_and_floor(case_id):
    """(truth at the 256x97 nodes, the 256x97 floor, the truth's max surface Mach), or
    (None, None, None) when either is missing (not run, not converged, not in the dataset)."""
    if case_id.startswith("s5-val-"):
        fine = {s.case_id: s for s in s5_validation_specs(FINE)}[case_id]
        coarse = {s.case_id: s for s in s5_validation_specs(TRAIN)}[case_id]
        t, f = load_case(SU2_ROOT, fine), load_case(SU2_ROOT, coarse)
        if t is None or f is None:
            return None, None, None
        return t["q"][::2, ::2], f["q"], float(t["max_surface_mach"])
    split = case_id.split("-")[0]
    got = {}
    for name, path in H5.items():
        if not path.exists():
            return None, None, None
        with h5py.File(path) as h:
            ids = list(h[split]["case_id"].asstr()[...])
            if case_id not in ids:
                return None, None, None
            k = ids.index(case_id)
            got[name] = (h[split]["q"][k], float(h[split]["max_surface_mach"][k]))
    return got["truth"][0][::2, ::2], got["floor"][0], got["truth"][1]


def wall_speed_max(q):
    return float(np.sqrt(q[:, 0, 1] ** 2 + q[:, 0, 2] ** 2).max())


def airfoil_score(q, case_id, specs):
    truth, floor, mmax = truth_and_floor(case_id)
    if truth is None:
        return {"status": "missing truth", "passed": False, "collapsed": None}
    s = specs[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    r = rmae_cp_mach(q, truth, X, Y, s.mach)
    regime = "supercritical" if mmax > 1.0 else "subcritical"
    return {"status": "ok", "rmae": r, "floor_rmae": rmae_cp_mach(floor, truth, X, Y, s.mach),
            "passed": r["cp"] <= GATE_RMAE and r["mach"] <= GATE_RMAE,
            "collapsed": wall_speed_max(q) < 0.5 * wall_speed_max(truth), "regime": regime,
            "s4": score_case(q, truth, X, Y, s.mach, s.alpha_deg, regime)}


def cylinder_score(q):
    err, cp = circle_cp_error(q, 0.1)
    X, Y = grid(Cylinder(), TRAIN)
    cl, cd, cm = wall_forces(X[:, 0], Y[:, 0], cp, 0.0)
    return {"status": "ok", "max_cp_error": err, "cl": cl, "cd": cd, "cm": cm,
            "passed": bool(err <= 0.1 and abs(cl) < 1e-2 and abs(cd) < 1e-2),
            "collapsed": wall_speed_max(q) < 0.5 * 2.0}


def score_run(q, case_id, specs):
    return cylinder_score(q) if case_id == CYLINDER else airfoil_score(q, case_id, specs)
