import json
from pathlib import Path

import numpy as np
import pytest

import apno.score.s5 as s5
from apno.geometry.ogrid import TRAIN, Cylinder, Section, grid

REPO = Path(__file__).parents[1]
DATA = REPO / "data" / "hdf5"


def _truth(case_id="s5-val-B"):
    s = s5.all_specs()[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    a = np.radians(s.alpha_deg)
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1 + 0.05 * np.sin(X)
    q[..., 1] = np.cos(a) + 0.1 * np.cos(Y)
    q[..., 2] = np.sin(a)
    q[..., 3] = 1 + 0.1 * np.cos(X) * np.exp(-Y ** 2)
    return q


def test_all_specs_hold_the_validation_and_comparison_cases():
    specs = s5.all_specs()
    want = {"s5-val-B", "s5-val-C", "s5-val-A", s5.ABLATION_AIRFOIL, *s5.STEP3_IN, *s5.STEP3_EXT}
    assert want <= set(specs)


def test_step3_in_range_cases_are_the_supercritical_showcase():
    show = json.loads((REPO / "results/s1/showcase.json").read_text())["cases"]
    assert tuple(c["case_id"] for c in show if c["regime"] == "supercritical") == s5.STEP3_IN


@pytest.mark.skipif(not (DATA / "512x193.h5").exists(), reason="dataset not built (data/hdf5)")
def test_step3_extrapolation_cases_follow_the_showcase_rule():
    import h5py
    with h5py.File(DATA / "512x193.h5") as f, h5py.File(DATA / "256x97.h5") as g:
        ids = list(f["test_ext"]["case_id"].asstr()[...])
        mm = f["test_ext"]["max_surface_mach"][...]
        both = set(g["test_ext"]["case_id"].asstr()[...])
    keep = [k for k, c in enumerate(ids) if c in both]
    ids, mm = [ids[k] for k in keep], mm[keep]
    order = np.argsort(mm, kind="stable")
    picked = tuple(ids[order[round(p * (len(ids) - 1))]] for p in (0.10, 0.50, 0.90))
    assert picked == s5.STEP3_EXT


def test_pinn_case_builds_sections_and_the_cylinder():
    specs = s5.all_specs()
    c = s5.pinn_case("s5-val-B", specs)
    assert (c.mach, c.alpha_deg, c.geometry) == (0.72, 3.0, Section(0.12, 0.0))
    cyl = s5.pinn_case(s5.CYLINDER, specs)
    assert (cyl.mach, cyl.alpha_deg, cyl.geometry) == (0.1, 0.0, Cylinder())


def test_airfoil_score_passes_the_truth_and_flags_a_collapse(monkeypatch):
    q = _truth()
    monkeypatch.setattr(s5, "truth_and_floor", lambda cid: (q, q.copy(), 0.9))
    ok = s5.airfoil_score(q, "s5-val-B", s5.all_specs())
    assert ok["status"] == "ok" and ok["passed"] and not ok["collapsed"]
    assert ok["rmae"] == {"cp": 0.0, "mach": 0.0} and ok["regime"] == "subcritical"
    rest = q.copy()
    rest[..., 1:3] = 0
    bad = s5.airfoil_score(rest, "s5-val-B", s5.all_specs())
    assert bad["collapsed"] and not bad["passed"]


def test_missing_truth_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(s5, "truth_and_floor", lambda cid: (None, None, None))
    got = s5.airfoil_score(np.zeros((256, 97, 4)), "test_ext-0005", s5.all_specs())
    assert got == {"status": "missing truth", "passed": False, "collapsed": None}


def test_cylinder_score_on_potential_flow():
    X, _ = grid(Cylinder(), TRAIN)
    th = -2 * np.pi * np.arange(256) / 256
    q = np.zeros(X.shape + (4,))
    q[..., 0], q[..., 1], q[..., 3] = 1.0, 1.0, 1.0
    q[:, 0, 3] = 1 + (1 - 4 * np.sin(th) ** 2) * 0.5 * 1.4 * 0.1 ** 2
    q[:, 0, 1], q[:, 0, 2] = 2 * np.abs(np.sin(th)), 0.0
    sc = s5.cylinder_score(q)
    assert sc["passed"] and not sc["collapsed"] and sc["max_cp_error"] < 1e-9
    q[:, 0, 1] = 0.0
    assert s5.cylinder_score(q)["collapsed"]
