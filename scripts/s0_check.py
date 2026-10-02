"""Stage S0: check SU2's accuracy and cost before the campaign (spec 4.6). Writes results/s0/.

Exits 1 if the force integrator fails to reproduce SU2 on any converged case, or if no case
converged (scorer check not performed): if the scorer cannot score SU2 against SU2, it cannot score
anything. Exits 2 if the scorer check passed but any case is NOT CONVERGED (S0 incomplete;
summary.json is still written). Exit 0 means S0 is complete. Run with `set -o pipefail` (or read
${PIPESTATUS[0]}) when piping through tee. Usage:
    SU2_BIN_DIR=... python scripts/s0_check.py [--workers 6]
"""

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from apno.geometry.ogrid import COARSE, FINE, TRAIN, Section, grid
from apno.physics.flow import nondimensionalise, pressure_coefficient
from apno.score.forces import wall_forces
from apno.settings import CONVERGENCE, settings_for
from apno.su2.restart import on_grid, read_restart
from apno.su2.run import CaseSpec, case_dir, run_campaign

CASES = {
    "sym": (0.50, 0.0, 0.12, 0.00),
    "sub0": (0.50, 0.0, 0.12, 0.02),
    "sub2": (0.50, 2.0, 0.12, 0.02),
    "super": (0.75, 2.0, 0.12, 0.02),
    "anchor": (0.80, 1.25, 0.12, 0.00),
}
ROOT = Path("runs/s0")
OUT = Path("results/s0")


def specs():
    out = []
    for name, (M, a, t, m) in CASES.items():
        for shape in (COARSE, TRAIN, FINE):
            out.append((CaseSpec(f"{name}-mg0", M, a, t, m, shape), 0))
        out.append((CaseSpec(f"{name}-mg3", M, a, t, m, TRAIN), 3))
    for name in ("sub2", "anchor"):
        M, a, t, m = CASES[name]
        for radius in (50.0, 200.0):
            out.append((CaseSpec(f"{name}-R{radius:.0f}", M, a, t, m, TRAIN, radius), 0))
    return out


def scorer_forces(spec):
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    f = on_grid(read_restart(case_dir(ROOT, spec) / "restart.dat"), X, Y)
    q = nondimensionalise(f["Density"], f["Momentum_x"], f["Momentum_y"], f["Energy"], spec.mach)
    cp = pressure_coefficient(q[3], spec.mach)
    return wall_forces(X[:, 0], Y[:, 0], cp[:, 0], spec.alpha_deg), X, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    su2_bin = Path(os.environ["SU2_BIN_DIR"])
    pairs = specs()
    mg = {s.case_id + s.level: m for s, m in pairs}
    res = run_campaign([s for s, _ in pairs], ROOT, su2_bin,
                       lambda s: replace(settings_for(s), mglevel=mg[s.case_id + s.level]),
                       args.workers, CONVERGENCE)
    by = {(r["case_id"], r["level"]): r for r in res}

    summary = {"runs": res, "scorer_check": [], "mesh": {}, "multigrid": {}, "radius": {},
               "known_answers": {}}
    worst = 0.0
    surf = {}
    for spec, _ in pairs:
        r = by.get((spec.case_id, spec.level))
        if not r or not r["converged"]:
            continue
        (cl, cd, cm), X, cp = scorer_forces(spec)
        d = max(abs(cl - r["cl"]), abs(cd - r["cd"]), abs(cm - r["cm"]))
        worst = max(worst, d)
        summary["scorer_check"].append({"case": spec.case_id, "level": spec.level, "max_diff": d})
        surf[(spec.case_id, spec.level)] = cp[:, 0]

    for name in CASES:
        rows = [by.get((f"{name}-mg0", lv)) for lv in ("128x49", "256x97", "512x193")]
        if not all(r and r["converged"] for r in rows):
            summary["mesh"][name] = "not all three levels converged"
            continue
        entry = {}
        for k in ("cl", "cd", "cm"):
            c, t, f = (r[k] for r in rows)
            ratio = abs(c - t) / abs(t - f) if t != f else math.inf
            entry[k] = {"128x49": c, "256x97": t, "512x193": f, "train_minus_fine": t - f,
                        "observed_order": math.log2(ratio) if 0 < ratio < math.inf else None}
        cp_t, cp_f = surf[(f"{name}-mg0", "256x97")], surf[(f"{name}-mg0", "512x193")][::2]
        entry["cp_train_vs_fine"] = {"l2": float(np.sqrt(np.mean((cp_t - cp_f) ** 2))),
                                     "max": float(np.abs(cp_t - cp_f).max())}
        summary["mesh"][name] = entry
        a, b = by.get((f"{name}-mg0", "256x97")), by.get((f"{name}-mg3", "256x97"))
        if a and b:
            summary["multigrid"][name] = {
                "mg0": {k: a[k] for k in ("wall_s", "cpu_s", "iterations", "converged", "cl")},
                "mg3": {k: b[k] for k in ("wall_s", "cpu_s", "iterations", "converged", "cl")}}
    def ok(case_id, level):
        r = by.get((case_id, level))
        return r if r and r["converged"] else None

    for name in ("sub2", "anchor"):
        row = {R: (ok(f"{name}-R{R}", "256x97") or {}).get("cl", "not converged")
               for R in ("50", "200")}
        row["100"] = (ok(f"{name}-mg0", "256x97") or {}).get("cl", "not converged")
        summary["radius"][name] = row
    ka = summary["known_answers"]
    NC = "not converged"
    r = ok("sym-mg0", "512x193")
    ka["sym_abs_cl_fine"] = abs(r["cl"]) if r else NC
    ka["subcritical_cd_counts_fine"] = {
        n: (1e4 * ok(f"{n}-mg0", "512x193")["cd"] if ok(f"{n}-mg0", "512x193") else NC)
        for n in ("sym", "sub0", "sub2")}
    M = CASES["sub2"][0]
    s0, s2 = ok("sub0-mg0", "512x193"), ok("sub2-mg0", "512x193")
    if s0 and s2:
        slope = (s2["cl"] - s0["cl"]) / math.radians(2)
        ka["lift_slope_ratio_to_prandtl_glauert"] = slope / (2 * math.pi / math.sqrt(1 - M * M))
    else:
        ka["lift_slope_ratio_to_prandtl_glauert"] = NC
    r = ok("anchor-mg0", "512x193")
    ka["anchor_cl_fine"] = r["cl"] if r else NC
    ka["vassberg_jameson_2010_cl"] = "read from the paper and recorded in docs/findings.md"
    summary["scorer_worst_abs_diff"] = worst

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2, default=str))
    failed = [f'{r["case_id"]}@{r["level"]}: {r["reason"]}' for r in res if not r["converged"]]
    print("NOT CONVERGED:", failed or "none")
    if not summary["scorer_check"]:
        print("SCORER CHECK NOT PERFORMED: no case converged")
        return 1
    if worst > 1e-6:
        print(f"SCORER CHECK FAILED: worst difference {worst:.3e} > 1e-6")
        return 1
    print(f"scorer check passed: worst difference {worst:.3e}")
    if failed:
        print("S0 INCOMPLETE: some cases did not converge")
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
