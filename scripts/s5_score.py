"""Stage S5 scores (S5 design §3.5, §4, §5.3-5.4): the gate table, the ablation verdicts, the
step-3 comparison and the costs, into results/s5/.

    python scripts/s5_score.py

Reads the S5 records (results/s5/), the S5 prediction files (runs/predictions_s5/) and, for the
comparison columns, the S4 prediction files (runs/predictions/, read only) and the truth.
results/s4 is read for SU2's and the FNO's costs and never written. A column whose prediction file
or truth is missing is reported as a status and left out of the group means, with the counts kept.

Ablation verdict (design §4): "needed" if the arm failed its pass criterion, collapsed, diverged or
could not be scored; "not needed" if it passed; "not run" if it has no record. Every arm's record
keeps its score (RMAE, near field, CL, collapse), losses and costs beside the verdict. The
comparison (§5.3): per case and method the S4 scores and the gate's RMAE; a step-3 run that
diverged (no prediction file) is the status "diverged", not "missing"; per method the mean
near-field ratio to the floor over the in-range and over the extrapolation cases, with the mean
|CL| and |CD| errors beside (not gated); degradation = the extrapolation mean over the in-range
mean, for the FNO, POD and the new PINN, on all six cases only: a method short of a case (diverged,
missing, unscored) has none, since a figure on the cases left is not like-for-like, and scores.md
gives its counts and diverged cases instead.

Costs (§5.4): per step-3 run its wall and CPU time and devices; the per-case cost is S4's rule
(GPU training at wall time, CPU training at CPU time), averaged over the runs that did not diverge
(a diverged run's time is cut short); the break-even (results/s5/break_even.png) puts the new
PINN (no fixed cost) beside SU2 per case at CPU time and the FNO (S4's fixed cost, per query).
"""

import json
import sys
from pathlib import Path

import numpy as np

from apno.data.predictions import datasets, read_predictions
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import pressure_coefficient
from apno.score.metrics import FLUX_LINES, score_case
from apno.score.report import (
    BY_CONSTRUCTION, BY_CONSTRUCTION_CELL, PHYSICS, break_even, plot_break_even, plot_showcase,
)
from apno.score.rmae import rmae_cp_mach
from apno.score.s5 import STEP3_EXT, STEP3_IN, all_specs, truth_and_floor

RES, PRED5, PRED = Path("results/s5"), Path("runs/predictions_s5"), Path("runs/predictions")
ARMS = ("A1", "A2", "A3", "A4", "A5", "A6a", "A6b")
METHODS = ("floor", "fno-256", "fno-128", "pod", "pinn-pure", "euler_av")
DEGRADED = ("fno-256", "fno-128", "pod", "euler_av")


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_runs():
    out = {}
    for p in sorted(RES.glob("runs*.json")):
        out.update(json.loads(p.read_text()))
    return out


def verdict(entry):
    if entry is None:
        return "not run"
    sc = entry.get("score") or {}
    if entry.get("diverged") or sc.get("status") != "ok":
        return "needed"
    return "needed" if (not sc.get("passed") or sc.get("collapsed")) else "not needed"


ABLATION_KEEP = ("diverged", "final", "stalled", "func_evals", "wall_s", "cpu_s", "device")


def ablation(runs):
    """Per arm the verdict, the score, the losses and the costs (None for an arm not run)."""
    return {a: {"verdict": verdict(runs.get(a)), "score": (runs.get(a) or {}).get("score"),
                **{k: (runs.get(a) or {}).get(k) for k in ABLATION_KEEP}}
            for a in ARMS}


def _column_file(method, case_id):
    return {"euler_av": (PRED5 / f"pinn-euler_av-{case_id}.h5", "q"),
            "fno-256": (PRED / "fno-n896-s0.h5", "q"),
            "fno-128": (PRED / "fno-n896-s0.h5", "q_128x49"),
            "pod": (PRED / "pod-n896.h5", "q"),
            "pinn-pure": (PRED / f"pinn-pure-{case_id}.h5", "q")}[method]


def method_q(method, case_id):
    """q of a comparison column for one case, or None if its file or dataset is missing."""
    path, ds = _column_file(method, case_id)
    if not path.exists() or ds not in datasets(path):
        return None
    return read_predictions(path, ds)[0].get(case_id)


def by_construction(method, case_id):
    """The physics checks a column's model satisfies identically, by S4's rule (spec 8.2,
    `apno.score.report.BY_CONSTRUCTION`, keyed on the prediction file's method and formulation):
    the S4 potential PINN's enthalpy and entropy. () for SU2, the FNO, POD and the S5 PINN, whose
    checks are all results."""
    if method == "floor":
        return ()
    path, ds = _column_file(method, case_id)
    if not path.exists():
        return ()
    attrs = read_predictions(path, ds)[1]
    return BY_CONSTRUCTION.get((attrs.get("method"), attrs.get("formulation", "euler")), ())


def score_column(method, case_id, specs, runs=None):
    """One case's scores for one method, or a status. A step-3 euler_av run recorded as diverged
    wrote no prediction file: it is "diverged", not "missing" (which reads as not run)."""
    if method == "euler_av" and ((runs or {}).get(case_id) or {}).get("diverged"):
        return {"status": "diverged"}
    truth, floor, mmax = truth_and_floor(case_id)
    if truth is None:
        return {"status": "missing truth"}
    q = floor if method == "floor" else method_q(method, case_id)
    if q is None:
        return {"status": "missing"}
    s = specs[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    t, flux = truth, FLUX_LINES
    if method == "fno-128":
        X, Y, t, flux = X[::2, ::2], Y[::2, ::2], truth[::2, ::2], tuple(j // 2 for j in FLUX_LINES)
    regime = "supercritical" if mmax > 1.0 else "subcritical"
    out = score_case(q, t, X, Y, s.mach, s.alpha_deg, regime, flux)
    if out["status"] == "ok":
        out["rmae"] = rmae_cp_mach(q, t, X, Y, s.mach)
        for k in by_construction(method, case_id):  # an identity is labelled, never a value
            out["physics"][k] = BY_CONSTRUCTION_CELL
    return out


GROUP_ERRORS = ("cl_err", "cd_err_counts")


GROUPS = (("in", STEP3_IN), ("ext", STEP3_EXT))


def group_ratios(table):
    """Per method and group: the mean near-field ratio to the floor and its count, the number of
    cases whose run diverged, and beside them, not gated, the mean |CL| and |CD| errors over the
    same cases (None when none has them)."""
    out = {}
    for m in METHODS[1:]:
        out[m] = {}
        for grp, cases in GROUPS:
            ok = [c for c in cases if table[c][m].get("status") == "ok"
                  and table[c]["floor"].get("status") == "ok"]
            r = [table[c][m]["near_field_rel_l2"] / table[c]["floor"]["near_field_rel_l2"]
                 for c in ok]
            out[m][grp] = float(np.mean(r)) if r else None
            out[m][f"n_{grp}"] = len(r)
            out[m][f"n_diverged_{grp}"] = sum(table[c][m].get("status") == "diverged"
                                              for c in cases)
            for k in GROUP_ERRORS:
                e = [abs(table[c][m][k]) for c in ok if table[c][m].get(k) is not None]
                out[m][f"{k}_{grp}"] = float(np.mean(e)) if e else None
    return out


def _complete(g):
    return all(g[f"n_{grp}"] == len(cases) for grp, cases in GROUPS)


def degradation(groups):
    """The extrapolation mean over the in-range mean, on all six cases (design §5.3), or None for
    a method short of a case: a figure on the cases left is not like-for-like with the others'."""
    return {m: (groups[m]["ext"] / groups[m]["in"]
                if _complete(groups[m]) and groups[m]["in"] else None)
            for m in DEGRADED}


def _degradation_cell(x, g):
    """The degradation, or for a method short of a case "—" with its counts and diverged cases."""
    if x is not None or not g:
        return _fmt(x)
    parts = []
    for grp, cases in GROUPS:
        n, d = g[f"n_{grp}"], g.get(f"n_diverged_{grp}", 0)
        if n < len(cases):
            parts.append(f"{grp} {n}/{len(cases)}" + (f", {d} diverged" if d else ""))
    return "—" + (f" ({'; '.join(parts)})" if parts else "")


def _run_cost(r):
    """S4's rule: GPU training at wall time, CPU training at CPU time."""
    return float(r["cpu_s"] if r.get("device", "cpu") == "cpu" else r["wall_s"])


def _basis(devices):
    if devices == {"cpu"}:
        return "cpu"
    if "cpu" not in devices:
        return "wall (GPU training, S4 rule)"
    return "per run: wall time for GPU training, CPU time for CPU training (S4 rule)"


def costs(runs, s4_costs):
    """The new PINN's per-case cost beside SU2's and the FNO's (S4's record). euler_av is None
    with no step-3 record; its means (per, per_wall, per_cpu) are over the runs that did not
    diverge, None if none finished; "runs" lists every recorded step-3 run."""
    su2, fno = s4_costs["costs"]["su2"], s4_costs["costs"]["fno"]
    out = {"euler_av": None, "su2": {"per_wall": su2["per_wall"], "per_cpu": su2["per_cpu"]},
           "fno": {"fixed": fno.get("fixed"), "per": fno["per"]}, "fno_per_query": fno["per"],
           "ratio_to_su2_cpu": None}
    step3 = {c: runs[c] for c in STEP3_IN + STEP3_EXT if c in runs}
    if not step3:
        return out
    done = [r for r in step3.values() if not r.get("diverged")]
    devices = {r.get("device", "cpu") for r in done}
    mean = (lambda xs: float(np.mean(xs))) if done else (lambda xs: None)
    per_wall, per_cpu = mean([r["wall_s"] for r in done]), mean([r["cpu_s"] for r in done])
    out["euler_av"] = {
        "n_runs": len(done), "n_diverged": len(step3) - len(done),
        "per": mean([_run_cost(r) for r in done]), "basis": _basis(devices) if done else None,
        "per_wall": per_wall, "per_cpu": per_cpu, "devices": sorted(devices),
        "runs": {c: {"wall_s": r.get("wall_s"), "cpu_s": r.get("cpu_s"),
                     "device": r.get("device"), "eval_device": r.get("eval_device"),
                     "diverged": bool(r.get("diverged")),
                     "cost_s": None if r.get("diverged") else _run_cost(r)}
                 for c, r in step3.items()}}
    if done:
        out["ratio_to_su2_cpu"] = {"wall": per_wall / su2["per_cpu"],
                                   "cpu": per_cpu / su2["per_cpu"]}
    return out


def break_even_curves(cost, n_max=5000):
    """Cumulative cost of n new cases: SU2 per case at CPU time, the FNO (S4's fixed cost, then
    per query; left out if S4 records no fixed cost) and the new PINN (no fixed cost, one training
    per case); None without a finished step-3 run."""
    ea = cost.get("euler_av")
    if not ea or ea.get("per") is None:
        return None
    c = {"su2": {"fixed": 0.0, "per": cost["su2"]["per_cpu"]}}
    if (cost.get("fno") or {}).get("fixed") is not None:
        c["fno"] = {"fixed": cost["fno"]["fixed"], "per": cost["fno"]["per"]}
    c["euler_av"] = {"fixed": 0.0, "per": ea["per"]}
    return break_even(c, n_max)


def _fmt(x, f="{:.4g}"):
    return "—" if x is None else f.format(x)


def _cell(x):
    return "—" if x is None else str(x)


def ablation_row(arm, v):
    """One ablation table row: the airfoil arms' RMAE, near field and CL error; for A6a (the
    cylinder) its max |ΔCp| against 1 - 4 sin²θ and its CL, against 0. Absent values are "—"."""
    sc = v.get("score") or {}
    r, s4 = sc.get("rmae") or {}, sc.get("s4") or {}
    cl = sc.get("cl") if arm == "A6a" else s4.get("cl_err")
    wall = v.get("wall_s")
    cells = [arm, v["verdict"], _cell(sc.get("status")), _fmt(r.get("cp")), _fmt(r.get("mach")),
             _fmt(s4.get("near_field_rel_l2")), _fmt(cl), _fmt(sc.get("max_cp_error")),
             _cell(sc.get("collapsed")), _fmt((v.get("final") or {}).get("total")),
             _fmt(None if wall is None else wall / 3600, "{:.2f}")]
    return "| " + " | ".join(cells) + " |"


def markdown(gate, abl, table, groups, degr, cost):
    out = ["## Gate (design §3)", "", "| config | passed | RMAE Cp | RMAE Mach | collapsed |",
           "|---|---|---|---|---|"]
    for name, e in ((gate or {}).get("configs") or {}).items():
        sc = e.get("score") or {}
        r = sc.get("rmae") or {}
        out.append(f"| {name} | {sc.get('passed')} | {_fmt(r.get('cp'))} | "
                   f"{_fmt(r.get('mach'))} | {sc.get('collapsed')} |")
    out += ["", "## Ablation (design §4)", "",
            "| arm | verdict | status | RMAE Cp | RMAE Mach | near field | CL err | max abs ΔCp | "
            "collapsed | final loss | wall h |", "|---" * 11 + "|"]
    out += [ablation_row(a, v) for a, v in abl.items()]
    out += ["", '"Not needed" means not needed given the rest of R: a one-at-a-time ablation '
            "misses interactions (design §4). One seed per arm."]
    out += ["", "## Comparison (design §5): near field, and ratio to the floor", "",
            "| case | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for c, row in table.items():
        fl = row["floor"].get("near_field_rel_l2") if row["floor"].get("status") == "ok" else None
        cells = []
        for m in METHODS:
            nf = row[m].get("near_field_rel_l2") if row[m].get("status") == "ok" else None
            ratio = nf / fl if nf is not None and fl else None
            cells.append(row[m].get("status") if nf is None
                         else f"{nf:.4f} ({_fmt(ratio, '{:.1f}')}x)")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out += ["", "| method | in-range ratio (n) | extrapolation ratio (n) | degradation | "
            "CL err in | CL err ext | CD err in (counts) | CD err ext (counts) |",
            "|---" * 8 + "|"]
    for m, g in groups.items():
        out.append(f"| {m} | {_fmt(g['in'])} ({g['n_in']}) | {_fmt(g['ext'])} ({g['n_ext']}) | "
                   f"{_degradation_cell(degr.get(m), g if m in DEGRADED else None)} | "
                   f"{_fmt(g.get('cl_err_in'))} | "
                   f"{_fmt(g.get('cl_err_ext'))} | {_fmt(g.get('cd_err_counts_in'))} | "
                   f"{_fmt(g.get('cd_err_counts_ext'))} |")
    out += ["", "Degradation: on all six cases only (design §5.3); a method short of a case "
            "shows — with its counts and diverged cases, not a figure on the cases left.",
            "", "CL and CD errors: the mean absolute error over the same cases as the ratio, "
            "reported beside, not gated (design §5.3)."]
    out += ["", "## Physics checks (design §5.3; S4's labels)", "",
            "| case | method | " + " | ".join(PHYSICS) + " |", "|---" * (len(PHYSICS) + 2) + "|"]
    for c, row in table.items():
        for m in METHODS:
            ph = row[m].get("physics") if row[m].get("status") == "ok" else None
            if ph is not None:
                cells = [ph[k] if isinstance(ph.get(k), str) else _fmt(ph.get(k)) for k in PHYSICS]
                out.append(f"| {c} | {m} | " + " | ".join(cells) + " |")
    out += ["", f'"{BY_CONSTRUCTION_CELL}" marks a check the model satisfies identically '
            "(spec 8.2): it is not a pass."]
    out += ["", "## Costs (design §5.4)", "", "```", json.dumps(cost, indent=2), "```", ""]
    return "\n".join(out)


def plots(specs, cases):
    for c in cases:
        truth, floor, _ = truth_and_floor(c)
        if truth is None:
            continue
        s = specs[c]
        X, _ = grid(Section(s.t, s.m), TRAIN)
        curves = {"truth": pressure_coefficient(truth[:, 0, 3], s.mach),
                  "su2 256x97": pressure_coefficient(floor[:, 0, 3], s.mach)}
        for m in ("fno-256", "pod", "pinn-pure", "euler_av"):
            q = method_q(m, c)
            if q is not None:
                curves[m] = pressure_coefficient(q[:, 0, 3], s.mach)
        plot_showcase(X[:, 0], curves, c, RES / f"showcase_s5_{c}.png")


def main(argv=None):
    RES.mkdir(parents=True, exist_ok=True)
    specs = all_specs()
    gate, runs = _load(RES / "gate.json"), load_runs()
    abl = ablation(runs)
    table = {c: {m: score_column(m, c, specs, runs) for m in METHODS}
             for c in STEP3_IN + STEP3_EXT}
    groups = group_ratios(table)
    degr = degradation(groups)
    cost = costs(runs, json.loads(Path("results/s4/costs.json").read_text()))
    (RES / "ablation.json").write_text(json.dumps(abl, indent=2, default=float))
    (RES / "scores.json").write_text(json.dumps(
        {"table": table, "groups": groups, "degradation": degr}, indent=2, default=float))
    (RES / "costs.json").write_text(json.dumps(cost, indent=2, default=float))
    (RES / "scores.md").write_text(markdown(gate, abl, table, groups, degr, cost))
    curves = break_even_curves(cost)
    if curves is not None:
        plot_break_even(curves, RES / "break_even.png")
    plots(specs, STEP3_IN + STEP3_EXT)
    print((RES / "scores.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
