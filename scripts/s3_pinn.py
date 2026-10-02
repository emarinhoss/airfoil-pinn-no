"""Stage S3: the PINN on the showcase cases (spec 7.3-7.5). Resumes by skipping finished files.

    python scripts/s3_pinn.py [--budget-hours 48] [--preflight-only]

Before any run starts, every planned configuration is timed for a few Adam steps and L-BFGS
closures on its own showcase case, the measurements and the projected wall time of every run are
written to results/s3/preflight.json beside the warm-up, and the stage REFUSES to start if the
projected total for the runs not yet finished exceeds --budget-hours. Spec 7.5 estimated one to
two hours per case; the final review measured viscous runs at days each with the nested Laplacian
(now folded into the fluxes), so the budget is a decision to be taken on the measurement, not a
default to be trusted. Euler runs come first.
"""

import argparse
import json
import sys
from pathlib import Path

from apno.data.predictions import write_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import (
    PinnConfig, evaluate_on_grid, projected_s, time_steps, train,
)

PRED, RES = Path("runs/predictions"), Path("results/s3")
BUDGET_HOURS = 48.0  # the stage's whole projected wall time; override deliberately, on record


def plan():
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    sup = [c for c in show if c["regime"] == "supercritical"]
    sub = [c for c in show if c["regime"] == "subcritical"]
    runs = [("euler", c["case_id"], PinnConfig()) for c in show]
    # spec 7.5's variant: gradient-norm balanced weights, on the median-Mach showcase case
    runs.append(("gradnorm", show[len(show) // 2]["case_id"], PinnConfig(grad_balance=True)))
    runs += [("visc1e-3", c["case_id"], PinnConfig(viscosity=1e-3)) for c in sup]
    if sup:
        top = max(sup, key=lambda c: c["max_surface_mach"])["case_id"]
        runs.append(("visc3e-4", top, PinnConfig(viscosity=3e-4)))
    if sub:
        runs.append(("nokutta", sub[0]["case_id"], PinnConfig(kutta=False)))
    return runs


def _case(spec):
    return PinnCase(spec.mach, spec.alpha_deg, Section(spec.t, spec.m))


def preflight(runs, specs, budget_hours, n_adam=5, n_closure=2):
    """Time each planned variant on its first case, project every unfinished run; True if within
    budget. Writes results/s3/preflight.json either way."""
    timing = {}
    for variant, cid, cfg in runs:
        if variant not in timing:
            timing[variant] = {"case_id": cid, **time_steps(_case(specs[cid]), cfg, n_adam,
                                                            n_closure)}
            print(f"preflight {variant} on {cid}: {timing[variant]}", flush=True)
    todo = [(v, c, cfg) for v, c, cfg in runs if not (PRED / f"pinn-{v}-{c}.h5").exists()]
    proj = [{"variant": v, "case_id": c, "hours": projected_s(cfg, timing[v]) / 3600}
            for v, c, cfg in todo]
    total = sum(r["hours"] for r in proj)
    out = {"n_adam_timed": n_adam, "n_closure_timed": n_closure, "timing_s": timing,
           "runs": proj, "projected_total_hours": total, "budget_hours": budget_hours,
           "within_budget": total <= budget_hours,
           "note": "wall time, serial; L-BFGS projected at one closure per iteration (a lower "
                   "bound)"}
    (RES / "preflight.json").write_text(json.dumps(out, indent=2))
    for r in proj:
        print(f"  {r['variant']:9s} {r['case_id']}: {r['hours']:.1f} h", flush=True)
    print(f"projected total {total:.1f} h for {len(proj)} runs against a budget of "
          f"{budget_hours:.1f} h", flush=True)
    return out["within_budget"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args(argv)
    specs = {s.case_id: s for s in case_specs("test_in", TRAIN)}
    RES.mkdir(parents=True, exist_ok=True)
    runs = plan()
    if not preflight(runs, specs, args.budget_hours):
        print("REFUSING TO START: the projected S3 wall time exceeds the budget. Reduce points, "
              "steps or viscous runs as a recorded spec decision, or raise --budget-hours "
              "deliberately.", flush=True)
        return 3
    if args.preflight_only:
        return 0
    record_path = RES / "pinn_runs.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for variant, cid, cfg in runs:
        path = PRED / f"pinn-{variant}-{cid}.h5"
        if path.exists():
            continue
        pinn, rec = train(_case(specs[cid]), cfg)
        q = evaluate_on_grid(pinn, TRAIN)
        write_predictions(path, [cid], q[None], {
            "method": "pinn", "tag": path.stem, "variant": variant, "seed": cfg.seed,
            "n_train": 0, "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"],
            "query_s": rec["wall_s"]})
        record[f"{variant}/{cid}"] = {"wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"],
                                      "final": rec["final"], "weights": rec["weights"],
                                      "history": rec["history"]}
        record_path.write_text(json.dumps(record, indent=2))
        print(variant, cid, rec["final"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
