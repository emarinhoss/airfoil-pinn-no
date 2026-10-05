"""Stage S3: the PINN on the showcase cases (spec 7.3-7.5). Resumes by skipping finished files.

    python scripts/s3_pinn.py [--budget-hours 48] [--preflight-only] [--shard K/N]
                              [--formulation potential|euler] [--preflight-file PATH] [--viscous]

The plan (spec 7 as amended 2026-10-05, the author's scope decision): the six showcase cases with
no artificial viscosity (variant "pure"), the gradient-norm balanced run on the median-Mach case
and the no-Kutta run on the first subcritical case, 8 runs. The artificial-viscosity variants
(visc1e-3 on every supercritical case, visc3e-4 on the strongest) are not planned; `--viscous`
appends them, so the nu term stays selectable on record. (Until 2026-10-05 the base variant was
named "euler", after the PINN's first formulation.)

The PINN is the full-potential one by default (spec 7 as amended 2026-10-04); `--formulation euler`
runs the Euler PINN, the recorded negative result, under the same plan. The formulation is recorded
in every preflight timing entry, run record and prediction file's attrs, and a non-default one
suffixes every file it writes (pinn-<variant>-<case>_euler.h5, pinn_runs_euler.json,
preflight_euler.json), so the two never resume from or overwrite each other's runs.

Before any run starts, every planned configuration is timed for a few Adam steps and L-BFGS
closures on its own showcase case, the measurements and the projected wall time of every run are
written to results/s3/preflight.json beside the warm-up, and the stage REFUSES to start if the
projected total for the runs not yet finished exceeds --budget-hours. `--preflight-file PATH`
reuses the per-variant timings of an earlier preflight file instead (one measured uncontended:
three shards preflighting at once measured one variant at 0.83, 1.14 and 29.3 s per Adam step
against 0.75 s alone, and refused on that noise); a variant it does not hold for this formulation
is timed as usual, its legacy name "euler" stands for "pure", and the budget check is the same.
Spec 7.5 estimated one to two hours per case; the final review measured viscous runs at days each,
so the budget is a decision to be taken on the measurement, not a default to be trusted.

The Adam phase runs on PinnConfig.device ('auto': the Apple GPU when MPS is available) and L-BFGS
on the CPU in float64; the preflight and the run record name the device of each phase.
`--shard K/N` runs the plan entries whose index in plan() order is K mod N, so N processes can run
S3 side by side: each times, budgets and records only its own runs, in its own
pinn_runs_shard{K}of{N}.json and preflight_shard{K}of{N}.json (0/1, the default, is the whole plan
and keeps the unsharded names). Shards running at once share the CPU and the GPU: a preflight
measures whatever load is present when it runs, so its projection holds only for that load.
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
    FORMULATIONS, PinnConfig, evaluate_on_grid, projected_s, resolve_device, time_steps, train,
)

PRED, RES = Path("runs/predictions"), Path("results/s3")
BUDGET_HOURS = 48.0  # the stage's whole projected wall time; override deliberately, on record


def plan(formulation="potential", viscous=False):
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    sup = [c for c in show if c["regime"] == "supercritical"]
    sub = [c for c in show if c["regime"] == "subcritical"]
    f = formulation
    runs = [("pure", c["case_id"], PinnConfig(formulation=f)) for c in show]
    # spec 7.5's variant: gradient-norm balanced weights, on the median-Mach showcase case
    runs.append(("gradnorm", show[len(show) // 2]["case_id"],
                 PinnConfig(grad_balance=True, formulation=f)))
    if sub:
        runs.append(("nokutta", sub[0]["case_id"], PinnConfig(kutta=False, formulation=f)))
    if viscous:  # not planned since the author's decision of 2026-10-05; kept selectable
        runs += [("visc1e-3", c["case_id"], PinnConfig(viscosity=1e-3, formulation=f))
                 for c in sup]
        if sup:
            top = max(sup, key=lambda c: c["max_surface_mach"])["case_id"]
            runs.append(("visc3e-4", top, PinnConfig(viscosity=3e-4, formulation=f)))
    return runs


def shard(runs, k, n):
    """The runs whose index in plan order is k mod n, in plan order."""
    return [r for i, r in enumerate(runs) if i % n == k]


def parse_shard(text):
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard wants K/N, got {text!r}") from None
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard needs 0 <= K < N, got {text!r}")
    return k, n


def _suffix(k, n, formulation="potential"):
    return ("" if n == 1 else f"_shard{k}of{n}") + _fsuffix(formulation)


def _fsuffix(formulation):
    return "" if formulation == "potential" else f"_{formulation}"


def prediction_path(variant, cid, formulation="potential"):
    return PRED / f"pinn-{variant}-{cid}{_fsuffix(formulation)}.h5"


def _case(spec):
    return PinnCase(spec.mach, spec.alpha_deg, Section(spec.t, spec.m))


LEGACY_VARIANTS = {"pure": "euler"}  # the base run's name in preflight files before 2026-10-05


def earlier_timings(path, formulation):
    """{variant: timing} from an earlier preflight file, for this formulation only, with legacy
    variant names mapped to today's."""
    held = json.loads(Path(path).read_text())["timing_s"]
    out = {}
    for name, t in held.items():
        if t.get("formulation", "euler") != formulation:
            continue
        now = next((v for v, old in LEGACY_VARIANTS.items() if old == name), name)
        if now not in held or now == name:
            out[now] = dict(t)
    return out


def preflight(runs, specs, budget_hours, n_adam=5, n_closure=2, shard_kn=(0, 1),
              formulation="potential", preflight_file=None):
    """Time each planned variant on its first case (or take its timing from `preflight_file`),
    project every unfinished run; True if within budget. Writes results/s3/preflight{suffix}.json
    either way, with the device of each phase."""
    reuse = earlier_timings(preflight_file, formulation) if preflight_file else {}
    timing = {}
    for variant, cid, cfg in runs:
        if variant in reuse and variant not in timing:
            timing[variant] = {**reuse[variant], "reused_from": str(preflight_file)}
            print(f"preflight {variant}: reused from {preflight_file}: {timing[variant]}",
                  flush=True)
        if variant not in timing:
            timing[variant] = {"case_id": cid, "formulation": cfg.formulation,
                               "adam_device": resolve_device(cfg.device), "lbfgs_device": "cpu",
                               **time_steps(_case(specs[cid]), cfg, n_adam, n_closure)}
            print(f"preflight {variant} on {cid}: {timing[variant]}", flush=True)
    todo = [(v, c, cfg) for v, c, cfg in runs
            if not prediction_path(v, c, cfg.formulation).exists()]
    proj = [{"variant": v, "case_id": c, "hours": projected_s(cfg, timing[v]) / 3600}
            for v, c, cfg in todo]
    total = sum(r["hours"] for r in proj)
    out = {"shard": f"{shard_kn[0]}/{shard_kn[1]}", "formulation": formulation,
           "timings_from": None if preflight_file is None else str(preflight_file),
           "n_adam_timed": n_adam,
           "n_closure_timed": n_closure, "timing_s": timing,
           "runs": proj, "projected_total_hours": total, "budget_hours": budget_hours,
           "within_budget": total <= budget_hours,
           "note": "wall time, serial; L-BFGS projected at one closure per iteration (a lower "
                   "bound)"}
    (RES / f"preflight{_suffix(*shard_kn, formulation)}.json").write_text(json.dumps(out, indent=2))
    for r in proj:
        print(f"  {r['variant']:9s} {r['case_id']}: {r['hours']:.1f} h", flush=True)
    print(f"projected total {total:.1f} h for {len(proj)} runs against a budget of "
          f"{budget_hours:.1f} h", flush=True)
    return out["within_budget"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="K/N",
                    help="run only the plan entries with index K mod N (default 0/1: all)")
    ap.add_argument("--formulation", choices=FORMULATIONS, default="potential",
                    help="the PINN's equations (default potential; euler is the negative result)")
    ap.add_argument("--preflight-file", type=Path, default=None, metavar="PATH",
                    help="reuse this earlier preflight's per-variant timings instead of timing")
    ap.add_argument("--viscous", action="store_true",
                    help="append the artificial-viscosity variants (not planned since 2026-10-05)")
    args = ap.parse_args(argv)
    specs = {s.case_id: s for s in case_specs("test_in", TRAIN)}
    RES.mkdir(parents=True, exist_ok=True)
    runs = shard(plan(args.formulation, args.viscous), *args.shard)
    if not preflight(runs, specs, args.budget_hours, shard_kn=args.shard,
                     formulation=args.formulation, preflight_file=args.preflight_file):
        print("REFUSING TO START: the projected S3 wall time exceeds the budget. Reduce points "
              "or steps as a recorded spec decision, or raise --budget-hours "
              "deliberately.", flush=True)
        return 3
    if args.preflight_only:
        return 0
    record_path = RES / f"pinn_runs{_suffix(*args.shard, args.formulation)}.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for variant, cid, cfg in runs:
        path = prediction_path(variant, cid, cfg.formulation)
        if path.exists():
            continue
        pinn, rec = train(_case(specs[cid]), cfg)
        q = evaluate_on_grid(pinn, TRAIN)
        write_predictions(path, [cid], q[None], {
            "method": "pinn", "tag": path.stem, "variant": variant,
            "formulation": cfg.formulation, "seed": cfg.seed,
            "n_train": 0, "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"],
            "query_s": rec["wall_s"], "train_device": rec["adam_device"],
            "adam_device": rec["adam_device"], "lbfgs_device": rec["lbfgs_device"]})
        record[f"{variant}/{cid}"] = {"formulation": rec["formulation"],
                                      "wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"],
                                      "adam_wall_s": rec["adam_wall_s"],
                                      "adam_device": rec["adam_device"],
                                      "lbfgs_device": rec["lbfgs_device"],
                                      "final": rec["final"], "weights": rec["weights"],
                                      "history": rec["history"],
                                      "te_dropped": rec.get("te_dropped", [])}
        record_path.write_text(json.dumps(record, indent=2))
        print(variant, cid, rec["final"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
