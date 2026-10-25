"""Score the library FNOs in runs/predictions_lib/ as S4 scores its contestants, beside S2's FNO.

The prediction files written by scripts/probes/probe_fno_libraries.py (one per library, training
set size and seed, variant = the library) are scored case by case with S4's own code
(`s4_score.load_truth`, `load_contestants`, `score_rows`), so every number means what it means in
results/s4: the truth is SU2 at 512x193 sampled at every second node, the floor is SU2's 256x97
solution, and a file's q_128x49 is scored on the 128x49 subgrid. S2's FNO files from
runs/predictions/ are scored alongside as references (`--reference`, default fno-n896-s0), with
their variant relabelled "s2" ("s2-lossB" for a loss-B file) so the tables tell them apart.

Writes results/fno_lib/scores.json and scores.md, never results/s4/: S4's records describe the
study's contestants and are not touched. scores.md carries, in order:
- the runs: each library run's train.json from runs/fno_lib/ (device, epochs, validation loss,
  resolution transfer, wall time) and each reference's row from results/s2/runs.json;
- the near-field error against the floor, per (contestant, grid, split, regime), as a ratio to the
  floor's mean on the same cases at 256x97 (S4's "x the floor");
- S4's three tables: forces, surface and near field; shocks; physics checks.

    python scripts/probes/score_fno_libraries.py
    python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1

Exit 2 if the truth files or the prediction files are missing, or if --out is S4's directory.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from apno.data.predictions import datasets, read_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN
from apno.score.report import (
    PHYSICS_METRICS, SHOCK_METRICS, aggregate, aggregate_physics, aggregate_shocks,
    markdown_table,
)

SCRIPTS = Path(__file__).resolve().parents[1]
S4_OUT = Path("results/s4")
RUN_FIELDS = ("device", "n_cases", "n_params", "epochs_run", "best_epoch", "best_val",
              "val_rel_l2_256x97", "transfer_val_rel_l2", "wall_s", "s_per_epoch", "query_s")


def _s4():
    spec = importlib.util.spec_from_file_location("s4_score", SCRIPTS / "s4_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_references(pred_dir, tags):
    """S2's FNO files by exact tag, relabelled variant "s2[-<variant>]"; missing tags skipped."""
    out, missing = [], []
    for tag in tags:
        p = Path(pred_dir) / f"{tag}.h5"
        if not p.exists():
            missing.append(tag)
            continue
        q, attrs = read_predictions(p)
        v = attrs.get("variant", "")
        attrs = {**attrs, "variant": f"s2-{v}" if v else "s2"}
        out.append((tag, "256x97", attrs, q))
        if "q_128x49" in datasets(p):
            out.append((tag, "128x49", attrs, read_predictions(p, "q_128x49")[0]))
    return out, missing


def run_records(lib_tags, runs_dir, ref_tags, s2_runs):
    """One row per scored tag: the library run's train.json, or the reference's S2 record."""
    s2 = {}
    if Path(s2_runs).exists():
        s2 = {r["tag"]: r for r in json.loads(Path(s2_runs).read_text())}
    rows = []
    for tag in lib_tags:
        p = Path(runs_dir) / tag / "train.json"
        rec = json.loads(p.read_text()) if p.exists() else None
        rows.append(_run_row(tag, rec, f"{rec.get('library')} {rec.get('version')}"
                             if rec else None, p))
    for tag in ref_tags:
        rows.append(_run_row(tag, s2.get(tag), "S2 FNO2d", Path(s2_runs)))
    return rows


def _run_row(tag, rec, source, path):
    row = {"tag": tag, "source": source or "no record", "record": str(path)}
    if rec:
        for k in RUN_FIELDS:
            if k in rec:
                row[k] = rec[k]
        if rec.get("wall_s") is not None and rec.get("epochs_run"):
            row["s_per_epoch"] = rec["wall_s"] / rec["epochs_run"]
        if "env" in rec:  # train.json is git-ignored, so its environment is kept here
            row["env"] = rec["env"]
    return row


def floor_ratios(groups):
    """Each non-floor group's near-field mean over the floor's on the same split and regime
    (the floor is at 256x97 only, so a 128x49 row is also divided by the 256x97 floor)."""
    floor = {(g["split"], g["regime"]): g.get("near_field_rel_l2_mean") for g in groups
             if g["method"] == "su2-256x97"}
    out = []
    for g in groups:
        if g["method"] == "su2-256x97":
            continue
        f = floor.get((g["split"], g["regime"]))
        m = g.get("near_field_rel_l2_mean")
        out.append({k: g[k] for k in ("method", "variant", "grid", "n_train", "seed", "split",
                                      "regime", "n_ok")}
                   | {"near_field_rel_l2_mean": m, "floor_mean": f,
                      "x_floor": m / f if m is not None and f else None})
    return out


def _table(rows, head):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r.get(h, "") for h in head]
        lines.append("| " + " | ".join(
            "" if c is None else f"{c:.4g}" if isinstance(c, float) else str(c)
            for c in cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred-dir", default="runs/predictions_lib")
    ap.add_argument("--runs", default="runs/fno_lib", help="the probe's train.json records")
    ap.add_argument("--reference", action="append", default=None,
                    help="an S2 tag in runs/predictions/ to score alongside (repeatable); "
                         "default fno-n896-s0")
    ap.add_argument("--reference-dir", default="runs/predictions")
    ap.add_argument("--s2-runs", default="results/s2/runs.json")
    ap.add_argument("--fine", default="data/hdf5/512x193.h5")
    ap.add_argument("--train", default="data/hdf5/256x97.h5")
    ap.add_argument("--out", default="results/fno_lib")
    args = ap.parse_args(argv)
    refs = ["fno-n896-s0"] if args.reference is None else args.reference

    out = Path(args.out)
    if out.resolve() == S4_OUT.resolve():
        print(f"refusing to write to {S4_OUT}: S4's records are not this scorer's to replace")
        return 2
    for p in (args.fine, args.train):
        if not Path(p).exists():
            print(f"no truth file at {p}: the scorer needs both SU2 levels (stage S1)")
            return 2
    s4 = _s4()
    libs = s4.load_contestants(args.pred_dir) if Path(args.pred_dir).is_dir() else []
    if not libs:
        print(f"no prediction files in {args.pred_dir}: run scripts/probes/probe_fno_libraries.py")
        return 2
    ref, missing = load_references(args.reference_dir, refs)
    for tag in missing:
        print(f"reference {tag}: no {args.reference_dir}/{tag}.h5, not scored")

    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    truth, floor, regimes = s4.load_truth(args.fine, args.train)
    contestants = [s4.floor_contestant(floor)] + libs + ref
    rows = s4.score_rows(contestants, truth, regimes, specs, {})
    groups = aggregate(rows)
    shock_groups, physics_groups = aggregate_shocks(rows), aggregate_physics(rows)
    ratios = floor_ratios(groups)
    lib_tags = sorted({tag for tag, *_ in libs})
    ref_tags = sorted({tag for tag, *_ in ref})
    runs = run_records(lib_tags, args.runs, ref_tags, args.s2_runs)
    unrecorded = [r["tag"] for r in runs if r["source"] == "no record"]
    if unrecorded:  # scored from its predictions, but its training record was not found
        print(f"no train.json under {args.runs} for {len(unrecorded)} scored run(s): "
              + ", ".join(unrecorded) + ". Their test scores are complete; their validation "
              "figures are blank until the records are copied there and this is run again.")

    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(
        {"runs": runs, "floor_ratios": ratios, "rows": rows, "groups": groups,
         "shock_groups": shock_groups, "physics_groups": physics_groups}, indent=2,
        default=float))
    (out / "scores.md").write_text(
        "# The library FNOs, scored as S4 scores its contestants\n\n"
        f"Command: `python scripts/probes/score_fno_libraries.py` over `{args.pred_dir}`, with "
        f"S2 reference{'s' if len(ref_tags) != 1 else ''} {', '.join(ref_tags) or 'none'} "
        "(variant `s2`). A library's variant is the library's name.\n\n"
        "## Runs\n\n"
        "best_val is loss A on the validation cases at 128x49; transfer is the relative L2 "
        "difference between the 256x97 prediction sampled at every second node and the 128x49 "
        "one. Wall times are on the device named: GPU wall time on different hardware is not "
        "one cost basis.\n\n"
        + _table(runs, ("tag", "source") + RUN_FIELDS) + "\n\n"
        "## Near field against the floor\n\n"
        "x_floor is the contestant's mean near-field relative L2 over the floor's (SU2 at 256x97) "
        "on the same split and regime. The floor exists at 256x97 only, so 128x49 rows are "
        "divided by it too.\n\n"
        + _table(ratios, ("method", "variant", "grid", "n_train", "seed", "split", "regime",
                          "n_ok", "near_field_rel_l2_mean", "floor_mean", "x_floor"))
        + "\n\n## Forces, surface and near field\n\n" + markdown_table(groups)
        + "\n\n## Shocks (supercritical cases)\n\n"
        "Pooled over every (case, surface) where the truth has a shock; a miss is a predicted "
        "surface with no supersonic compression.\n\n"
        + markdown_table(shock_groups, SHOCK_METRICS) + "\n\n## Physics checks\n\n"
        "Mass flux is |net flux| through closed grid line j. n_nonpositive_state counts cases "
        "with rho <= 0 or p <= 0 somewhere in the near field.\n\n"
        + markdown_table(physics_groups, PHYSICS_METRICS) + "\n")
    print((out / "scores.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
