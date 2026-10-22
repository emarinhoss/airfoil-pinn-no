"""Stage S4: score every prediction file, the floor and SU2 itself; write tables, plots, costs.

Prediction files are identified by their tag (the file stem). Costs, the showcase and the
data-efficiency curve read exact tags: fno-n896-s0 and pod-n896 for cost and showcase, fno-n<n>-s0
and pod-n<n> for efficiency. S4 refuses to run if two files share (method, variant, n_train, seed)
on any case, since aggregation would pool them silently. A file holding q_128x49 is also scored at
128x49 against the truth sampled at every fourth fine-grid node (grid "128x49"), beside its
256x97 score, so the FNO's resolution-transfer error is not charged to the method unseen.

Costs: SU2 per case is its median CPU time on the test cases. The FNO and POD fixed costs are the
campaign plus model search plus final training ("fixed", the one the break-even plot uses) and,
beside it, the campaign plus final training only ("fixed_final_fit_only"). The FNO's search is its
tuning runs (results/s2/tuning.json); POD's is its mode-and-smoothing search. Training on the GPU
is costed at wall time, because process_time() cannot see GPU work; CPU training at CPU time.
The PINN, whose Adam phase runs on the GPU and whose L-BFGS phase runs on the CPU, is costed by the
same rule per phase: Adam wall time plus L-BFGS CPU time (the run's CPU time less the Adam phase's
own). Runs that predate the Adam phase's recorded CPU time take the Adam wall time from their S3
run record (results/s3/pinn_runs*.json) and are costed as a labelled bracket: the upper bound
counts the Adam phase's host CPU time as L-BFGS time, the lower bound (`per_lower`) assumes that
host CPU time was as long as the Adam wall time. Every method also carries its wall-time and
CPU-time figures beside the basis the break-even uses (spec 8.2 asks for both).
"""

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from apno.data.build import regime
from apno.data.predictions import datasets, read_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import pressure_coefficient
from apno.score.metrics import FLUX_LINES, score_case
from apno.score.report import (
    PHYSICS_METRICS, SHOCK_METRICS, aggregate, aggregate_physics, aggregate_shocks, break_even,
    markdown_table, plot_break_even, plot_efficiency, plot_showcase,
)

OUT = Path("results/s4")
PINN_RECORDS = "results/s3/pinn_runs*.json"
COST_TAGS = {"fno": "fno-n896-s0", "pod": "pod-n896"}
COARSE_FLUX_LINES = tuple(j // 2 for j in FLUX_LINES)


def _group(path, split):
    with h5py.File(path) as f:
        g = f[split]
        return dict(zip(g["case_id"].asstr()[...], g["q"][...])), dict(
            zip(g["case_id"].asstr()[...], g["max_surface_mach"][...]))


def load_contestants(pred_dir):
    """[(tag, grid, attrs, {case_id: q})] for every prediction file; refuses colliding keys."""
    out, seen = [], defaultdict(dict)
    for p in sorted(Path(pred_dir).glob("*.h5")):
        tag = p.stem
        q, attrs = read_predictions(p)
        key = (attrs["method"], attrs.get("variant", ""), int(attrs.get("n_train", 0)),
               int(attrs.get("seed", 0)))
        for cid in q:
            if cid in seen[key]:
                raise SystemExit(f"{p} and {seen[key][cid]} both predict {cid} for (method, "
                                 f"variant, n_train, seed) = {key}; aggregation would pool them. "
                                 "Move one aside or give it a distinct variant.")
            seen[key][cid] = p
        out.append((tag, "256x97", attrs, q))
        if "q_128x49" in datasets(p):
            out.append((tag, "128x49", attrs, read_predictions(p, "q_128x49")[0]))
    return out


def load_truth(fine_h5="data/hdf5/512x193.h5", train_h5="data/hdf5/256x97.h5"):
    """(truth, floor, regimes) keyed by case id, for the test cases converged at both levels:
    the truth is SU2 at 512x193 sampled at every second node, the floor SU2 at 256x97."""
    truth, floor, regimes = {}, {}, {}
    for split in ("test_in", "test_ext"):
        t, mach = _group(fine_h5, split)
        fl, _ = _group(train_h5, split)
        for cid in t:
            if cid in fl:
                truth[cid], floor[cid] = t[cid][::2, ::2], fl[cid]
                regimes[cid] = regime(mach[cid])
    return truth, floor, regimes


def floor_contestant(floor):
    """SU2's own 256x97 solution as a contestant: the floor every method is measured against."""
    return ("su2-256x97", "256x97",
            {"method": "su2-256x97", "variant": "floor", "n_train": 0, "seed": 0}, floor)


def score_rows(contestants, truth, regimes, specs, grids):
    """One row per (contestant, case): every truth case for a field method, a PINN's own cases
    only. A file's q_128x49 is scored on the 128x49 subgrid against the truth sampled there.
    `grids` caches each case's 256x97 grid and is filled as cases are met."""
    rows = []
    for tag, gname, attrs, preds in contestants:
        scope = list(preds) if attrs["method"] == "pinn" else list(truth)
        for cid in scope:
            if cid not in truth:
                continue
            s = specs[cid]
            if cid not in grids:
                grids[cid] = grid(Section(s.t, s.m), TRAIN)
            X, Y = grids[cid]
            t, flux = truth[cid], FLUX_LINES
            if gname == "128x49":
                X, Y, t, flux = X[::2, ::2], Y[::2, ::2], t[::2, ::2], COARSE_FLUX_LINES
            rows.append({"tag": tag, "method": attrs["method"], "variant": attrs.get("variant", ""),
                         "formulation": attrs.get("formulation", "euler"),
                         "grid": gname, "n_train": int(attrs.get("n_train", 0)),
                         "seed": int(attrs.get("seed", 0)), "split": cid.split("-")[0],
                         "regime": regimes[cid], "case_id": cid,
                         "score": score_case(preds.get(cid), t, X, Y, s.mach, s.alpha_deg,
                                             regimes[cid], flux)})
    return rows


def _train_cost(attrs):
    """(seconds, basis): CPU time for CPU training, wall time for GPU training."""
    if attrs.get("train_device", "cpu") == "cpu":
        return float(attrs["train_cpu_s"]), "cpu"
    return float(attrs["train_wall_s"]), "wall"


def _tuning_cost(path):
    rows = json.loads(Path(path).read_text())["rows"]
    total = 0.0
    for r in rows:
        total += r["cpu_s"] if r.get("device", "cpu") == "cpu" else r["wall_s"]
    return total, len(rows)


def _tuning_bases(path):
    """(CPU seconds, wall seconds) summed over the tuning runs."""
    rows = json.loads(Path(path).read_text())["rows"]
    return sum(r["cpu_s"] for r in rows), sum(r["wall_s"] for r in rows)


def load_pinn_records(pattern=PINN_RECORDS):
    """S3's run records, keyed "<formulation>:<variant>/<case_id>"."""
    out = {}
    for p in sorted(Path(".").glob(pattern)):
        for key, rec in json.loads(p.read_text()).items():
            out[f"{rec.get('formulation', 'euler')}:{key}"] = rec
    return out


def _pinn_run_cost(attrs, record):
    """One PINN run's wall, CPU and per-phase cost (Adam at wall time when it ran on the GPU,
    L-BFGS at CPU time), and the per-phase lower bound when the Adam CPU time is unrecorded."""
    wall, cpu = float(attrs["train_wall_s"]), float(attrs["train_cpu_s"])
    dev = attrs.get("adam_device", attrs.get("train_device", "cpu"))
    if dev == "cpu" and attrs.get("lbfgs_device", "cpu") == "cpu":
        return {"wall": wall, "cpu": cpu, "phase": cpu, "lower": None, "basis": "cpu"}
    record = record or {}
    adam_wall = attrs.get("train_adam_wall_s", record.get("adam_wall_s"))
    if adam_wall is None:
        raise SystemExit(f"{attrs.get('tag', 'a pinn file')}: Adam ran on {dev} but neither its "
                         "attrs (train_adam_wall_s) nor its S3 run record (adam_wall_s) hold the "
                         "Adam phase's wall time, so S4 cannot cost it per phase")
    adam_wall = float(adam_wall)
    adam_cpu = attrs.get("train_adam_cpu_s", record.get("adam_cpu_s"))
    if adam_cpu is not None:
        return {"wall": wall, "cpu": cpu, "phase": adam_wall + cpu - float(adam_cpu),
                "lower": None, "basis": f"per-phase: Adam wall time ({dev}) + L-BFGS CPU time"}
    return {"wall": wall, "cpu": cpu, "phase": adam_wall + cpu, "lower": cpu,
            "basis": f"per-phase upper bound: Adam wall time ({dev}) + the run's whole CPU time "
                     "(the Adam phase's own CPU time was not recorded); per_lower is the run's "
                     "CPU time, the per-phase cost if the Adam phase's host CPU time equalled its "
                     "wall time"}


def pinn_cost(contestants, records=None):
    """The PINN's cost: no fixed cost, and per use the mean over its base runs (variant "pure",
    the plan's six showcase cases without artificial viscosity) of the per-phase cost, with the
    mean wall and CPU times beside it; None if there are no base runs."""
    records = records or {}
    runs = []
    for _, gname, a, preds in contestants[1:]:
        if gname != "256x97" or a["method"] != "pinn" or a.get("variant") != "pure":
            continue
        key = f"pure/{next(iter(preds))}" if preds else None
        rec = records.get(f"{a.get('formulation', 'euler')}:{key}", records.get(key))
        runs.append(_pinn_run_cost(a, rec))
    if not runs:
        return None
    out = {"fixed": 0.0, "per": float(np.mean([r["phase"] for r in runs])),
           "basis": "; ".join(sorted({r["basis"] for r in runs})),
           "per_wall": float(np.mean([r["wall"] for r in runs])),
           "per_cpu": float(np.mean([r["cpu"] for r in runs])), "n_runs": len(runs)}
    if any(r["lower"] is not None for r in runs):
        out["per_lower"] = float(np.mean([r["phase"] if r["lower"] is None else r["lower"]
                                          for r in runs]))
    return out


def costs_record(by_tag, campaign, su2_per, tuning_path, campaign_wall=None, su2_per_wall=None):
    costs = {"su2": {"fixed": 0.0, "per": su2_per, "basis": "cpu", "per_cpu": su2_per,
                     "per_wall": su2_per_wall}}
    for method, tag in COST_TAGS.items():
        if tag not in by_tag:
            continue
        a = by_tag[tag]
        final, basis = _train_cost(a)
        if method == "fno":
            if not Path(tuning_path).exists():
                raise SystemExit(f"{tuning_path} not found: the FNO's fixed cost includes its "
                                 "tuning runs, so S4 needs stage S2's tuning record")
            search, n_search = _tuning_cost(tuning_path)
            with_search, final_only = final + search, final
            search_cpu, search_wall = _tuning_bases(tuning_path)
            note = f"campaign + {n_search} tuning runs + final training ({tag})"
        else:
            with_search, final_only = final, float(a["fit_cpu_s"])
            search_cpu = search_wall = 0.0  # POD's search is inside its training time
            note = f"campaign + mode/smoothing search including the final fit ({tag})"
        costs[method] = {"fixed": campaign + with_search,
                         "fixed_final_fit_only": campaign + final_only,
                         "per": float(a["query_s"]), "train_cost_basis": basis, "fixed_is": note,
                         "fixed_cpu": campaign + search_cpu + float(a["train_cpu_s"]),
                         "fixed_wall": (None if campaign_wall is None else
                                        campaign_wall + search_wall + float(a["train_wall_s"]))}
    return costs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    truth, floor, regimes = load_truth()
    contestants = [floor_contestant(floor)] + load_contestants("runs/predictions")
    grids = {}
    rows = score_rows(contestants, truth, regimes, specs, grids)
    # SU2's own physics checks on the truth itself (spec 8.2: every column including SU2)
    for cid, q in truth.items():
        s = specs[cid]
        X, Y = grids.get(cid) or grid(Section(s.t, s.m), TRAIN)
        rows.append({"tag": "su2-512x193", "method": "su2-512x193", "variant": "truth",
                     "grid": "256x97", "n_train": 0, "seed": 0, "split": cid.split("-")[0],
                     "regime": regimes[cid], "case_id": cid,
                     "score": score_case(q, q, X, Y, s.mach, s.alpha_deg, regimes[cid])})
    groups = aggregate(rows)
    shock_groups, physics_groups = aggregate_shocks(rows), aggregate_physics(rows)
    (OUT / "scores.json").write_text(json.dumps(
        {"rows": rows, "groups": groups, "shock_groups": shock_groups,
         "physics_groups": physics_groups}, indent=2, default=float))
    (OUT / "scores.md").write_text(
        "## Forces, surface and near field\n\n"
        "The su2-512x193 / truth rows score the truth against itself: their error columns are zero "
        "by construction; only their physics checks carry information.\n\n"
        + markdown_table(groups) + "\n\n## Shocks (supercritical cases)\n\n"
        "Pooled over every (case, surface) where the truth has a shock; a miss is a predicted "
        "surface with no supersonic compression.\n\n"
        + markdown_table(shock_groups, SHOCK_METRICS) + "\n\n## Physics checks\n\n"
        "'by construction' marks a check the contestant satisfies identically (spec 8.2): it is "
        "not counted as a pass. Mass flux is |net flux| through closed grid line j. "
        "n_nonpositive_state counts cases with rho <= 0 or p <= 0 somewhere in the near field: "
        "entropy is undefined there, so the entropy columns are over the other cases only.\n\n"
        + markdown_table(physics_groups, PHYSICS_METRICS) + "\n")

    # costs, in seconds
    su2_glob = "runs/su2/256x97/*/result.json"
    results = [json.loads(p.read_text()) for p in Path(".").glob(su2_glob)]
    test_cpu = [r["cpu_s"] for r in results if r["case_id"].startswith("test")]
    if not test_cpu:
        raise SystemExit(f"no 'test*' result.json under {su2_glob}: found {len(results)} "
                         "result.json files in total; cannot cost SU2 per query")
    per_su2 = float(np.median(test_cpu))
    per_su2_wall = float(np.median([r["wall_s"] for r in results
                                    if r["case_id"].startswith("test")]))

    # paid for: every val case and every train case among the first n Sobol points, failed or not
    def paid(r, n):
        return r["case_id"].startswith("val") or (
            r["case_id"].startswith("train") and int(r["case_id"][-4:]) < n)

    campaign = {n: sum(r["cpu_s"] for r in results if paid(r, n))
                for n in (64, 128, 256, 512, 896)}
    campaign_wall = {n: sum(r["wall_s"] for r in results if paid(r, n))
                     for n in (64, 128, 256, 512, 896)}
    by_tag = {tag: attrs for tag, gname, attrs, _ in contestants[1:] if gname == "256x97"}
    costs = costs_record(by_tag, campaign[896], per_su2, Path("results/s2/tuning.json"),
                         campaign_wall[896], per_su2_wall)
    pinn = pinn_cost(contestants, load_pinn_records())
    if pinn:
        costs["pinn"] = pinn
    (OUT / "costs.json").write_text(json.dumps(
        {"costs": costs, "campaign_cpu_s": campaign, "campaign_wall_s": campaign_wall,
         "break_even_uses": "fixed: campaign + model search + final training, for FNO and POD; "
                            "per: SU2 CPU time, the PINN per phase (see its basis)",
         "bases": "per_cpu / per_wall and fixed_cpu / fixed_wall: the same costs on CPU time "
                  "alone and on wall time alone (the campaign's wall time is summed over its "
                  "concurrent workers)"},
        indent=2))
    plot_break_even(break_even(costs, 5000), OUT / "break_even.png")

    # data efficiency on test_in, by exact tag
    pts = {}
    for method, fmt in (("fno", "fno-n{}-s0"), ("pod", "pod-n{}")):
        n_paid, err, n_cases = [], [], []
        for n in (64, 128, 256, 512, 896):
            tag = fmt.format(n)
            if tag not in by_tag:
                continue
            vals = [r["score"]["near_field_rel_l2"] for r in rows
                    if r["tag"] == tag and r["grid"] == "256x97" and r["split"] == "test_in"
                    and r["score"]["status"] == "ok"]
            if not vals:
                print(f"data efficiency: dropping {tag} (no test_in case scored ok)")
            n_paid.append(n)
            err.append(float(np.mean(vals)) if vals else float("nan"))
            n_cases.append(int(by_tag[tag].get("n_cases", n)))
        pts[method] = (n_paid, err, n_cases)
    fl = [r["score"]["near_field_rel_l2"] for r in rows
          if r["method"] == "su2-256x97" and r["split"] == "test_in"]
    plot_efficiency(pts, float(np.mean(fl)), OUT / "data_efficiency.png")

    # showcase wall pressure, by exact tag (plus every PINN file)
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    shown = set(COST_TAGS.values())
    for c in show:
        cid = c["case_id"]
        s = specs[cid]
        X, _ = grids.get(cid) or grid(Section(s.t, s.m), TRAIN)
        curves = {"truth": pressure_coefficient(truth[cid][:, 0, 3], s.mach),
                  "su2 256x97": pressure_coefficient(floor[cid][:, 0, 3], s.mach)}
        for tag, gname, attrs, preds in contestants[1:]:
            if gname == "256x97" and cid in preds and (attrs["method"] == "pinn" or tag in shown):
                label = f"{attrs['method']} {attrs.get('variant', '')}".strip()
                curves[label] = pressure_coefficient(preds[cid][:, 0, 3], s.mach)
        plot_showcase(X[:, 0], curves, f"{cid}  max M {c['max_surface_mach']:.2f}",
                      OUT / f"showcase_{cid}.png")
    print((OUT / "scores.md").read_text())


if __name__ == "__main__":
    main()
