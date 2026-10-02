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


def costs_record(by_tag, campaign, su2_per, tuning_path):
    costs = {"su2": {"fixed": 0.0, "per": su2_per}}
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
            note = f"campaign + {n_search} tuning runs + final training ({tag})"
        else:
            with_search, final_only = final, float(a["fit_cpu_s"])
            note = f"campaign + mode/smoothing search including the final fit ({tag})"
        costs[method] = {"fixed": campaign + with_search,
                         "fixed_final_fit_only": campaign + final_only,
                         "per": float(a["query_s"]), "train_cost_basis": basis, "fixed_is": note}
    return costs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    truth, floor, regimes = {}, {}, {}
    for split in ("test_in", "test_ext"):
        t, mach = _group("data/hdf5/512x193.h5", split)
        fl, _ = _group("data/hdf5/256x97.h5", split)
        for cid in t:
            if cid in fl:
                truth[cid], floor[cid] = t[cid][::2, ::2], fl[cid]
                regimes[cid] = regime(mach[cid])
    contestants = [("su2-256x97", "256x97",
                    {"method": "su2-256x97", "variant": "floor", "n_train": 0, "seed": 0}, floor)]
    contestants += load_contestants("runs/predictions")
    rows = []
    grids = {}
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
                         "grid": gname, "n_train": int(attrs.get("n_train", 0)),
                         "seed": int(attrs.get("seed", 0)), "split": cid.split("-")[0],
                         "regime": regimes[cid], "case_id": cid,
                         "score": score_case(preds.get(cid), t, X, Y, s.mach, s.alpha_deg,
                                             regimes[cid], flux)})
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
        "not counted as a pass. Mass flux is |net flux| through closed grid line j.\n\n"
        + markdown_table(physics_groups, PHYSICS_METRICS) + "\n")

    # costs, in seconds
    su2_glob = "runs/su2/256x97/*/result.json"
    results = [json.loads(p.read_text()) for p in Path(".").glob(su2_glob)]
    test_cpu = [r["cpu_s"] for r in results if r["case_id"].startswith("test")]
    if not test_cpu:
        raise SystemExit(f"no 'test*' result.json under {su2_glob}: found {len(results)} "
                         "result.json files in total; cannot cost SU2 per query")
    per_su2 = float(np.median(test_cpu))
    # paid for: every val case and every train case among the first n Sobol points, failed or not
    campaign = {n: sum(r["cpu_s"] for r in results
                       if r["case_id"].startswith("val")
                       or (r["case_id"].startswith("train") and int(r["case_id"][-4:]) < n))
                for n in (64, 128, 256, 512, 896)}
    by_tag = {tag: attrs for tag, gname, attrs, _ in contestants[1:] if gname == "256x97"}
    costs = costs_record(by_tag, campaign[896], per_su2, Path("results/s2/tuning.json"))
    pinn = [_train_cost(a)[0] for _, gname, a, _ in contestants[1:]
            if gname == "256x97" and a["method"] == "pinn" and a.get("variant") == "euler"]
    if pinn:
        costs["pinn"] = {"fixed": 0.0, "per": float(np.mean(pinn))}
    (OUT / "costs.json").write_text(json.dumps(
        {"costs": costs, "campaign_cpu_s": campaign,
         "break_even_uses": "fixed: campaign + model search + final training, for FNO and POD"},
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
