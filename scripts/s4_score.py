"""Stage S4: score every prediction file, the floor and SU2 itself; write tables, plots, costs."""

import json
from pathlib import Path

import h5py
import numpy as np

from apno.data.build import regime
from apno.data.predictions import read_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import pressure_coefficient
from apno.score.metrics import score_case
from apno.score.report import (
    aggregate, break_even, markdown_table, plot_break_even, plot_efficiency, plot_showcase,
)

OUT = Path("results/s4")


def _group(path, split):
    with h5py.File(path) as f:
        g = f[split]
        return dict(zip(g["case_id"].asstr()[...], g["q"][...])), dict(
            zip(g["case_id"].asstr()[...], g["max_surface_mach"][...]))


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
    contestants = [({"method": "su2-256x97", "variant": "floor", "n_train": 0, "seed": 0}, floor)]
    for p in sorted(Path("runs/predictions").glob("*.h5")):
        q, attrs = read_predictions(p)
        contestants.append((attrs, q))
    rows = []
    for attrs, preds in contestants:
        scope = list(preds) if attrs["method"] == "pinn" else list(truth)
        for cid in scope:
            if cid not in truth:
                continue
            s = specs[cid]
            X, Y = grid(Section(s.t, s.m), TRAIN)
            rows.append({"method": attrs["method"], "variant": attrs.get("variant", ""),
                         "n_train": int(attrs.get("n_train", 0)), "seed": int(attrs.get("seed", 0)),
                         "split": cid.split("-")[0], "regime": regimes[cid], "case_id": cid,
                         "score": score_case(preds.get(cid), truth[cid], X, Y, s.mach,
                                             s.alpha_deg, regimes[cid])})
    # SU2's own physics checks on the truth itself (spec 8.2: every column including SU2)
    for cid, q in truth.items():
        s = specs[cid]
        X, Y = grid(Section(s.t, s.m), TRAIN)
        rows.append({"method": "su2-512x193", "variant": "truth", "n_train": 0, "seed": 0,
                     "split": cid.split("-")[0], "regime": regimes[cid], "case_id": cid,
                     "score": score_case(q, q, X, Y, s.mach, s.alpha_deg, regimes[cid])})
    groups = aggregate(rows)
    (OUT / "scores.json").write_text(json.dumps({"rows": rows, "groups": groups}, indent=2,
                                                default=float))
    (OUT / "scores.md").write_text(markdown_table(groups) + "\n")

    # costs, in CPU seconds
    su2_glob = "runs/su2/256x97/*/result.json"
    results = [json.loads(p.read_text()) for p in Path(".").glob(su2_glob)]
    test_cpu = [r["cpu_s"] for r in results if r["case_id"].startswith("test")]
    if not test_cpu:
        raise SystemExit(f"no 'test*' result.json under {su2_glob}: found {len(results)} "
                         "result.json files in total; cannot cost SU2 per query")
    per_su2 = float(np.median(test_cpu))
    campaign = {n: sum(r["cpu_s"] for r in results
                       if r["case_id"].startswith("val")
                       or (r["case_id"].startswith("train") and int(r["case_id"][-4:]) < n))
                for n in (64, 128, 256, 512, 896)}
    costs = {"su2": {"fixed": 0.0, "per": per_su2}}
    for attrs, _ in contestants[1:]:
        if attrs["method"] in ("fno", "pod") and attrs["n_train"] == 896 and attrs["seed"] == 0 \
                and not attrs.get("variant"):
            costs[attrs["method"]] = {"fixed": campaign[896] + attrs["train_cpu_s"],
                                      "per": attrs["query_s"]}
    pinn = [a["train_cpu_s"] for a, _ in contestants if a["method"] == "pinn"
            and a.get("variant") == "euler"]
    if pinn:
        costs["pinn"] = {"fixed": 0.0, "per": float(np.mean(pinn))}
    (OUT / "costs.json").write_text(json.dumps({"costs": costs, "campaign_cpu_s": campaign},
                                               indent=2))
    plot_break_even(break_even(costs, 5000), OUT / "break_even.png")

    # data efficiency on test_in
    pts = {}
    for method in ("fno", "pod"):
        gs = sorted((g for g in groups if g["method"] == method and g["split"] == "test_in"
                     and g["seed"] == 0 and not g["variant"]), key=lambda g: g["n_train"])
        by_n = {}
        for g in gs:
            by_n.setdefault(g["n_train"], []).append((g["n_ok"], g.get("near_field_rel_l2_mean")))
        n = sorted(by_n)
        err = []
        for m in n:
            total = sum(k for k, _ in by_n[m])
            if total == 0:
                print(f"data efficiency: dropping {method} n_train={m} (n_ok = 0 in every group)")
                err.append(float("nan"))
            else:
                err.append(sum(k * e for k, e in by_n[m] if e is not None) / total)
        pts[method] = (n, err)
    fl = [r["score"]["near_field_rel_l2"] for r in rows
          if r["method"] == "su2-256x97" and r["split"] == "test_in"]
    plot_efficiency(pts, float(np.mean(fl)), OUT / "data_efficiency.png")

    # showcase wall pressure
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    for c in show:
        cid = c["case_id"]
        s = specs[cid]
        X, _ = grid(Section(s.t, s.m), TRAIN)
        curves = {"truth": pressure_coefficient(truth[cid][:, 0, 3], s.mach),
                  "su2 256x97": pressure_coefficient(floor[cid][:, 0, 3], s.mach)}
        for attrs, preds in contestants[1:]:
            if cid in preds and (attrs["method"] == "pinn" or
                                 (attrs["n_train"] == 896 and attrs["seed"] == 0)):
                label = f"{attrs['method']} {attrs.get('variant', '')}".strip()
                curves[label] = pressure_coefficient(preds[cid][:, 0, 3], s.mach)
        plot_showcase(X[:, 0], curves, f"{cid}  max M {c['max_surface_mach']:.2f}",
                      OUT / f"showcase_{cid}.png")
    print((OUT / "scores.md").read_text())


if __name__ == "__main__":
    main()
