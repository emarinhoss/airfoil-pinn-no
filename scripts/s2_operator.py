"""Stage S2: POD and FNO. Subcommands run independently and resume by skipping finished tags.

    python scripts/s2_operator.py tune        # 8 configurations, picks one by validation loss
    python scripts/s2_operator.py final       # 3 seeds at 896, loss B, efficiency prefixes
    python scripts/s2_operator.py pod         # POD at every prefix

Tuning selects on validation only, so tuning runs write NO test predictions: only the final runs
(fno-n896-s0, ...) and POD reach runs/predictions, each under its own tag. A run whose validation
loss is never finite is recorded as diverged, with no prediction file, and the others go on. Every
FNO run records its resolution transfer: the same model at 128x49 against 256x97 sampled at every
second node, on the validation cases; final runs also store their test predictions at 128x49
(dataset q_128x49) so S4 scores both resolutions.
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from apno.data.predictions import write_predictions
from apno.data.sampling import PREFIXES, prefix_indices
from apno.models.pod import PodModel
from apno.models.targets import Standardiser, from_targets, to_targets
from apno.models.train_fno import (
    FnoConfig, diverged, finite_or_none, load_features_only, load_model, load_split, predict,
    train,
)

H5 = Path("data/hdf5/256x97.h5")
RUNS, PRED, RES = Path("runs/fno"), Path("runs/predictions"), Path("results/s2")
TUNING = [FnoConfig(width=w, modes_i=mi, modes_j=mj, epochs=100)
          for w, mi, mj in itertools.product((32, 48), (16, 24), (8, 12))]


def _dump(path, obj):
    path.write_text(json.dumps(finite_or_none(obj), indent=2, allow_nan=False))


def transfer_rel_l2(q_fine, q_coarse, inputs):
    """Mean over cases of the relative L2 difference, in target space, between the prediction at
    256x97 sampled at every second node and the prediction at 128x49 (0 for an exact transfer)."""
    a = to_targets(q_fine[:, ::2, ::2], inputs).reshape(len(inputs), -1)
    b = to_targets(q_coarse, inputs).reshape(len(inputs), -1)
    return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))


def _predict_both(model, std, split):
    f = load_features_only(H5, split)
    q, sec = predict(model, std, f["xy"], f["inputs"])
    qc, _ = predict(model, std, f["xy"][:, ::2, ::2], f["inputs"])
    return f, q, qc, sec


def _run_fno(tag, cfg, variant="", write=True):
    d = RUNS / tag
    if (d / "train.json").exists():
        res = json.loads((d / "train.json").read_text())
        if res["cfg"] != asdict(cfg):
            raise ValueError(f"{tag}: cached train.json cfg differs from the requested cfg; "
                             f"refusing to reuse (remove {d} deliberately to retrain)")
    else:
        res = train(cfg, H5, d)
    row = {"tag": tag, "status": "ok", "best_val": res["best_val"],
           "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
           "wall_s": res["wall_s"], "cpu_s": res["cpu_s"], "device": res.get("device", "cpu"),
           "n_cases": res.get("n_cases"), "cfg": asdict(cfg)}
    if diverged(res):
        row["status"] = "diverged"  # recorded, not skipped or replaced (spec 8.4)
        print(f"{tag}: DIVERGED (no finite validation loss); no prediction file", flush=True)
        return row
    model, std, _ = load_model(res["checkpoint"])
    f, qv, qvc, _ = _predict_both(model, std, "val")
    row["transfer_val_rel_l2"] = transfer_rel_l2(qv, qvc, f["inputs"])
    if write:
        ids, q, qc, secs = [], [], [], []
        for split in ("test_in", "test_ext"):
            f, a, b, sec = _predict_both(model, std, split)
            ids += f["case_id"]
            q.append(a)
            qc.append(b)
            secs.append(sec)
        q, qc, sec = np.concatenate(q), np.concatenate(qc), float(np.mean(secs))
        write_predictions(PRED / f"{tag}.h5", ids, q, {
            "method": "fno", "tag": tag, "n_train": cfg.n_train, "n_cases": row["n_cases"],
            "seed": cfg.seed, "variant": variant, "train_device": row["device"],
            "train_wall_s": res["wall_s"], "train_cpu_s": res["cpu_s"], "query_s": sec},
            extra={"q_128x49": qc})
        row["query_s"] = sec
    return row


def tune():
    rows = [_run_fno(f"tune-{k}", c, write=False) for k, c in enumerate(TUNING)]
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        _dump(RES / "tuning.json", {"rows": rows, "selected": None})
        raise SystemExit("every tuning configuration diverged; nothing to select")
    best = min(ok, key=lambda r: r["best_val"])
    _dump(RES / "tuning.json", {"rows": rows, "selected": best["tag"]})
    print(json.dumps({r["tag"]: r["best_val"] for r in rows}, indent=2), "selected", best["tag"])


def final():
    sel = json.loads((RES / "tuning.json").read_text())["selected"]
    base = replace(TUNING[int(sel.split("-")[1])], epochs=300)
    rows = [_run_fno(f"fno-n896-s{s}", replace(base, seed=s)) for s in (0, 1, 2)]
    rows.append(_run_fno("fno-n896-s0-lossB", replace(base, loss="B"), variant="lossB"))
    rows += [_run_fno(f"fno-n{n}-s0", replace(base, n_train=n)) for n in PREFIXES[:-1]]
    _dump(RES / "runs.json", rows)


def pod():
    tr, va = load_split(H5, "train"), load_split(H5, "val")
    yv = to_targets(va["q"], va["inputs"])
    rows = []
    for n in PREFIXES:
        idx = prefix_indices(tr["case_id"], n)  # the first n Sobol points minus failures
        q_tr, x_tr = tr["q"][idx], tr["inputs"][idx]
        std = Standardiser().fit(to_targets(q_tr, x_tr))
        y = std.transform(to_targets(q_tr, x_tr))
        t0, c0 = time.perf_counter(), time.process_time()
        best = None
        for k, s in itertools.product((8, 16, 32, 64, 128, 256), (0.0, 1e-6, 1e-4, 1e-2)):
            if k >= len(idx):
                continue
            m = PodModel(k, s).fit(x_tr, y)
            err = float(np.mean(np.linalg.norm((std.inverse(m.predict(va["inputs"])) - yv)
                                               .reshape(len(yv), 4, -1), axis=2)
                                / np.linalg.norm(yv.reshape(len(yv), 4, -1), axis=2)))
            if best is None or err < best[0]:
                best = (err, k, s, m)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        t1, c1 = time.perf_counter(), time.process_time()
        PodModel(best[1], best[2]).fit(x_tr, y)  # the selected model alone: final-fit cost
        fit_wall, fit_cpu = time.perf_counter() - t1, time.process_time() - c1
        ids, qs, t_q = [], [], []
        for split in ("test_in", "test_ext"):
            f = load_features_only(H5, split)
            t1 = time.perf_counter()
            qs.append(from_targets(std.inverse(best[3].predict(f["inputs"])), f["inputs"]))
            t_q.append((time.perf_counter() - t1) / len(f["inputs"]))
            ids += f["case_id"]
        write_predictions(PRED / f"pod-n{n}.h5", ids, np.concatenate(qs), {
            "method": "pod", "tag": f"pod-n{n}", "n_train": n, "n_cases": int(len(idx)),
            "seed": 0, "variant": "", "train_device": "cpu",
            "train_wall_s": wall, "train_cpu_s": cpu, "fit_wall_s": fit_wall,
            "fit_cpu_s": fit_cpu, "query_s": float(np.mean(t_q))})
        rows.append({"n": n, "n_cases": int(len(idx)), "val_err": best[0], "modes": best[1],
                     "smoothing": best[2], "wall_s": wall, "cpu_s": cpu, "fit_cpu_s": fit_cpu})
        print(rows[-1])
    _dump(RES / "pod.json", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tune", "final", "pod"))
    RES.mkdir(parents=True, exist_ok=True)
    {"tune": tune, "final": final, "pod": pod}[ap.parse_args().what]()
