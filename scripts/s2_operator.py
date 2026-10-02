"""Stage S2: POD and FNO. Subcommands run independently and resume by skipping finished tags.

    python scripts/s2_operator.py tune        # 8 configurations, picks one by validation loss
    python scripts/s2_operator.py final       # 3 seeds at 896, loss B, efficiency prefixes
    python scripts/s2_operator.py pod         # POD at every prefix
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from apno.data.predictions import write_predictions
from apno.data.sampling import PREFIXES
from apno.models.pod import PodModel
from apno.models.targets import Standardiser, from_targets, to_targets
from apno.models.train_fno import (
    FnoConfig, load_features_only, load_model, load_split, predict, train,
)

H5 = Path("data/hdf5/256x97.h5")
RUNS, PRED, RES = Path("runs/fno"), Path("runs/predictions"), Path("results/s2")
TUNING = [FnoConfig(width=w, modes_i=mi, modes_j=mj, epochs=100)
          for w, mi, mj in itertools.product((32, 48), (16, 24), (8, 12))]


def _predict_tests(model, std):
    out = {}
    for split in ("test_in", "test_ext"):
        f = load_features_only(H5, split)
        q, sec = predict(model, std, f["xy"], f["inputs"])
        out[split] = (f["case_id"], q, sec)
    ids = out["test_in"][0] + out["test_ext"][0]
    q = np.concatenate([out["test_in"][1], out["test_ext"][1]])
    return ids, q, float(np.mean([out[s][2] for s in out]))


def _run_fno(tag, cfg, variant=""):
    d = RUNS / tag
    if (d / "train.json").exists():
        res = json.loads((d / "train.json").read_text())
        if res["cfg"] != asdict(cfg):
            raise ValueError(f"{tag}: cached train.json cfg differs from the requested cfg; "
                             f"refusing to reuse (remove {d} deliberately to retrain)")
    else:
        res = train(cfg, H5, d)
    model, std, _ = load_model(res["checkpoint"])
    ids, q, sec = _predict_tests(model, std)
    write_predictions(PRED / f"{tag}.h5", ids, q, {
        "method": "fno", "n_train": cfg.n_train, "seed": cfg.seed, "variant": variant,
        "train_wall_s": res["wall_s"], "train_cpu_s": res["cpu_s"], "query_s": sec})
    return {"tag": tag, "best_val": res["best_val"], "epochs_run": res["epochs_run"],
            "wall_s": res["wall_s"], "cpu_s": res["cpu_s"], "query_s": sec, "cfg": asdict(cfg)}


def tune():
    rows = [_run_fno(f"tune-{k}", c) for k, c in enumerate(TUNING)]
    best = min(rows, key=lambda r: r["best_val"])
    (RES / "tuning.json").write_text(json.dumps({"rows": rows, "selected": best["tag"]}, indent=2))
    print(json.dumps({r["tag"]: r["best_val"] for r in rows}, indent=2), "selected", best["tag"])


def final():
    sel = json.loads((RES / "tuning.json").read_text())["selected"]
    base = replace(TUNING[int(sel.split("-")[1])], epochs=300)
    rows = [_run_fno(f"fno-n896-s{s}", replace(base, seed=s)) for s in (0, 1, 2)]
    rows.append(_run_fno("fno-n896-s0-lossB", replace(base, loss="B"), variant="lossB"))
    rows += [_run_fno(f"fno-n{n}-s0", replace(base, n_train=n)) for n in PREFIXES[:-1]]
    (RES / "runs.json").write_text(json.dumps(rows, indent=2))


def pod():
    tr, va = load_split(H5, "train"), load_split(H5, "val")
    rows = []
    for n in PREFIXES:
        std = Standardiser().fit(to_targets(tr["q"][:n], tr["inputs"][:n]))
        y = std.transform(to_targets(tr["q"][:n], tr["inputs"][:n]))
        yv = to_targets(va["q"], va["inputs"])
        t0, c0 = time.perf_counter(), time.process_time()
        best = None
        for k, s in itertools.product((8, 16, 32, 64, 128, 256), (0.0, 1e-6, 1e-4, 1e-2)):
            if k >= n:
                continue
            m = PodModel(k, s).fit(tr["inputs"][:n], y)
            err = float(np.mean(np.linalg.norm((std.inverse(m.predict(va["inputs"])) - yv)
                                               .reshape(len(yv), 4, -1), axis=2)
                                / np.linalg.norm(yv.reshape(len(yv), 4, -1), axis=2)))
            if best is None or err < best[0]:
                best = (err, k, s, m)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        ids, qs, t_q = [], [], []
        for split in ("test_in", "test_ext"):
            f = load_features_only(H5, split)
            t1 = time.perf_counter()
            qs.append(from_targets(std.inverse(best[3].predict(f["inputs"])), f["inputs"]))
            t_q.append((time.perf_counter() - t1) / len(f["inputs"]))
            ids += f["case_id"]
        write_predictions(PRED / f"pod-n{n}.h5", ids, np.concatenate(qs), {
            "method": "pod", "n_train": n, "seed": 0, "variant": "",
            "train_wall_s": wall, "train_cpu_s": cpu, "query_s": float(np.mean(t_q))})
        rows.append({"n": n, "val_err": best[0], "modes": best[1], "smoothing": best[2],
                     "wall_s": wall})
        print(rows[-1])
    (RES / "pod.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tune", "final", "pod"))
    RES.mkdir(parents=True, exist_ok=True)
    {"tune": tune, "final": final, "pod": pod}[ap.parse_args().what]()
