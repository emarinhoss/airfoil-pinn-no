"""Stage S2: POD and FNO. Subcommands run independently and resume by skipping finished tags.

    python scripts/s2_operator.py tune        # 8 configurations, picks one by validation loss
    python scripts/s2_operator.py final       # 3 seeds at 896, loss B, efficiency prefixes
    python scripts/s2_operator.py pod         # POD at every prefix
    python scripts/s2_operator.py repredict   # FNO files under the current padding rule

Tuning selects on validation only, so tuning runs write NO test predictions: only the final runs
(fno-n896-s0, ...) and POD reach runs/predictions, each under its own tag. A run whose validation
loss is never finite is recorded as diverged, with no prediction file, and the others go on. Every
FNO run records its resolution transfer: the same model at 128x49 against 256x97 sampled at every
second node, on the validation cases; final runs also store their test predictions at 128x49
(dataset q_128x49) so S4 scores both resolutions.

`repredict` brings the final runs' 256x97 predictions and recorded transfers to the current radial
padding rule (spec 6.3, amended 2026-10-26: 16 rows at 256x97, where the runs were made with 17),
from the same checkpoints, without training. Training at 128x49 pads 8 rows under both rules, so
the models, their costs and their 128x49 predictions are unchanged. Each file must first be
reproduced under the old rule, to 1e-4 relative, or it is left alone and the command exits 4. The
old files are kept in runs/predictions_rows17/, outside S4's reach; the old transfers stay in
runs.json as transfer_val_rel_l2_rows17. A file already under the current rule is skipped.
"""

import argparse
import datetime
import itertools
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from apno.data.predictions import read_predictions, write_predictions
from apno.data.sampling import PREFIXES, prefix_indices
from apno.models.fno import radial_pad, radial_pad_period_matched
from apno.models.pod import PodModel
from apno.models.targets import Standardiser, from_targets, to_targets
from apno.models.train_fno import (
    FnoConfig, diverged, finite_or_none, load_features_only, load_model, load_split, predict,
    train,
)

H5 = Path("data/hdf5/256x97.h5")
RUNS, PRED, RES = Path("runs/fno"), Path("runs/predictions"), Path("results/s2")
ROWS17 = Path("runs/predictions_rows17")  # the final runs' files as made with 17 rows at 256x97
REPRODUCED = 1e-4  # largest relative difference from the file still counted as reproduced
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


def _max_rel_diff(ids, q, ref):
    b = np.stack([ref[c] for c in ids])
    return float(np.abs(q - b).max() / np.abs(b).max())


def repredict():
    """The final runs' 256x97 test predictions and validation transfers under the current padding
    rule; 0, or 4 if a file did not reproduce under the old rule (module docstring)."""
    rows = json.loads((RES / "runs.json").read_text())
    rows17, rows_now = radial_pad_period_matched(97), radial_pad(97)
    log, bad = [], []
    for row in rows:
        tag, path = row["tag"], PRED / f"{row['tag']}.h5"
        if row["status"] != "ok" or not path.exists():
            continue
        q_file, attrs = read_predictions(path)
        qc_file, _ = read_predictions(path, "q_128x49")
        model, std, _ = load_model(RUNS / tag / "best.pt")
        if attrs.get("radial_rows_256x97") != rows_now:
            model.pad_rule = radial_pad_period_matched
            ids, q, qc = [], [], []
            for split in ("test_in", "test_ext"):
                f, a, b, _ = _predict_both(model, std, split)
                ids += f["case_id"]
                q.append(a)
                qc.append(b)
            check = {"q": _max_rel_diff(ids, np.concatenate(q), q_file),
                     "q_128x49": _max_rel_diff(ids, np.concatenate(qc), qc_file)}
            entry = {"tag": tag, "reproduces_file_at_old_rows": check,
                     "reproduced": max(check.values()) <= REPRODUCED}
            log.append(entry)
            if not entry["reproduced"]:
                bad.append(tag)
                print(f"{tag}: NOT REPRODUCED at {rows17} rows {check}; left as it is", flush=True)
                continue
            model.pad_rule = radial_pad
            q, secs = [], []
            for split in ("test_in", "test_ext"):
                f = load_features_only(H5, split)
                a, sec = predict(model, std, f["xy"], f["inputs"])
                q.append(a)
                secs.append(sec)
            q = np.concatenate(q)
            entry["max_rel_change"] = _max_rel_diff(ids, q, q_file)
            ROWS17.mkdir(parents=True, exist_ok=True)
            if not (ROWS17 / path.name).exists():
                shutil.copy2(path, ROWS17 / path.name)
            tmp = path.with_suffix(".tmp")
            write_predictions(tmp, ids, q, {
                **attrs, "radial_rows_256x97": rows_now, "radial_rows_256x97_before": rows17,
                "repredicted": datetime.date.today().isoformat(),
                "repredict_query_s": float(np.mean(secs))},
                extra={"q_128x49": np.stack([qc_file[c] for c in ids])})
            os.replace(tmp, path)
            print(f"{tag}: {len(ids)} test cases at {rows_now} rows; reproduced at {rows17} rows "
                  f"to {max(check.values()):.1e}; largest change {entry['max_rel_change']:.3f}",
                  flush=True)
        if row.get("radial_rows_256x97") != rows_now:
            f, qv, qvc, _ = _predict_both(model, std, "val")
            row.setdefault("transfer_val_rel_l2_rows17", row["transfer_val_rel_l2"])
            row["transfer_val_rel_l2"] = transfer_rel_l2(qv, qvc, f["inputs"])
            row["radial_rows_256x97"] = rows_now
            _dump(RES / "runs.json", rows)
    if log:
        _dump(RES / f"repredict_{datetime.date.today().isoformat()}.json",
              {"rows": rows_now, "rows_before": rows17,
               "reproduced_if_max_rel_diff_at_most": REPRODUCED, "files": log})
    if bad:
        print(f"not re-predicted: {', '.join(bad)}")
        return 4
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tune", "final", "pod", "repredict"))
    RES.mkdir(parents=True, exist_ok=True)
    sys.exit({"tune": tune, "final": final, "pod": pod,
              "repredict": repredict}[ap.parse_args().what]())
