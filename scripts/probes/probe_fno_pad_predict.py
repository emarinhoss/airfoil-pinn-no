"""S2's FNO2d, re-evaluated on the test cases with a different number of padding rows at 256x97.

The padding sweep (docs/findings.md, 2026-10-26) found that 16 radial padding rows at 256x97,
where `FNO2d`'s rule uses 17, lower the transfer and the validation near field for every checkpoint.
The padding does not change training at 128x49, so S2's own trained models are re-evaluated
here, with no retraining: each checkpoint predicts the test cases at 256x97 with `--rows` rows,
through the same override the sweep uses (`probe_fno_pad_sweep.rows_at`). Test inputs are read
through `load_features_only`, as S2 reads them; no test field is read.

Before it is trusted, each checkpoint also predicts the test cases at its own 17 rows. Those
predictions are compared with S2's committed prediction file (`runs/predictions/<tag>.h5`), and
the largest relative difference is recorded. That shows this re-evaluation reproduces S2's
pipeline, so the only change is the row count.

The new files go to runs/predictions_pad<rows>/<tag>-pad<rows>.h5 (variant `pad<rows>`), never
runs/predictions/, which S4 reads in full. Score them beside S2's files with the library scorer:

    python scripts/probes/probe_fno_pad_predict.py           # S2's seven loss-A runs, 16 rows
    python scripts/probes/score_fno_libraries.py --pred-dir runs/predictions_pad16 \\
        --out results/fno_pad16 --reference fno-n896-s0 --reference fno-n896-s1 \\
        --reference fno-n896-s2 --reference fno-n64-s0 --reference fno-n128-s0 \\
        --reference fno-n256-s0 --reference fno-n512-s0

The scorer will say the pad16 files have no training record: they are S2's models, whose records
are S2's (results/s2/runs.json). The reproduction check is written to
results/fno_pad<rows>/predict.json; the script exits 4 if a checkpoint does not reproduce S2's file.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import probe_fno_libraries as probe
from apno.data.predictions import read_predictions, write_predictions
from apno.models.train_fno import load_features_only
from probe_fno_pad_sweep import load_checkpoint, native_rows, rows_at

DEFAULT_CHECKPOINTS = ("runs/fno/fno-n*-s[0-9]/best.pt",)  # S2's loss-A runs; not -lossB
REPRODUCED = 1e-4  # largest relative difference from S2's file still counted as reproduced


def predict_test(model, std, h5_path, rows=None):
    """(case ids, q at 256x97) for test_in and test_ext; rows None = the model's own padding."""
    with h5py.File(h5_path) as f:
        splits = [s for s in ("test_in", "test_ext") if s in f]
    ids, qs = [], []
    for split in splits:
        feat = load_features_only(h5_path, split)
        if rows is None:
            q, _ = probe.predict(model, std, feat["xy"], feat["inputs"])
        else:
            with rows_at(model, rows):
                q, _ = probe.predict(model, std, feat["xy"], feat["inputs"])
        ids += feat["case_id"]
        qs.append(q)
    return ids, np.concatenate(qs)


def reproduction(ids, q_native, s2_file):
    """The largest relative difference between this run's own-rows predictions and S2's file, on
    the cases both hold; None if there is no file."""
    if not Path(s2_file).exists():
        return None
    ref, _ = read_predictions(s2_file)
    common = [k for k, c in enumerate(ids) if c in ref]
    if not common:
        return None
    a = q_native[common]
    b = np.stack([ref[ids[k]] for k in common])
    return {"cases": len(common), "max_rel_diff": float(np.abs(a - b).max() / np.abs(b).max())}


def run(paths, rows, h5_path, device, out_dir, check_dir, log=print):
    out_dir = Path(out_dir)
    records = []
    for path in paths:
        model, std, tag, source = load_checkpoint(path)
        model = model.to(device)
        t0 = time.perf_counter()
        ids, q_native = predict_test(model, std, h5_path)
        _, q_rows = predict_test(model, std, h5_path, rows)
        check = reproduction(ids, q_native, Path(check_dir) / f"{tag}.h5")
        cfg = model_cfg(path)
        out = out_dir / f"{tag}-pad{rows}.h5"
        write_predictions(out, ids, q_rows,
                          {"method": "fno", "variant": f"pad{rows}", "n_train": cfg["n_train"],
                           "seed": cfg["seed"], "rows": rows, "source_tag": tag,
                           "native_rows": native_rows(model)})
        rec = {"tag": tag, "source": source, "checkpoint": str(path), "rows": rows,
               "native_rows": native_rows(model), "n_cases": len(ids), "device": str(device),
               "wall_s": time.perf_counter() - t0, "prediction": str(out),
               "reproduces_s2": check,
               "reproduced": None if check is None else check["max_rel_diff"] <= REPRODUCED}
        records.append(rec)
        note = ("no S2 file to check against" if check is None else
                f"own rows against S2's file: {check['max_rel_diff']:.2e} over {check['cases']} "
                "cases" + ("" if rec["reproduced"] else " -- NOT REPRODUCED"))
        log(f"{tag}: {len(ids)} test cases at {rows} rows -> {out}; {note}")
    return records


def model_cfg(path):
    import torch
    return torch.load(path, weights_only=True, map_location="cpu")["cfg"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", help="best.pt files or globs; default: S2's seven "
                    "loss-A runs, runs/fno/fno-n*-s[0-9]/best.pt")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--h5", default="data/hdf5/256x97.h5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check-dir", default="runs/predictions",
                    help="S2's prediction files, to check the own-rows predictions against")
    ap.add_argument("--out-dir", default=None, help="default runs/predictions_pad<rows>")
    ap.add_argument("--results", default=None, help="default results/fno_pad<rows>")
    args = ap.parse_args(argv)
    if args.rows < 0:
        ap.error("--rows must be non-negative")
    out_dir = Path(args.out_dir or f"runs/predictions_pad{args.rows}")
    if out_dir.resolve() == Path("runs/predictions").resolve():
        print("refusing to write into runs/predictions/, which S4 reads in full")
        return 2
    patterns = args.checkpoints or DEFAULT_CHECKPOINTS
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not Path(args.h5).exists():
        print(f"no dataset at {args.h5}")
        return 2
    if not paths:
        print("no checkpoint matches " + ", ".join(patterns))
        return 2
    device = probe.resolve_device(args.device)
    records = run(paths, args.rows, args.h5, device, out_dir, args.check_dir)
    results = Path(args.results or f"results/fno_pad{args.rows}")
    results.mkdir(parents=True, exist_ok=True)
    (results / "predict.json").write_text(json.dumps(
        {"rows": args.rows, "reproduced_if_max_rel_diff_at_most": REPRODUCED,
         "records": records}, indent=1))
    bad = [r["tag"] for r in records if r["reproduced"] is False]
    if bad:
        print(f"NOT REPRODUCED at own rows: {', '.join(bad)}. Their {args.rows}-row files are "
              "written but should not be scored until the difference is understood.")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
