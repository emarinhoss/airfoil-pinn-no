"""Prediction files: one per (method, configuration, seed), q at 256x97 for the test cases.

A file may hold further datasets of the same cases (`extra`), e.g. the FNO's q_128x49, the same
model evaluated on the 128x49 subgrid it was trained on, so the resolution transfer is scored.
"""

from pathlib import Path

import h5py
import numpy as np


def write_predictions(path, case_ids, q, attrs, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["q"] = np.asarray(q, dtype=np.float64)
        for name, arr in (extra or {}).items():
            f[name] = np.asarray(arr, dtype=np.float64)
        f.create_dataset("case_id", data=list(case_ids), dtype=h5py.string_dtype())
        for k, v in attrs.items():
            f.attrs[k] = v


def read_predictions(path, dataset="q"):
    with h5py.File(path) as f:
        ids = list(f["case_id"].asstr()[...])
        q = f[dataset][...]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    return dict(zip(ids, q)), attrs


def datasets(path):
    """The per-case datasets a prediction file holds ('q' first)."""
    with h5py.File(path) as f:
        names = [k for k in f if k != "case_id"]
    return sorted(names, key=lambda k: (k != "q", k))
