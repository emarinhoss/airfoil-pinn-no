"""Prediction files: one per (method, configuration, seed), q at 256x97 for the test cases."""

import h5py
import numpy as np


def write_predictions(path, case_ids, q, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["q"] = np.asarray(q, dtype=np.float64)
        f.create_dataset("case_id", data=list(case_ids), dtype=h5py.string_dtype())
        for k, v in attrs.items():
            f.attrs[k] = v


def read_predictions(path):
    with h5py.File(path) as f:
        ids = list(f["case_id"].asstr()[...])
        q = f["q"][...]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    return dict(zip(ids, q)), attrs
