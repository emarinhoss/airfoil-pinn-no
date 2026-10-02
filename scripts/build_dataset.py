"""Build data/hdf5/{256x97,512x193}.h5 from runs/su2 and pick the PINN showcase cases.

Writes results/s1/failures_<level>.json and results/s1/showcase.json.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from apno.data.build import build_level, regime, showcase
from apno.data.sampling import case_specs
from apno.geometry.ogrid import FINE, TRAIN, level_name

ROOT, OUT, RES = Path("runs/su2"), Path("data/hdf5"), Path("results/s1")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    plan = {TRAIN: (("train", "val", "test_in", "test_ext"), np.float32),
            FINE: (("test_in", "test_ext"), np.float64)}
    for shape, (splits, dtype) in plan.items():
        lv = level_name(shape)
        fails = build_level({s: case_specs(s, shape) for s in splits}, ROOT, OUT / f"{lv}.h5",
                            dtype)
        (RES / f"failures_{lv}.json").write_text(json.dumps(fails, indent=2))
        print(lv, {s: len(v) for s, v in fails.items()}, "failures")
    with h5py.File(OUT / "512x193.h5") as f, h5py.File(OUT / "256x97.h5") as t:
        g = f["test_in"]
        ids = list(g["case_id"].asstr()[...])
        # a test case counts only if it converged on BOTH grids (it needs truth and floor)
        both = set(t["test_in"]["case_id"].asstr()[...])
        keep = [k for k, c in enumerate(ids) if c in both]
        mach = g["max_surface_mach"][...][keep]
        ids = [ids[k] for k in keep]
        picked = showcase(ids, mach)
        lookup = dict(zip(ids, mach))
    out = {"rule": "sorted by fine-grid max surface Mach; index round(p*(n-1))",
           "n_survivors": len(ids), "cases": [
               {"case_id": c, "max_surface_mach": float(lookup[c]),
                "regime": regime(lookup[c])} for c in picked]}
    (RES / "showcase.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
