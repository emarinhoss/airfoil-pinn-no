"""Raw SU2 case directories -> one HDF5 file per grid level, deterministically (spec 5.1-5.3).

Only converged cases are written; every failure is returned with its reason, never silently
dropped. Each case's restart nodes are checked against a freshly generated grid, so a node-ordering
mistake fails loudly instead of corrupting data.
"""

import json

import h5py
import numpy as np

from apno.geometry.ogrid import Section, grid
from apno.physics.flow import nondimensionalise
from apno.su2.restart import on_grid, read_restart
from apno.su2.run import case_dir

SHOWCASE_PERCENTILES = (0.10, 0.30, 0.50, 0.70, 0.85, 0.95)


def regime(max_surface_mach):
    return "supercritical" if max_surface_mach > 1.0 else "subcritical"


def showcase(case_ids, max_mach):
    n = len(case_ids)
    if n < len(SHOWCASE_PERCENTILES):
        raise ValueError(f"need at least 6 surviving test cases, have {n}")
    order = np.argsort(np.asarray(max_mach), kind="stable")
    return [case_ids[order[round(p * (n - 1))]] for p in SHOWCASE_PERCENTILES]


def load_case(root, spec):
    d = case_dir(root, spec)
    if not (d / "result.json").exists():
        return None
    res = json.loads((d / "result.json").read_text())
    if not res["converged"]:
        return None
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    f = on_grid(read_restart(d / "restart.dat"), X, Y)
    q = nondimensionalise(f["Density"], f["Momentum_x"], f["Momentum_y"], f["Energy"], spec.mach)
    return {"case_id": spec.case_id, "xy": np.stack([X, Y], -1), "q": np.stack(q, -1),
            "inputs": np.array([spec.mach, spec.alpha_deg, spec.t, spec.m]),
            "forces": np.array([res["cl"], res["cd"], res["cm"]]),
            "max_surface_mach": res["max_surface_mach"]}


def build_level(specs_by_split, root, out_path, dtype):
    failures = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for split, specs in specs_by_split.items():
            cases, failures[split] = [], []
            for spec in specs:
                c = load_case(root, spec)
                if c is None:
                    rj = case_dir(root, spec) / "result.json"
                    why = json.loads(rj.read_text())["reason"] if rj.exists() else "not run"
                    failures[split].append({"case_id": spec.case_id, "reason": why,
                                            "inputs": spec.inputs()})
                else:
                    cases.append(c)
            if not cases:
                reasons = sorted({x["reason"] for x in failures[split]})[:5]
                raise ValueError(f"split {split!r}: no converged cases among "
                                 f"{len(failures[split])}; first reasons: {reasons}")
            g = f.create_group(split)
            g["xy"] = np.stack([c["xy"] for c in cases]).astype(dtype)
            g["q"] = np.stack([c["q"] for c in cases]).astype(dtype)
            g["inputs"] = np.stack([c["inputs"] for c in cases])
            g["forces_su2"] = np.stack([c["forces"] for c in cases])
            g["max_surface_mach"] = np.array([c["max_surface_mach"] for c in cases])
            g.create_dataset("case_id", data=[c["case_id"] for c in cases],
                             dtype=h5py.string_dtype())
    tmp.replace(out_path)
    return failures
