"""Stages S2 and S4 executed end to end on a small synthetic dataset at the real resolutions.

The synthetic fields are smooth and analytic; nothing here measures accuracy. What is checked is
the bookkeeping that per-function tests cannot see: which runs write prediction files, which files
S4 costs, pools and plots, and that a diverged run is recorded rather than blocking the stage.
"""

import importlib.util
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apno.data.sampling import case_specs
from apno.geometry.ogrid import FINE, TRAIN, Section, grid
from apno.models import train_fno
from apno.models.train_fno import FnoConfig
from apno.physics.flow import local_mach

SCRIPTS = Path(__file__).parents[1] / "scripts"
FAILED_TRAIN = (2, 5)  # Sobol points that "did not converge"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _field(spec, shape):
    X, Y = grid(Section(spec.t, spec.m), shape)
    a = np.radians(spec.alpha_deg)
    bump = np.exp(-((X - 0.5) ** 2 + Y**2))
    q = np.stack([1 + 0.02 * bump, np.cos(a) + 0.05 * bump, np.sin(a) + 0.01 * Y * bump,
                  1 - 0.05 * bump], -1)
    return np.stack([X, Y], -1), q


def _write_level(path, splits, shape, dtype):
    with h5py.File(path, "w") as f:
        for split, specs in splits.items():
            xy, q = zip(*(_field(s, shape) for s in specs))
            q = np.stack(q)
            g = f.create_group(split)
            g["xy"] = np.stack(xy).astype(dtype)
            g["q"] = q.astype(dtype)
            g["inputs"] = np.array([[s.mach, s.alpha_deg, s.t, s.m] for s in specs])
            g["forces_su2"] = np.zeros((len(specs), 3))
            mach = [float(local_mach(*(qq[:, 0, k] for k in range(4)), s.mach).max())
                    for qq, s in zip(q, specs)]
            mach[0] = 1.05  # one case labelled supercritical, so the shock tables run
            g["max_surface_mach"] = np.array(mach)
            g.create_dataset("case_id", data=[s.case_id for s in specs],
                             dtype=h5py.string_dtype())


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("stages")
    specs = {"train": [s for k, s in enumerate(case_specs("train", TRAIN)[:18])
                       if k not in FAILED_TRAIN],
             "val": case_specs("val", TRAIN)[:4], "test_in": case_specs("test_in", TRAIN)[:7],
             "test_ext": case_specs("test_ext", TRAIN)[:2]}
    (root / "data/hdf5").mkdir(parents=True)
    _write_level(root / "data/hdf5/256x97.h5", specs, TRAIN, np.float32)
    _write_level(root / "data/hdf5/512x193.h5",
                 {k: [s.__class__(s.case_id, s.mach, s.alpha_deg, s.t, s.m, FINE) for s in v]
                  for k, v in specs.items() if k.startswith("test")}, FINE, np.float64)
    paid = case_specs("train", TRAIN)[:18] + specs["val"] + specs["test_in"] + specs["test_ext"]
    for k, s in enumerate(paid):
        d = root / "runs/su2/256x97" / s.case_id
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"case_id": s.case_id, "cpu_s": 100.0 + k}))
    with h5py.File(root / "data/hdf5/512x193.h5") as f:
        mach = dict(zip(f["test_in"]["case_id"].asstr()[...], f["test_in"]["max_surface_mach"]))
    (root / "results/s1").mkdir(parents=True)
    (root / "results/s1/showcase.json").write_text(json.dumps({"cases": [
        {"case_id": c, "max_surface_mach": float(m), "regime": "subcritical"}
        for c, m in list(mach.items())[:6]]}))
    return root


def test_s2_then_s4(staged, monkeypatch, capsys):
    monkeypatch.chdir(staged)
    monkeypatch.setattr(train_fno, "default_device", lambda: "cpu")
    real_loss = train_fno.loss_fn

    def loss(pred, true, cfg):  # the width-4 configuration never has a finite loss
        return real_loss(pred, true, cfg) * (float("nan") if cfg.width == 4 else 1.0)

    monkeypatch.setattr(train_fno, "loss_fn", loss)
    s2 = _module("s2_operator")
    tiny = dict(modes_i=4, modes_j=3, layers=2, epochs=2, patience=1, lr=0.0, batch=8)
    monkeypatch.setattr(s2, "TUNING", [FnoConfig(width=8, **tiny), FnoConfig(width=4, **tiny)])
    s2.RES.mkdir(parents=True, exist_ok=True)

    s2.tune()
    s2.tune()  # resumes over the cached diverged run instead of dying in load_model
    tuning = json.loads(Path("results/s2/tuning.json").read_text(), parse_constant=float)
    assert tuning["selected"] == "tune-0"
    assert [r["status"] for r in tuning["rows"]] == ["ok", "diverged"]
    assert tuning["rows"][1]["best_val"] is None
    assert not list(Path("runs/predictions").glob("tune-*"))  # tuning selects on val only
    s2.final()
    s2.pod()
    preds = sorted(p.stem for p in Path("runs/predictions").glob("*.h5"))
    assert not [p for p in preds if p.startswith("tune")]
    assert "fno-n896-s0" in preds and "pod-n896" in preds and "fno-n64-s0" in preds
    runs = json.loads(Path("results/s2/runs.json").read_text())
    assert all(r["transfer_val_rel_l2"] >= 0 for r in runs)
    assert {r["n_cases"] for r in runs} == {16}  # 18 Sobol points paid for, 2 failed

    s4 = _module("s4_score")
    s4.main()
    out = Path("results/s4")
    costs = json.loads((out / "costs.json").read_text())
    campaign = costs["campaign_cpu_s"]["896"]
    fno_attrs = h5py.File("runs/predictions/fno-n896-s0.h5").attrs
    tune_cpu = sum(r["cpu_s"] for r in tuning["rows"])
    fno = costs["costs"]["fno"]
    assert fno["fixed"] == pytest.approx(campaign + tune_cpu + fno_attrs["train_cpu_s"])
    assert fno["fixed_final_fit_only"] == pytest.approx(campaign + fno_attrs["train_cpu_s"])
    assert fno["per"] == pytest.approx(fno_attrs["query_s"]) and fno["train_cost_basis"] == "cpu"
    pod_attrs = h5py.File("runs/predictions/pod-n896.h5").attrs
    assert costs["costs"]["pod"]["fixed_final_fit_only"] == pytest.approx(
        campaign + pod_attrs["fit_cpu_s"])

    scores = json.loads((out / "scores.json").read_text())
    g = [g for g in scores["groups"] if g["method"] == "fno" and g["seed"] == 0
         and g["n_train"] == 896 and g["variant"] == "" and g["grid"] == "256x97"
         and g["split"] == "test_in"]
    assert sum(x["n_ok"] for x in g) == 7  # one model's seven test_in cases, nothing pooled
    assert any(x["grid"] == "128x49" and x["method"] == "fno" for x in scores["groups"])
    assert scores["shock_groups"] and scores["physics_groups"]
    md = (out / "scores.md").read_text()
    assert "## Shocks" in md and "## Physics checks" in md
    assert len(list(out.glob("showcase_*.png"))) == 6
    assert (out / "data_efficiency.png").exists() and (out / "break_even.png").exists()

    shutil.copy("runs/predictions/fno-n896-s0.h5", "runs/predictions/stray.h5")
    with pytest.raises(SystemExit, match="pool"):
        s4.main()


def test_s4_costs_the_pure_pinn_runs_only():
    s4 = _module("s4_score")

    def pinn(variant, wall_s, grid="256x97", **extra):
        attrs = {"method": "pinn", "variant": variant, "train_device": "mps",
                 "train_wall_s": wall_s, "train_cpu_s": 1.0, **extra}
        return (f"pinn-{variant}", grid, attrs, {})

    contestants = [("su2-256x97", "256x97", {"method": "su2-256x97"}, {}),
                   pinn("pure", 100.0), pinn("pure", 300.0), pinn("gradnorm", 1e4),
                   pinn("nokutta", 1e4), pinn("euler", 1e4), pinn("pure", 1e4, grid="128x49"),
                   ("fno-n896-s0", "256x97", {"method": "fno", "variant": ""}, {})]
    assert s4.pinn_cost(contestants) == {"fixed": 0.0, "per": 200.0}
    assert s4.pinn_cost(contestants[:1] + contestants[3:5]) is None
