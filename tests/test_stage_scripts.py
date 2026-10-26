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
        (d / "result.json").write_text(json.dumps({"case_id": s.case_id, "cpu_s": 100.0 + k,
                                                   "wall_s": 110.0 + k}))
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
    # both bases, beside the one the break-even uses (spec 8.2: wall time and CPU time; review 2 F5)
    campaign_wall = costs["campaign_wall_s"]["896"]
    assert campaign_wall == pytest.approx(campaign + 10.0 * 18 + 10.0 * 4)  # wall = cpu + 10 here
    tune_wall = sum(r["wall_s"] for r in tuning["rows"])
    assert fno["fixed_cpu"] == pytest.approx(campaign + tune_cpu + fno_attrs["train_cpu_s"])
    assert fno["fixed_wall"] == pytest.approx(
        campaign_wall + tune_wall + fno_attrs["train_wall_s"])
    su2 = costs["costs"]["su2"]
    assert su2["per_cpu"] == su2["per"] and su2["per_wall"] == pytest.approx(su2["per"] + 10.0)
    assert su2["basis"] == "cpu"
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


def test_s2_repredict_moves_the_final_runs_to_the_current_padding_rule(staged, tmp_path,
                                                                      monkeypatch, capsys):
    """Runs made under the 17-row rule are re-predicted at 16 rows from their checkpoints: the
    old file must reproduce first, the 128x49 predictions, costs and other attributes are kept,
    the old files and transfers are kept aside, and a second run does nothing (spec 6.3,
    amended 2026-10-26)."""
    import torch

    from apno.models import fno
    from apno.models.train_fno import load_features_only, load_model, predict

    root = tmp_path / "s2"
    shutil.copytree(staged / "data", root / "data")
    monkeypatch.chdir(root)
    monkeypatch.setattr(train_fno, "default_device", lambda: "cpu")
    s2 = _module("s2_operator")
    tiny = dict(width=8, modes_i=4, modes_j=3, layers=2, epochs=2, patience=1, lr=0.0, batch=8)
    monkeypatch.setattr(s2, "TUNING", [FnoConfig(**tiny)])
    monkeypatch.setattr(s2, "PREFIXES", (64, 896))  # one efficiency prefix is enough here
    s2.RES.mkdir(parents=True)
    (s2.RES / "tuning.json").write_text(json.dumps({"rows": [], "selected": "tune-0"}))
    with monkeypatch.context() as m:  # the runs as S2 made them, under the 17-row rule
        m.setattr(fno, "radial_pad", fno.radial_pad_period_matched)
        s2.final()
        for ck in Path("runs/fno").glob("*/best.pt"):  # weights of a trained scale
            state = torch.load(ck, weights_only=True)
            state["model"] = {k: v * 50 if v.is_complex() else v
                              for k, v in state["model"].items()}
            torch.save(state, ck)
        s2.final()  # re-predicts from the scaled checkpoints
        assert load_model("runs/fno/fno-n896-s0/best.pt")[0].pad_rule(97) == 17
    old_runs = {r["tag"]: r for r in json.loads(Path("results/s2/runs.json").read_text())}
    tags = sorted(old_runs)
    old = {}
    for t in tags:
        with h5py.File(f"runs/predictions/{t}.h5") as f:
            old[t] = ({k: f[k][...] for k in ("q", "q_128x49")}, dict(f.attrs))
    with h5py.File("runs/predictions/fno-n64-s0.h5", "r+") as f:  # not S2's file any more
        f["q"][...] = f["q"][...] * 1.01

    assert s2.repredict() == 4
    assert "fno-n64-s0: NOT REPRODUCED" in capsys.readouterr().out
    with h5py.File("runs/predictions/fno-n64-s0.h5") as f:
        assert "radial_rows_256x97" not in f.attrs
    assert not Path("runs/predictions_rows17/fno-n64-s0.h5").exists()
    runs = {r["tag"]: r for r in json.loads(Path("results/s2/runs.json").read_text())}
    for t in tags:
        if t == "fno-n64-s0":
            continue
        with h5py.File(f"runs/predictions/{t}.h5") as f:
            q, qc, attrs = f["q"][...], f["q_128x49"][...], dict(f.attrs)
            ids = list(f["case_id"].asstr()[...])
        q_old, a_old = old[t][0]["q"], old[t][1]
        assert np.array_equal(qc, old[t][0]["q_128x49"])
        assert float(np.abs(q - q_old).max() / np.abs(q_old).max()) > 1e-4
        assert attrs["radial_rows_256x97"] == 16 and attrs["radial_rows_256x97_before"] == 17
        for k in ("train_cpu_s", "train_wall_s", "query_s", "n_train", "seed", "variant"):
            assert attrs[k] == a_old[k]
        with h5py.File(f"runs/predictions_rows17/{t}.h5") as f:
            assert np.array_equal(f["q"][...], q_old)
        model, std, _ = load_model(f"runs/fno/{t}/best.pt")  # the current rule, from scratch
        ref = np.concatenate([predict(model, std, g["xy"], g["inputs"])[0] for g in (
            load_features_only("data/hdf5/256x97.h5", s) for s in ("test_in", "test_ext"))])
        assert ids == [c for s in ("test_in", "test_ext") for c in
                       load_features_only("data/hdf5/256x97.h5", s)["case_id"]]
        assert np.allclose(q, ref, rtol=0, atol=1e-6 * np.abs(ref).max())
        r = runs[t]
        assert r["transfer_val_rel_l2_rows17"] == old_runs[t]["transfer_val_rel_l2"]
        assert r["transfer_val_rel_l2"] != r["transfer_val_rel_l2_rows17"]
        assert r["radial_rows_256x97"] == 16
    assert "radial_rows_256x97" not in runs["fno-n64-s0"]
    log = json.loads(next(Path("results/s2").glob("repredict_*.json")).read_text())
    assert {e["tag"]: e["reproduced"] for e in log["files"]}["fno-n896-s0"] is True

    Path("runs/predictions/fno-n64-s0.h5").unlink()  # a rerun skips what is done
    before = {t: Path(f"runs/predictions/{t}.h5").stat().st_mtime_ns for t in tags
              if t != "fno-n64-s0"}
    assert s2.repredict() == 0
    assert before == {t: Path(f"runs/predictions/{t}.h5").stat().st_mtime_ns for t in before}
    assert json.loads(Path("results/s2/runs.json").read_text()) == list(runs.values())


def _pinn(variant, wall_s, cpu_s=1.0, grid="256x97", device="mps", cid="test_in-0022", **extra):
    attrs = {"method": "pinn", "variant": variant, "train_device": device, "adam_device": device,
             "lbfgs_device": "cpu", "train_wall_s": wall_s, "train_cpu_s": cpu_s, **extra}
    return (f"pinn-{variant}-{cid}", grid, attrs, {cid: None})


def test_s4_costs_the_pure_pinn_runs_only_and_per_phase():
    # GPU Adam is costed at its wall time and CPU L-BFGS at its CPU time, the scorer's own rule
    # applied per phase (review 2, F5): 60 + (50 - 20) = 90 and 200 + (150 - 40) = 310
    s4 = _module("s4_score")
    phase = dict(train_adam_wall_s=60.0, train_adam_cpu_s=20.0)
    contestants = [("su2-256x97", "256x97", {"method": "su2-256x97"}, {}),
                   _pinn("pure", 100.0, 50.0, **phase),
                   _pinn("pure", 300.0, 150.0, train_adam_wall_s=200.0, train_adam_cpu_s=40.0),
                   _pinn("gradnorm", 1e4, **phase), _pinn("nokutta", 1e4, **phase),
                   _pinn("euler", 1e4, **phase), _pinn("pure", 1e4, grid="128x49", **phase),
                   ("fno-n896-s0", "256x97", {"method": "fno", "variant": ""}, {})]
    c = s4.pinn_cost(contestants)
    assert c["fixed"] == 0.0 and c["per"] == 200.0 and c["n_runs"] == 2
    assert c["per_wall"] == 200.0 and c["per_cpu"] == 100.0
    assert c["basis"].startswith("per-phase") and "upper bound" not in c["basis"]
    assert "per_lower" not in c
    assert s4.pinn_cost(contestants[:1] + contestants[3:5]) is None


def test_s4_pinn_cost_without_the_adam_cpu_time_is_a_labelled_bracket():
    # the S3 runs predate adam_cpu_s: the Adam wall time comes from the committed run record, and
    # the per-phase cost is bracketed: upper adam_wall + cpu, lower cpu (host CPU <= Adam wall)
    s4 = _module("s4_score")
    contestants = [("su2-256x97", "256x97", {"method": "su2-256x97"}, {}),
                   _pinn("pure", 100.0, 50.0)]
    with pytest.raises(SystemExit, match="adam_wall_s"):
        s4.pinn_cost(contestants)
    c = s4.pinn_cost(contestants, records={"pure/test_in-0022": {"adam_wall_s": 60.0}})
    assert c["per"] == 110.0 and c["per_lower"] == 50.0 and "upper bound" in c["basis"]
    cpu_only = s4.pinn_cost([contestants[0], _pinn("pure", 100.0, 70.0, device="cpu")])
    assert cpu_only["per"] == 70.0 and cpu_only["basis"] == "cpu"
