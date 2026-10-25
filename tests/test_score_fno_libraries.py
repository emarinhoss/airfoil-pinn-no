"""The library-FNO scorer on a small synthetic dataset at the real resolutions.

Two prediction files make the expected scores exact: one predicts the truth itself (near-field
error 0, so 0 x the floor) and one predicts the floor (1 x the floor). S2's FNO is scored beside
them as a reference. Nothing here measures accuracy; what is checked is the bookkeeping: S4's
code does the scoring, the outputs go to results/fno_lib/ and never results/s4/, the runs table
reads the probe's records, and a missing input is an exit status, not a traceback.
"""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from apno.data.predictions import write_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import FINE, TRAIN, Section, grid

SCRIPT = Path(__file__).parents[1] / "scripts" / "probes" / "score_fno_libraries.py"
SPLITS = {"test_in": 3, "test_ext": 1}


def _module():
    spec = importlib.util.spec_from_file_location("score_fno_libraries", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _field(s, shape, bump_scale):
    X, Y = grid(Section(s.t, s.m), shape)
    a = np.radians(s.alpha_deg)
    bump = bump_scale * np.exp(-((X - 0.5) ** 2 + Y**2))
    return np.stack([1 + 0.02 * bump, np.cos(a) + 0.05 * bump, np.sin(a) + 0.01 * Y * bump,
                     1 - 0.05 * bump], -1)


def _write_level(path, shape, dtype, bump_scale):
    with h5py.File(path, "w") as f:
        for split, n in SPLITS.items():
            specs = case_specs(split, TRAIN)[:n]
            g = f.create_group(split)
            g["q"] = np.stack([_field(s, shape, bump_scale) for s in specs]).astype(dtype)
            g["max_surface_mach"] = np.array([1.05] + [0.7] * (n - 1))  # one supercritical
            g.create_dataset("case_id", data=[s.case_id for s in specs],
                             dtype=h5py.string_dtype())


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data/hdf5").mkdir(parents=True)
    # the floor differs from the truth (bump 0.9 against 1.0), so its error is not zero
    _write_level("data/hdf5/512x193.h5", FINE, np.float64, 1.0)
    _write_level("data/hdf5/256x97.h5", TRAIN, np.float32, 0.9)
    ids, truth, floor = [], [], []
    with h5py.File("data/hdf5/512x193.h5") as t, h5py.File("data/hdf5/256x97.h5") as fl:
        for split in SPLITS:
            ids += list(t[split]["case_id"].asstr()[...])
            truth.append(t[split]["q"][...][:, ::2, ::2])
            floor.append(fl[split]["q"][...].astype(np.float64))
    truth, floor = np.concatenate(truth), np.concatenate(floor)
    lib = Path("runs/predictions_lib")
    write_predictions(lib / "neuralop-n896-s0.h5", ids, truth,
                      {"method": "fno", "variant": "neuralop", "n_train": 896, "seed": 0},
                      extra={"q_128x49": truth[:, ::2, ::2]})
    write_predictions(lib / "apno-n896-s0.h5", ids, floor,
                      {"method": "fno", "variant": "apno", "n_train": 896, "seed": 0})
    write_predictions(lib / "apno-n896-s0-nopatience.h5", ids, truth,  # a rerun beside it
                      {"method": "fno", "variant": "apno-nopatience", "n_train": 896, "seed": 0})
    write_predictions("runs/predictions/fno-n896-s0.h5", ids, floor,
                      {"method": "fno", "variant": "", "n_train": 896, "seed": 0},
                      extra={"q_128x49": floor[:, ::2, ::2]})
    rec = Path("runs/fno_lib/apno-n896-s0")
    rec.mkdir(parents=True)
    (rec / "train.json").write_text(json.dumps(
        {"library": "apno", "version": "0.1.0", "device": "cuda", "n_cases": 881,
         "n_params": 3555476, "epochs_run": 300, "best_epoch": 280, "best_val": 0.0125,
         "val_rel_l2_256x97": 0.02, "transfer_val_rel_l2": 0.03, "wall_s": 240.0,
         "query_s": 0.004, "env": {"torch": "2.9.0+cu126", "gpu": "NVIDIA A100-SXM4-40GB",
                                    "NVIDIA_TF32_OVERRIDE": "0"}, "log": []}))
    Path("results/s2").mkdir(parents=True)
    Path("results/s2/runs.json").write_text(json.dumps(
        [{"tag": "fno-n896-s0", "device": "mps", "n_cases": 881, "epochs_run": 300,
          "best_epoch": 278, "best_val": 0.0126, "transfer_val_rel_l2": 0.0348,
          "wall_s": 2160.0, "query_s": 0.0536}]))
    return tmp_path


def test_scores_with_s4_code_and_writes_only_its_own_directory(staged):
    assert _module().main([]) == 0
    assert not Path("results/s4").exists()
    out = json.loads(Path("results/fno_lib/scores.json").read_text())
    md = Path("results/fno_lib/scores.md").read_text()
    for section in ("## Runs", "## Near field against the floor", "## Forces, surface",
                    "## Shocks", "## Physics checks"):
        assert section in md
    ratio = {(r["variant"], r["grid"], r["split"], r["regime"]): r["x_floor"]
             for r in out["floor_ratios"]}
    for split, regimes in (("test_in", ("subcritical", "supercritical")),
                           ("test_ext", ("supercritical",))):
        for reg in regimes:
            assert ratio[("neuralop", "256x97", split, reg)] == pytest.approx(0.0, abs=1e-12)
            assert ratio[("neuralop", "128x49", split, reg)] == pytest.approx(0.0, abs=1e-12)
            assert ratio[("apno", "256x97", split, reg)] == pytest.approx(1.0)
            assert ratio[("s2", "256x97", split, reg)] == pytest.approx(1.0)
    assert ("apno", "128x49", "test_in", "subcritical") not in ratio  # no q_128x49 in the file
    assert ("s2", "128x49", "test_in", "subcritical") in ratio
    assert {r["variant"] for r in out["rows"]} == {"floor", "neuralop", "apno", "apno-nopatience",
                                                   "s2"}
    assert ratio[("apno-nopatience", "256x97", "test_in", "subcritical")] == pytest.approx(0.0,
                                                                                          abs=1e-12)


def test_runs_table_reads_the_probe_records_and_s2(staged):
    _module().main([])
    runs = {r["tag"]: r for r in json.loads(Path("results/fno_lib/scores.json").read_text())
            ["runs"]}
    assert runs["apno-n896-s0"]["source"] == "apno 0.1.0"
    assert runs["apno-n896-s0"]["device"] == "cuda"
    assert runs["apno-n896-s0"]["s_per_epoch"] == pytest.approx(0.8)
    assert runs["apno-n896-s0"]["env"]["gpu"] == "NVIDIA A100-SXM4-40GB"  # kept from train.json
    assert runs["apno-n896-s0-nopatience"]["source"] == "no record"
    assert runs["neuralop-n896-s0"]["source"] == "no record"
    assert runs["fno-n896-s0"]["source"] == "S2 FNO2d"
    assert runs["fno-n896-s0"]["s_per_epoch"] == pytest.approx(7.2)


def test_runs_scored_without_a_training_record_are_named(staged, capsys):
    _module().main([])
    out = capsys.readouterr().out
    assert "no train.json under runs/fno_lib for 2 scored run(s)" in out
    assert "apno-n896-s0-nopatience, neuralop-n896-s0." in out


def test_a_missing_reference_is_reported_and_skipped(staged, capsys):
    assert _module().main(["--reference", "fno-n64-s0"]) == 0
    assert "reference fno-n64-s0: no runs/predictions/fno-n64-s0.h5" in capsys.readouterr().out
    rows = json.loads(Path("results/fno_lib/scores.json").read_text())["rows"]
    assert "s2" not in {r["variant"] for r in rows}


@pytest.mark.parametrize("setup, args", [
    (lambda: None, ["--out", "results/s4"]),
    (lambda: [p.unlink() for p in Path("runs/predictions_lib").glob("*.h5")], []),
    (lambda: Path("data/hdf5/512x193.h5").unlink(), []),
])
def test_refusals_exit_2_and_write_nothing(staged, setup, args):
    setup()
    assert _module().main(args) == 2
    assert not Path("results/fno_lib").exists() and not Path("results/s4").exists()
