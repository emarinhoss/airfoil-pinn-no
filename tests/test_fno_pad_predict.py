"""Re-evaluating S2's FNO2d on the test cases with another padding row count: the own-rows
predictions reproduce S2's file, the new rows change them, a mismatch with S2's file stops the
script, and nothing is written into runs/predictions/."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import probe_fno_libraries as probe  # noqa: E402
import probe_fno_pad_predict as pp  # noqa: E402
from apno.data.predictions import read_predictions, write_predictions  # noqa: E402
from apno.models.fno import FNO2d  # noqa: E402
from apno.models.targets import Standardiser  # noqa: E402
from apno.models.train_fno import FnoConfig, load_features_only  # noqa: E402

CFG = FnoConfig(width=8, modes_i=4, modes_j=3, layers=2, n_train=4, seed=0)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A stand-in dataset, an S2-style checkpoint and S2's prediction file made from it."""
    monkeypatch.chdir(tmp_path)
    h5 = probe.write_standin(tmp_path / "d.h5", {"train": 2, "val": 2, "test_in": 3,
                                                 "test_ext": 2})
    torch.manual_seed(0)
    model = FNO2d(width=8, modes_i=4, modes_j=3, layers=2).eval()
    with torch.no_grad():  # spectral weights of a trained scale, so the padding matters
        for p in model.parameters():
            if p.is_complex():
                p.mul_(50.0)
    std = {"mean": [0.0, 0.0, 0.1, 0.1], "std": [0.1, 0.1, 0.5, 0.5]}
    Path("runs/fno/fno-n4-s0").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "std": std, "cfg": CFG.__dict__},
               "runs/fno/fno-n4-s0/best.pt")
    ids, qs = [], []
    for split in ("test_in", "test_ext"):  # as S2 wrote it: the model's own padding
        f = load_features_only(h5, split)
        qs.append(probe.predict(model, Standardiser.from_state(std), f["xy"], f["inputs"])[0])
        ids += f["case_id"]
    write_predictions("runs/predictions/fno-n4-s0.h5", ids, np.concatenate(qs),
                      {"method": "fno", "variant": "", "n_train": 4, "seed": 0})
    return h5


def _args(h5, *extra):
    return ["--h5", str(h5), "--device", "cpu", *extra]


def test_own_rows_reproduce_s2_and_17_rows_differ(staged):
    """FNO2d's own rows are 16 since 2026-10-26; 17 is the rule its committed record used."""
    assert pp.main(_args(staged, "--rows", "17")) == 0
    rec = json.loads(Path("results/fno_pad17/predict.json").read_text())["records"][0]
    assert rec["tag"] == "fno-n4-s0" and rec["source"] == "S2" and rec["native_rows"] == 16
    assert rec["reproduced"] and rec["reproduces_s2"]["cases"] == 5
    assert rec["reproduces_s2"]["max_rel_diff"] < 1e-6
    new, attrs = read_predictions("runs/predictions_pad17/fno-n4-s0-pad17.h5")
    old, _ = read_predictions("runs/predictions/fno-n4-s0.h5")
    assert set(new) == set(old)
    diff = max(float(np.abs(new[c] - old[c]).max()) for c in new)
    assert diff > 1e-4  # one row more at 256x97 moves the prediction
    assert attrs["variant"] == "pad17" and attrs["rows"] == 17 and attrs["n_train"] == 4
    assert attrs["source_tag"] == "fno-n4-s0" and attrs["native_rows"] == 16


def test_16_rows_is_s2s_prediction(staged):
    assert pp.main(_args(staged)) == 0
    new, _ = read_predictions("runs/predictions_pad16/fno-n4-s0-pad16.h5")
    old, _ = read_predictions("runs/predictions/fno-n4-s0.h5")
    assert max(float(np.abs(new[c] - old[c]).max()) for c in new) < 1e-6


def test_a_mismatch_with_s2s_file_stops_the_script(staged, capsys):
    with h5py.File("runs/predictions/fno-n4-s0.h5", "r+") as f:
        f["q"][...] = f["q"][...] * 1.01
    assert pp.main(_args(staged)) == 4
    assert "NOT REPRODUCED" in capsys.readouterr().out
    rec = json.loads(Path("results/fno_pad16/predict.json").read_text())["records"][0]
    assert rec["reproduced"] is False


def test_no_s2_file_is_recorded_not_failed(staged):
    Path("runs/predictions/fno-n4-s0.h5").unlink()
    assert pp.main(_args(staged)) == 0
    rec = json.loads(Path("results/fno_pad16/predict.json").read_text())["records"][0]
    assert rec["reproduces_s2"] is None and rec["reproduced"] is None


def test_the_files_load_beside_s2s_in_the_scorer(staged):
    """The new variant does not collide with S2's when the library scorer loads both."""
    pp.main(_args(staged))
    spec = importlib.util.spec_from_file_location(
        "s4_score", Path(__file__).resolve().parents[1] / "scripts" / "s4_score.py")
    s4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s4)
    loaded = s4.load_contestants("runs/predictions_pad16")
    assert [(t, g, a["variant"]) for t, g, a, _ in loaded] == [
        ("fno-n4-s0-pad16", "256x97", "pad16")]


def test_refusals(staged):
    assert pp.main(_args(staged, "--out-dir", "runs/predictions")) == 2
    assert not Path("runs/predictions/fno-n4-s0-pad16.h5").exists()
    with pytest.raises(SystemExit):
        pp.main(_args(staged, "--rows", "-1"))
    assert pp.main(["nothing/*.pt", *_args(staged)]) == 2
    assert pp.main(_args(Path("missing.h5"))) == 2
