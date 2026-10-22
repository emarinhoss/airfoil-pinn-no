"""The library-FNO probe: each wrapper is an operator on the O-grid's index space with the same
contract as FNO2d, and the probe trains end to end on a stand-in dataset. Library tests skip where
the library is not installed (the conda env does not carry them)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import probe_fno_libraries as probe  # noqa: E402
from apno.models.train_fno import FnoConfig  # noqa: E402

SMALL = FnoConfig(width=8, modes_i=4, modes_j=3, layers=2)


def _skip_unless(library):
    if library != "apno":
        pytest.importorskip(library)


@pytest.mark.parametrize("library", probe.LIBRARIES)
def test_shapes_at_both_resolutions(library):
    _skip_unless(library)
    m = probe.build_model(library, SMALL).eval()
    for ni, nj in ((128, 49), (256, 97)):
        assert m(torch.randn(2, 6, ni, nj)).shape == (2, 4, ni, nj)


@pytest.mark.parametrize("library", probe.LIBRARIES)
def test_roll_equivariance_around_the_airfoil(library):
    """No positional embedding along i, so rolling the input around the airfoil rolls the output."""
    _skip_unless(library)
    torch.manual_seed(0)
    m = probe.build_model(library, SMALL).eval()
    x = torch.randn(1, 6, 32, 17)
    with torch.no_grad():
        a = m(torch.roll(x, 5, dims=2))
        b = torch.roll(m(x), 5, dims=2)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_physicsnemo_padding_follows_the_resolution():
    pytest.importorskip("physicsnemo")
    m = probe.build_model("physicsnemo", SMALL).eval()
    for nj in (49, 97):
        m(torch.randn(1, 6, 16, nj))
        assert m.net.spec_encoder.pad == [0, probe.radial_pad(nj)]


def test_library_lines_counts_the_model_modules():
    assert probe.library_lines("apno") == {"apno.models.fno": 67}


def test_standin_dataset_has_the_design_ids_and_finite_fields(tmp_path):
    path = probe.write_standin(tmp_path / "d.h5", {"train": 3, "val": 2})
    import h5py
    with h5py.File(path) as f:
        assert list(f["train"]["case_id"].asstr()[...]) == ["train-0000", "train-0001",
                                                            "train-0002"]
        q = f["train"]["q"][...]
        assert q.shape == (3, 256, 97, 4) and np.isfinite(q).all()
        assert f["val"]["inputs"].shape == (2, 4)


@pytest.mark.parametrize("library", probe.LIBRARIES)
def test_probe_trains_end_to_end_on_the_standin(library, tmp_path, monkeypatch):
    _skip_unless(library)
    monkeypatch.chdir(tmp_path)
    path = probe.write_standin(tmp_path / "d.h5", {"train": 4, "val": 2, "test_in": 2})
    cfg = FnoConfig(width=8, modes_i=4, modes_j=3, layers=2, epochs=2, batch=2, n_train=4)
    model, std, rec = probe.train(library, cfg, path, tmp_path / "run", torch.device("cpu"))
    assert rec["epochs_run"] == 2 and rec["best_epoch"] >= 0 and not rec["diverged"]
    assert rec["library"] == library and rec["n_params"] > 0
    assert json.loads((tmp_path / "run" / "train.json").read_text())["best_val"] == rec["best_val"]
    out = probe.write_test_predictions(model, std, path, f"{library}-test", library, cfg)
    assert out == Path("runs/predictions_lib") / f"{library}-test.h5" and out.exists()


def _tiny_args(path, runs, *extra):
    return ["--library", "apno", "--h5", str(path), "--n-train", "4", "--epochs", "1",
            "--width", "8", "--modes-i", "4", "--modes-j", "3", "--layers", "2", "--batch", "2",
            "--device", "cpu", "--runs", str(runs), *extra]


def test_a_suffixed_rerun_has_its_own_tag_variant_and_files(tmp_path, monkeypatch):
    """A rerun (e.g. early stopping off) is scored beside the run it repeats, not over it."""
    import h5py
    monkeypatch.chdir(tmp_path)
    path = probe.write_standin(tmp_path / "d.h5", {"train": 4, "val": 2, "test_in": 2})
    assert probe.main(_tiny_args(path, tmp_path / "runs", "--seed", "1")) == 0
    assert probe.main(_tiny_args(path, tmp_path / "runs", "--seed", "1", "--patience", "300",
                                 "--suffix", "nopatience")) == 0
    for tag, variant in (("apno-n4-s1", "apno"), ("apno-n4-s1-nopatience", "apno-nopatience")):
        assert (tmp_path / "runs" / tag / "train.json").exists()
        with h5py.File(Path("runs/predictions_lib") / f"{tag}.h5") as f:
            assert f.attrs["variant"] == variant
    rec = json.loads((tmp_path / "runs/apno-n4-s1-nopatience/train.json").read_text())
    assert rec["cfg"]["patience"] == 300


def test_the_record_says_what_it_ran_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    path = probe.write_standin(tmp_path / "d.h5", {"train": 4, "val": 2})
    probe.main(_tiny_args(path, tmp_path / "runs"))
    env = json.loads((tmp_path / "runs/apno-n4-s0/train.json").read_text())["env"]
    assert env["torch"] == torch.__version__ and env["NVIDIA_TF32_OVERRIDE"] == "0"
    assert isinstance(env["cudnn_allow_tf32"], bool) and "gpu" not in env  # CPU: no GPU name


def test_a_suffix_must_be_lower_case_letters_and_digits(tmp_path):
    with pytest.raises(SystemExit):
        probe.main(_tiny_args(tmp_path / "d.h5", tmp_path / "runs", "--suffix", "no-patience"))
