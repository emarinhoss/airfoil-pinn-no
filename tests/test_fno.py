import json

import h5py
import numpy as np
import pytest
import torch

from apno.models.fno import FNO2d, SpectralConv2d, radial_pad
from apno.models import train_fno
from apno.models.train_fno import (
    FnoConfig, load_features_only, load_model, load_split, loss_fn, predict, relative_l2, train,
)


def test_shapes_at_both_resolutions():
    m = FNO2d(width=8, modes_i=4, modes_j=3, layers=2).eval()
    for ni, nj in ((128, 49), (256, 97)):
        assert m(torch.randn(2, 6, ni, nj)).shape == (2, 4, ni, nj)


def test_radial_dft_period_is_the_same_at_every_resolution():
    assert [radial_pad(nj) for nj in (49, 97, 193)] == [8, 17, 35]
    assert {(nj + radial_pad(nj)) / (nj - 1) for nj in (49, 97, 193)} == {19 / 16}


@pytest.mark.parametrize("kj", (2, 6, 11))
def test_a_wrap_smooth_radial_mode_transfers_from_128x49_to_256x97(kj):
    """One spectral layer, on a field periodic over the padded radial period, gives the same values
    at the shared nodes of both resolutions: the learned mode is the same physical mode."""
    torch.manual_seed(0)
    conv = SpectralConv2d(1, 1, modes_i=4, modes_j=kj + 1).double()
    outs = []
    for ni, nj in ((128, 49), (256, 97)):
        pad = radial_pad(nj)
        eta = torch.arange(nj + pad, dtype=torch.float64) / (nj - 1)
        xi = torch.arange(ni, dtype=torch.float64) / ni
        period = (nj + pad) / (nj - 1)
        f = torch.cos(2 * torch.pi * kj * eta / period + 0.3)[None, :] * torch.cos(
            2 * torch.pi * 3 * xi)[:, None]
        with torch.no_grad():
            outs.append(conv(f[None, None])[0, 0, :, :nj])
    coarse, fine = outs
    rel = (coarse - fine[::2, ::2]).abs().max() / coarse.abs().max()
    assert rel < 1e-3


def test_roll_equivariance_around_the_airfoil():
    torch.manual_seed(0)
    m = FNO2d(width=8, modes_i=6, modes_j=4, layers=3).double().eval()
    x = torch.randn(1, 6, 32, 17, dtype=torch.float64)
    with torch.no_grad():
        a = m(torch.roll(x, 5, dims=2))
        b = torch.roll(m(x), 5, dims=2)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def _tiny_h5(path, n=16, ni=16, nj=9, skip=()):
    """Train ids skip the Sobol indices in `skip`, as failed cases would."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for split in ("train", "val", "test_in"):
            g = f.create_group(split)
            xy = np.stack(np.meshgrid(np.linspace(1, 3, ni), np.linspace(1, 2, nj),
                                      indexing="ij"), -1)
            g["xy"] = np.broadcast_to(xy, (n, ni, nj, 2)).astype(np.float32)
            inp = np.column_stack([rng.uniform(.3, .8, n), rng.uniform(-2, 6, n),
                                   rng.uniform(.08, .16, n), rng.uniform(0, .04, n)])
            g["inputs"] = inp
            q = np.ones((n, ni, nj, 4)) + 0.05 * inp[:, None, None, :1] * xy[None, ..., :1]
            g["q"] = q.astype(np.float32)
            ks = [k for k in range(n + len(skip)) if split != "train" or k not in skip][:n]
            g.create_dataset("case_id", data=[f"{split}-{k:04d}" for k in ks],
                             dtype=h5py.string_dtype())


def test_loader_refuses_test_splits(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    with pytest.raises(PermissionError):
        load_split(tmp_path / "d.h5", "test_in")
    feats = load_features_only(tmp_path / "d.h5", "test_in")
    assert "q" not in feats and feats["xy"].shape[0] == 16


def test_training_smoke(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=3, patience=10, n_train=16,
                    batch=4)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1, device="cpu")
    assert out["epochs_run"] == 3 and np.isfinite(out["best_val"]) and not out["diverged"]
    assert out["device"] == "cpu" and out["n_cases"] == 16
    model, std, _ = load_model(out["checkpoint"], device="cpu")
    f = load_features_only(tmp_path / "d.h5", "test_in")
    q, sec = predict(model, std, f["xy"], f["inputs"])
    assert q.shape == (16, 16, 9, 4) and np.isfinite(q).all() and sec > 0


def test_training_prefix_is_the_sobol_prefix_minus_failures(tmp_path):
    _tiny_h5(tmp_path / "d.h5", skip=(2, 5))  # Sobol points 2 and 5 failed to converge
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=1, n_train=8, batch=4)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1, device="cpu")
    assert out["n_cases"] == 6
    assert out["train_case_ids"] == [f"train-{k:04d}" for k in (0, 1, 3, 4, 6, 7)]


def test_a_run_that_never_has_a_finite_loss_is_diverged_and_strict_json(tmp_path, monkeypatch):
    _tiny_h5(tmp_path / "d.h5")

    def nan_loss(pred, true, cfg):
        return (pred * float("nan")).mean()

    monkeypatch.setattr(train_fno, "loss_fn", nan_loss)
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=2, n_train=16, batch=8)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1, device="cpu")
    assert out["diverged"] and out["best_val"] is None and out["best_epoch"] == -1
    assert not (tmp_path / "run" / "best.pt").exists()

    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")

    saved = json.loads((tmp_path / "run" / "train.json").read_text(), parse_constant=refuse)
    assert saved["diverged"] and saved["log"][0]["val"] is None


def test_loss_b_adds_the_weighted_wall_row():
    torch.manual_seed(1)
    p, t = torch.randn(2, 4, 8, 5), torch.randn(2, 4, 8, 5)
    a = loss_fn(p, t, FnoConfig(loss="A"))
    b = loss_fn(p, t, FnoConfig(loss="B", wall_weight=10.0))
    torch.testing.assert_close(b, a + 10.0 * relative_l2(p[..., :1], t[..., :1]))


def test_early_stopping_after_patience_epochs_without_improvement(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=50, patience=2, lr=0.0,
                    n_train=16, batch=8)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1, device="cpu")
    assert out["best_epoch"] == 0 and out["epochs_run"] == 3


def test_default_device_is_the_apple_gpu_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert train_fno.default_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert train_fno.default_device() == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS on this machine")
def test_training_on_mps_records_the_device(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=1, n_train=16, batch=8)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1, device="mps")
    assert out["device"] == "mps" and np.isfinite(out["best_val"])
    model, std, _ = load_model(out["checkpoint"], device="mps")
    f = load_features_only(tmp_path / "d.h5", "test_in")
    q, _ = predict(model, std, f["xy"], f["inputs"])
    assert np.isfinite(q).all()
