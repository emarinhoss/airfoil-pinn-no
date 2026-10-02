import h5py
import numpy as np
import pytest
import torch

from apno.models.fno import FNO2d
from apno.models.train_fno import FnoConfig, load_features_only, load_split, predict, train


def test_shapes_at_both_resolutions():
    m = FNO2d(width=8, modes_i=4, modes_j=3, layers=2).eval()
    for ni, nj in ((128, 49), (256, 97)):
        assert m(torch.randn(2, 6, ni, nj)).shape == (2, 4, ni, nj)


def test_roll_equivariance_around_the_airfoil():
    torch.manual_seed(0)
    m = FNO2d(width=8, modes_i=6, modes_j=4, layers=3).double().eval()
    x = torch.randn(1, 6, 32, 17, dtype=torch.float64)
    with torch.no_grad():
        a = m(torch.roll(x, 5, dims=2))
        b = torch.roll(m(x), 5, dims=2)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def _tiny_h5(path, n=16, ni=16, nj=9):
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
            g.create_dataset("case_id", data=[f"{split}-{k}" for k in range(n)],
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
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1)
    assert out["epochs_run"] == 3 and np.isfinite(out["best_val"])
    from apno.models.train_fno import load_model
    model, std, _ = load_model(out["checkpoint"])
    f = load_features_only(tmp_path / "d.h5", "test_in")
    q, sec = predict(model, std, f["xy"], f["inputs"])
    assert q.shape == (16, 16, 9, 4) and np.isfinite(q).all() and sec > 0
