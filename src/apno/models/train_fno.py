"""Training and inference for the FNO (spec 6.4).

Leakage guard: `load_split` refuses test splits; test cases are only ever read through
`load_features_only`, which never returns field values. Training runs on the training grid sampled
at every second node (128x49 from 256x97), and evaluation runs at 256x97.

The training set for n_train is the first n_train Sobol points minus those that failed to converge
(spec 5.3), selected by the index in the case id; train.json records how many cases that is
(n_cases) and which. Training runs on `device` (default: the Apple GPU when MPS is available, else
the CPU, spec 6.4); process_time() cannot see GPU work, so a GPU run's cost is its wall time.
A run whose validation loss is never finite writes no checkpoint and is marked diverged; train.json
is strict JSON, with non-finite floats written as null.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from apno.data.sampling import prefix_indices
from apno.models.fno import FNO2d
from apno.models.targets import Standardiser, features, from_targets, to_targets

TRAINABLE = ("train", "val")


@dataclass
class FnoConfig:
    width: int = 32
    modes_i: int = 24
    modes_j: int = 12
    layers: int = 4
    loss: str = "A"
    wall_weight: float = 10.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 8
    epochs: int = 300
    patience: int = 30
    seed: int = 0
    n_train: int = 896


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def finite_or_none(obj):
    """obj with every non-finite float replaced by None, so json.dumps writes strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_none(v) for v in obj]
    return obj


def diverged(res):
    """True if a train() record has no usable checkpoint: no finite validation loss, ever."""
    best = res.get("best_val")
    return res.get("best_epoch", -1) < 0 or best is None or not math.isfinite(best)


def load_split(h5_path, split):
    if split not in TRAINABLE:
        raise PermissionError(f"split {split!r} is not readable for training")
    with h5py.File(h5_path) as f:
        g = f[split]
        return {"xy": g["xy"][...], "q": g["q"][...], "inputs": g["inputs"][...],
                "case_id": list(g["case_id"].asstr()[...])}


def load_features_only(h5_path, split):
    with h5py.File(h5_path) as f:
        g = f[split]
        return {"xy": g["xy"][...], "inputs": g["inputs"][...],
                "case_id": list(g["case_id"].asstr()[...])}


def relative_l2(pred, true, eps=1e-6):
    num = (pred - true).flatten(2).norm(dim=2)
    return (num / (true.flatten(2).norm(dim=2) + eps)).mean()


def loss_fn(pred, true, cfg):
    loss = relative_l2(pred, true)
    if cfg.loss == "B":
        loss = loss + cfg.wall_weight * relative_l2(pred[..., :1], true[..., :1])
    return loss


def _tensors(d, std, sub, idx=None):
    sl = slice(None) if idx is None else idx
    x = features(d["xy"][sl][:, ::sub, ::sub], d["inputs"][sl])
    y = std.transform(to_targets(d["q"][sl][:, ::sub, ::sub], d["inputs"][sl]))
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def train(cfg, h5_path, out_dir, subsample=2, device=None):
    device = device or default_device()
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = load_split(h5_path, "train"), load_split(h5_path, "val")
    idx = prefix_indices(tr["case_id"], cfg.n_train)
    if idx.size == 0:
        raise ValueError(f"no converged training case among the first {cfg.n_train} Sobol points")
    std = Standardiser().fit(to_targets(tr["q"][idx], tr["inputs"][idx]))
    xt, yt = (t.to(device) for t in _tensors(tr, std, subsample, idx))
    xv, yv = (t.to(device) for t in _tensors(va, std, subsample))
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    best, best_epoch, log = float("inf"), -1, []
    t0, c0 = time.perf_counter(), time.process_time()
    epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt)).to(device)
        for k in range(0, len(xt), cfg.batch):
            b = perm[k : k + cfg.batch]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b], cfg).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xv[k : k + 32]), yv[k : k + 32], cfg).item()
                                 for k in range(0, len(xv), 32)]))
        log.append({"epoch": epoch, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"model": state, "std": std.state(), "cfg": asdict(cfg)},
                       out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            break
    out = {"best_val": best, "best_epoch": best_epoch, "epochs_run": epoch + 1,
           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "device": str(device), "n_cases": int(idx.size),
           "train_case_ids": [tr["case_id"][k] for k in idx],
           "checkpoint": str(out_dir / "best.pt"), "cfg": asdict(cfg), "log": log}
    out = finite_or_none(out)
    out["diverged"] = diverged(out)
    (out_dir / "train.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    return out


def load_model(checkpoint, device=None):
    ck = torch.load(checkpoint, weights_only=True, map_location="cpu")
    cfg = FnoConfig(**ck["cfg"])
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float()
    model.load_state_dict(ck["model"])
    return model.to(device or default_device()).eval(), Standardiser.from_state(ck["std"]), cfg


def predict(model, std, xy, inputs, batch=8):
    """q at the nodes of xy, on the model's device, and the wall seconds per case."""
    device = next(model.parameters()).device
    x = torch.from_numpy(features(xy, inputs))
    t0 = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([model(x[k : k + batch].to(device)).cpu()
                       for k in range(0, len(x), batch)]).numpy()
    sec = (time.perf_counter() - t0) / len(x)
    return from_targets(std.inverse(z.astype(np.float64)), inputs), sec
