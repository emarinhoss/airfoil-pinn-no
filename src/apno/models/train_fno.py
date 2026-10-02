"""Training and inference for the FNO (spec 6.4).

Leakage guard: `load_split` refuses test splits; test cases are only ever read through
`load_features_only`, which never returns field values. Training runs on the training grid sampled
at every second node (128x49 from 256x97), and evaluation runs at 256x97.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

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


def _tensors(d, std, sub, n=None):
    sl = slice(None, n)
    x = features(d["xy"][sl, ::sub, ::sub], d["inputs"][sl])
    y = std.transform(to_targets(d["q"][sl, ::sub, ::sub], d["inputs"][sl]))
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def train(cfg, h5_path, out_dir, subsample=2):
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = load_split(h5_path, "train"), load_split(h5_path, "val")
    std = Standardiser().fit(to_targets(tr["q"][: cfg.n_train], tr["inputs"][: cfg.n_train]))
    xt, yt = _tensors(tr, std, subsample, cfg.n_train)
    xv, yv = _tensors(va, std, subsample)
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    best, best_epoch, log = float("inf"), -1, []
    t0, c0 = time.perf_counter(), time.process_time()
    epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), cfg.batch):
            idx = perm[k : k + cfg.batch]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx], cfg).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xv[k : k + 32]), yv[k : k + 32], cfg).item()
                                 for k in range(0, len(xv), 32)]))
        log.append({"epoch": epoch, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            torch.save({"model": model.state_dict(), "std": std.state(), "cfg": asdict(cfg)},
                       out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            break
    out = {"best_val": best, "best_epoch": best_epoch, "epochs_run": epoch + 1,
           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "checkpoint": str(out_dir / "best.pt"), "cfg": asdict(cfg), "log": log}
    (out_dir / "train.json").write_text(json.dumps(out, indent=2))
    return out


def load_model(checkpoint):
    ck = torch.load(checkpoint, weights_only=True)
    cfg = FnoConfig(**ck["cfg"])
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float()
    model.load_state_dict(ck["model"])
    return model.eval(), Standardiser.from_state(ck["std"]), cfg


def predict(model, std, xy, inputs, batch=8):
    x = torch.from_numpy(features(xy, inputs))
    t0 = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([model(x[k : k + batch]) for k in range(0, len(x), batch)]).numpy()
    sec = (time.perf_counter() - t0) / len(x)
    return from_targets(std.inverse(z.astype(np.float64)), inputs), sec
