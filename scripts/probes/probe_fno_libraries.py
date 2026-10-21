"""The FNO of two open-source libraries, trained on this study's data through its own pipeline.

Three contestants share one training loop, so a difference in validation loss is a difference in
the operator, not in the loop:

- `apno`: this study's `FNO2d` (src/apno/models/fno.py), the control, which should reproduce
  results/s2/runs.json to the seed;
- `neuralop`: `neuralop.models.FNO` (the NeuralOperator library, NVIDIA and Caltech), with its
  grid positional embedding off (the features already carry the node position) and a radial domain
  padding of the same fraction as FNO2d's, applied symmetrically because that is the only mode it
  has;
- `physicsnemo`: `physicsnemo.models.fno.FNO` (NVIDIA PhysicsNeMo), with its coordinate features
  off and its fixed row padding reset at every forward pass to `radial_pad(NJ)`, so the radial DFT
  period is the same at 128x49 and 256x97, as FNO2d's is.

Each one takes the same six feature channels and predicts the same four standardised targets, is
trained at 128x49 (256x97 sampled at every second node) on the prefix training set of spec 5.3,
loss A, and is evaluated at 256x97 on the validation cases. Its record carries the best validation
loss, the relative L2 at both resolutions, the resolution transfer of scripts/s2_operator.py, the
parameter count, the wall and CPU time, and the lines of library code behind the model. Test
predictions go to runs/predictions_lib/<tag>.h5, in the S4 format but outside runs/predictions/,
which S4 reads in full.

    python scripts/probes/probe_fno_libraries.py --library neuralop --n-train 896 --epochs 300
    python scripts/probes/probe_fno_libraries.py --library physicsnemo --n-train 896 --epochs 300
    python scripts/probes/probe_fno_libraries.py --library apno --n-train 896 --epochs 300

`--smoke DIR` writes a stand-in dataset to DIR (the real grids and Sobol inputs, with an analytic
stream-plus-doublet-plus-vortex field in place of SU2) and runs on it, to check a library runs end
to end where the SU2 data is absent. Its numbers say nothing about airfoils.
"""

import argparse
import inspect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from apno.constants import GAMMA
from apno.data.predictions import write_predictions
from apno.data.sampling import case_specs, prefix_indices
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.models.fno import FNO2d, radial_pad
from apno.models.targets import Standardiser, features, from_targets, to_targets
from apno.models.train_fno import (
    FnoConfig, _tensors, finite_or_none, load_features_only, load_split, loss_fn, relative_l2,
)

LIBRARIES = ("apno", "neuralop", "physicsnemo")
RUNS, PRED = Path("runs/fno_lib"), Path("runs/predictions_lib")
IN_CH, OUT_CH = 6, 4


# ---------------------------------------------------------------- the three operators

class NeuralopFno(nn.Module):
    """neuralop.models.FNO configured to keep the same modes as FNO2d.

    n_modes counts the modes kept along each dimension in total, centred by an fftshift, so
    2*modes_i along i is the same set as FNO2d's modes_i positive plus modes_i negative. The last
    dimension is the real FFT's and keeps n_modes[-1] // 2 + 1, so 2*modes_j - 2 keeps modes_j.
    domain_padding is a fraction of the resolution padded on BOTH ends of each dimension.
    """

    def __init__(self, cfg):
        super().__init__()
        from neuralop.models import FNO
        frac = radial_pad(TRAIN[1] // 2 + 1) / 2 / (TRAIN[1] // 2 + 1)  # 4/49: 8 rows at 49 split
        self.net = FNO(n_modes=(2 * cfg.modes_i, 2 * cfg.modes_j - 2), in_channels=IN_CH,
                       out_channels=OUT_CH, hidden_channels=cfg.width, n_layers=cfg.layers,
                       positional_embedding=None, domain_padding=[0.0, frac])

    def forward(self, x):
        return self.net(x)


class PhysicsnemoFno(nn.Module):
    """physicsnemo.models.fno.FNO with the radial padding re-derived from the resolution.

    The library pads a fixed number of rows at the end of each dimension, set at construction, so
    a model trained at 128x49 would carry 8 rows to 256x97 and change the radial DFT period. The
    encoder keeps the padding in two plain attributes, reset here on every call.
    """

    def __init__(self, cfg):
        super().__init__()
        from physicsnemo.models.fno import FNO
        self.net = FNO(in_channels=IN_CH, out_channels=OUT_CH, dimension=2,
                       latent_channels=cfg.width, num_fno_layers=cfg.layers,
                       num_fno_modes=[cfg.modes_i, cfg.modes_j], padding=[0, radial_pad(49)],
                       coord_features=False, decoder_layers=1, decoder_layer_size=128,
                       decoder_activation_fn="gelu")

    def forward(self, x):
        pad = radial_pad(x.shape[-1])
        enc = self.net.spec_encoder
        enc.pad, enc.ipad = [0, pad], [None, -pad]
        return self.net(x)


def build_model(library, cfg):
    if library == "apno":
        return FNO2d(in_ch=IN_CH, out_ch=OUT_CH, width=cfg.width, modes_i=cfg.modes_i,
                     modes_j=cfg.modes_j, layers=cfg.layers)
    if library == "neuralop":
        return NeuralopFno(cfg)
    if library == "physicsnemo":
        return PhysicsnemoFno(cfg)
    raise ValueError(f"unknown library {library!r}; one of {LIBRARIES}")


def library_lines(library):
    """Lines of library source behind the model: the modules its classes are defined in."""
    if library == "apno":
        import apno.models.fno as m
        return {m.__name__: len(inspect.getsource(m).splitlines())}
    if library == "neuralop":
        from neuralop.layers import (channel_mlp, embeddings, fno_block, padding,
                                     skip_connections, spectral_convolution)
        from neuralop.models import fno
        mods = (fno, fno_block, spectral_convolution, padding, embeddings, channel_mlp,
                skip_connections)
    else:
        from physicsnemo.models.fno import fno
        from physicsnemo.nn.module import fno_layers, spectral_layers
        mods = (fno, fno_layers, spectral_layers)
    return {m.__name__: len(inspect.getsource(m).splitlines()) for m in mods}


def library_version(library):
    if library == "apno":
        return "0.1.0"
    mod = __import__("neuralop" if library == "neuralop" else "physicsnemo")
    return getattr(mod, "__version__", "?")


# ---------------------------------------------------------------- training, as train_fno.train

def transfer_rel_l2(q_fine, q_coarse, inputs):
    """scripts/s2_operator.transfer_rel_l2: 256x97 at every second node against 128x49."""
    a = to_targets(q_fine[:, ::2, ::2], inputs).reshape(len(inputs), -1)
    b = to_targets(q_coarse, inputs).reshape(len(inputs), -1)
    return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))


def predict(model, std, xy, inputs, batch=8):
    device = next(model.parameters()).device
    x = torch.from_numpy(features(xy, inputs))
    t0 = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([model(x[k:k + batch].to(device)).cpu()
                       for k in range(0, len(x), batch)]).numpy()
    sec = (time.perf_counter() - t0) / len(x)
    return from_targets(std.inverse(z.astype(np.float64)), inputs), sec


def train(library, cfg, h5_path, out_dir, device, subsample=2):
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = load_split(h5_path, "train"), load_split(h5_path, "val")
    idx = prefix_indices(tr["case_id"], cfg.n_train)
    if idx.size == 0:
        raise ValueError(f"no training case among the first {cfg.n_train} Sobol points")
    std = Standardiser().fit(to_targets(tr["q"][idx], tr["inputs"][idx]))
    xt, yt = (t.to(device) for t in _tensors(tr, std, subsample, idx))
    xv, yv = (t.to(device) for t in _tensors(va, std, subsample))
    model = build_model(library, cfg).float().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    best, best_epoch, log = float("inf"), -1, []
    t0, c0 = time.perf_counter(), time.process_time()
    epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt)).to(device)
        for k in range(0, len(xt), cfg.batch):
            b = perm[k:k + cfg.batch]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b], cfg).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xv[k:k + 32]), yv[k:k + 32], cfg).item()
                                 for k in range(0, len(xv), 32)]))
        log.append({"epoch": epoch, "val": val, "wall_s": time.perf_counter() - t0})
        print(f"epoch {epoch:4d}  val {val:.5f}  {time.perf_counter() - t0:7.1f} s", flush=True)
        if val < best:
            best, best_epoch = val, epoch
            # neuralop's BaseModel.state_dict() adds a '_metadata' dict (its constructor
            # arguments); the tensors alone reload into a model built from cfg.
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()
                     if torch.is_tensor(v)}
            torch.save({"model": state, "std": std.state(), "cfg": asdict(cfg),
                        "library": library}, out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            break
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    if best_epoch >= 0:
        model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True)["model"])
    model.eval()
    # Validation at both resolutions, in target space, as S2 records it.
    q_fine, query_s = predict(model, std, va["xy"], va["inputs"])
    q_coarse, _ = predict(model, std, va["xy"][:, ::2, ::2], va["inputs"])
    yv_true = to_targets(va["q"], va["inputs"])
    rel = {f"val_rel_l2_{tag}": float(relative_l2(
        torch.from_numpy(to_targets(q, inp)), torch.from_numpy(y)))
        for tag, q, inp, y in (("256x97", q_fine, va["inputs"], yv_true),
                               ("128x49", q_coarse, va["inputs"], yv_true[:, :, ::2, ::2]))}
    out = {"library": library, "version": library_version(library), "status": "ok",
           "best_val": best, "best_epoch": best_epoch, "epochs_run": epoch + 1,
           "wall_s": wall, "cpu_s": cpu, "device": str(device), "n_cases": int(idx.size),
           "n_params": n_params, "library_lines": library_lines(library), **rel,
           "transfer_val_rel_l2": transfer_rel_l2(q_fine, q_coarse, va["inputs"]),
           "query_s": query_s, "cfg": asdict(cfg), "log": log}
    out = finite_or_none(out)
    out["diverged"] = best_epoch < 0 or out["best_val"] is None
    (out_dir / "train.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    return model, std, out


def write_test_predictions(model, std, h5_path, tag, library, cfg):
    with h5py.File(h5_path) as f:
        splits = [s for s in ("test_in", "test_ext") if s in f]
    ids, q, qc = [], [], []
    for split in splits:
        feat = load_features_only(h5_path, split)
        q.append(predict(model, std, feat["xy"], feat["inputs"])[0])
        qc.append(predict(model, std, feat["xy"][:, ::2, ::2], feat["inputs"])[0])
        ids += feat["case_id"]
    if not ids:
        return None
    path = PRED / f"{tag}.h5"
    write_predictions(path, ids, np.concatenate(q),
                      {"method": "fno", "variant": library, "n_train": cfg.n_train,
                       "seed": cfg.seed, "library_version": library_version(library)},
                      extra={"q_128x49": np.concatenate(qc)})
    return path


# ---------------------------------------------------------------- the stand-in dataset

def standin_fields(X, Y, mach, alpha_deg, t):
    """A uniform stream plus a regularised doublet and a vortex about mid-chord, with a
    Prandtl-Glauert factor and first-order isentropic p and rho. Smooth, not a flow solution."""
    a, beta = np.radians(alpha_deg), np.sqrt(1.0 - mach * mach)
    dx, dy = X - 0.5, Y
    r2 = dx * dx + dy * dy + 0.25
    kappa, gam = 2.0 * t, np.pi * np.sin(a)
    u = np.cos(a) + (-kappa * (dx * dx - dy * dy) / r2 ** 2 + gam * dy / (2 * np.pi * r2)) / beta
    v = np.sin(a) + (-2.0 * kappa * dx * dy / r2 ** 2 - gam * dx / (2 * np.pi * r2)) / beta
    cp = 1.0 - (u * u + v * v)
    p = 1.0 + 0.5 * GAMMA * mach * mach * cp
    rho = 1.0 + 0.5 * mach * mach * cp
    return np.stack([rho, u, v, p], -1)


def write_standin(path, counts):
    """counts: {split: n}. Real ids, inputs and grids from the design; analytic fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for split, n in counts.items():
            specs = case_specs(split, TRAIN)[:n]
            xy, q = [], []
            for s in specs:
                X, Y = grid(Section(s.t, s.m), TRAIN)
                xy.append(np.stack([X, Y], -1))
                q.append(standin_fields(X, Y, s.mach, s.alpha_deg, s.t))
            g = f.create_group(split)
            g["xy"] = np.asarray(xy, dtype=np.float32)
            g["q"] = np.asarray(q, dtype=np.float32)
            g["inputs"] = np.array([[s.mach, s.alpha_deg, s.t, s.m] for s in specs])
            g.create_dataset("case_id", data=[s.case_id for s in specs],
                             dtype=h5py.string_dtype())
    return path


# ---------------------------------------------------------------- entry point

def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", choices=LIBRARIES, required=True)
    ap.add_argument("--h5", default="data/hdf5/256x97.h5")
    ap.add_argument("--n-train", type=int, default=896)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--modes-i", type=int, default=24)
    ap.add_argument("--modes-j", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--smoke", metavar="DIR", default=None,
                    help="write a stand-in dataset under DIR and run on it (no SU2 data needed)")
    args = ap.parse_args(argv)

    cfg = FnoConfig(width=args.width, modes_i=args.modes_i, modes_j=args.modes_j,
                    layers=args.layers, loss="A", batch=args.batch, epochs=args.epochs,
                    patience=args.patience, seed=args.seed, n_train=args.n_train)
    h5_path = Path(args.h5)
    if args.smoke:
        h5_path = write_standin(Path(args.smoke) / "standin_256x97.h5",
                                {"train": 48, "val": 8, "test_in": 8, "test_ext": 4})
        cfg = FnoConfig(**{**asdict(cfg), "n_train": min(cfg.n_train, 48), "epochs": args.epochs})
        print(f"stand-in dataset {h5_path} (analytic fields, NOT SU2)")
    if not h5_path.exists():
        print(f"no dataset at {h5_path}; run scripts/build_dataset.py or pass --smoke DIR")
        return 2
    device = resolve_device(args.device)
    tag = f"{args.library}-n{cfg.n_train}-s{cfg.seed}" + ("-smoke" if args.smoke else "")
    out_dir = Path(args.runs) / tag
    print(f"{tag}: {args.library} {library_version(args.library)} on {device}, {h5_path}")
    model, std, rec = train(args.library, cfg, h5_path, out_dir, device)
    path = None if args.smoke else write_test_predictions(model, std, h5_path, tag,
                                                          args.library, cfg)
    print(json.dumps({k: v for k, v in rec.items() if k != "log"}, indent=2))
    print(f"record {out_dir / 'train.json'}; predictions {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
