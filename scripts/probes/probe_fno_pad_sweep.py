"""How does an FNO's error at 256x97 depend on its radial padding there? No training: checkpoints.

docs/findings.md (2026-10-25) found that NeuralOperator's smaller resolution transfer comes from its
padding, and only through the row count at the 256x97 evaluation grid: 16 rows where `FNO2d`'s
period-matched rule (`radial_pad_period_matched`) used 17. The padding does not change training at
128x49, where both pad 8 rows. (Since 2026-10-26 FNO2d's own rule is 16 rows at 256x97, so its
native rows are 16 in a sweep made after that; the committed sweep was made at 17.) So the
dependence can be measured on the existing checkpoints: each is evaluated on the validation cases at
256x97 with every row count in `--rows`, its 128x49 prediction kept at the native 8 rows, and
compared with SU2's 256x97 solution.

Only the row count at 256x97 is varied. Every layer commutes with a circular shift of the padded
radial column, so where the rows go does not matter, only how many
(tests/test_fno_ablation.py); the override pads at the far end for every model type.

Per checkpoint and row count, two numbers, both on the validation cases (the test set is not
read, so a row count chosen here can be scored on it afterwards):
- `val_rel_l2_256x97`: the mean over cases of the relative L2 error against SU2 at 256x97, in
  target space, as the probe's `train.json` records it;
- `transfer`: the relative L2 difference between the 256x97 prediction at every second node and
  the 128x49 one, as S2 and the probe record it;
- `near_field`: S4's near-field measure (`score_case`'s `near_field_rel_l2`: the state less the
  free stream, weighted by node area within 2.5 chords of mid-chord), against SU2 at 256x97, the
  only solution the validation cases have. Its mean over all cases, and over the subcritical and
  supercritical ones (by the SU2 256x97 solution's largest surface Mach number). It was added on
  2026-10-26 because the whole-field error and S4's test near field disagreed (docs/findings.md).

    python scripts/probes/probe_fno_pad_sweep.py                       # the default checkpoints
    python scripts/probes/probe_fno_pad_sweep.py --rows 8-24 runs/fno_lib/abl-L-n896-s*/best.pt

Each checkpoint's sweep is cached in runs/fno_lib/pad_sweep/ and skipped on a rerun that asks for
no new row counts. Writes results/fno_lib/pad_sweep.json, pad_sweep.md and pad_sweep.png.
"""

import argparse
import glob
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

import h5py

import probe_fno_libraries as probe
from apno.data.build import regime
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.models.fno import FNO2d, radial_pad_period_matched
from apno.models.targets import Standardiser, to_targets
from apno.models.train_fno import FnoConfig, load_split, relative_l2
from apno.physics.flow import freestream_q
from apno.score.metrics import NEAR_RADIUS, node_areas

EVAL_NJ = 97  # the 256x97 grid's radial size
DEFAULT_CHECKPOINTS = (
    "runs/fno/fno-n896-s[0-9]/best.pt",                    # S2's FNO2d, three seeds, MPS
    "runs/fno_lib/abl-none-n896-s*/best.pt",               # FNO2d, five seeds, A100
    "runs/fno_lib/abl-MPIL-n896-s*/best.pt",               # NeuralOperator's FNO, five seeds
    "runs/fno_lib/neuralop-n896-s*-nopatience/best.pt",    # NeuralOperator's own class, five
)
CACHE = Path("runs/fno_lib/pad_sweep")
RECORD_VERSION = 2  # 2: near field added (2026-10-26); older cache files are swept again
NEAR = ("near_field", "near_field_subcritical", "near_field_supercritical")
OUT = Path("results/fno_lib")


def parse_rows(text):
    """'0-34' or '8,16,17' or a mix ('8-12,16,17') -> sorted unique ints >= 0."""
    rows = set()
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        rows.update(range(int(a), int(b) + 1) if b else [int(a)])
    if not rows or min(rows) < 0:
        raise ValueError(f"rows {text!r}: give non-negative counts, e.g. 0-34 or 8,16,17")
    return sorted(rows)


# ---------------------------------------------------------------- the padding override

class _FarEndRows(torch.nn.Module):
    """Pads `rows` zero rows at the far end at radial size `nj`; any other size goes to `native`."""

    def __init__(self, native, nj, rows):
        super().__init__()
        self.native, self.nj, self.rows = native, nj, rows
        self._own = False

    def pad(self, x):
        self._own, self._nj = x.shape[-1] == self.nj, x.shape[-1]
        return F.pad(x, (0, self.rows)) if self._own else self.native.pad(x)

    def unpad(self, x):
        return x[..., : self._nj] if self._own else self.native.unpad(x)


@contextmanager
def rows_at(model, rows, nj=EVAL_NJ):
    """Evaluate `model` with `rows` padding rows at radial size `nj` and its own rule elsewhere."""
    if isinstance(model, FNO2d):
        own = model.pad_rule
        model.pad_rule = lambda n: rows if n == nj else own(n)
        try:
            yield
        finally:
            model.pad_rule = own
    elif isinstance(model, probe.PhysicsnemoFno):
        def rule(n):
            return rows if n == nj else radial_pad_period_matched(n)

        with mock.patch.object(probe, "radial_pad_period_matched", rule):
            yield
    else:
        holder, name = ((model.net, "domain_padding") if isinstance(model, probe.NeuralopFno)
                        else (model, "padding"))  # AblationFno
        native = getattr(holder, name)
        setattr(holder, name, _FarEndRows(native, nj, rows))
        try:
            yield
        finally:
            setattr(holder, name, native)


def native_rows(model, nj=EVAL_NJ):
    """The rows the model pads at radial size nj on its own: 16 for FNO2d's rule at 97 (17 before
    2026-10-26, and for the ablation's FNO2d padding and the PhysicsNeMo wrapper)."""
    if isinstance(model, FNO2d):
        return model.pad_rule(nj)
    if isinstance(model, probe.PhysicsnemoFno):
        return radial_pad_period_matched(nj)
    pad = getattr(model, "padding", None) or getattr(getattr(model, "net", None),
                                                     "domain_padding", None)
    if hasattr(pad, "domain_padding"):  # NeuralOperator's DomainPadding
        return 2 * round(pad.domain_padding[-1] * nj)
    return pad.rule(nj)  # the ablation's one-sided padding


# ---------------------------------------------------------------- checkpoints

def load_checkpoint(path):
    """(model, standardiser, tag, source) for an S2 checkpoint (runs/fno/<tag>/best.pt) or a
    probe one (runs/fno_lib/<tag>/best.pt), on the CPU."""
    path = Path(path)
    ck = torch.load(path, weights_only=True, map_location="cpu")
    cfg, tag = FnoConfig(**ck["cfg"]), path.parent.name
    library = ck.get("library", "apno")
    factors = ""
    if library == "ablate":
        m = re.match(r"^abl-(none|[MPIL]+)-", tag)
        if not m:
            raise ValueError(f"{path}: an ablation checkpoint needs its factors in the tag")
        factors = "" if m.group(1) == "none" else m.group(1)
    model = probe.build_model(library, cfg, factors).float()
    model.load_state_dict(ck["model"])
    source = "S2" if "library" not in ck else library
    return model.eval(), Standardiser.from_state(ck["std"]), tag, source


def group_of(tag, source):
    """The checkpoint's group: its tag without the seed, and S2's marked as such."""
    g = re.sub(r"-s\d+", "", tag)
    return f"S2 {g}" if source == "S2" else g


# ---------------------------------------------------------------- the sweep

def near_field_setup(inputs, sub=1):
    """Per case, S4's near-field mask, node-area weights and free stream, on the 256x97 grid built
    as S4 builds it (sub=2: its 128x49 subgrid)."""
    out = []
    for mach, alpha, t, m in np.asarray(inputs, dtype=float):
        X, Y = grid(Section(t, m), TRAIN)
        X, Y = X[::sub, ::sub], Y[::sub, ::sub]
        near = np.hypot(X - 0.5, Y) <= NEAR_RADIUS
        out.append((near, node_areas(X, Y)[near][:, None], freestream_q(alpha)))
    return out


def near_field_rel_l2(pred_q, true_q, setup):
    """`score_case`'s near_field_rel_l2 for each case (tests/test_fno_pad_sweep.py checks it)."""
    vals = []
    for p, t, (near, w, q_inf) in zip(pred_q, true_q, setup):
        dp, dt = p[near] - q_inf, t[near] - q_inf
        den = np.sqrt(np.sum(w * dt * dt))
        vals.append(float(np.sqrt(np.sum(w * (dp - dt) ** 2)) / den) if den else 0.0)
    return np.array(vals)


def _near_means(per_case, regimes, prefix="near_field"):
    out = {prefix: float(per_case.mean())}
    for reg in ("subcritical", "supercritical"):
        sel = regimes == reg
        out[f"{prefix}_{reg}"] = float(per_case[sel].mean()) if sel.any() else None
    return out


def sweep_one(model, std, val, rows, device):
    """{rows: {"val_rel_l2_256x97", "transfer", "near_field", ...}} and the native 128x49 errors,
    on the val split. val carries "regime", one per case."""
    model = model.to(device)
    xy, inputs, regimes = val["xy"], val["inputs"], np.asarray(val["regime"])
    y_true = to_targets(val["q"], inputs)
    fine_nf = val.get("_near_fine") or near_field_setup(inputs)
    coarse_nf = val.get("_near_coarse") or near_field_setup(inputs, sub=2)
    q_coarse, _ = probe.predict(model, std, xy[:, ::2, ::2], inputs)
    out = {"val_rel_l2_128x49": float(relative_l2(
        torch.from_numpy(to_targets(q_coarse, inputs)), torch.from_numpy(y_true[:, :, ::2, ::2]))),
        **_near_means(near_field_rel_l2(q_coarse, val["q"][:, ::2, ::2], coarse_nf), regimes,
                      "near_field_128x49")}
    per = {}
    for r in rows:
        with rows_at(model, r):
            q_fine, _ = probe.predict(model, std, xy, inputs)
        per[r] = {"val_rel_l2_256x97": float(relative_l2(
            torch.from_numpy(to_targets(q_fine, inputs)), torch.from_numpy(y_true))),
            "transfer": probe.transfer_rel_l2(q_fine, q_coarse, inputs),
            **_near_means(near_field_rel_l2(q_fine, val["q"], fine_nf), regimes)}
    out["rows"] = per
    return out


def _cached(path, rows):
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    if rec.get("version") != RECORD_VERSION:
        return None
    return rec if set(rows) <= {int(r) for r in rec["rows"]} else None


def run(checkpoints, rows, h5_path, device, cache=CACHE, log=print):
    val = load_split(h5_path, "val")
    with h5py.File(h5_path) as f:  # SU2's own largest surface Mach number, per validation case
        val["regime"] = [regime(m) for m in f["val"]["max_surface_mach"][...]]
    val["_near_fine"] = near_field_setup(val["inputs"])
    val["_near_coarse"] = near_field_setup(val["inputs"], sub=2)
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for path in checkpoints:
        model, std, tag, source = load_checkpoint(path)
        cfile = cache / f"{source}-{tag}.json"
        rec = _cached(cfile, rows)
        if rec is None:
            t0 = time.perf_counter()
            res = sweep_one(model, std, val, rows, device)
            rec = {"version": RECORD_VERSION, "tag": tag, "source": source,
                   "group": group_of(tag, source),
                   "n_subcritical": int(sum(r == "subcritical" for r in val["regime"])),
                   "checkpoint": str(path), "native_rows": native_rows(model),
                   "n_val": len(val["inputs"]), "device": str(device),
                   "wall_s": time.perf_counter() - t0, **res}
            cfile.write_text(json.dumps(rec, indent=1))
            log(f"{tag} ({source}): {len(rows)} row counts in {rec['wall_s']:.0f} s")
        else:
            log(f"{tag} ({source}): cached")
        rec["rows"] = {int(r): v for r, v in rec["rows"].items() if int(r) in rows}
        records.append(rec)
    return records


# ---------------------------------------------------------------- the report

def group_curves(records, rows):
    """{group: {"n", "native_rows", metric: [mean over the group's checkpoints per row count]}}"""
    groups = {}
    for rec in records:
        groups.setdefault(rec["group"], []).append(rec)
    out = {}
    for g, recs in groups.items():
        out[g] = {"n": len(recs), "native_rows": sorted({r["native_rows"] for r in recs}),
                  "val_rel_l2_128x49": float(np.mean([r["val_rel_l2_128x49"] for r in recs]))}
        for m in NEAR:
            vals = [r.get(f"near_field_128x49{m[len('near_field'):]}") for r in recs]
            out[g][f"{m}_128x49"] = (float(np.mean(vals)) if all(v is not None for v in vals)
                                     else None)
        for m in ("val_rel_l2_256x97", "transfer", *NEAR):
            vals = [[r["rows"][k].get(m) for r in recs] for k in rows]
            out[g][m] = [float(np.mean(v)) if all(x is not None for x in v) else None
                         for v in vals]
    return out


METRICS = (("val_rel_l2_256x97", "Whole-field error against SU2 at 256x97 (target space)"),
           ("transfer", "Transfer (256x97 at every second node against 128x49)"),
           ("near_field", "Near field against SU2 at 256x97, all validation cases"),
           ("near_field_subcritical", "Near field against SU2 at 256x97, subcritical cases"),
           ("near_field_supercritical", "Near field against SU2 at 256x97, supercritical cases"))
SHORT = {"val_rel_l2_256x97": "whole field", "transfer": "transfer", "near_field": "near field",
         "near_field_subcritical": "near field, sub", "near_field_supercritical": "near field, sup"}


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


def report(records, rows):
    curves = group_curves(records, rows)
    show = sorted({r for r in (0, 8, 12, 14, 15, 16, 17, 18, 20, 24, 32) if r in rows}
                  | {rows[0], rows[-1]})
    n_sub = {r.get("n_subcritical") for r in records} - {None}
    lines = ["# FNO padding sweep at 256x97", "",
             "Command: `python scripts/probes/probe_fno_pad_sweep.py` (rows "
             f"{rows[0]} to {rows[-1]}). Validation cases only "
             f"({records[0]['n_val'] if records else 0}, of which "
             f"{', '.join(map(str, sorted(n_sub))) or '?'} subcritical by SU2's 256x97 solution). "
             "Each checkpoint is evaluated at 256x97 with each number of radial padding rows, its "
             "128x49 prediction at its own padding (8 rows). The near field is S4's measure "
             "against SU2 at 256x97, the only solution the validation cases have. Group means "
             "over seeds.",
             "", "## Best row count per group", "",
             "Each cell: the best row count, its value, and the value at the group's own rows.", "",
             "| group | checkpoints | own rows | 128x49: whole field / near field | "
             + " | ".join(SHORT[m] for m, _ in METRICS) + " |",
             "|---|---|---|---|" + "---|" * len(METRICS)]
    for g, c in sorted(curves.items()):
        own = c["native_rows"][0]
        cells = []
        for m, _ in METRICS:
            vals = c[m]
            if any(v is None for v in vals):
                cells.append("")
                continue
            k = int(np.argmin(vals))
            at_own = _fmt(vals[rows.index(own)]) if own in rows else ""
            cells.append(f"{rows[k]}: {vals[k]:.4f} (own {at_own})")
        lines.append(f"| {g} | {c['n']} | {own} | {c['val_rel_l2_128x49']:.4f} / "
                     f"{_fmt(c.get('near_field_128x49'))} | " + " | ".join(cells) + " |")
    for m, name in METRICS:
        lines += ["", f"## {name}, group means", "",
                  "| rows | " + " | ".join(sorted(curves)) + " |",
                  "|---|" + "---|" * len(curves)]
        for r in show:
            k = rows.index(r)
            lines.append(f"| {r} | " + " | ".join(_fmt(curves[g][m][k])
                                                  for g in sorted(curves)) + " |")
    return "\n".join(lines) + "\n", curves


def plot(curves, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (m, name) in zip(axes, METRICS):
        for g, c in sorted(curves.items()):
            if all(v is not None for v in c[m]):
                ax.plot(rows, c[m], marker=".", label=f"{g} ({c['n']})")
        for r, style in ((16, ":"), (17, "--")):
            ax.axvline(r, color="grey", linestyle=style, linewidth=0.8)
        ax.set_xlabel("radial padding rows at 256x97 (FNO2d's rule 17 until 2026-10-26, "
                      "NeuralOperator 16)")
        ax.set_title(name, fontsize=9)
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[-1].axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", help="best.pt files or globs; default: S2's three "
                    "FNO2d, abl-none, abl-MPIL and NeuralOperator's five each")
    ap.add_argument("--rows", default="0-34", help="row counts at 256x97, e.g. 0-34 or 8,16,17")
    ap.add_argument("--h5", default="data/hdf5/256x97.h5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    try:
        rows = parse_rows(args.rows)
    except ValueError as e:
        ap.error(str(e))
    patterns = args.checkpoints or DEFAULT_CHECKPOINTS
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not Path(args.h5).exists():
        print(f"no dataset at {args.h5}")
        return 2
    if not paths:
        print("no checkpoint matches " + ", ".join(patterns))
        return 2
    device = probe.resolve_device(args.device)
    print(f"{len(paths)} checkpoints, rows {rows[0]} to {rows[-1]} ({len(rows)}), on {device}")
    records = run(paths, rows, args.h5, device)
    text, curves = report(records, rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pad_sweep.json").write_text(json.dumps(
        {"rows": rows, "records": records, "groups": curves}, indent=1))
    (out / "pad_sweep.md").write_text(text)
    plot(curves, rows, out / "pad_sweep.png")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
