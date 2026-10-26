"""Why does one padding row change the FNO's wall pressure at the leading and trailing edges?

Going from 16 to 17 radial padding rows at 256x97 raises the trained FNO's wall Cp by up to 0.12
just behind the leading-edge stagnation point and lowers it by 0.15 to 0.27 at the trailing edge,
with about +0.005 elsewhere (docs/findings.md, 2026-10-26). An untrained FNO2d on the same grid and
features spreads its change along the wall instead, so the pattern comes from what the network
learned. This probe asks which property of the trained network's own field, at each wall node,
predicts the size of the change there. No training: S2's checkpoints, on the validation cases.

Per case, delta = Cp at 17 rows minus Cp at 16 rows along the wall. Each candidate is computed from
the 16-row prediction (or the input features) at the same wall nodes:

- `cp_wall`: the wall Cp itself, its departure from free stream;
- `cp_wall_minus_far`: wall Cp minus the far-field row's, the drop across the column;
- `cp_off_wall`: Cp at the first row off the wall minus the wall's, the radial slope there;
- `cp_along_wall`: |dCp/di| along the wall, how sharp the pressure is around the airfoil;
- `dx_k`: the position feature dx*k at the wall (-0.41 at the leading edge, +0.41 at the trailing).

For each, a straight line delta = a + b * candidate per case, and its R^2.

`shift` tests one reading of the result. Lengthening the radial DFT by a row stretches every learned
radial filter, and to first order that adds a term proportional to the field's radial gradient: a
small radial shift. If that is what happens, the change in every channel q, at every row j, is
delta_q = a + b * dq/dj, with the same b (the shift, in rows) for all four channels and a high R^2.
dq/dj is the forward difference at the wall and the central one elsewhere, from the 16-row field.
Per case, channel and row, b and R^2 are fitted over the wall-parallel line of nodes, and the
medians over cases are reported. The median over cases is
reported by regime, with all five together as an upper bound. The probe also records how the change
spreads outward from the wall, per output channel, at the leading edge, mid-chord on each surface,
and the trailing edge. Finally, it records how much of the wall's total |delta| lies in the leading-
and trailing-edge regions, against their share of the wall nodes.

    python scripts/probes/probe_fno_drag_locate.py                  # S2's fno-n896-s0, s1, s2
    python scripts/probes/probe_fno_drag_locate.py --untrained      # and an untrained FNO2d

`--untrained` adds an FNO2d of S2's sizes with random weights (seed 0, spectral weights scaled by
50, about a trained network's size), using the first checkpoint's output standardisation, as a
control for what the grid and features alone do.

Writes results/fno_drag/locate.json, locate.md and locate.png.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import probe_fno_libraries as probe
from apno.models.fno import FNO2d
from apno.models.targets import features
from apno.physics.flow import pressure_coefficient
from probe_fno_drag import DEFAULT_CHECKPOINTS, load_val, region_of
from probe_fno_pad_sweep import load_checkpoint, rows_at

CANDIDATES = ("cp_wall", "cp_wall_minus_far", "cp_off_wall", "cp_along_wall", "dx_k")
CHANNELS = ("rho", "u", "v", "p")
PROFILE_ROWS = (0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 96)
SHIFT_ROWS = (0, 1, 2, 4, 8, 16, 32)


def candidates(q16, feat, mach):
    """Each candidate along the wall of one case. q16: (NI, NJ, 4); feat: (6, NI, NJ)."""
    cp = pressure_coefficient(q16[:, :, 3], mach)
    wall = cp[:, 0]
    return {"cp_wall": wall, "cp_wall_minus_far": wall - cp[:, -1],
            "cp_off_wall": cp[:, 1] - wall,
            "cp_along_wall": 0.5 * np.abs(np.roll(wall, -1) - np.roll(wall, 1)),
            "dx_k": feat[0, :, 0]}


def r2(y, cols):
    """R^2 of the least-squares line (or plane) y = a + sum b_k x_k."""
    A = np.column_stack([np.ones_like(y), *cols])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - A @ coef) ** 2) / ss) if ss > 0 else float("nan")


def radial_difference(q, j):
    """dq/dj along row j of one channel's field q (NI, NJ): forward at the wall, central inside."""
    return q[:, 1] - q[:, 0] if j == 0 else 0.5 * (q[:, j + 1] - q[:, j - 1])


def shift_fit(delta, grad):
    """(b, R^2) of delta = a + b * grad."""
    A = np.column_stack([np.ones_like(grad), grad])
    coef, *_ = np.linalg.lstsq(A, delta, rcond=None)
    return float(coef[1]), r2(delta, [grad])


def columns(xw):
    """Wall indices of the leading edge, mid-chord lower and upper, and the trailing edge."""
    ni = len(xw)
    lower, upper = np.arange(ni // 2), np.arange(ni // 2, ni)
    return {"leading edge": int(np.argmin(xw)),
            "mid-chord lower": int(lower[np.argmin(np.abs(xw[lower] - 0.5))]),
            "mid-chord upper": int(upper[np.argmin(np.abs(xw[upper] - 0.5))]),
            "trailing edge": 0}


def _med(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def locate_one(model, std, val, device):
    model = model.to(device)
    xy, inputs, regimes = val["xy"], val["inputs"], val["regime"]
    with rows_at(model, 16):
        q16, _ = probe.predict(model, std, xy, inputs)
    with rows_at(model, 17):
        q17, _ = probe.predict(model, std, xy, inputs)
    feat = features(xy, inputs)
    fits = {c: [] for c in (*CANDIDATES, "all five")}
    share, profiles, deltas = [], {}, []
    shifts = np.full((len(inputs), len(CHANNELS), len(SHIFT_ROWS), 2), np.nan)
    for k, (mach, *_rest) in enumerate(inputs):
        for c in range(len(CHANNELS)):
            for n, j in enumerate(SHIFT_ROWS):
                shifts[k, c, n] = shift_fit(q17[k, :, j, c] - q16[k, :, j, c],
                                            radial_difference(q16[k, :, :, c], j))
        delta = (pressure_coefficient(q17[k, :, 0, 3], mach)
                 - pressure_coefficient(q16[k, :, 0, 3], mach))
        deltas.append(delta)
        cand = candidates(q16[k], feat[k], mach)
        for c in CANDIDATES:
            fits[c].append(r2(delta, [cand[c]]))
        fits["all five"].append(r2(delta, [cand[c] for c in CANDIDATES]))
        xw = val["_wall"][k][0]
        reg = region_of(xw)
        ends = (reg == "leading edge") | (reg == "trailing edge")
        share.append((float(np.abs(delta[ends]).sum() / np.abs(delta).sum()), float(ends.mean())))
        for name, i in columns(xw).items():
            d = np.abs(q17[k, i] - q16[k, i])  # (NJ, 4)
            profiles.setdefault(name, []).append(d[list(PROFILE_ROWS)])
    sub = regimes == "subcritical"
    r2_out = {}
    for c, v in fits.items():
        v = np.asarray(v)
        r2_out[c] = {"subcritical": _med(v[sub]), "supercritical": _med(v[~sub]), "all": _med(v)}
    share = np.asarray(share)
    deltas = np.asarray(deltas)
    shift = {}
    for c, ch in enumerate(CHANNELS):
        shift[ch] = {"rows": list(SHIFT_ROWS),
                     "b_median": [_med(shifts[:, c, n, 0]) for n in range(len(SHIFT_ROWS))],
                     "b_q25": [float(np.nanpercentile(shifts[:, c, n, 0], 25))
                               for n in range(len(SHIFT_ROWS))],
                     "b_q75": [float(np.nanpercentile(shifts[:, c, n, 0], 75))
                               for n in range(len(SHIFT_ROWS))],
                     "r2_median": [_med(shifts[:, c, n, 1]) for n in range(len(SHIFT_ROWS))],
                     "r2_median_subcritical": [_med(shifts[sub, c, n, 1])
                                               for n in range(len(SHIFT_ROWS))]}
    return {
        "shift": shift,
        "r2_median": r2_out,
        "ends_share_of_abs_change": {"median": _med(share[:, 0]),
                                     "share_of_wall_nodes": _med(share[:, 1])},
        "profile_rows": list(PROFILE_ROWS),
        "profile_mean_abs_change": {name: {ch: np.mean(p, 0)[:, n].tolist()
                                           for n, ch in enumerate(CHANNELS)}
                                    for name, p in profiles.items()},
        "wall_delta_cp_mean_subcritical": deltas[sub].mean(0).tolist() if sub.any() else None,
        "wall_x_mean": np.mean([w[0] for w in val["_wall"]], 0).tolist(),
        "wall_cp16_mean_subcritical": np.mean(
            [pressure_coefficient(q16[k, :, 0, 3], inputs[k, 0]) for k in np.flatnonzero(sub)],
            0).tolist() if sub.any() else None,
    }


def report(records):
    lines = ["# Where the 16-to-17-row change acts on the FNO's wall pressure", "",
             "Validation cases. delta = wall Cp at 17 rows minus at 16, per case. Written by "
             "`python scripts/probes/probe_fno_drag_locate.py`.", ""]
    for rec in records:
        s = rec["ends_share_of_abs_change"]
        lines += [f"## {rec['tag']}", "",
                  f"Leading- and trailing-edge regions (x/c < 0.05 or > 0.95): "
                  f"{100 * s['share_of_wall_nodes']:.0f} % of the wall nodes and "
                  f"{100 * s['median']:.0f} % of the wall's total |delta| (median over cases).",
                  "", "Median R^2 of delta = a + b * candidate, per case:", "",
                  "| candidate | subcritical | supercritical | all |", "|---|---|---|---|"]
        for c, v in rec["r2_median"].items():
            lines.append(f"| {c} | " + " | ".join(
                "—" if v[k] is None else f"{v[k]:.2f}"
                for k in ("subcritical", "supercritical", "all")) + " |")
        if "shift" in rec:
            sh = rec["shift"]
            rows = sh["p"]["rows"]
            lines += ["", "Shift fit, delta_q = a + b * dq/dj per case, channel and row j: "
                      "median b (rows) [interquartile range] and median R^2:", "",
                      "| channel | " + " | ".join(f"j = {j}" for j in rows) + " |",
                      "|---|" + "---|" * len(rows)]
            for ch, v in sh.items():
                lines.append(f"| {ch} | " + " | ".join(
                    "—" if b is None else f"{b:+.3f} [{lo:+.3f}, {hi:+.3f}], R^2 {r:.2f}"
                    for b, lo, hi, r in zip(v["b_median"], v["b_q25"], v["b_q75"],
                                            v["r2_median"])) + " |")
        lines += ["", "Mean |change| from 16 to 17 rows by row j (0 = wall), channel p:", "",
                  "| column | " + " | ".join(str(j) for j in rec["profile_rows"]) + " |",
                  "|---|" + "---|" * len(rec["profile_rows"])]
        for name, prof in rec["profile_mean_abs_change"].items():
            lines.append(f"| {name} | " + " | ".join(f"{v:.1e}" for v in prof["p"]) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for rec in records:
        x = np.asarray(rec["wall_x_mean"])
        if rec["wall_delta_cp_mean_subcritical"] is not None:
            i = np.arange(len(x))
            axes[0].plot(i, rec["wall_delta_cp_mean_subcritical"], label=f"{rec['tag']} delta Cp")
            axes[0].plot(i, 0.1 * np.asarray(rec["wall_cp16_mean_subcritical"]), ":",
                         label=f"{rec['tag']} 0.1 x Cp (16 rows)")
        for name, prof in rec["profile_mean_abs_change"].items():
            axes[1].plot(np.asarray(rec["profile_rows"]) + 1, prof["p"], marker="o", ms=3,
                         label=f"{rec['tag'][-2:]} {name}")
    rec = records[0]
    names = list(rec["r2_median"])
    for k, r in enumerate(records):
        axes[2].bar(np.arange(len(names)) + 0.27 * k, [r["r2_median"][c]["subcritical"] or 0
                                                         for c in names], 0.27,
                    label=r["tag"])
    axes[0].set(xlabel="wall index i (0 = trailing edge, lower surface first)",
                ylabel="Cp", title="wall Cp change 16 -> 17 rows, subcritical mean")
    axes[0].legend(fontsize=6)
    axes[1].set(xscale="log", yscale="log", xlabel="row j + 1 (1 = wall)",
                ylabel="|p(17) - p(16)|", title="change through the column, mean over cases")
    axes[1].legend(fontsize=5)
    axes[2].set_xticks(np.arange(len(names)) + 0.27, names, rotation=20, fontsize=7)
    axes[2].set(ylabel="median R^2", title="what predicts delta along the wall (subcritical)")
    axes[2].legend(fontsize=6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def untrained_like(model, scale=50.0, seed=0):
    """An FNO2d of the same sizes with random weights, spectral weights scaled to a trained size."""
    torch.manual_seed(seed)
    w = model.spectral[0].w_pos
    m = FNO2d(width=w.shape[0], modes_i=w.shape[2], modes_j=w.shape[3],
              layers=len(model.spectral)).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.is_complex():
                p.mul_(scale)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", help="best.pt files or globs; default: S2's "
                    "fno-n896-s0, s1, s2")
    ap.add_argument("--h5", default="data/hdf5/256x97.h5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fno_drag")
    ap.add_argument("--untrained", action="store_true",
                    help="also an untrained FNO2d of S2's sizes, as a control")
    args = ap.parse_args(argv)
    if not Path(args.h5).exists():
        print(f"no dataset at {args.h5}")
        return 2
    patterns = args.checkpoints or DEFAULT_CHECKPOINTS
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        print("no checkpoint matches " + ", ".join(patterns))
        return 2
    device = probe.resolve_device(args.device)
    val = load_val(args.h5)
    models = [(path, *load_checkpoint(path)) for path in paths]
    if args.untrained:
        models.append(("untrained", untrained_like(models[0][1]), models[0][2], "untrained",
                       "untrained"))
    records = []
    for path, model, std, tag, source in models:
        t0 = time.perf_counter()
        rec = {"tag": tag, "source": source, "checkpoint": str(path), "device": str(device),
               "n_val": len(val["inputs"]),
               "n_subcritical": int((val["regime"] == "subcritical").sum()),
               **locate_one(model, std, val, device)}
        rec["wall_s"] = time.perf_counter() - t0
        records.append(rec)
        print(f"{tag}: located in {rec['wall_s']:.0f} s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "locate.json").write_text(json.dumps({"records": records}, indent=1))
    (out / "locate.md").write_text(report(records))
    plot(records, out / "locate.png")
    print(f"wrote {out}/locate.json, locate.md and locate.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
