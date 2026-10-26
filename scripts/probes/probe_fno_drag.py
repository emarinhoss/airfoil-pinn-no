"""Where does the FNO's drag offset at 256x97 come from? No training: S2's checkpoints, validation.

At 256x97 the FNO's in-range drag error is +15 to +17 counts at the median with 17 radial padding
rows and -17 to -18 with 16, while the same models at 128x49 are within 0.4 counts of zero
(docs/findings.md, 2026-10-26). On the test cases the mean of the 16- and 17-row drag errors is the
128x49 one to within a few counts, case by case, which would follow if the drag at 256x97 were
linear in the row count and crossed the 128x49 value near 16.5 rows. That has two points behind it.
This probe measures it over a range of row counts, on the validation cases, so the test set is not
read.

The drag offset of a prediction is its CD at 256x97 minus the same model's CD at 128x49, in counts.
It needs no truth: it is what the resolution transfer does to the drag. Forces come from the wall
row alone, so the offset is the wall pressure's. Per checkpoint it records:

- per row count: the mean drag offset, by regime, and the near field against SU2 at 256x97 and the
  transfer, as the padding sweep computes them (probe_fno_pad_sweep);
- per case, a straight line fitted to the offset over `--fit` rows: its slope (counts per row), the
  row count where it crosses zero, and the largest residual, which says how straight it is;
- the prediction averaged over 16 and 17 rows, scored the same way. Forces are linear in the wall
  pressure, so its drag offset is the mean of the two; its near field and transfer are not;
- where on the wall the change from 16 to 17 rows acts: the mean change in wall Cp along x/c on
  each surface, and its drag by region (leading edge x/c < 0.05, trailing edge x/c > 0.95, the rest
  of each surface), from S4's vertex-lumped normals (`score.forces`).

    python scripts/probes/probe_fno_drag.py                    # S2's fno-n896-s0, s1, s2
    python scripts/probes/probe_fno_drag.py --rows 10-22 --fit 14-19 runs/fno/fno-n896-s0/best.pt

Writes results/fno_drag/drag.json, drag.md and drag.png.

Two modes need no dataset or checkpoint:

- `--from-scores`: the committed test scores at 17 rows (S4 at commit 7b39c58) and 16 rows
  (results/s4/scores.json), per case: the two drag errors, the 128x49 one, and the row count where
  the straight line through the two points meets the 128x49 value. Writes from_scores.json and .md.
- `--synthetic`: the transfer against the row count for one spectral layer on fields that are smooth
  across the wrap or stop at the far-field row, and for an untrained FNO2d (spectral weights scaled
  to a trained size) on the stand-in dataset's features. Writes synthetic.json and .md.
"""

import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import probe_fno_libraries as probe
from apno.data.build import regime
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.models.fno import FNO2d, SpectralConv2d
from apno.models.train_fno import features, load_features_only, load_split
from apno.physics.flow import pressure_coefficient
from apno.score.forces import vertex_normals, wall_forces
from probe_fno_pad_sweep import (load_checkpoint, near_field_rel_l2, near_field_setup, parse_rows,
                                 rows_at)

DEFAULT_CHECKPOINTS = ("runs/fno/fno-n896-s[0-9]/best.pt",)
LE, TE = 0.05, 0.95  # x/c bounds of the leading- and trailing-edge regions
REGIONS = ("leading edge", "upper", "lower", "trailing edge")


def walls(inputs, sub=1):
    """Per case, the wall nodes (x, y) of S4's grid, and of its 128x49 subgrid for sub=2."""
    out = []
    for _, _, t, m in np.asarray(inputs, dtype=float):
        X, Y = grid(Section(t, m), TRAIN)
        out.append((X[::sub, 0], Y[::sub, 0]))
    return out


def drag_counts(q, wall, inputs):
    """CD in counts for each case, as score_case computes it (S4's one force integrator)."""
    return np.array([1e4 * wall_forces(xw, yw, pressure_coefficient(qq[:, 0, 3], mach), alpha)[1]
                     for qq, (xw, yw), (mach, alpha, _, _) in zip(q, wall, inputs)])


def drag_by_node(dcp, xw, yw, alpha_deg):
    """Each wall node's share, in counts, of the drag of a wall-Cp change dcp; it sums to the CD
    change wall_forces gives."""
    n = vertex_normals(xw, yw)
    a = np.radians(alpha_deg)
    return -1e4 * np.asarray(dcp) * (n[:, 0] * np.cos(a) + n[:, 1] * np.sin(a))


def region_of(xw):
    """The region of each wall node: i = 0 is the trailing edge, the lower surface comes first."""
    ni = len(xw)
    reg = np.where(np.arange(ni) < ni // 2, "lower", "upper").astype(object)
    reg[xw < LE] = "leading edge"
    reg[xw > TE] = "trailing edge"
    return reg


def line_fit(rows, offsets):
    """Per case: slope (counts per row), the row count where the line crosses zero, and the
    largest absolute residual. offsets: (cases, rows)."""
    r = np.asarray(rows, dtype=float)
    A = np.stack([r, np.ones_like(r)], 1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(offsets).T, rcond=None)
    slope, icpt = coef
    resid = np.abs(np.asarray(offsets) - (slope[:, None] * r + icpt[:, None])).max(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = np.where(slope != 0, -icpt / slope, np.nan)
    return slope, cross, resid


def _q(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "n": int(v.size)} if v.size else None


def _by_regime(per_case, regimes):
    out = {"all": float(np.mean(per_case))}
    for reg in ("subcritical", "supercritical"):
        sel = regimes == reg
        out[reg] = float(np.mean(per_case[sel])) if sel.any() else None
    return out


def _score(q_fine, q_coarse, val, cd_coarse, setup):
    """Drag offset, drag error against SU2 at 256x97, near field and transfer of one 256x97
    prediction."""
    inputs, regimes = val["inputs"], val["regime"]
    cd = drag_counts(q_fine, val["_wall"], inputs)
    off = cd - cd_coarse
    return off, {"drag_offset_mean": _by_regime(off, regimes),
                 "drag_offset_median": float(np.median(off)),
                 "drag_err_vs_su2_256_median": float(np.median(cd - val["_cd_su2"])),
                 "near_field": _by_regime(near_field_rel_l2(q_fine, val["q"], setup), regimes),
                 "transfer": probe.transfer_rel_l2(q_fine, q_coarse, inputs)}


def probe_one(model, std, val, rows, fit, device):
    model = model.to(device)
    xy, inputs, regimes = val["xy"], val["inputs"], val["regime"]
    q_coarse, _ = probe.predict(model, std, xy[:, ::2, ::2], inputs)
    cd_coarse = drag_counts(q_coarse, val["_wall_coarse"], inputs)
    preds, per, offsets = {}, {}, {}
    for r in sorted(set(rows) | set(fit) | {16, 17}):
        with rows_at(model, r):
            preds[r], _ = probe.predict(model, std, xy, inputs)
        offsets[r], per[r] = _score(preds[r], q_coarse, val, cd_coarse, val["_near_fine"])
    avg = 0.5 * (preds[16] + preds[17])
    off_avg, rec_avg = _score(avg, q_coarse, val, cd_coarse, val["_near_fine"])
    slope, cross, resid = line_fit(fit, np.stack([offsets[r] for r in fit], 1))
    sub = regimes == "subcritical"

    # where the change from 16 to 17 rows acts on the wall
    dcp = np.stack([pressure_coefficient(preds[17][k, :, 0, 3], m) -
                    pressure_coefficient(preds[16][k, :, 0, 3], m)
                    for k, m in enumerate(inputs[:, 0])])
    shares = np.stack([drag_by_node(dcp[k], *val["_wall"][k], inputs[k, 1])
                       for k in range(len(inputs))])
    regions = {name: [] for name in REGIONS}
    for k, (xw, _) in enumerate(val["_wall"]):
        reg = region_of(xw)
        for name in REGIONS:
            regions[name].append(float(shares[k][reg == name].sum()))
    return {
        "cd_128x49_minus_su2_256_median": float(np.median(cd_coarse - val["_cd_su2"])),
        "near_field_128x49": _by_regime(near_field_rel_l2(
            q_coarse, val["q"][:, ::2, ::2], val["_near_coarse"]), regimes),
        "rows": {r: per[r] for r in sorted(per) if r in rows or r in (16, 17)},
        "avg_16_17": rec_avg,
        "fit": {"rows": list(fit),
                "slope_counts_per_row": {"all": _q(slope), "subcritical": _q(slope[sub]),
                                         "supercritical": _q(slope[~sub])},
                "crossing_rows": {"all": _q(cross), "subcritical": _q(cross[sub]),
                                  "supercritical": _q(cross[~sub])},
                "max_residual_counts": _q(resid)},
        "change_16_to_17": {
            "drag_counts_mean": float(shares.sum(1).mean()),
            "by_region_mean": {k: float(np.mean(v)) for k, v in regions.items()},
            "by_region_mean_subcritical": {k: float(np.mean(np.asarray(v)[sub]))
                                           for k, v in regions.items()} if sub.any() else None,
            "wall_cp_mean": dcp.mean(0).tolist(),
            "wall_cp_mean_subcritical": dcp[sub].mean(0).tolist() if sub.any() else None,
            "wall_x_mean": np.mean([w[0] for w in val["_wall"]], 0).tolist()},
        "_offsets": {r: offsets[r].tolist() for r in sorted(offsets)},
    }


def load_val(h5_path):
    val = load_split(h5_path, "val")
    with h5py.File(h5_path) as f:
        val["regime"] = np.array([regime(m) for m in f["val"]["max_surface_mach"][...]])
    val["_near_fine"] = near_field_setup(val["inputs"])
    val["_near_coarse"] = near_field_setup(val["inputs"], sub=2)
    val["_wall"] = walls(val["inputs"])
    val["_wall_coarse"] = walls(val["inputs"], sub=2)
    val["_cd_su2"] = drag_counts(val["q"], val["_wall"], val["inputs"])
    return val


def _f(v, fmt="{:.2f}"):
    return "—" if v is None else fmt.format(v)


def report(records, rows):
    lines = ["# The FNO's drag offset at 256x97 against the radial padding rows", "",
             "Validation cases. Drag offset: CD at 256x97 minus the same model's CD at 128x49, "
             "in counts (no truth involved). Written by `python scripts/probes/probe_fno_drag.py`.",
             ""]
    for rec in records:
        f = rec["fit"]
        lines += [f"## {rec['tag']}", "",
                  f"{rec['n_val']} validation cases, of which {rec['n_subcritical']} subcritical. "
                  f"CD at 128x49 minus SU2 at 256x97, median: "
                  f"{rec['cd_128x49_minus_su2_256_median']:+.1f} counts.", "",
                  "| rows | drag offset, mean (sub / super) | median | near field (sub / super) | "
                  "transfer |", "|---|---|---|---|---|"]
        for r, v in sorted(rec["rows"].items(), key=lambda kv: int(kv[0])):
            d, n = v["drag_offset_mean"], v["near_field"]
            lines.append(f"| {r} | {_f(d['subcritical'], '{:+.1f}')} / "
                         f"{_f(d['supercritical'], '{:+.1f}')} | {v['drag_offset_median']:+.1f} | "
                         f"{_f(n['subcritical'], '{:.4f}')} / {_f(n['supercritical'], '{:.4f}')} | "
                         f"{v['transfer']:.4f} |")
        a, nc = rec["avg_16_17"], rec["near_field_128x49"]
        lines += [f"| mean of 16 and 17 | {_f(a['drag_offset_mean']['subcritical'], '{:+.1f}')} / "
                  f"{_f(a['drag_offset_mean']['supercritical'], '{:+.1f}')} | "
                  f"{a['drag_offset_median']:+.1f} | "
                  f"{_f(a['near_field']['subcritical'], '{:.4f}')} / "
                  f"{_f(a['near_field']['supercritical'], '{:.4f}')} | {a['transfer']:.4f} |",
                  f"| 128x49 (own grid) | 0 | 0 | {_f(nc['subcritical'], '{:.4f}')} / "
                  f"{_f(nc['supercritical'], '{:.4f}')} | — |", ""]
        lines += [f"Straight line through the offset over rows {f['rows'][0]}-{f['rows'][-1]}, "
                  "per case (median [interquartile range]):", ""]
        for reg in ("subcritical", "supercritical", "all"):
            s, c = f["slope_counts_per_row"][reg], f["crossing_rows"][reg]
            if s:
                lines.append(f"- {reg} ({s['n']}): slope {s['median']:+.1f} "
                             f"[{s['q25']:+.1f}, {s['q75']:+.1f}] counts per row; crosses zero "
                             f"at {c['median']:.2f} [{c['q25']:.2f}, {c['q75']:.2f}] rows")
        m = f["max_residual_counts"]
        lines += [f"- largest residual from the line: {m['median']:.2f} "
                  f"[{m['q25']:.2f}, {m['q75']:.2f}] counts", ""]
        ch = rec["change_16_to_17"]
        lines += ["Drag of the wall-Cp change from 16 to 17 rows, by region, mean over cases "
                  f"(total {ch['drag_counts_mean']:+.1f} counts): "
                  + ", ".join(f"{k} {v:+.1f}" for k, v in ch["by_region_mean"].items()), ""]
    return "\n".join(lines) + "\n"


def plot(records, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for rec in records:
        rs = sorted(int(r) for r in rec["rows"])
        for reg, ls in (("subcritical", "-"), ("supercritical", "--")):
            v = [rec["rows"][r]["drag_offset_mean"][reg] for r in rs]
            if v[0] is not None:
                axes[0].plot(rs, v, ls, marker="o", ms=3, label=f"{rec['tag']} {reg}")
        ch = rec["change_16_to_17"]
        x = np.asarray(ch["wall_x_mean"])
        dcp = np.asarray(ch["wall_cp_mean_subcritical"] or ch["wall_cp_mean"])
        half = len(x) // 2
        axes[1].plot(x[:half], dcp[:half], label=f"{rec['tag']} lower")
        axes[1].plot(x[half:], dcp[half:], "--", label=f"{rec['tag']} upper")
        reg = ch["by_region_mean_subcritical"] or ch["by_region_mean"]
        axes[2].bar([f"{k}\n{rec['tag'][-2:]}" for k in reg], list(reg.values()))
    axes[0].axhline(0, color="k", lw=0.5)
    for r in (16, 17):
        axes[0].axvline(r, color="grey", lw=0.5, ls=":")
    axes[0].set(xlabel="radial padding rows at 256x97", ylabel="drag offset (counts)",
                title="CD(256x97) - CD(128x49), mean over cases")
    axes[0].legend(fontsize=6)
    axes[1].set(xlabel="x/c", ylabel="Cp(17 rows) - Cp(16 rows)",
                title="wall Cp change, mean over subcritical cases")
    axes[1].legend(fontsize=6)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(ylabel="drag (counts)", title="drag of that change by region (subcritical)")
    axes[2].tick_params(axis="x", labelsize=6)
    fig.savefig(path, dpi=130)
    plt.close(fig)


S4_AT_17_ROWS = "7b39c58"  # the last commit whose results/s4/scores.json is at 17 rows


def from_scores(new_path="results/s4/scores.json", old_text=None):
    """Per FNO run and test split: the drag errors at 16 rows, 17 rows and 128x49, and where the
    line through the two 256x97 points meets the 128x49 value. old_text: the 17-row scores.json
    (default: read from git at S4_AT_17_ROWS)."""
    if old_text is None:
        old_text = subprocess.run(["git", "show", f"{S4_AT_17_ROWS}:results/s4/scores.json"],
                                  capture_output=True, text=True, check=True).stdout

    def rows(text):
        return {(r["tag"], r["grid"], r["case_id"]): r["score"]["cd_err_counts"]
                for r in json.loads(text)["rows"]
                if r["method"] == "fno" and r["score"].get("status") == "ok"}

    new, old = rows(Path(new_path).read_text()), rows(old_text)
    out = []
    for tag in sorted({t for t, g, _ in new if g == "256x97"}):
        for split in ("test_in", "test_ext"):
            ids = sorted(c for t, g, c in new if t == tag and g == "256x97" and c.startswith(split))
            if not ids or any((tag, "256x97", c) not in old for c in ids):
                continue
            e16 = np.array([new[(tag, "256x97", c)] for c in ids])
            e17 = np.array([old[(tag, "256x97", c)] for c in ids])
            e128 = np.array([new[(tag, "128x49", c)] for c in ids])
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = 16 + (e16 - e128) / (e16 - e17)
            out.append({"tag": tag, "split": split, "n": len(ids),
                        "median_16": float(np.median(e16)), "median_17": float(np.median(e17)),
                        "median_128x49": float(np.median(e128)),
                        "neg_16": int((e16 < 0).sum()), "neg_17": int((e17 < 0).sum()),
                        "mean_abs_16": float(np.abs(e16).mean()),
                        "mean_abs_17": float(np.abs(e17).mean()),
                        "mean_abs_avg": float(np.abs((e16 + e17) / 2).mean()),
                        "mean_abs_128x49": float(np.abs(e128).mean()),
                        "avg_minus_128x49": _q((e16 + e17) / 2 - e128),
                        "slope_17_minus_16": _q(e17 - e16),
                        "crossing_rows": _q(cross)})
    return out


def report_from_scores(recs):
    lines = ["# The FNO's test drag errors at 16 and 17 padding rows and at 128x49", "",
             f"17 rows: results/s4/scores.json at commit {S4_AT_17_ROWS}; 16 rows and 128x49: the "
             "current file. Counts. Crossing: the row count where the line through the 16- and "
             "17-row errors meets the 128x49 error, per case (median [interquartile range]). "
             "Written by `python scripts/probes/probe_fno_drag.py --from-scores`.", "",
             "| run | split | n | median 17 / 16 / 128x49 | negative 17 / 16 | mean abs 17 / 16 / "
             "average / 128x49 | (16 + 17)/2 - 128x49 | 17 - 16 | crossing |",
             "|---|---|---|---|---|---|---|---|---|"]
    def q(d, f="{:+.1f}"):
        return f"{f.format(d['median'])} [{f.format(d['q25'])}, {f.format(d['q75'])}]"

    for r in recs:
        lines.append(
            f"| {r['tag']} | {r['split']} | {r['n']} | {r['median_17']:+.1f} / "
            f"{r['median_16']:+.1f} / {r['median_128x49']:+.1f} | {r['neg_17']} / {r['neg_16']} | "
            f"{r['mean_abs_17']:.1f} / {r['mean_abs_16']:.1f} / {r['mean_abs_avg']:.1f} / "
            f"{r['mean_abs_128x49']:.1f} | {q(r['avg_minus_128x49'])} | "
            f"{q(r['slope_17_minus_16'])} | {q(r['crossing_rows'], '{:.2f}')} |")
    return "\n".join(lines) + "\n"


def _layer_transfer(conv, field, rows, wrap_smooth):
    """One spectral layer: 128x49 (8 rows) against 256x97 with `rows` rows at shared nodes."""
    outs = []
    for nj, pad in ((49, 8), (97, rows)):
        n = nj + pad
        eta = torch.arange(n, dtype=torch.float64) / (nj - 1)
        if wrap_smooth:
            g = torch.cos(2 * torch.pi * 3 * eta * (nj - 1) / n + 0.3)
        else:
            g = field(eta) * (torch.arange(n) < nj)  # data on [0, 1], then the zero rows
        xi = torch.arange(32 if nj == 49 else 64, dtype=torch.float64)
        f = g[None, :] * torch.cos(2 * torch.pi * xi / len(xi))[:, None]
        with torch.no_grad():
            outs.append(conv(f[None, None])[0, 0, :, :nj])
    c, fine = outs
    return float((c - fine[::2, ::2]).norm() / c.norm())


def synthetic(rows=range(13, 21), seeds=3, scales=(20.0, 50.0, 150.0)):
    rows = list(rows)
    torch.manual_seed(0)
    conv = SpectralConv2d(1, 1, modes_i=4, modes_j=8).double()
    fields = {"constant 1": lambda e: torch.ones_like(e), "1 + eta": lambda e: 1 + e,
              "exp(-3 eta)": lambda e: torch.exp(-3 * e)}
    layer = {name: [_layer_transfer(conv, f, r, False) for r in rows]
             for name, f in fields.items()}
    layer["cos, smooth across the wrap"] = [_layer_transfer(conv, None, r, True) for r in rows]
    with tempfile.TemporaryDirectory() as d:
        h5 = probe.write_standin(Path(d) / "s.h5", {"val": 6})
        feat = load_features_only(h5, "val")
    x = torch.from_numpy(features(feat["xy"], feat["inputs"])).float()
    xc = x[..., ::2, ::2]
    norm = lambda t: t.flatten(1).norm(dim=1)  # noqa: E731
    untrained = {}
    for scale in scales:
        res = np.zeros((seeds, len(rows)))
        for seed in range(seeds):
            torch.manual_seed(seed)
            m = FNO2d(width=48, modes_i=24, modes_j=8, layers=4).eval()
            with torch.no_grad():
                for p in m.parameters():
                    if p.is_complex():
                        p.mul_(scale)
                yc = m(xc)
                for k, r in enumerate(rows):
                    with rows_at(m, r):
                        yf = m(x)[..., ::2, ::2]
                    res[seed, k] = float((norm(yf - yc) / norm(yc)).mean())
        untrained[f"spectral weights x{scale:g}"] = res.mean(0).tolist()
    return {"rows": rows, "one_layer": layer, "untrained_fno2d": untrained}


def _vertex(rows, v):
    """Where two straight branches through the points either side of the minimum meet."""
    k = int(np.argmin(v))
    if k == 0 or k == len(v) - 1:
        return None
    if v[k - 1] < v[k + 1]:
        k -= 1  # the minimum's pair is (k, k+1) with k the lower row count
    if k == 0 or k + 2 >= len(v):
        return None
    sl, sr = v[k] - v[k - 1], v[k + 2] - v[k + 1]  # left (negative) and right (positive) slopes
    return float((v[k + 1] - v[k] + sl * rows[k] - sr * rows[k + 1]) / (sl - sr))


def report_synthetic(res):
    rows = res["rows"]
    lines = ["# Transfer against the row count at 256x97, without training", "",
             "Relative L2 difference between the 128x49 output (8 rows) and the 256x97 output at "
             "its shared nodes. Vertex: where straight branches through the two points either side "
             "of the minimum meet. Written by `python scripts/probes/probe_fno_drag.py "
             "--synthetic`.", "",
             "| | " + " | ".join(str(r) for r in rows) + " | vertex |",
             "|---|" + "---|" * (len(rows) + 1)]
    for title, block in (("one spectral layer", res["one_layer"]),
                         ("untrained FNO2d", res["untrained_fno2d"])):
        for name, v in block.items():
            vx = _vertex(rows, np.asarray(v))
            lines.append(f"| {title}: {name} | " + " | ".join(f"{x:.4f}" for x in v)
                         + f" | {'—' if vx is None else f'{vx:.2f}'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="*", help="best.pt files or globs; default: S2's "
                    "fno-n896-s0, s1, s2")
    ap.add_argument("--rows", default="12-21", help="row counts at 256x97 to score")
    ap.add_argument("--fit", default="14-19", help="row counts the straight line is fitted over")
    ap.add_argument("--h5", default="data/hdf5/256x97.h5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/fno_drag")
    ap.add_argument("--from-scores", action="store_true",
                    help="analyse the committed 16- and 17-row test scores; no data needed")
    ap.add_argument("--synthetic", action="store_true",
                    help="one spectral layer and an untrained FNO2d; no data needed")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.from_scores or args.synthetic:
        out.mkdir(parents=True, exist_ok=True)
        if args.from_scores:
            recs = from_scores()
            (out / "from_scores.json").write_text(json.dumps(recs, indent=1))
            (out / "from_scores.md").write_text(report_from_scores(recs))
            print(report_from_scores(recs))
        if args.synthetic:
            res = synthetic()
            (out / "synthetic.json").write_text(json.dumps(res, indent=1))
            (out / "synthetic.md").write_text(report_synthetic(res))
            print(report_synthetic(res))
        return 0
    rows, fit = parse_rows(args.rows), parse_rows(args.fit)
    if len(fit) < 2:
        ap.error("--fit needs at least two row counts")
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
    records = []
    for path in paths:
        model, std, tag, source = load_checkpoint(path)
        t0 = time.perf_counter()
        res = probe_one(model, std, val, rows, fit, device)
        rec = {"tag": tag, "source": source, "checkpoint": str(path), "device": str(device),
               "n_val": len(val["inputs"]),
               "n_subcritical": int((val["regime"] == "subcritical").sum()),
               "wall_s": time.perf_counter() - t0, **res}
        records.append(rec)
        print(f"{tag}: {len(rec['rows'])} row counts in {rec['wall_s']:.0f} s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drag.json").write_text(json.dumps(
        {"rows": rows, "fit": fit, "case_ids": list(val["case_id"]),
         "regimes": val["regime"].tolist(), "records": records}, indent=1))
    (out / "drag.md").write_text(report(records, rows))
    plot(records, rows, out / "drag.png")
    print(f"wrote {out}/drag.json, drag.md and drag.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
