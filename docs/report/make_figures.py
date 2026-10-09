"""Figures for docs/report/report.tex, written to docs/report/figures/*.pdf.

Run from the repository root:  python docs/report/make_figures.py

Reads the committed records (results/s1, results/s3, results/s4, results/s5) and, for the
flow-field and wall-pressure figures, the git-ignored dataset and predictions (data/hdf5,
runs/predictions, runs/predictions_s5, runs/su2, runs/warmup_q*.npy), so those figures need a
machine that has run the stages.

One colour per contestant throughout: FNO, POD and PINN take the first three categorical slots;
SU2 (the truth and the floor) is drawn in ink, as the reference every contestant is read against.
"""

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from apno.data.predictions import read_predictions  # noqa: E402
from apno.data.sampling import case_specs  # noqa: E402
from apno.geometry.ogrid import TRAIN, Section, grid  # noqa: E402
from apno.physics.flow import local_mach, pressure_coefficient  # noqa: E402

OUT = Path("docs/report/figures")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
FNO, POD, PINN, EULER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FAIL = "#d03b3b"  # status: critical (did not converge), always with a label
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

TEXTWIDTH = 6.5  # inches, the report's \textwidth

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def specs():
    return {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}


def truth_and_floor():
    truth, floor, mmax = {}, {}, {}
    for split in ("test_in", "test_ext"):
        with h5py.File("data/hdf5/512x193.h5") as f, h5py.File("data/hdf5/256x97.h5") as g:
            ids_f = list(f[split]["case_id"].asstr()[...])
            ids_g = list(g[split]["case_id"].asstr()[...])
            for k, cid in enumerate(ids_f):
                if cid in ids_g:
                    truth[cid] = f[split]["q"][k][::2, ::2]
                    floor[cid] = g[split]["q"][ids_g.index(cid)]
                    mmax[cid] = float(f[split]["max_surface_mach"][k])
    return truth, floor, mmax


def showcase_ids():
    return [c["case_id"] for c in json.loads(Path("results/s1/showcase.json").read_text())["cases"]]


# 1. The O-grid
def fig_grid():
    X, Y = grid(Section(0.12, 0.02), TRAIN)
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.05),
                            gridspec_kw={"width_ratios": [1.0, 1.55, 1.0]})
    views = [((-110, 110), (-110, 110), 8, 4, "(a) whole domain"),
             ((-0.45, 1.45), (-0.55, 0.55), 2, 2, "(b) near the section, every 2nd line"),
             ((0.93, 1.03), (-0.035, 0.035), 1, 1, "(c) trailing edge")]
    for ax, (xl, yl, si, sj, title) in zip(axs, views):
        Xs, Ys = X[::si, ::sj], Y[::si, ::sj]
        Xc, Yc = np.vstack([Xs, Xs[:1]]), np.vstack([Ys, Ys[:1]])
        ax.plot(Xc, Yc, color=MUTED, lw=0.3)
        ax.plot(Xc.T, Yc.T, color=MUTED, lw=0.3)
        ax.plot(np.r_[X[:, 0], X[0, 0]], np.r_[Y[:, 0], Y[0, 0]], color=INK, lw=0.9)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("x / c")
    axs[0].set_ylabel("y / c")
    save(fig, "grid.pdf")


# 2. The case design
def fig_design():
    pool = case_specs("train", TRAIN) + case_specs("val", TRAIN)
    tin, text = case_specs("test_in", TRAIN), case_specs("test_ext", TRAIN)
    failed = set()
    for lv in ("256x97", "512x193"):
        for split in json.loads(Path(f"results/s1/failures_{lv}.json").read_text()).values():
            failed |= {f["case_id"] for f in split}
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    axes_of = [("mach", "alpha_deg", "Mach number M", "angle of attack α (°)"),
               ("t", "m", "thickness t", "maximum camber m")]
    for ax, (a, b, la, lb) in zip(axs, axes_of):
        g = lambda S, k: np.array([getattr(s, k) for s in S])  # noqa: E731
        ax.scatter(g(pool, a), g(pool, b), s=3, color=MUTED, lw=0,
                   label=f"training + validation ({len(pool):,})")
        ax.scatter(g(tin, a), g(tin, b), s=14, facecolors="none", edgecolors=INK, lw=0.7,
                   label=f"test, in range ({len(tin)})")
        ax.scatter(g(text, a), g(text, b), s=14, marker="^", color=INK2, lw=0,
                   label=f"test, extrapolation ({len(text)})")
        F = [s for s in pool + tin + text if s.case_id in failed]
        ax.scatter(g(F, a), g(F, b), s=16, marker="x", color=FAIL, lw=0.9,
                   label=f"did not converge ({len(F)} cases)")
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
    axs[0].axvline(0.80, color=AXIS, lw=0.6)
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "design.pdf")


# 3. Mach fields, truth against each contestant
def fig_fields(truth, S):
    cases = [("test_in-0022", "subcritical"), ("test_in-0040", "supercritical")]
    cols = [("SU2 512×193 (truth)", None), ("FNO", "fno-n896-s0"), ("POD", "pod-n896"),
            ("PINN", "pinn-pure-{cid}")]
    fig, axs = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.0), sharex=True, sharey=True)
    scales = {"subcritical": (0.0, 0.6, np.arange(0.05, 0.6, 0.05)),
              "supercritical": (0.0, 1.7, np.arange(0.1, 1.7, 0.1))}
    for r, (cid, reg) in enumerate(cases):
        s = S[cid]
        X, Y = grid(Section(s.t, s.m), TRAIN)
        lo, hi, iso = scales[reg]
        for c, (label, tag) in enumerate(cols):
            ax = axs[r, c]
            if tag is None:
                q = truth[cid]
            else:
                path = Path("runs/predictions") / (tag.format(cid=cid) + ".h5")
                preds, _ = read_predictions(path)
                q = preds[cid]
            M = local_mach(q[..., 0], q[..., 1], q[..., 2], q[..., 3], s.mach)
            Xc, Yc, Mc = (np.vstack([A, A[:1]]) for A in (X, Y, M))
            pm = ax.pcolormesh(Xc, Yc, Mc, cmap=SEQ, vmin=lo, vmax=hi, shading="gouraud",
                               rasterized=True)
            ax.contour(Xc, Yc, Mc, levels=iso, colors="white", linewidths=0.35, alpha=0.8)
            if reg == "supercritical":
                ax.contour(Xc, Yc, Mc, levels=[1.0], colors=INK, linewidths=0.8)
            ax.fill(X[:, 0], Y[:, 0], color="white", lw=0)
            ax.plot(np.r_[X[:, 0], X[0, 0]], np.r_[Y[:, 0], Y[0, 0]], color=INK, lw=0.6)
            ax.set_xlim(-0.4, 1.5)
            ax.set_ylim(-0.6, 0.9)
            ax.set_aspect("equal")
            ax.grid(False)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(f"{cid}\n{reg}, M∞ {s.mach:.2f}")
        cb = fig.colorbar(pm, ax=axs[r], shrink=0.92, pad=0.015)
        cb.set_label("local Mach" + (" (black: M = 1)" if reg == "supercritical" else ""),
                     fontsize=7)
        cb.outline.set_visible(False)
    save(fig, "fields.pdf")


# 4. The cylinder warm-up: the Euler PINN at rest, the potential PINN passing
def fig_warmup():
    th = 2 * np.pi * np.arange(256) / 256
    exact = 1 - 4 * np.sin(th) ** 2
    pot = pressure_coefficient(np.load("runs/warmup_q.npy")[:, 0, 3], 0.1)
    eul = pressure_coefficient(np.load("runs/warmup_q_euler_2026-10-02.npy")[:, 0, 3], 0.1)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.3))
    d = np.degrees(th)
    ax.plot(d, exact, color=INK, lw=1.0, label="potential flow, 1 − 4 sin²θ")
    ax.plot(d, pot, color=PINN, lw=2.2, alpha=0.85,
            label="full-potential PINN (passed, max |ΔCp| 0.027)")
    ax.plot(d, eul, color=EULER, lw=1.6, ls=(0, (4, 1.5)),
            label="Euler PINN (fluid at rest, max |ΔCp| 3.94)")
    ax.set_xlabel("θ (°), clockwise from the rear stagnation point")
    ax.set_ylabel("wall Cp")
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.invert_yaxis()
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.25))
    save(fig, "warmup.pdf")


# 5. Wall pressure on the six showcase cases
def fig_showcase(truth, floor, mmax, S):
    ids = showcase_ids()
    fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.1), sharex=True)
    for ax, cid in zip(axs.ravel(), ids):
        s = S[cid]
        X, _ = grid(Section(s.t, s.m), TRAIN)
        xw = np.r_[X[:, 0], X[0, 0]]
        cp = lambda q: np.r_[pressure_coefficient(q[:, 0, 3], s.mach),  # noqa: E731
                             pressure_coefficient(q[0, 0, 3], s.mach)]
        ax.plot(xw, cp(truth[cid]), color=INK, lw=1.0, label="SU2 512×193 (truth)", zorder=5)
        ax.plot(xw, cp(floor[cid]), color=INK2, lw=0.9, ls=(0, (3, 1.5)),
                label="SU2 256×97 (floor)", zorder=4)
        for tag, color, label in (("fno-n896-s0", FNO, "FNO"), ("pod-n896", POD, "POD"),
                                  (f"pinn-pure-{cid}", PINN, "PINN")):
            preds, _ = read_predictions(Path("runs/predictions") / f"{tag}.h5")
            ax.plot(xw, cp(preds[cid]), color=color, lw=1.2, label=label)
        ax.invert_yaxis()
        reg = "supercritical" if mmax[cid] > 1 else "subcritical"
        ax.set_title(f"{cid}: M∞ {s.mach:.2f}, α {s.alpha_deg:.1f}°\n"
                     f"max surface M {mmax[cid]:.2f} ({reg})", loc="left")
    for ax in axs[1]:
        ax.set_xlabel("x / c")
    for ax in axs[:, 0]:
        ax.set_ylabel("Cp")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "showcase.pdf")


# 6. Near-field error against the floor, per set and regime
def fig_nearfield(groups):
    def val(method, variant, grid_, split, regime, n=None, seed=None):
        for g in groups:
            if (g["method"], g["variant"], g["grid"], g["split"], g["regime"]) == \
                    (method, variant, grid_, split, regime) and \
                    (n is None or g["n_train"] == n) and (seed is None or g["seed"] == seed):
                return g["near_field_rel_l2_mean"]
        return None

    panels = [("test_in", "subcritical", "In range, subcritical (37)"),
              ("test_in", "supercritical", "In range, supercritical (24)"),
              ("test_ext", "supercritical", "Extrapolation, supercritical (14)")]
    rows = [("PINN (3 cases)", PINN, "o", ("pinn", "pure", "256x97", None, None)),
            ("POD", POD, "o", ("pod", "", "256x97", 896, 0)),
            ("FNO at 256×97", FNO, "o", ("fno", "", "256x97", 896, 0)),
            ("FNO at 128×49", FNO, "D", ("fno", "", "128x49", 896, 0))]
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.85), sharey=True)
    for ax, (split, regime, title) in zip(axs, panels):
        fl = val("su2-256x97", "floor", "256x97", split, regime)
        ax.axvline(fl, color=INK, lw=0.9)
        ax.text(fl, len(rows) - 0.35, f" floor {fl:.4f}", color=INK, fontsize=7, va="bottom")
        for k, (label, color, mk, (m, v, gr, n, sd)) in enumerate(rows):
            e = val(m, v, gr, split, regime, n, sd)
            if e is None:
                continue
            ax.plot([fl, e], [k, k], color=GRID, lw=1.0, zorder=1)
            ax.plot(e, k, mk, color=color, ms=6 if mk == "o" else 5, mec="white", mew=0.8,
                    zorder=3)
            ax.text(e * 1.18, k, f"{e / fl:.1f}×", va="center", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.005, 2.0)
        ax.set_ylim(-0.6, len(rows) - 0.1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("near-field relative L2 error")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    save(fig, "nearfield.pdf")


# 7. Lift parity
def fig_cl_parity(rows):
    sets = [(("test_in", "subcritical"), "o", "in range, subcritical"),
            (("test_in", "supercritical"), "^", "in range, supercritical"),
            (("test_ext", "supercritical"), "s", "extrapolation")]
    panels = [("FNO (n = 896, seed 0)", FNO, lambda r: r["tag"] == "fno-n896-s0"),
              ("POD (n = 896)", POD, lambda r: r["tag"] == "pod-n896"),
              ("PINN (all 8 runs)", PINN, lambda r: r["method"] == "pinn")]
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3), sharex=True, sharey=True)
    for ax, (title, color, sel) in zip(axs, panels):
        ax.plot([-0.6, 1.9], [-0.6, 1.9], color=AXIS, lw=0.8, zorder=1)
        for (split, regime), mk, label in sets:
            R = [r for r in rows if sel(r) and r["grid"] == "256x97" and r["split"] == split
                 and r["regime"] == regime and r["score"].get("status") == "ok"]
            if not R:
                continue
            t = np.array([r["score"]["cl_true"] for r in R])
            p = t + np.array([r["score"]["cl_err"] for r in R])
            filled = split == "test_in"
            ax.scatter(t, p, s=16, marker=mk, facecolors=color if filled else "none",
                       edgecolors="white" if filled else color, lw=0.6 if filled else 0.9,
                       label=label, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("CL, SU2 truth")
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 1.9)
        ax.set_ylim(-0.6, 1.9)
    axs[0].set_ylabel("CL, predicted")
    axs[0].legend(loc="upper left", handletextpad=0.2)
    save(fig, "cl_parity.pdf")


# 8. Data efficiency
def fig_efficiency(rows, groups):
    ns = (64, 128, 256, 512, 896)

    def mean_nf(tag):
        v = [r["score"]["near_field_rel_l2"] for r in rows if r["tag"] == tag
             and r["grid"] == "256x97" and r["split"] == "test_in"]
        return float(np.mean(v))

    fl = float(np.mean([r["score"]["near_field_rel_l2"] for r in rows
                        if r["method"] == "su2-256x97" and r["split"] == "test_in"]))
    fno = [mean_nf(f"fno-n{n}-s0") for n in ns]
    pod = [mean_nf(f"pod-n{n}") for n in ns]
    seeds = [mean_nf(f"fno-n896-s{k}") for k in (1, 2)]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.6, 2.4))
    ax.axhline(fl, color=INK, lw=0.9)
    ax.text(70, fl * 1.06, f"floor: SU2 256×97 against 512×193 ({fl:.4f})", fontsize=7, color=INK)
    ax.plot(ns, pod, "-o", color=POD, ms=5, mec="white", mew=0.8)
    ax.plot(ns, fno, "-o", color=FNO, ms=5, mec="white", mew=0.8)
    ax.plot([896, 896], seeds, "o", color=FNO, ms=3.5, mfc="white", mew=0.9)
    ax.text(990, pod[-1], "POD", color=INK2, va="center", fontsize=7.5)
    ax.text(990, fno[-1], "FNO", color=INK2, va="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_yticks([0.015, 0.02, 0.03, 0.05, 0.1, 0.15], ["0.015", "0.02", "0.03", "0.05", "0.1",
                                                          "0.15"])
    ax.set_xlim(55, 1300)
    ax.set_xlabel("SU2 training cases paid for, n")
    ax.set_ylabel("near-field relative L2\n(test, in range, mean of 61)")
    save(fig, "efficiency.pdf")


# 9. Break-even
def fig_break_even():
    c = json.loads(Path("results/s4/costs.json").read_text())["costs"]
    n = np.arange(0, 5001)
    h = {k: (c[k]["fixed"] + n * c[k]["per"]) / 3600 for k in ("su2", "fno", "pod", "pinn")}
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
    ax = axs[0]
    m = n <= 2000
    ax.plot(n[m], h["su2"][m], color=INK, lw=1.2)
    ax.plot(n[m], h["fno"][m], color=FNO, lw=1.4)
    ax.plot(n[m], h["pod"][m], color=POD, lw=1.2, ls=(0, (3, 1.5)))
    for k, color, lab, xy in (("fno", FNO, "FNO", (80, 88)), ("pod", POD, "POD", (1250, 35))):
        nb = int(np.argmax(h[k] < h["su2"]))
        ax.plot(nb, h["su2"][nb], "o", color=color, ms=5, mec="white", mew=0.8, zorder=4)
        ax.annotate(f"{lab}: fixed cost {h[k][0]:.1f} h,\nbreaks even at n = {nb:,}",
                    (nb, h["su2"][nb]), xytext=xy, fontsize=7, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    ax.text(1600, h["su2"][1600] + 3, "just run SU2", fontsize=7, color=INK, ha="center")
    ax.set_xlabel("new cases answered, n")
    ax.set_ylabel("cumulative cost (hours)")
    ax.set_title("(a) FNO and POD against SU2", loc="left")
    ax = axs[1]
    for k, color, lab, ls in (("pinn", PINN, "PINN (per phase, upper bound)", "-"),
                              ("su2", INK, "SU2", "-"), ("fno", FNO, "FNO", "-"),
                              ("pod", POD, "POD", (0, (3, 1.5)))):
        ax.plot(n[1:], h[k][1:], color=color, lw=1.3, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("new cases answered, n")
    ax.set_title("(b) all four methods, log scales", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "break_even.pdf")


# 10. PINN training histories
def fig_pinn_history():
    R = {}
    for p in sorted(Path("results/s3").glob("pinn_runs_shard*of2.json")):
        R.update(json.loads(p.read_text()))
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.35))
    ax = axs[0]
    for key, ls, lab in (("pure/test_in-0022", "-", "with the Kutta condition (pure)"),
                         ("nokutta/test_in-0022", (0, (3, 1.5)), "without it (nokutta)")):
        H = [h for h in R[key]["history"] if h["step"] != "lbfgs"]
        ax.plot([h["step"] for h in H], [h["residual"] for h in H], color=PINN, ls=ls, lw=1.3,
                label=lab)
        ax.plot(31500, R[key]["final"]["residual"], "o" if ls == "-" else "s", color=PINN,
                ms=4.5, mfc="white" if ls != "-" else PINN, mew=0.9)
    ax.axvline(30000, color=AXIS, lw=0.6)
    ax.text(29600, 3, "end of\nL-BFGS", fontsize=6.5, color=INK2, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("residual loss")
    ax.set_title("(a) test_in-0022, with and without the Kutta term", loc="left")
    ax.legend(loc="lower left")
    ax = axs[1]
    r = R["gradnorm/test_in-0004"]
    H = [h for h in r["history"] if h["step"] != "lbfgs"]
    st = [h["step"] for h in H]
    ax.plot(st, [h["w_wall"] * h["wall"] for h in H], color=INK2, lw=1.3,
            label="weighted wall term")
    ax.plot(st, [h["residual"] for h in H], color=INK, lw=1.0, ls=(0, (3, 1.5)),
            label="residual")
    ax.plot(st, [max(h["w_kutta"] * h["kutta"], 1e-6) for h in H], color=MUTED, lw=1.0,
            ls=(0, (1, 1.2)), label="weighted Kutta term")
    f, w = r["final"], r["weights"]
    ax.plot(31500, w["wall"] * f["wall"], "o", color=INK2, ms=4.5)
    ax.plot(31500, f["residual"], "o", color=INK, ms=4.5, mfc="white", mew=0.9)
    ax.axvline(30000, color=AXIS, lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e5)
    ax.set_xlabel("Adam step")
    ax.set_title("(b) gradnorm on test_in-0004", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "pinn_history.pdf")


# S5: the gate on validation case B
def fig_s5_gate():
    from apno.score.s5 import truth_and_floor as s5_truth_and_floor
    cid, mach = "s5-val-B", 0.72
    truth, floor, mmax = s5_truth_and_floor(cid)
    preds, _ = read_predictions(Path("runs/predictions_s5/pinn-euler_av-T1.h5"))
    gate = json.loads(Path("results/s5/gate.json").read_text())["configs"]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
    ax = axs[0]
    X, _ = grid(Section(0.12, 0.0), TRAIN)
    xw = np.r_[X[:, 0], X[0, 0]]
    cp = lambda q: np.r_[pressure_coefficient(q[:, 0, 3], mach),  # noqa: E731
                         pressure_coefficient(q[0, 0, 3], mach)]
    ax.plot(xw, cp(truth), color=INK, lw=1.0, label="SU2 512×193 (truth)", zorder=5)
    ax.plot(xw, cp(floor), color=INK2, lw=0.9, ls=(0, (3, 1.5)), label="SU2 256×97 (floor)",
            zorder=4)
    ax.plot(xw, cp(preds[cid]), color=EULER, lw=1.4, label="S5 PINN, T1 (T2–T4 the same)")
    ax.invert_yaxis()
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp")
    ax.set_title(f"(a) wall pressure, case B\nM∞ 0.72, α 3°, max surface M {mmax:.2f}",
                 loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.56, 0.0))
    ax = axs[1]
    start = []
    for name, e in gate.items():
        H = e["history"]
        y = [H[0]["loss_before"]] + [h["total"] for h in H]
        ax.plot(np.arange(len(y)), y, color=EULER, lw=1.2)
        start.append(f"{name} {y[0]:.4g}")
    ax.text(30, 300, "initial loss: " + ", ".join(start), fontsize=6.5, color=INK2, va="center")
    ax.axhline(0.0469, color=MUTED, lw=0.8, ls=(0, (3, 1.5)))
    ax.text(800, 0.0469 * 0.7, "wall-tangency term of the free stream, 0.0469",
            fontsize=6.5, color=INK2, ha="right", va="top")
    ax.text(800, 0.0508 * 1.35, "all four, from the first outer iteration on: 0.0508",
            fontsize=6.5, color=INK2, ha="right")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 900)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3e3)
    ax.set_xlabel("outer iterations completed (20 per epoch, 800 in all)")
    ax.set_ylabel("loss")
    ax.set_title("(b) loss of T1–T4 on the batch just trained", loc="left")
    fig.tight_layout()
    save(fig, "s5_gate.pdf")


# Appendix figures (teaching): activations, spectral truncation, collocation points, learning curves
def fig_activations():
    from scipy.stats import norm
    z = np.linspace(-3, 3, 601)
    th = np.tanh(z)
    acts = [("tanh (the PINN)", PINN, "-", th, 1 - th**2, -2 * th * (1 - th**2)),
            ("GELU (the FNO)", FNO, "-", z * norm.cdf(z), norm.cdf(z) + z * norm.pdf(z),
             norm.pdf(z) * (2 - z**2)),
            ("ReLU (for comparison)", INK2, (0, (3, 1.5)), np.maximum(z, 0), (z > 0) * 1.0,
             np.zeros_like(z))]
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.95), sharex=True)
    for k, (title, ylab) in enumerate([("(a) σ(z)", "σ"), ("(b) first derivative σ′(z)", "σ′"),
                                       ("(c) second derivative σ″(z)", "σ″")]):
        ax = axs[k]
        for name, color, ls, *vals in acts:
            ax.plot(z, vals[k], color=color, ls=ls, lw=1.4, label=name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("z")
    axs[0].legend(loc="upper left")
    axs[2].annotate("ReLU: 0 everywhere\n(a Dirac spike at 0)", (1.5, 0), xytext=(0.4, 0.55),
                    fontsize=6.5, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    fig.tight_layout()
    save(fig, "activations.pdf")


def fig_spectral_truncation(truth, S):
    cid = "test_in-0040"
    s = S[cid]
    q = truth[cid][::2, ::2]  # the 128x49 nodes: the grid the FNO trains on
    X, _ = grid(Section(s.t, s.m), TRAIN)
    xw = X[::2, 0]
    cp = pressure_coefficient(q[:, 0, 3], s.mach)
    c = np.fft.fft(cp)
    k = np.fft.fftfreq(cp.size, 1.0 / cp.size)
    keep = np.abs(k) < 24
    cp24 = np.fft.ifft(np.where(keep, c, 0)).real
    up = np.arange(64, 128)  # leading edge to trailing edge along the upper surface
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    ax.plot(xw[up], cp[up], color=INK, lw=1.1, label="SU2 truth, 128 nodes around the wall")
    ax.plot(xw[up], cp24[up], color=FNO, lw=1.3, label="kept: the 24 lowest modes of each sign")
    ax.invert_yaxis()
    ax.set_xlabel("x / c (upper surface)")
    ax.set_ylabel("Cp")
    ax.set_title("(a) one spectral filter acting on wall Cp", loc="left")
    ax.legend(loc="lower left")
    ax = axs[1]
    kp = np.arange(0, 65)
    amp = np.abs(c[:65]) / cp.size
    ax.axvspan(-0.5, 23.5, color=GRID, lw=0)
    ax.plot(kp, amp, "o", color=INK, ms=2.8, mew=0)
    ax.text(11.5, amp.max() * 0.35, "kept", ha="center", fontsize=7, color=INK2)
    ax.text(44, amp.max() * 0.35, "discarded", ha="center", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("mode k around the airfoil (cycles per circuit)")
    ax.set_ylabel("|ĉ_k|")
    ax.set_title("(b) the spectrum of the same signal", loc="left")
    fig.tight_layout()
    save(fig, "spectral_truncation.pdf")


def fig_collocation(S):
    import torch

    from apno.geometry.ogrid import map_xy, stretch_ratio
    cid = "test_in-0022"
    s = S[cid]
    geom = Section(s.t, s.m)
    gen = torch.Generator().manual_seed(0)
    xi = torch.rand(20000, generator=gen, dtype=torch.float32)  # the run's first draw, seed 0
    eta = 1.0 - torch.rand(20000, generator=gen, dtype=torch.float32)
    x, y = (t.numpy() for t in map_xy(xi.double(), eta.double(), geom, 100.0,
                                      stretch_ratio(100.0), xp=torch))
    drop = np.hypot(x - 1.0, y) < 0.02
    near = np.hypot(x - 0.5, y) <= 2.5
    Xw, Yw = grid(geom, TRAIN)
    wall = (np.r_[Xw[:, 0], Xw[0, 0]], np.r_[Yw[:, 0], Yw[0, 0]])
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.25),
                            gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, (xl, yl, ms, title) in zip(axs, [
            ((-0.6, 1.6), (-0.6, 0.6), 0.6,
             f"(a) {near.sum():,} of 20,000 points lie within 2.5 chords"),
            ((0.94, 1.04), (-0.035, 0.035), 4.0,
             f"(b) trailing edge: {drop.sum()} points dropped")]):
        ax.plot(x[~drop], y[~drop], ".", color=PINN, ms=ms, mew=0)
        ax.plot(x[drop], y[drop], "x", color=INK, ms=3.0, mew=0.7)
        ax.fill(*wall, color="white", lw=0, zorder=3)
        ax.plot(*wall, color=INK, lw=0.8, zorder=4)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("x / c")
    circ = plt.Circle((1.0, 0.0), 0.02, fill=False, color=INK2, lw=0.8, ls=(0, (3, 1.5)))
    axs[1].add_patch(circ)
    axs[0].set_ylabel("y / c")
    save(fig, "collocation.pdf")


def fig_learning_curves():
    ns = (64, 128, 256, 512, 896)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal: n, light to dark
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.3))
    for n, color in zip(ns, ramp):
        d = json.loads(Path(f"runs/fno/fno-n{n}-s0/train.json").read_text())
        e = [r["epoch"] for r in d["log"]]
        v = [r["val"] for r in d["log"]]
        ax.plot(e, v, color=color, lw=1.2)
        ax.text(303, v[-1], f"n = {n}", color=INK2, fontsize=7, va="center")
        ax.plot(d["best_epoch"], d["best_val"], "o", color=color, ms=3.5, mec="white", mew=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0, 345)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss A")
    save(fig, "learning_curves.pdf")


def main():
    scores = json.loads(Path("results/s4/scores.json").read_text())
    rows, groups = scores["rows"], scores["groups"]
    S = specs()
    fig_grid()
    fig_design()
    fig_nearfield(groups)
    fig_cl_parity(rows)
    fig_efficiency(rows, groups)
    fig_break_even()
    fig_pinn_history()
    truth, floor, mmax = truth_and_floor()
    fig_fields(truth, S)
    fig_warmup()
    fig_showcase(truth, floor, mmax, S)
    fig_activations()
    fig_spectral_truncation(truth, S)
    fig_collocation(S)
    fig_learning_curves()
    fig_s5_gate()


if __name__ == "__main__":
    main()
