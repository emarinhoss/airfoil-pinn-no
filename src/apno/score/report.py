"""Aggregation, tables and plots for stage S4. Every training run is reported, not only the best.

Three tables, every one split by method, configuration, grid, test set and regime (spec 8.4):
forces, surface and near field (`aggregate`); shocks on supercritical cases (`aggregate_shocks`),
pooled over the surfaces where the truth has a shock, with misses counted; and the physics checks
(`aggregate_physics`). A check a contestant satisfies by construction (BY_CONSTRUCTION) is printed
as "by construction", never as a value that reads as a pass (spec 8.2).
"""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KEYS = ("method", "variant", "grid", "n_train", "seed", "split", "regime")
SCALARS = ("cl_err", "cd_err_counts", "cm_err", "cp_l2", "cp_max", "near_field_rel_l2")
PHYSICS = ("entropy_max", "entropy_mean", "enthalpy_max_dev", "wall_normal_rms")
# Keyed on (method, formulation): the prediction file's `formulation` attr, default "euler".
# Euler PINN (spec 7.3): free-stream total enthalpy and a tangent wall velocity make those two
# checks identities. Potential PINN (spec 7.8): enthalpy and entropy are identities, but wall
# tangency is soft, so wall_normal_rms is a real result and must print as a value.
BY_CONSTRUCTION = {
    ("pinn", "euler"): ("enthalpy_max_dev", "wall_normal_rms"),
    ("pinn", "potential"): ("enthalpy_max_dev", "entropy_max", "entropy_mean"),
}
BY_CONSTRUCTION_CELL = "by construction"


def _groups(rows, extra=()):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in KEYS + tuple(extra))].append(r["score"])
    return sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))


def aggregate(rows):
    out = []
    for key, scores in _groups(rows):
        g = dict(zip(KEYS, key))
        ok = [s for s in scores if s["status"] == "ok"]
        for status in ("ok", "missing", "non-finite"):
            g[f"n_{status}"] = sum(s["status"] == status for s in scores)
        for m in SCALARS:
            vals = np.array([s[m] for s in ok if m in s], dtype=float)
            if vals.size:
                g[f"{m}_mean"] = float(vals.mean())
                g[f"{m}_median"] = float(np.median(vals))
                g[f"{m}_absmax"] = float(np.abs(vals).max())
        out.append(g)
    return out


def _stats(prefix, vals, signed=False):
    vals = np.asarray(vals, dtype=float)
    if not vals.size:
        return {}
    if signed:
        return {f"{prefix}_median": float(np.median(vals)),
                f"{prefix}_absmax": float(np.abs(vals).max())}
    return {f"{prefix}_median": float(np.median(vals)), f"{prefix}_max": float(vals.max())}


def aggregate_shocks(rows):
    """Per group of supercritical rows: shocks in the truth, misses, |x_err|, Mach-ahead error and
    overshoot, pooled over every (case, surface) where the truth has a shock."""
    out = []
    for key, scores in _groups(r for r in rows if r.get("regime") == "supercritical"):
        g = dict(zip(KEYS, key))
        ok = [s for s in scores if s["status"] == "ok"]
        found = [v for s in ok for v in s.get("shock", {}).values()]
        hit = [v for v in found if not v["missed"]]
        g.update(n_supercritical=len(scores), n_scored=len(ok), n_true_shocks=len(found),
                 n_missed=len(found) - len(hit))
        g.update(_stats("abs_x_err", [abs(v["x_err"]) for v in hit]))
        g.update(_stats("mach_ahead_err", [v["mach_ahead_err"] for v in hit], signed=True))
        g.update(_stats("overshoot", [v["overshoot"] for v in found]))
        out.append(g)
    return out


def aggregate_physics(rows):
    """Per group: median and maximum of each physics check, and of |mass flux| per grid line."""
    out = []
    for key, scores in _groups(rows, extra=("formulation",)):
        g = dict(zip(KEYS, key[:-1]))
        formulation = key[-1] or "euler"
        ok = [s["physics"] for s in scores if s["status"] == "ok"]
        g["n_ok"] = len(ok)
        # entropy is undefined on a case with rho <= 0 or p <= 0 somewhere in the near field: its
        # statistics are over the cases where it is defined, and this counts the others
        g["n_nonpositive_state"] = sum(p.get("n_nonpositive", 0) > 0 for p in ok)
        skip = BY_CONSTRUCTION.get((g["method"], formulation), ())
        for m in PHYSICS:
            if m in skip:
                g[f"{m}_median"] = g[f"{m}_max"] = BY_CONSTRUCTION_CELL
            else:
                g.update(_stats(m, [p[m] for p in ok if p[m] is not None]))
        lines = sorted({j for p in ok for j in p["mass_flux"]})
        for k, j in enumerate(lines):
            g.update(_stats(f"mass_flux_line{k}", [abs(p["mass_flux"][j]) for p in ok]))
            g[f"mass_flux_line{k}_j"] = j
        out.append(g)
    return out


def break_even(costs, n_max):
    n = np.arange(n_max + 1)
    return {k: c["fixed"] + n * c["per"] for k, c in costs.items()}


SHOCK_METRICS = ("n_supercritical", "n_scored", "n_true_shocks", "n_missed", "abs_x_err_median",
                 "abs_x_err_max", "mach_ahead_err_median", "mach_ahead_err_absmax",
                 "overshoot_median", "overshoot_max")
PHYSICS_METRICS = ("n_ok", "n_nonpositive_state") + \
    tuple(f"{m}_{s}" for m in PHYSICS for s in ("median", "max")) + \
    tuple(f"mass_flux_line{k}_{s}" for k in range(4) for s in ("j", "median", "max"))


def markdown_table(groups, metrics=("n_ok", "n_missing", "n_non-finite", "near_field_rel_l2_mean",
                                    "cl_err_absmax", "cd_err_counts_absmax", "cp_max_mean")):
    head = list(KEYS) + list(metrics)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in groups:
        cells = [g.get(h, "") for h in head]
        lines.append("| " + " | ".join(f"{c:.4g}" if isinstance(c, float) else str(c)
                                       for c in cells) + " |")
    return "\n".join(lines)


def plot_break_even(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, y in curves.items():
        ax.plot(np.arange(len(y)), y / 3600, label=k, ls="--" if k == "su2" else "-")
    ax.set(xlabel="new cases answered",
           ylabel="cumulative cost, hours (CPU time; GPU training at wall time)", yscale="log",
           title="Cost to answer n new cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_efficiency(points, floor, path):
    """points: method -> (n paid for, error[, n_cases trained on]); n_cases labels each point."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, pt in points.items():
        n, err = pt[0], pt[1]
        ax.plot(n, err, "o-", label=method)
        for x, y, c in zip(n, err, pt[2] if len(pt) > 2 else ()):
            if np.isfinite(y):
                ax.annotate(f"{c}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhline(floor, color="k", ls=":", label="floor: SU2 256x97 vs 512x193")
    ax.set(xscale="log", yscale="log",
           xlabel="SU2 training cases paid for (label: converged cases trained on)",
           ylabel="near-field relative L2 (test_in, mean)", title="Data efficiency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_showcase(xw, curves, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, cp in curves.items():
        ax.plot(xw, cp, ".-" if label == "truth" else "-", ms=2, label=label)
    ax.invert_yaxis()
    ax.set(xlabel="x/c", ylabel="Cp", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
