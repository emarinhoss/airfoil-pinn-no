"""Aggregation, tables and plots for stage S4. Every training run is reported, not only the best."""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KEYS = ("method", "variant", "n_train", "seed", "split", "regime")
SCALARS = ("cl_err", "cd_err_counts", "cm_err", "cp_l2", "cp_max", "near_field_rel_l2")


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEYS)].append(r["score"])
    out = []
    for key, scores in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
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


def break_even(costs, n_max):
    n = np.arange(n_max + 1)
    return {k: c["fixed"] + n * c["per"] for k, c in costs.items()}


def markdown_table(groups, metrics=("near_field_rel_l2_mean", "cl_err_absmax",
                                    "cd_err_counts_absmax", "cp_max_mean")):
    head = list(KEYS) + ["n_ok", "n_missing", "n_non-finite"] + list(metrics)
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
    ax.set(xlabel="new cases answered", ylabel="cumulative CPU hours", yscale="log",
           title="Cost to answer n new cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_efficiency(points, floor, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, (n, err) in points.items():
        ax.plot(n, err, "o-", label=method)
    ax.axhline(floor, color="k", ls=":", label="floor: SU2 256x97 vs 512x193")
    ax.set(xscale="log", yscale="log", xlabel="SU2 training cases",
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
