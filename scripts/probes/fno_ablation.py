"""Which of NeuralOperator's differences from FNO2d gives it the smaller resolution transfer?

docs/findings.md (2026-10-24) found NeuralOperator's FNO loses less than this study's `FNO2d`
between the 128x49 training grid and the 256x97 evaluation grid, in all five seeds of each, and
left the cause open. `AblationFno` is one model with a switch per difference, so each can be
changed alone. With every switch off it is `FNO2d`; with every switch on it is NeuralOperator
2.0.0's FNO as the probe configures it. Both are pinned by tests that copy weights across and
compare outputs.

The factors, each "on" meaning NeuralOperator's choice:

- M  channel MLP: after the spectral layer, a two-layer channel MLP (W -> W/2 -> W, GELU) plus a
     soft-gated skip from the block's input, then GELU (not after the last layer). Off: none.
- P  padding: zero rows on both radial ends, a fixed fraction (4/49 of NJ each side, so 4 + 4 at
     128x49 and 8 + 8 at 256x97). Off: FNO2d's `radial_pad(NJ)` rows at the far end only, which
     keeps the radial DFT period at 19/16 in eta at every NJ.
- I  spectral initialisation and bias placement: complex weights of standard deviation
     sqrt(2 / 2W) (0.144 at W 48), and the bias added to the spectral output, drawn with the same
     deviation; the skip has no bias. Off: deviation 1 / W^2 (4.3e-4), no spectral bias, the skip
     convolution carrying PyTorch's default bias. The two placements give the same functions; only
     the initial values differ.
- L  lifting and projection: two-layer lifting (6 -> 2W -> W, GELU) and projection (W -> 2W -> 4).
     Off: a linear lifting (6 -> W) and FNO2d's projection (W -> 128 -> 4).

The FFT normalisation is not a factor: NeuralOperator's "forward" and FNO2d's "backward" scale the
transform pair by the same 1/N in total, so the outputs are the same to rounding
(tests/test_fno_ablation.py). The kept Fourier modes are the same set in both.

The analysis is fixed before the runs: `python scripts/probes/fno_ablation.py summary` reads
results/fno_lib/scores.json and reports, per configuration, the mean over seeds of the validation
transfer (primary), the best validation loss, and the near field against the floor at 256x97 on
in-range cases; then each factor's effect added alone to the all-off model and removed alone from
the all-on one, with a two-sided permutation test over the seeds.
"""

import argparse
import itertools
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from torch import nn

from apno.models.fno import radial_pad

FACTORS = "MPIL"
IN_CH, OUT_CH = 6, 4
FNO2D_PROJ_HIDDEN = 128
NEURALOP_PAD_FRACTION = 4 / 49  # per side; the probe's NeuralopFno uses the same


def canonical(factors):
    """The factor letters in the order M, P, I, L; '' and 'none' mean none. Rejects others."""
    s = "" if factors in (None, "none") else factors.upper()
    bad = set(s) - set(FACTORS)
    if bad or len(set(s)) != len(s):
        raise ValueError(f"factors {factors!r}: each of {', '.join(FACTORS)} at most once")
    return "".join(f for f in FACTORS if f in s)


def variant_name(factors):
    return f"abl-{canonical(factors) or 'none'}"


class _OneSidedRadialPadding(nn.Module):
    """FNO2d's padding: radial_pad(NJ) zero rows after the far-field row, none at the wall."""

    def pad(self, x):
        self._nj = x.shape[-1]
        return F.pad(x, (0, radial_pad(self._nj)))

    def unpad(self, x):
        return x[..., : self._nj]


class AblationFno(nn.Module):
    def __init__(self, cfg, factors=""):
        super().__init__()
        from neuralop.layers.channel_mlp import ChannelMLP
        from neuralop.layers.padding import DomainPadding
        from neuralop.layers.skip_connections import Flattened1dConv, SoftGating
        from neuralop.layers.spectral_convolution import SpectralConv

        self.factors = canonical(factors)
        on = set(self.factors)
        w, n_layers = cfg.width, cfg.layers
        n_modes = (2 * cfg.modes_i, 2 * cfg.modes_j - 2)  # the same modes as FNO2d keeps
        self.lift = (ChannelMLP(IN_CH, w, hidden_channels=2 * w, n_layers=2) if "L" in on
                     else ChannelMLP(IN_CH, w, n_layers=1))
        self.padding = (DomainPadding([0.0, NEURALOP_PAD_FRACTION]) if "P" in on
                        else _OneSidedRadialPadding())
        if "I" in on:
            self.convs = nn.ModuleList(SpectralConv(w, w, n_modes) for _ in range(n_layers))
        else:
            self.convs = nn.ModuleList(SpectralConv(w, w, n_modes, bias=False,
                                                    init_std=1.0 / (w * w))
                                       for _ in range(n_layers))
        self.skips = nn.ModuleList(Flattened1dConv(w, w, 1, bias="I" not in on)
                                   for _ in range(n_layers))
        if "M" in on:
            self.mlps = nn.ModuleList(ChannelMLP(w, hidden_channels=round(0.5 * w), n_layers=2)
                                      for _ in range(n_layers))
            self.gates = nn.ModuleList(SoftGating(w, w) for _ in range(n_layers))
        self.proj = ChannelMLP(w, OUT_CH, hidden_channels=2 * w if "L" in on else FNO2D_PROJ_HIDDEN,
                               n_layers=2)

    def forward(self, x):
        """NeuralOperator's FNOBlocks.forward_with_postactivation when M is on; FNO2d's layer
        (spectral + skip, then GELU except after the last layer) when it is off."""
        h = self.padding.pad(self.lift(x))
        last = len(self.convs) - 1
        for k, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            y = conv(h) + skip(h)
            if k < last:
                y = F.gelu(y)
            if "M" in self.factors:
                y = self.mlps[k](y) + self.gates[k](h)
                if k < last:
                    y = F.gelu(y)
            h = y
        return self.proj(self.padding.unpad(h))


# ---------------------------------------------------------------- the planned analysis

_TAG = re.compile(r"^abl-(none|[MPIL]+)-n(\d+)-s(\d+)$")


def permutation_p(a, b):
    """Two-sided permutation test of the difference in means, exhaustive over the splits."""
    a, b = list(map(float, a)), list(map(float, b))
    x, k = np.array(a + b), len(a)
    if comb(len(x), k) > 200_000:
        raise ValueError("too many splits for an exhaustive test")
    obs = abs(np.mean(a) - np.mean(b))
    hits = total = 0
    for idx in itertools.combinations(range(len(x)), k):
        m = np.zeros(len(x), bool)
        m[list(idx)] = True
        hits += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12
        total += 1
    return hits / total


def collect(scores, n_train=896):
    """{code: {seed: {"transfer", "best_val", "sub", "sup"}}} for the ablation runs in scores,
    sub and sup being the near field over the floor at 256x97 on in-range cases."""
    xf = {(r["variant"], r["seed"], r["grid"], r["split"], r["regime"]): r["x_floor"]
          for r in scores["floor_ratios"] if r["n_train"] == n_train}
    out = {}
    for run in scores["runs"]:
        m = _TAG.match(run["tag"])
        if not m or int(m.group(2)) != n_train:
            continue
        code, seed = m.group(1), int(m.group(3))
        v = f"abl-{code}"
        out.setdefault(code, {})[seed] = {
            "transfer": run.get("transfer_val_rel_l2"), "best_val": run.get("best_val"),
            "sub": xf.get((v, seed, "256x97", "test_in", "subcritical")),
            "sup": xf.get((v, seed, "256x97", "test_in", "supercritical"))}
    return out


METRICS = (("transfer", "validation transfer"), ("best_val", "best validation loss"),
           ("sub", "x floor, 256x97, in range, subcritical"),
           ("sup", "x floor, 256x97, in range, supercritical"))


def _vals(cfg, metric):
    return [r[metric] for r in cfg.values() if r[metric] is not None]


def summary(scores):
    """The configurations' means, then the add-one and remove-one effects of each factor."""
    data = collect(scores)
    lines = ["# FNO transfer ablation", "",
             "Factors: M channel MLP, P symmetric padding, I spectral initialisation and bias "
             "placement, L two-layer lifting and projection (scripts/probes/fno_ablation.py). "
             "abl-none is FNO2d, abl-MPIL NeuralOperator's FNO.", "",
             "## Configurations (mean over seeds)", "",
             "| configuration | seeds | " + " | ".join(name for _, name in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    order = sorted(data, key=lambda c: (c != "none", len(c), [FACTORS.index(f) for f in c
                                                              if f in FACTORS]))
    for code in order:
        cells = [f"{np.mean(v):.4g}" if (v := _vals(data[code], m)) else "" for m, _ in METRICS]
        lines.append(f"| abl-{code} | {', '.join(map(str, sorted(data[code])))} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## Effects", "",
              "Add-one: abl-X against abl-none. Remove-one: abl-MPIL without X against abl-MPIL. "
              "Effect is the difference in means (configuration with the factor minus without); "
              "p is a two-sided exhaustive permutation test over the seeds.", "",
              "| factor | design | with | without | n | " + " | ".join(
                  f"{name} effect | p" for _, name in METRICS) + " |",
              "|---|---|---|---|---|" + "---|---|" * len(METRICS)]
    for f in FACTORS:
        for design, with_, without in (("add-one", f, "none"),
                                       ("remove-one", "MPIL", canonical(FACTORS.replace(f, ""))
                                        or "none")):
            if with_ not in data or without not in data:
                continue
            cells = []
            for m, _ in METRICS:
                a, b = _vals(data[with_], m), _vals(data[without], m)
                if len(a) >= 2 and len(b) >= 2:
                    cells += [f"{np.mean(a) - np.mean(b):+.4g}", f"{permutation_p(a, b):.3f}"]
                else:
                    cells += ["", ""]
            lines.append(f"| {f} | {design} | abl-{with_} | abl-{without} | "
                         f"{len(data[with_])} + {len(data[without])} | " + " | ".join(cells) + " |")
    missing = [c for c in ["none", *FACTORS, "MPIL",
                           *(canonical(FACTORS.replace(f, "")) for f in FACTORS)]
               if c not in data]
    if missing:
        lines += ["", "Not yet run: " + ", ".join(f"abl-{c}" for c in missing) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="the FNO transfer ablation's planned analysis")
    ap.add_argument("command", choices=("summary",))
    ap.add_argument("--scores", default="results/fno_lib/scores.json")
    ap.add_argument("--out", default="results/fno_lib/ablation.md")
    args = ap.parse_args(argv)
    if not Path(args.scores).exists():
        print(f"no {args.scores}: run scripts/probes/score_fno_libraries.py first")
        return 2
    text = summary(json.loads(Path(args.scores).read_text()))
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
