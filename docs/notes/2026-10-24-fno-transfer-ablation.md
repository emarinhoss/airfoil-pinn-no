# 2026-10-24 — The FNO transfer ablation: design, runs and planned analysis

Over five seeds each, NeuralOperator's FNO loses less than this study's `FNO2d` between the 128x49
training grid and the 256x97 evaluation grid (validation transfer 0.0255 against 0.0374, no
overlap; `docs/findings.md`, 2026-10-24). This ablation asks which of its differences does that.
It is written down before any of its runs.

## The four differences

`scripts/probes/fno_ablation.py` holds `AblationFno`, one model with a switch for each difference.
"On" is NeuralOperator's choice, "off" is `FNO2d`'s:

| factor | on (NeuralOperator) | off (`FNO2d`) |
|---|---|---|
| M, channel MLP | after the spectral layer, a channel MLP (W → W/2 → W, GELU) plus a soft-gated skip from the block's input, then GELU | none |
| P, padding | zero rows on both radial ends, 4/49 of NJ each (4 + 4 at 128x49, 8 + 8 at 256x97) | `radial_pad(NJ)` rows at the far end only (8 and 17), keeping the radial DFT period at 19/16 in η |
| I, initialisation | spectral weights of deviation √(2/2W), 0.144 at W 48; bias on the spectral output | deviation 1/W², 4.3e-4; bias on the skip convolution |
| L, lifting and projection | 6 → 96 → 48 lifting, 48 → 96 → 4 projection | 6 → 48 lifting, 48 → 128 → 4 projection |

The FFT normalisation is not one: both scale the transform pair by 1/N in total
(`docs/findings.md`, corrections entry of 2026-10-24).

**What the tests establish** (`tests/test_fno_ablation.py`):

- With all four off, `AblationFno` gives `FNO2d`'s outputs from the same weights, at 128x49 and
  256x97, and has its parameter count (3,555,476). Its random initial values are drawn from the same
  distributions as `FNO2d`'s.
- With all four on, it gives the probe's NeuralOperator model's outputs from the same weights, at
  both grids, and has its parameter count (3,568,468).
- Every one of the 16 combinations keeps the operator's contract: both resolutions, and an exact
  roll equivariance around the airfoil.
- Each switch does what the table says: padding rows, initial deviations and bias placement, and
  module sizes.

## The runs

Every run uses the S2 configuration (width 48, modes 24 × 8, 4 layers, loss A, batch 8, AdamW 1e-3
with cosine decay over 300 epochs), 896 training cases, early stopping off, on the Colab A100 with
TF32 off. Five seeds, 0 to 4, per configuration.

| phase | configurations | runs | about |
|---|---|---|---|
| 1, controls | `abl-none` (= `FNO2d`), `abl-MPIL` (= NeuralOperator) | 10 | 70 min |
| 1, add one to `FNO2d` | `abl-M`, `abl-P`, `abl-I`, `abl-L` | 20 | 2.3 h |
| 2, remove one from NeuralOperator | `abl-PIL`, `abl-MIL`, `abl-MPL`, `abl-MPI` | 20 | 2.3 h |

The controls come first. They show that the one model reproduces both ends when trained, not only
when given copied weights. **Stop if they fail.** That means `abl-none`'s mean transfer outside
`FNO2d`'s five-seed range (0.0339 to 0.0451), or `abl-MPIL`'s outside NeuralOperator's (0.0222 to
0.0271).

Phase 2 is worth running if phase 1 is ambiguous. That happens if no single factor closes most
of the gap alone, or if two do. Add-one shows whether a factor is enough on its own; remove-one
shows whether NeuralOperator needs it. The two differ when factors interact.

### On Colab

As in `docs/notes/2026-10-22-colab-fno.md`, steps 2 to 5 (runtime, clone, links, installs, TF32
off), then pull this branch. The loop skips any run whose record already exists, so after a
disconnect the same cell picks up where it stopped. A record is written only when a run finishes.

```python
%cd /content/airfoil-pinn-no
!git pull
%env NVIDIA_TF32_OVERRIDE=0
import os
PHASE1 = ("none", "MPIL", "M", "P", "I", "L")
PHASE2 = ("PIL", "MIL", "MPL", "MPI")
for factors in PHASE1:
    for seed in range(5):
        tag = f"abl-{factors}-n896-s{seed}"
        if os.path.exists(f"runs/fno_lib/{tag}/train.json"):
            print("done", tag)
            continue
        !python scripts/probes/probe_fno_libraries.py --library ablate --factors {factors} --n-train 896 --seed {seed} --epochs 300 --patience 300 --device cuda 2>&1 | tee runs/{tag}.log
```

For phase 2, run the same cell with `PHASE2` in the outer loop.

### Back on the Mac

Copy the new `runs/fno_lib/abl-*/` folders and `runs/predictions_lib/abl-*.h5` files from Drive.
Then score everything and run the planned analysis:

```bash
git pull
python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2
python scripts/probes/fno_ablation.py summary
git add results/fno_lib && git commit -m "Probes: scores of the FNO transfer ablation" && git push
```

## The planned analysis

`python scripts/probes/fno_ablation.py summary` writes `results/fno_lib/ablation.md` from
`results/fno_lib/scores.json`:

- **Per configuration**, the mean over its seeds of four quantities. The validation transfer is
  primary: it is measured on validation cases, so choosing among configurations does not touch
  the test set. The other three are the best validation loss, and the near field against the
  floor at 256x97 on in-range subcritical and supercritical test cases.
- **Per factor**, the add-one effect (`abl-X` minus `abl-none`) and the remove-one effect
  (`abl-MPIL` minus `abl-MPIL` without X), each with a two-sided exhaustive permutation test over
  the seeds. At five seeds against five, p = 0.008 is the smallest possible.

The gap to explain is 0.0119 in mean transfer (0.0374 to 0.0255). A factor accounts for the
transfer if adding it alone closes most of that gap, and removing it alone from NeuralOperator
reopens most of it.

**Predictions.** The study records its predictions before the runs. If you want this ablation
tested against one, add a dated findings entry with it before phase 1 starts. Two observations
from earlier runs might inform one, neither of them evidence about the cause. PhysicsNeMo starts
its spectral weights at FNO2d's small scale and matches its padding, and it had the largest
transfer of the three (0.0468, one seed). NeuralOperator's symmetric padding was expected
(2026-10-21) to transfer worse, and it transferred better.
