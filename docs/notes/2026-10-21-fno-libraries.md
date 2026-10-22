# 2026-10-21 — The FNO of two open-source libraries, read and run against this study's own

This is a note, not a findings entry: nothing here is measured on SU2 data. It records what
NeuralOperator 2.0.0 (`neuraloperator`, NVIDIA and Caltech) and NVIDIA PhysicsNeMo 2.2.2
(`nvidia-physicsnemo`) do when asked for a 2D Fourier neural operator, how each maps onto the
67-line `FNO2d` of `src/apno/models/fno.py`, what it took to put each one into this study's
pipeline, and how to run the real comparison. The probe is
`scripts/probes/probe_fno_libraries.py`; its tests are `tests/test_probe_fno_libraries.py`.
The real comparison needs `data/hdf5/256x97.h5`, which is git-ignored, so it has not run yet; the
numbers at the end come from a stand-in dataset and say nothing about airfoils.

## What an FNO is, in this study's terms

Every FNO here takes the six feature channels of `apno.models.targets.features` on the O-grid's
(i, j) index space and returns the four standardised targets at the same nodes:

1. **Lift**: a pointwise map from 6 channels to a width W of latent channels.
2. **L Fourier layers**, each `h ← σ(K(h) + W₁h)`: K truncates the 2D FFT of h to the lowest
   modes, multiplies each kept mode by a learned complex W×W matrix, and inverts the FFT; W₁ is a
   pointwise linear map (a 1×1 convolution), the skip path that carries what the kept modes do not.
3. **Project**: a pointwise map from W channels to the 4 outputs.

Because K acts on wavenumbers, not on nodes, the same weights evaluate at any resolution, which is
how a model trained at 128×49 is run at 256×97. Everything a library adds is a variation on the
three parts: how the lift and projection are built, what sits beside K in a layer, how the
non-periodic radial direction is padded before the FFT, and whether node coordinates are appended
to the input.

## The three, component by component

| Component | `FNO2d` (this study) | `neuralop.models.FNO` 2.0.0 | `physicsnemo.models.fno.FNO` 2.2.2 |
|---|---|---|---|
| Lift | 1×1 conv, 6 → W | 2-layer channel MLP, 6 → 2W → W, GELU | 2-layer 1×1 convs, 6 → W/2 → W, GELU |
| Kept modes | `modes_i` positive and negative along i, `modes_j` along j (real FFT) | `n_modes` is the total per dimension, centred by `fftshift`; the last dimension keeps `n_modes[-1] // 2 + 1` | `modes1` positive and negative, `modes2` along the last: the same rule as `FNO2d` |
| Spectral weights | one complex tensor per sign, N(0, 1/(W·W)) | one complex tensor over the centred block, N(0, √(2/(2W))); optional CP, Tucker or TT factorisation | real pairs (…, 2) viewed as complex, U(0, 1/(W·W)) (the original Li et al. initialisation) |
| Beside K | 1×1 conv | 1×1 conv skip, plus a 2-layer channel MLP (hidden W/2) with a soft-gating skip | 1×1 conv |
| Non-linearity | GELU, none after the last layer | GELU after K and after the channel MLP, none after the last layer | GELU, none after the last layer |
| Normalisation | none | none by default (AdaIN, group or instance norm available) | none |
| Radial padding | zero rows at the far end only, `radial_pad(NJ)` = round((NJ − 1)·3/16) − 1, so the DFT period is 19/16 in η at every NJ | a fraction of the resolution, padded on **both** ends of each dimension, cached per resolution | a fixed number of rows at the far end, set at construction, the same at every resolution |
| Position input | none: the features already carry the node position | a [0, 1] grid in each dimension appended to the input, on by default | a [0, 1] meshgrid appended to the input, on by default |
| Projection | 1×1 conv W → 128, GELU, 1×1 conv → 4 | 2-layer channel MLP, W → 2W → 4 | pointwise fully connected net, W → 32 → 4, SiLU, by default |
| FFT normalisation | `backward` (torch default) | `forward` | `backward` |
| Parameters at the S2 configuration (W 48, modes 24×8, 4 layers) | 3,555,476 (complex counted once; 7,094,420 real) | 3,568,468 | 7,095,452 real: `FNO2d` plus the two-layer lift |
| Lines of library source behind the model | 67 | 2,285 over 7 modules | 2,014 over 3 modules |

Two of the differences matter for this study and were neutralised in the probe; the rest were
left as the libraries ship them, because the point is to see the library's operator:

- **Position input.** Both libraries append a [0, 1] grid along i, which makes the operator
  depend on where the trailing edge sits in the index space. `FNO2d` is exactly equivariant to a
  roll around the airfoil (`tests/test_fno.py`), and the features already carry (x, y). The probe
  turns the embedding off in both (`positional_embedding=None`, `coord_features=False`), and the
  tests check that all three are roll-equivariant to 1e-5.
- **Radial padding.** `FNO2d` chooses its padding so a learned radial mode means the same physical
  mode at 128×49 and 256×97 (the module docstring explains the 19/16 period). PhysicsNeMo pads a
  fixed row count, so a model trained with 8 rows at 49 would carry 8 rows to 97 and change the
  period; the probe's wrapper resets the encoder's `pad` and `ipad` attributes to `radial_pad(NJ)`
  on every call, and a test pins it. NeuralOperator pads a fraction, so the period scales, but
  only symmetrically: the probe uses 4/49 on each end, which puts zero rows against the wall too.
  Its period is (NJ + 2·round(4NJ/49))/(NJ − 1): 57/48 at 49 and 113/96 at 97, the 0.88 % mismatch
  the `FNO2d` docstring describes for the earlier rule. Its transfer error is therefore expected
  to be a little worse than `FNO2d`'s, and the record measures it.
  *(Outcome 2026-10-22, on the S1 data: it was smaller, 0.0259 against 0.0346 to 0.0381
  for `FNO2d`; `docs/findings.md`, 2026-10-22.)*

Mode counts were matched exactly: `n_modes=(2·modes_i, 2·modes_j − 2)` in NeuralOperator keeps
the same set as `FNO2d`'s `modes_i` and `modes_j`.

## How hard it was

**Install.** Both come from PyPI and ran on a CPU-only Linux box with Python 3.13 and torch
2.14.1 without any fix. The cost is the footprint, not the steps:

| | `neuraloperator` 2.0.0 | `nvidia-physicsnemo` 2.2.2 |
|---|---|---|
| Direct dependencies | 10 (tensorly, tensorly-torch, opt-einsum, wandb, zencfg, configmypy, …) | 25 (warp-lang, timm, torchvision, onnx, hydra-core, s3fs, pandas, tensordict, …) |
| Package on disk | 2.4 MB | 25 MB |
| Import side effects | none | prints a Warp CUDA warning at every import on a machine without a GPU |

torch from PyPI brought 4.4 GB of CUDA libraries that this box cannot use; the conda
`airfoil-ml` environment on the Mac avoids that.

**Code.** The wrappers that make each library's model answer `FNO2d`'s contract are 12 and 14
lines. The friction was in four places, each found by running, not by reading the documentation:

1. NeuralOperator's `n_modes` counts both signs, and its last dimension is halved again: passing
   `FNO2d`'s numbers straight through keeps the wrong set.
2. NeuralOperator's `state_dict()` carries a `_metadata` dict of constructor arguments, which a
   plain `torch.save` / `torch.load(weights_only=True)` round trip rejects. The probe saves the
   tensors only; they reload into a model built from the same configuration.
3. NeuralOperator pads symmetrically with no one-sided option; PhysicsNeMo pads one-sidedly but
   by a fixed count. Neither matches `FNO2d`'s rule without a wrapper.
4. PhysicsNeMo's `FNO.__init__` types `padding` as an `int`, but the encoder accepts a per-axis
   list, which is what the probe needs to pad j and not i.

Everything else in the probe is this study's own pipeline reused unchanged: `features`,
`to_targets`, `Standardiser`, `prefix_indices`, `load_split`, `loss_fn`, `relative_l2` and the
training loop of `train_fno.train`, copied so that a model from any library can be dropped in.

## The stand-in run (not SU2 data)

`--smoke DIR` builds a dataset with the real Sobol inputs and the real 256×97 O-grids of the first
48 train, 8 val, 8 test_in and 4 test_ext cases, but with an analytic field (uniform stream, a
regularised doublet and a vortex about mid-chord, a Prandtl-Glauert factor, first-order isentropic
p and ρ) in place of the SU2 solution. It checks that each library trains, checkpoints, evaluates
at both resolutions and writes a prediction file. Run here on 4 CPU cores, 30 epochs, the S2
configuration (W 48, modes 24×8, 4 layers, loss A, batch 8, seed 0):

Command, per library L, with DIR a scratch directory:
`python scripts/probes/probe_fno_libraries.py --library L --smoke DIR --runs DIR/runs --epochs 30 --device cpu`;
the numbers are read from `DIR/runs/L-n48-s0-smoke/train.json`.

| library | best val loss (128×49) | epoch | val rel. L2 at 256×97 | transfer (256×97 vs 128×49) | wall | CPU | s per query at 256×97 |
|---|---|---|---|---|---|---|---|
| `apno` (`FNO2d`) | 0.2879 | 21 | 0.2714 | 0.0231 | 78 s | 302 s | 0.091 |
| `neuralop` | 0.2844 | 23 | 0.2674 | 0.0138 | 113 s | 400 s | 0.150 |
| `physicsnemo` | 0.2840 | 22 | 0.2682 | 0.0265 | 93 s | 355 s | 0.159 |

The stand-in field is smooth and nearly linear in its inputs, and 30 epochs on 48 cases is far
from convergence, so these losses are not a ranking of the operators; they show that all three
descend at the same rate through the same loop and that each one transfers across resolutions.
The transfer did not come out as the padding analysis above expects: NeuralOperator's symmetric
padding gave the smallest transfer error of the three on this field, not a larger one. The real
data decides that; the expectation stands as written so the outcome can be checked against it.
NeuralOperator's layer carries the extra channel MLP, which is where its 45 % longer epoch goes.
The runs live in a scratch directory, not in `runs/` or `results/`; nothing in this section is
a result.

## Running the real comparison

On the Mac, in the `airfoil-ml` environment, with the S1 dataset present:

```bash
pip install neuraloperator nvidia-physicsnemo          # once; ~30 packages
python scripts/probes/probe_fno_libraries.py --library apno        --n-train 896 --epochs 300
python scripts/probes/probe_fno_libraries.py --library neuralop    --n-train 896 --epochs 300
python scripts/probes/probe_fno_libraries.py --library physicsnemo --n-train 896 --epochs 300
```

Each writes `runs/fno_lib/<library>-n896-s0/train.json` and
`runs/predictions_lib/<library>-n896-s0.h5` (test_in and test_ext at 256×97, plus `q_128x49`), in
the S4 format but outside `runs/predictions/`, which S4 reads in full. The `apno` run is the
control: through the same loop it should reproduce `results/s2/runs.json` for `fno-n896-s0` to
the seed (best_val 0.01260 at epoch 278, transfer 0.0348, 2,160 s on MPS). If it does not, the
loop differs from S2's and the other two numbers are not comparable to S2 either.

What to compare, per library, from `train.json`: `best_val` (loss A on the validation cases at
128×49), `val_rel_l2_256x97`, `transfer_val_rel_l2`, `wall_s` and `n_params`. To score the test
predictions as S4 does, run `python scripts/probes/score_fno_libraries.py`, which writes
`results/fno_lib/`. *(Updated 2026-10-22: this said to point `load_contestants` in
`scripts/s4_score.py` at `runs/predictions_lib` in a separate run; the scorer now does it.)*

What the comparison would show: whether a library's FNO, with the same modes and width, reaches
the same validation loss and transfer as `FNO2d` on this data, and at what cost. What it would
not show: anything about the library's other models (GINO, UNO, the transformer operators), or
about the libraries' own training loops, which the probe does not use.
