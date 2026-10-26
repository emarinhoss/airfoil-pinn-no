# airfoil-pinn-no

A PINN against a neural operator for 2D inviscid flow around NACA 4-digit airfoils, measured against
SU2.

Four contestants run on the same cases. They are scored on the same nodes with one force
integrator, and the cost of each is recorded:

- **SU2**, Euler CFD: the reference, and the source of the training data.
- **FNO**: a Fourier neural operator on the O-grid, trained on up to 896 SU2 cases.
- **POD**: proper orthogonal decomposition, with each mode's weight interpolated over the inputs by
  a thin-plate-spline RBF.
- **PINN**: trained separately on each test case from the flow equations alone, as a full-potential
  formulation.

The input box covers Mach 0.30–0.80, angle of attack −2° to 6°, thickness 0.08–0.16 and camber
0–0.04, with the camber position fixed at 0.4. An extrapolation set covers Mach 0.80–0.85.

Truth is SU2 on a 512×193 grid. **The floor** is SU2's own 256×97 solution scored against that
truth: it is the best a 256×97 answer can be expected to do.

## Status

Stages S0 to S4 all ran between 2026-10-01 and 2026-10-06. Every result is in
[docs/findings.md](docs/findings.md), with the command that produced each figure.

These predictions were recorded before any model ran:

| Prediction | Outcome |
|---|---|
| 1. For a single new case, SU2 beats the PINN on cost by one to two orders of magnitude. | **Held on wall time only** (88.5×). **Not held** on CPU time or per phase (174–220×): SU2 wins by more than predicted. |
| 2. The FNO beats "just run SU2" only after about as many new cases as it took to train it (~1,000). | **Held.** The FNO breaks even at 1,126 new cases. |
| 3. Without the Kutta condition, the PINN's residuals look acceptable while its lift is wrong. | **Partly.** Without the condition the lift is 84 % low, but with it the lift is still 77 % low. |

Near-field error on the test sets, against the floor:

- **In range, subcritical**: the FNO is 1.3× the floor at its own 128×49 training grid and 2.1× at
  the 256×97 evaluation grid. There it is level with POD on the near field, and POD's lift and drag
  errors are smaller.
- **In range, supercritical**: the FNO is 1.6× the floor, and 3.7× better than POD.
- **Extrapolation**: the FNO is 3.4× the floor and POD 7.9×. The FNO's drag errors reach 204
  counts.
- **PINN**: 26–65× the floor. It under-predicts lift on five of its six showcase cases.

The FNO's radial padding at 256×97 is 16 rows, since the design was amended on 2026-10-26
(spec §6.3). With the original 17 rows, its three figures above were 5.3×, 2.4× and 3.6×. One
padding row also flips a drag bias of about 17 counts on the in-range cases, from high to low: a
leading-edge pressure error, with the best row count near 16.5 for every model. The record is in the
findings, 2026-10-24 to 2026-10-26.

The scope changed twice during the work. Both changes are dated amendments in the spec:

- The Euler PINN converged to fluid at rest, so it was replaced by the full-potential PINN (§7.8,
  2026-10-04). The Euler version is kept runnable as the recorded negative result.
- The artificially viscous PINN runs were dropped (§7.9, 2026-10-05).

Two literature checks were closed on 2026-10-07:

- **Vassberg & Jameson (2010)**: their grid-converged lift for the NACA 0012 anchor (Mach 0.8,
  1.25°) is 0.348–0.356. SU2 here gives 0.326 at 512×193, 6–8 % low. So the truth used for
  scoring carries its own discretisation error on transonic lifting cases. Every contestant is
  scored against the same truth, so the comparisons between them are unaffected.
- **Wassing, Langer & Bekemeyer (arXiv:2408.17364)**: they train an Euler PINN for transonic
  NACA 0012 flow to within about 1 % of a finite-volume reference, using localised artificial
  viscosity. So this study's failed Euler PINN reflects how it was set up here, not a limit of
  Euler PINNs in general. Which of the differences matters is still open.

## Setup

```bash
conda env create -f environment.yml
conda activate airfoil-ml
pip install -e . --no-deps
```

SU2 (8.5.0, the `SU2_CFD` binary) is found only through the `SU2_BIN_DIR` environment variable.
Tests that need it skip when it is unset.

## Tests

```bash
python -m pytest -q    # tests marked su2 need SU2_BIN_DIR
ruff check .
```

## Running the pipeline

Run every command from the repository root. Each stage is resumable: rerunning a command skips
finished work.

| Stage | What | Command |
|---|---|---|
| S0 | Check SU2's accuracy and cost; freeze the settings | `SU2_BIN_DIR=... python scripts/s0_check.py` |
| S1 | Data campaign: 1,104 cases at 256×97 and 80 at 512×193 | `SU2_BIN_DIR=... python scripts/s1_campaign.py train`, then `fine` |
| S1 | Build the HDF5 dataset and pick the PINN showcase cases | `python scripts/build_dataset.py` |
| S2 | FNO tuning, final runs and seeds; POD; FNO predictions under the current padding rule | `python scripts/s2_operator.py tune`, `final`, `pod`, `repredict` |
| S3a | PINN cylinder warm-up | `python scripts/s3_warmup.py` |
| S3 | PINN showcase runs (8 runs; shardable) | `python scripts/s3_pinn.py --shard K/N` |
| S4 | Score everything; write tables, plots and costs | `python scripts/s4_score.py` |
| S5 | SU2 reference for the validation case | `SU2_BIN_DIR=... python scripts/s5_su2.py` |
| S5 | Euler PINN: the gate, then the ablation and comparison | `python scripts/s5_pinn.py gate`, then `steps` |
| S5 | Score S5 against the FNO and the floor | `python scripts/s5_score.py` |

S0 to S3 take hours to days on a laptop. S4 only scores the existing predictions.

## Layout

```
src/apno/
  geometry/   NACA section as a continuous curve; O-grid map, nested grids, .su2 writer
  su2/        config, resumable campaign driver, binary restart reader, convergence rule
  physics/    flow quantities; Euler and full-potential residuals in grid coordinates
  data/       Sobol design and splits; raw cases -> HDF5; prediction files
  models/     POD, FNO and its training, the two PINN formulations and their training
  score/      the one force integrator, per-case metrics, aggregation and plots
scripts/      one entry point per stage
tests/        pytest suite, with a fake SU2 for the driver tests
docs/         findings.md, the design spec and the implementation plan
results/      committed outputs of every stage (s0 to s4)
data/, runs/  git-ignored: HDF5 datasets, SU2 cases, checkpoints, predictions (regenerable)
```
