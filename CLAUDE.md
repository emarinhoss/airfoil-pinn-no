# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A measured comparison of four ways to get 2D inviscid flow around NACA 4-digit airfoils: SU2 Euler
CFD, a Fourier neural operator (FNO) trained on SU2 data, a POD baseline, and a PINN trained per
case from the equations alone. All are scored on the same nodes against fine-grid SU2, with a cost
record.

- Design: `docs/superpowers/specs/2026-10-01-airfoil-pinn-vs-neural-operator-design.md`. It is
  amended through dated sections (§7.8, §7.9), not by silent edits.
- Plan, task by task: `docs/superpowers/plans/2026-10-01-airfoil-pinn-vs-neural-operator.md`.
- Every measured result: `docs/findings.md`.

Stages S0–S4 have all run (to 2026-10-06), and the work is merged into `main`. S5 ran its gate on
2026-10-08 and is merged into `main` too. No configuration passed, because every one is trapped in a uniform
field, so S5 stopped after step 1. The probes behind that are in `scripts/probes/probe_s5_*.py`
and the findings. The remote is `origin`,
`git@github.com:emarinhoss/airfoil-pinn-no.git`, a public repository (pushed by the author's
decision of 2026-10-07). `references/` holds copyrighted papers: it is git-ignored and must never be
committed.

## Commands

Run everything from the repo root, because the scripts use relative paths. The environment is conda
`airfoil-ml` (Python 3.11). A non-interactive shell needs
`export PATH=/Users/esousa/miniforge3/envs/airfoil-ml/bin:$PATH`.

```bash
conda env create -f environment.yml && pip install -e . --no-deps
python -m pytest -q                                        # ~45 s
python -m pytest tests/test_run.py::test_crash_records_failure
python -m pytest -m su2                                    # real SU2; needs SU2_BIN_DIR
ruff check .                                               # E, F, W at line length 100, pinned
```

- **SU2** is found only through `SU2_BIN_DIR` (SU2 8.5.0; on this machine
  `/Users/esousa/theLab/bwb-tanker-scratch/bin`). Tests marked `su2` skip when it is unset.
- **The driver tests** run `tests/fake_su2.py` instead of SU2. Its behaviour is set by
  `FAKE_SU2_MODE` (`ok`, `crash`, `nan`, `sigkill`, `segv`, ...).
- **The `spike_dir` tests** read a real SU2 run from `APNO_SPIKE_DIR` (default
  `~/theLab/bwb-tanker-study/cache/su2_spike`) and skip if it is absent.

### Stage scripts

There is one entry point per stage. Each is resumable: a rerun skips finished work.

| Stage | Command | Writes |
|---|---|---|
| S0 | `SU2_BIN_DIR=... python scripts/s0_check.py` | `results/s0/`; exit 1 = scorer check failed, 2 = a case did not converge |
| S1 | `SU2_BIN_DIR=... python scripts/s1_campaign.py train` (then `fine`) | `runs/su2/<level>/<case_id>/` |
| S1 | `python scripts/build_dataset.py` | `data/hdf5/{256x97,512x193}.h5`, `results/s1/` |
| S2 | `python scripts/s2_operator.py tune\|final\|pod` | `runs/fno/`, `runs/predictions/`, `results/s2/` |
| S3a | `python scripts/s3_warmup.py --device cpu --tag T` | `results/s3/warmup_T.json` |
| S3 | `python scripts/s3_pinn.py --shard K/N --preflight-file results/s3/preflight.json` | `runs/predictions/pinn-*.h5`, `results/s3/` |
| S4 | `python scripts/s4_score.py` | `results/s4/` |
| S5 | `SU2_BIN_DIR=... python scripts/s5_su2.py` | `runs/su2/<level>/s5-val-*/`, `results/s5/su2.json`; exit 1 = unfinished (re-run), 2 = none converged |
| S5 | `python scripts/s5_pinn.py gate`, then `steps [--shard K/N]` | `runs/predictions_s5/`, `results/s5/`; exit 2 = no validation case or truth, 3 = over budget, 4 = a scoring failure |
| S5 | `python scripts/s5_score.py` | `results/s5/` |

S0–S3 take hours to days. The S1 campaign ran over two days on 8 SU2 workers, the warm-up took
about 8 h, and each PINN run takes 5–6 h. Do not launch them unless asked. S4 only scores and can
be rerun cheaply. The shell launchers that were actually used are `runs/*.sh`. They are git-ignored,
so the findings quote the ones that matter.

## Architecture

The pipeline in `src/apno`: `data.sampling` → `su2` → `data.build` → `models` →
`data.predictions` → `score`.

- **Cases** (`data/sampling.py`) come from a Sobol design with fixed seeds over the input box
  (`BOX`: M, α, t, m; camber position fixed at 0.4). The splits are train, val, test_in, and
  test_ext (M 0.80–0.85). The case id carries the Sobol index, which is how the prefix training
  sets (`PREFIXES`, 64…896) are selected.
- **Grids** (`geometry/`): the NACA section is one continuous curve in ξ, and the O-grid map is
  (ξ, η) → (x, y) with a 100-chord outer radius. There are three levels: `FINE` 512×193, `TRAIN`
  256×97 and `COARSE` 128×49. Each coarser level is exactly every second node of the next, which is
  how truth is sampled onto the evaluation grid.
- **SU2 driver** (`su2/`):
  - `run.py` runs the campaign: one directory per case, run in a process pool. An atomically
    written `result.json` is what marks a case done.
  - A killed run is never treated as a result.
  - Resume refuses settings that differ from the recorded ones.
  - Convergence is judged by our own rule (`convergence.judge`), not by SU2's flag.
  - `config.py` writes only keywords from `VERIFIED_KEYWORDS`.
- **Frozen settings** (`settings.py`, `FROZEN = True`): S0 froze these values. Changing one needs
  a dated findings entry, a rerun of every affected case, and an update to `tests/test_settings.py`
  in the same commit.
- **Dataset** (`data/build.py`): one HDF5 file per level, with one group per split.
  - Only converged cases are written. Failures go to `results/s1/failures_<level>.json` with their
    reasons.
  - Every restart's nodes are checked against a freshly generated grid.
- **FNO** (`models/fno.py`, `train_fno.py`): trains at 128×49 (256×97 sampled every second node)
  and is evaluated at 256×97.
  - Final runs also store `q_128x49`, so S4 scores the resolution transfer separately.
  - Radial padding is `radial_pad(NJ)` = round(3(NJ − 1)/16) − 1 rows, 8 at 128×49 and 17 at
    256×97. It keeps the radial DFT period at 19/16 in η at every resolution, so a learned
    radial mode is the same physical mode at every level. The earlier (NJ − 1)/6 rule matched
    the padded span instead and was replaced on 2026-10-01 (spec §6.3, amended by the final
    review). The 2026-10-25 ablation found that, on the trained networks, 16 rows at
    256x97 transfer much better than this rule's 17 (`docs/findings.md`); the rule is unchanged.
    `scripts/probes/probe_fno_pad_sweep.py` evaluates existing checkpoints at 256x97 over a range
    of row counts, without training, into `results/fno_lib/pad_sweep.*`. `scripts/probes/probe_fno_pad_predict.py`
    re-predicts S2's test cases with another row count into `runs/predictions_pad<rows>/`.
  - `scripts/probes/probe_fno_libraries.py` trains the FNO of `neuraloperator` or
    `nvidia-physicsnemo` through the same pipeline, for comparison with `FNO2d`
    (`docs/notes/2026-10-21-fno-libraries.md`). Its outputs go to `runs/fno_lib/` and
    `runs/predictions_lib/`, never `runs/predictions/`. Running it on a Colab GPU:
    `docs/notes/2026-10-22-colab-fno.md`. `scripts/probes/score_fno_libraries.py` scores those
    files with S4's code (`load_truth`, `load_contestants`, `score_rows` in `s4_score.py`), beside
    S2's FNO, into `results/fno_lib/`, never `results/s4/`. `scripts/probes/fno_ablation.py` switches NeuralOperator's
    four differences from `FNO2d` on one at a time (`--library ablate --factors MPIL`), for the
    transfer ablation (`docs/notes/2026-10-24-fno-transfer-ablation.md`).
- **POD** (`models/pod.py`): a linear basis whose mode weights are interpolated over the inputs by
  a thin-plate-spline RBF.
- **PINN**: the PINN has two formulations, selected by `PinnConfig.formulation`.
  - `"potential"` is the default since 2026-10-04 (`models/pinn_potential.py`). It is full
    potential: a learned circulation Γ with a quarter-chord vortex term, an equal-speed Kutta
    condition, and the residual excluded within 0.02 chord of the trailing edge.
  - `"euler"` (`models/pinn.py` + `physics/euler.py`) is the original. It is kept runnable as the
    documented negative result: it converges to fluid at rest. A non-default formulation adds a
    suffix to every file it writes, so the two never resume from or overwrite each other.
  - `train_pinn.train` runs Adam in float32 on the Apple GPU (`device="auto"`), then L-BFGS and
    grid evaluation in float64 on the CPU, because MPS has no float64.
  - `s3_pinn.py` times every planned run first and refuses with exit 3 if the projection exceeds
    `--budget-hours`.
- **S5 PINN** (`models/pinn_euler_av.py`, `models/train_pinn_av.py`, `physics/viscosity.py`): an
  Euler PINN built to Wassing, Langer & Bekemeyer's recipe, for stage S5. It has primitive
  outputs, Fourier features of the 512x193 grid indices, collocation at its nodes, mini-batch
  L-BFGS, and artificial viscosity (closed-form |A|) localised by a shock sensor times a
  stagnation sensor. Its residual is div F − D, the dissipative sign. The design is
  `docs/superpowers/specs/2026-10-07-s5-euler-pinn-av-design.md`; `score/s5.py` holds the cases,
  truth loading and run scores shared by the S5 scripts.
  - S5 prediction files go to `runs/predictions_s5/`, never `runs/predictions/`, which S4's
    scorer reads in full.
  - A run in `results/s5/gate.json` or any `runs*.json` is finished and never retrained.
- **Predictions** (`data/predictions.py`): one HDF5 file per contestant, `runs/predictions/<tag>.h5`.
  S4 identifies contestants and costs by exact tag (`fno-n896-s0`, `pod-n896`,
  `pinn-<variant>-<case>`). It refuses two files with the same (method, variant, n_train, seed).
- **Scoring** (`score/`):
  - `forces.py` is the only force integrator. Its vertex-lumped normals reproduce SU2's CL, CD and
    CMz to 5e-11; do not add another.
  - `metrics.score_case` scores one case, and `report.py` does the aggregation, the tables, the
    break-even curves and the plots.
  - Truth is SU2 at 512×193 sampled every second node. "The floor" is SU2's own 256×97 solution
    scored against that truth.

## Invariants that are easy to break

- **Node ordering**: node id `j * NI + i`, and arrays are indexed `[i, j]`. `i = 0` is the trailing
  edge, the lower surface comes first, and `j = 0` is the wall.
- **Non-dimensional states**: every array the models and the scorer see is non-dimensional
  (ρ/ρ∞, u/U∞, v/U∞, p/p∞; free stream in `constants.py`). The PINNs compute p in ρ∞U∞² units
  internally, and `to_dataset_q` converts it.
- **Precision**: truth (`512x193.h5`) is float64 and only the scorer reads it. `256x97.h5` is
  float32.
- **Leakage guards** (tested): `train_fno.load_split` raises on test splits, and test inputs are
  read only through `load_features_only`. `PinnCase` carries no flow-field data.
- **Committed records**: `scripts/s3_warmup.py` refuses to overwrite an existing record. Use `--tag`.
- **Ignored data**: `data/` (~1 GB) and `runs/` (~7 GB) are git-ignored but must be regenerable
  from the scripts. S4 reads `data/hdf5/`, `runs/predictions/` and `runs/su2/`. `results/` and
  `docs/` are committed.

## Findings and docs conventions

`docs/findings.md` is the evidence record, and the following rules hold throughout it:

- Entries are dated (`## YYYY-MM-DD — topic`) and appended.
- No figure appears without the command that produced it, written inline.
- The predictions in the 2026-10-01 entry are never edited. Their outcomes are written below them.
- A correction is made in place and marked, e.g. `*(Corrected 2026-10-06, second final review: this
  said "...")*`, and listed in a later corrections entry.
- Results are reported as measured, failures included, at the precision the record supports.

Commit after each task with the tests passing. Commit messages lead with the stage or area,
e.g. `S4: ...`, `PINN: ...`, `Findings: ...`.
