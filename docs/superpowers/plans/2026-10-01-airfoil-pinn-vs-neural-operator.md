# Airfoil family: PINN vs neural operator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a 2D Euler airfoil-family dataset with SU2 on one nested O-grid, train a hand-written FNO (with a POD baseline) and a per-case PINN on it, and score all of them against fine-grid SU2 truth with one scorer and a cost record.

**Architecture:** One continuous O-grid map `(ξ, η) → (x, y)` defines the geometry for the mesh generator, the FNO's grid and the PINN's coordinates. A resumable driver runs SU2 into one directory per case, with a `result.json` written last. A deterministic builder turns those directories into HDF5 files, one group per split. Models write prediction files; one scorer reads truth, the floor and every prediction file, and writes tables and plots into `results/`.

**Tech Stack:** Python 3.11, NumPy, SciPy (`qmc.Sobol`, `RBFInterpolator`), h5py, PyTorch ≥ 2.5 (CPU; MPS if available), Matplotlib, SymPy (tests only), pytest, ruff. SU2 8.5.0 binaries at `$SU2_BIN_DIR`.

**Spec:** `docs/superpowers/specs/2026-10-01-airfoil-pinn-vs-neural-operator-design.md` (commit `102bad6`). Read it beside this plan; section numbers below (§n) refer to it.

## Global Constraints

- Repository `~/theLab/airfoil-pinn-no`, local only, never pushed. *(2026-10-07: superseded by the
  author's decision to push to GitHub; spec §9.1.)* Package `apno` under `src/`.
- Conda environment `airfoil-ml`, Python 3.11; install with `pip install -e . --no-deps`.
- SU2 found **only** through `SU2_BIN_DIR`; tests that need it carry `@pytest.mark.su2` and skip when it is unset.
- Input box, verbatim from §4.1: M ∈ [0.30, 0.80]; α ∈ [−2°, 6°]; t ∈ [0.08, 0.16]; m ∈ [0, 0.04]; p = 0.4 fixed. Extrapolation set: M ∈ [0.80, 0.85].
- Grids: fine 512 × 193, training 256 × 97, coarse 128 × 49; outer radius 100 chords; training-grid first wall spacing 2.0 × 10⁻³ chords.
- Free stream for SU2: p∞ = 101,325 Pa, T∞ = 288.15 K, γ = 1.4, R = 287.058 J/(kg·K). Every array the models see is non-dimensional: ρ/ρ∞, u/U∞, v/U∞, p/p∞.
- Node ordering everywhere: node id `j * NI + i`, arrays indexed `[i, j]`; `i = 0` is the trailing edge, the lower surface comes first, and `j = 0` is the wall.
- Truth is float64 and is read only by the scorer. The training-grid HDF5 file is float32.
- `results/` and `docs/` are committed. `data/` and `runs/` are git-ignored and must be regenerable from scripts.
- No figure goes into `docs/findings.md` without the command that produced it.
- Lint: `ruff check .` with line length 100 and rules `E`, `F`, `W`, pinned in `pyproject.toml`. Commit after every task, with tests passing.

## Spec amendments made by this plan (Task 1, Step 6 applies them to the spec text)

1. **§4.2 outer-boundary construction.** The far point is the circle point at polar angle **−2πξ**, not "the same angle about mid-chord". Measured on 2026-10-01 at 256 × 97: the spike's construction folds the mesh (9,792 inverted cells at t = 0.08, m = 0.04) and puts near-wall grid lines at a median 16–21° to the wall. The uniform-angle construction has no inverted cell at any (t, m) corner of the box, with a median 53–60°; only the last ~1 % of chord at the trailing edge has lines below 10°.
2. **§8.1 force integration.** The integration is **vertex-lumped**: each wall node carries half of each adjacent edge's outward normal. Measured on the spike's own solution, it reproduces SU2's CL, CD and CMz to 5 × 10⁻¹¹. The edge-midpoint trapezoid matches CL and CD but misses CMz by 1.6 × 10⁻⁵, which fails the 10⁻⁶ check.
3. **§6.3 radial padding.** The padding is (NJ − 1)/6 rows: 8 at 128 × 49, 16 at 256 × 97. The padded domain then covers the same span of η at every resolution, which is what lets one trained model run at both resolutions.
4. **§5.2 HDF5 layout.** Each file holds one group per split (`train`, `val`, `test_in`, `test_ext`) instead of one `split` label dataset, and stores only converged cases. Failures are listed in `results/s1/failures_<level>.json`. A loader can then refuse a test group by name.

## Review Focus

These are failure modes the spec implies but no feature test exercises. Each is pinned by the named test.

1. **An SU2 run that crashes or diverges (exit ≠ 0, NaN in the history).** The case is recorded as failed with the reason and never treated as data. Test: `tests/test_run.py::test_crash_records_failure` and `::test_nan_history_records_failure` (Task 9).
2. **A case killed mid-run** (stale `restart.dat` and no `result.json`, or a leftover `result.json.tmp`). The case reruns from scratch and the stale files cannot be read as its result. Test: `tests/test_run.py::test_stale_files_rerun` and `::test_tmp_result_is_not_done` (Task 9).
3. **A history file that is missing, empty or lacks a column.** The verdict is failed with a stated reason, never an exception that aborts the campaign. Test: `tests/test_convergence.py::test_missing_and_empty_history` (Task 5).
4. **Test cases that fail convergence, leaving fewer than 64.** The showcase rule runs on the survivors, and the failures are listed. Test: `tests/test_build.py::test_showcase_on_survivors` (Task 13).
5. **A prediction file missing a test case, or holding NaN** (for example a PINN that diverged). The scorer reports `missing` or `non-finite` for that case and still scores every other one. Test: `tests/test_metrics.py::test_missing_and_nonfinite_predictions` (Task 21).

---

## File structure

```
pyproject.toml, environment.yml, .gitignore
src/apno/
  __init__.py
  constants.py             γ, R, p∞, T∞
  settings.py              the campaign settings, frozen by S0 (Task 11)
  geometry/naca.py         section(xi, t, m, p, xp) — the continuous curve
  geometry/ogrid.py        Section, Cylinder, map_xy, grid, cell_areas, write_su2
  su2/restart.py           read_restart, write_restart, on_grid
  su2/convergence.py       read_history, judge, Verdict
  su2/config.py            Su2Settings, render_config, VERIFIED_KEYWORDS
  su2/run.py               CaseSpec, run_case, run_campaign
  physics/flow.py          nondimensionalise, Cp, Mach, entropy, enthalpy, normal_shock
  physics/euler.py         torch residual in (ξ, η), optional artificial viscosity
  data/sampling.py         Sobol design, splits, case specs
  data/build.py            raw → HDF5, regime, showcase rule
  data/predictions.py      write/read prediction files
  models/targets.py        output scaling + Standardiser; FNO input features
  models/pod.py            PodModel
  models/fno.py            SpectralConv2d, FNO2d
  models/train_fno.py      FnoConfig, data loading (refuses test), train, predict
  models/pinn.py           PinnCase, PinnNet, Pinn
  models/train_pinn.py     PinnConfig, losses, train, evaluate_on_grid
  score/forces.py          wall_forces, vertex_normals
  score/metrics.py         score_case and its parts
  score/report.py          aggregation, tables, plots
scripts/  s0_check.py, s1_campaign.py, build_dataset.py, s2_operator.py,
          s3_warmup.py, s3_pinn.py, s4_score.py
tests/    conftest.py, fake_su2.py, test_*.py
docs/     findings.md (+ the spec and this plan)
results/  s0/ s1/ s2/ s3/ s4/   (committed)
```

---

### Task 1: Repository scaffold, environment, findings log, spec amendments

**Files:**
- Create: `pyproject.toml`, `environment.yml`, `.gitignore`, `src/apno/__init__.py`, `src/apno/constants.py`, `tests/conftest.py`, `tests/test_scaffold.py`, `docs/findings.md`
- Modify: `docs/superpowers/specs/2026-10-01-airfoil-pinn-vs-neural-operator-design.md`

**Interfaces:**
- Produces: `apno.constants.GAMMA = 1.4`, `R_GAS = 287.058`, `P_INF = 101325.0`, `T_INF = 288.15`; pytest fixtures `su2_bin` (a `Path`, or a skip) and `spike_dir` (a `Path`, or a skip); markers `su2` and `slow`.

- [ ] **Step 1: Write the project files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "apno"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W"]  # pinned: a wider default elsewhere on this machine fires 25 style rules

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "su2: needs real SU2 binaries via SU2_BIN_DIR",
  "slow: takes more than a minute",
]
```

`environment.yml`:
```yaml
name: airfoil-ml
channels: [conda-forge]
dependencies:
  - python=3.11
  - pytorch>=2.5
  - numpy
  - scipy
  - h5py
  - matplotlib
  - sympy
  - pytest
  - ruff
  - pip
```

`.gitignore`:
```
data/
runs/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
```

`src/apno/__init__.py`:
```python
"""Airfoil family: a PINN and a neural operator, head-to-head."""
```

`src/apno/constants.py`:
```python
"""Physical constants shared by SU2's configuration and every non-dimensionalisation."""

GAMMA = 1.4
R_GAS = 287.058  # J/(kg K), SU2's default GAS_CONSTANT
P_INF = 101325.0  # Pa
T_INF = 288.15  # K
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
import os
from pathlib import Path

import pytest

SPIKE_DIR = Path(
    os.environ.get(
        "APNO_SPIKE_DIR", str(Path.home() / "theLab/bwb-tanker-study/cache/su2_spike")
    )
)


@pytest.fixture
def su2_bin() -> Path:
    value = os.environ.get("SU2_BIN_DIR")
    if not value:
        pytest.skip("SU2_BIN_DIR not set")
    return Path(value)


@pytest.fixture
def spike_dir() -> Path:
    """The 2026-10-01 NACA 0012 spike: a real SU2 restart, history and surface file."""
    if not (SPIKE_DIR / "restart.dat").exists():
        pytest.skip(f"spike files not present at {SPIKE_DIR}")
    return SPIKE_DIR
```

- [ ] **Step 3: Write the failing test `tests/test_scaffold.py`**

```python
from apno import constants


def test_constants():
    assert constants.GAMMA == 1.4
    assert constants.R_GAS == 287.058
    assert constants.P_INF == 101325.0
    assert constants.T_INF == 288.15
```

- [ ] **Step 4: Create the environment and run the test**

Run:
```bash
cd ~/theLab/airfoil-pinn-no
conda env create -f environment.yml
conda activate airfoil-ml
pip install -e . --no-deps
python -m pytest tests/test_scaffold.py -v
```
Expected: 1 passed. Before `pip install` the test fails with `ModuleNotFoundError: No module named 'apno'`.

Then check the Apple GPU, which the spec leaves open (§10). Run this in your own terminal, not from inside an agent sandbox:
```bash
python -c "import torch; print('mps', torch.backends.mps.is_available())"
```
Record the answer in Step 5's findings entry.

- [ ] **Step 5: Write `docs/findings.md`, with the predictions recorded before any model runs**

```markdown
# Findings

Dated entries. Every figure carries the command that produced it. Predictions are recorded before
the run that tests them and are never edited afterwards; outcomes are appended below them.

## 2026-10-01 — predictions (spec §8.5), recorded before any model runs

1. For a single new case, SU2 beats the PINN on cost by one to two orders of magnitude.
2. The FNO beats "just run SU2" only after roughly as many new cases as it took to train it
   (about a thousand), unless the data-efficiency curve shows far fewer training cases suffice.
3. Without the Kutta condition, the PINN's residuals look acceptable while its lift is wrong.

## 2026-10-01 — environment

- Apple GPU (MPS) available in the author's terminal: <write True or False from Task 1 Step 4>.
- Measured while planning (from the spike in ~/theLab/bwb-tanker-study/cache/su2_spike):
  the far-point-at-same-angle O-grid folds (9,792 inverted cells at t = 0.08, m = 0.04);
  vertex-lumped force integration reproduces SU2's CL/CD/CMz to 5e-11.
```

When you write the entry, replace the angle-bracket text with the printed value. It is the only value the plan leaves for you to fill in, because only your own terminal can measure it.

- [ ] **Step 6: Apply the four spec amendments**

Edit the spec, replacing each passage exactly.

(a) In §4.2, replace
```
x_f the point on a circle
  of radius 100 chords about mid-chord at the same angle about mid-chord (the spike's construction),
  and s(η) a geometric stretching.
```
with
```
x_f the point on a circle
  of radius 100 chords about mid-chord at polar angle −2πξ, and s(η) a geometric stretching.
  (Amended 2026-10-01 by the implementation plan: the spike's "same angle about mid-chord"
  construction folds the mesh — 9,792 inverted cells at t = 0.08, m = 0.04 — and the uniform angle
  has no inverted cell at any corner of the box.)
```

(b) In §8.1, replace `trapezoidal integration of (p − p∞) over the wall edges` with `vertex-lumped integration of (p − p∞) over the wall (each node carries half of each adjacent edge's outward normal; measured to reproduce SU2 on the spike to 5 × 10⁻¹¹, where the edge-midpoint trapezoid misses CMz by 1.6 × 10⁻⁵)`.

(c) In §6.3, replace `the field is
  zero-padded by 8 rows before the FFT and cropped after.` with `the field is
  zero-padded by (NJ − 1)/6 rows (8 at 128 × 49) before the FFT and cropped after, so the padded
  domain spans the same η at every resolution.`

(d) At the end of §5.2's table paragraph, add: `Amended by the plan: each file holds one group per split and only converged cases; failures are listed in results/s1/failures_<level>.json.`

- [ ] **Step 7: Lint and commit**

```bash
ruff check . && python -m pytest tests/test_scaffold.py -q
git add pyproject.toml environment.yml .gitignore src tests docs
git commit -m "Scaffold: package, environment, findings log with predictions, spec amendments"
```

---

### Task 2: The NACA 4-digit section as one continuous curve

**Files:**
- Create: `src/apno/geometry/__init__.py` (empty), `src/apno/geometry/naca.py`
- Test: `tests/test_naca.py`

**Interfaces:**
- Produces: `section(xi, t, m, p=0.4, xp=np) -> (x, y)`. Takes ξ ∈ [0, 1) as a NumPy array or torch tensor (pass `xp=torch` for tensors) and returns the wall point. ξ = 0 is the trailing edge, ξ ∈ [0, 0.5] runs along the lower surface to the leading edge, and the upper surface follows. Also `section_s(s, t, m, p=0.4, xp=np)`, the same curve as a function of s = cos(πξ). Its derivative is finite at the trailing edge, where dx/dξ = dy/dξ = 0, so the PINN's wall normal (Task 18) differentiates with respect to s.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from apno.geometry.naca import section


def test_trailing_edge_closed_and_leading_edge_at_origin():
    for t, m in ((0.12, 0.0), (0.08, 0.04)):
        x, y = section(np.array([0.0, 0.5]), t, m)
        assert x[0] == pytest.approx(1.0, abs=1e-14) and y[0] == pytest.approx(0.0, abs=1e-14)
        assert x[1] == pytest.approx(0.0, abs=1e-14) and y[1] == pytest.approx(0.0, abs=1e-14)


def test_symmetric_section_is_mirror_symmetric():
    xi = np.linspace(0.01, 0.49, 50)
    x1, y1 = section(xi, 0.12, 0.0)
    x2, y2 = section(1.0 - xi, 0.12, 0.0)
    np.testing.assert_allclose(x1, x2, atol=1e-14)
    np.testing.assert_allclose(y1, -y2, atol=1e-14)
    assert np.all(y1 < 0)  # lower surface first


def test_thickness_is_t():
    xi = np.linspace(0.5, 1.0, 20001)[:-1]
    _, y = section(xi, 0.12, 0.0)
    assert 2 * y.max() == pytest.approx(0.12, rel=5e-3)


def test_pair_midpoint_is_the_camber_line():
    # points at xi and 1 - xi share s**2, so they straddle the same camber-line point
    s = np.sqrt(0.4)
    xi = np.array([np.arccos(s) / np.pi])
    xl, yl = section(xi, 0.10, 0.04)
    xu, yu = section(1.0 - xi, 0.10, 0.04)
    assert 0.5 * (xl[0] + xu[0]) == pytest.approx(0.4, abs=1e-12)
    assert 0.5 * (yl[0] + yu[0]) == pytest.approx(0.04, abs=1e-12)


def test_numpy_and_torch_agree():
    xi = np.linspace(0.0, 1.0, 101)[:-1]
    xn, yn = section(xi, 0.14, 0.03)
    xt, yt = section(torch.tensor(xi, dtype=torch.float64), 0.14, 0.03, xp=torch)
    np.testing.assert_allclose(xn, xt.numpy(), atol=1e-14)
    np.testing.assert_allclose(yn, yt.numpy(), atol=1e-14)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_naca.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apno.geometry'`.

- [ ] **Step 3: Implement `src/apno/geometry/naca.py`**

```python
"""A NACA 4-digit section as one continuous closed curve of xi in [0, 1).

xi = 0 is the trailing edge. xi in [0, 0.5] runs along the lower surface to the leading edge and
xi in [0.5, 1) back along the upper surface (the spike's ordering, clockwise). With s = cos(pi xi),
the chord station is x = s**2 and sqrt(x) = |s|, so the curve has no square-root singularity at
the leading edge. The trailing edge is closed (x**4 coefficient -0.1036). Works on NumPy arrays and
torch tensors alike, so the mesh generator and the PINN share one definition of the geometry.
"""

import numpy as np

A = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)


def section(xi, t, m, p=0.4, xp=np):
    return section_s(xp.cos(np.pi * xi), t, m, p, xp)


def section_s(s, t, m, p=0.4, xp=np):
    """The same curve as a function of s = cos(pi xi) in [-1, 1]. d/dxi vanishes at the trailing
    edge (ds/dxi = -pi sin(pi xi) = 0 there); d/ds does not, so normals are taken in s."""
    x = s * s
    poly = A[1] + A[2] * x + A[3] * x * x + A[4] * x * x * x
    ys = -5.0 * t * (A[0] * s + s * xp.abs(s) * poly)  # signed half-thickness, negative = lower
    if m == 0:
        return x, ys
    fore = x < p
    yc = xp.where(
        fore,
        m / p**2 * (2 * p * x - x * x),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x),
    )
    dyc = xp.where(fore, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    k = 1.0 / xp.sqrt(1.0 + dyc * dyc)
    return x - ys * dyc * k, yc + ys * k
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_naca.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/geometry tests/test_naca.py
git commit -m "Geometry: NACA 4-digit section as one continuous curve (numpy and torch)"
```

---

### Task 3: The O-grid map, nested grids and the `.su2` writer

**Files:**
- Create: `src/apno/geometry/ogrid.py`
- Test: `tests/test_ogrid.py`

**Interfaces:**
- Consumes: `section` (Task 2).
- Produces:
  - `FINE = (512, 193)`, `TRAIN = (256, 97)`, `COARSE = (128, 49)`, `FAR_RADIUS = 100.0`, `WALL_SPACING = 2.0e-3`; `level_name(shape) -> "256x97"`.
  - `Section(t, m, p=0.4)`, with `.wall(xi, xp)` and `.wall_s(s, xp)`; `Cylinder(radius=0.5).wall(xi, xp)`.
  - `stretch_ratio(radius=FAR_RADIUS) -> float`; `stretch(eta, ratio, xp=np)`.
  - `map_xy(xi, eta, geom, radius=FAR_RADIUS, ratio=None, xp=np) -> (x, y)`.
  - `grid(geom, shape=TRAIN, radius=FAR_RADIUS) -> (X, Y)`, each of shape `(NI, NJ)`.
  - `cell_areas(X, Y) -> (NI, NJ-1)`; `write_su2(path, X, Y)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from apno.geometry.naca import section
from apno.geometry.ogrid import (
    COARSE, FAR_RADIUS, FINE, TRAIN, Cylinder, Section, cell_areas, grid, map_xy, stretch,
    stretch_ratio, write_su2,
)

CORNERS = [Section(t, m) for t in (0.08, 0.16) for m in (0.0, 0.04)]


@pytest.mark.parametrize("sec", CORNERS, ids=lambda s: f"t{s.t}-m{s.m}")
@pytest.mark.parametrize("shape", [FINE, TRAIN, COARSE])
def test_every_cell_positive_at_box_corners(sec, shape):
    X, Y = grid(sec, shape)
    assert cell_areas(X, Y).min() > 0


@pytest.mark.parametrize("radius", [50.0, 200.0])
def test_positive_at_other_radii(radius):
    for sec in CORNERS:
        X, Y = grid(sec, TRAIN, radius)
        assert cell_areas(X, Y).min() > 0


def test_grids_nest_exactly():
    sec = Section(0.12, 0.02)
    Xf, Yf = grid(sec, FINE)
    Xt, Yt = grid(sec, TRAIN)
    Xc, Yc = grid(sec, COARSE)
    assert np.array_equal(Xf[::2, ::2], Xt) and np.array_equal(Yf[::2, ::2], Yt)
    assert np.array_equal(Xf[::4, ::4], Xc) and np.array_equal(Yf[::4, ::4], Yc)


def test_wall_on_section_and_outer_circle():
    sec = Section(0.10, 0.03)
    X, Y = grid(sec, TRAIN)
    xw, yw = section(np.arange(256) / 256, 0.10, 0.03)
    np.testing.assert_array_equal(X[:, 0], xw)
    np.testing.assert_array_equal(Y[:, 0], yw)
    np.testing.assert_allclose(np.hypot(X[:, -1] - 0.5, Y[:, -1]), FAR_RADIUS, rtol=1e-13)


def test_training_grid_first_spacing():
    r = stretch_ratio()
    assert stretch(1 / 96, r) * FAR_RADIUS == pytest.approx(2.0e-3, rel=1e-9)


def test_symmetric_section_gives_mirror_grid():
    X, Y = grid(Section(0.12, 0.0), TRAIN)
    mirror = (-np.arange(256)) % 256
    np.testing.assert_allclose(X[mirror], X, atol=1e-12)
    np.testing.assert_allclose(Y[mirror], -Y, atol=1e-12)


def test_torch_map_equals_numpy_map():
    sec = Section(0.12, 0.04)
    xi = np.random.default_rng(0).random(200)
    eta = np.random.default_rng(1).random(200)
    xn, yn = map_xy(xi, eta, sec)
    xt, yt = map_xy(torch.tensor(xi), torch.tensor(eta), sec, xp=torch)
    np.testing.assert_allclose(xn, xt.numpy(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(yn, yt.numpy(), rtol=1e-13, atol=1e-13)


def test_cylinder_grid_is_polar():
    X, Y = grid(Cylinder(), COARSE)
    np.testing.assert_allclose(np.hypot(X[:, 0] - 0.5, Y[:, 0]), 0.5, rtol=1e-14)
    assert cell_areas(X, Y).min() > 0


def test_write_su2_layout(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    write_su2(tmp_path / "mesh.su2", X, Y)
    lines = (tmp_path / "mesh.su2").read_text().splitlines()
    assert lines[0] == "NDIME= 2" and lines[1] == f"NELEM= {128 * 48}"
    k = lines.index(f"NPOIN= {128 * 49}")
    i, j = 5, 3
    x, y, nid = lines[k + 1 + j * 128 + i].split()
    assert int(nid) == j * 128 + i
    assert float(x) == pytest.approx(X[i, j], abs=1e-14)
    assert float(y) == pytest.approx(Y[i, j], abs=1e-14)
    assert "MARKER_TAG= airfoil" in lines and "MARKER_TAG= farfield" in lines


def test_write_su2_refuses_folded_grid(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    X[:, 1], X[:, 2] = X[:, 2].copy(), X[:, 1].copy()
    Y[:, 1], Y[:, 2] = Y[:, 2].copy(), Y[:, 1].copy()
    with pytest.raises(ValueError, match="non-positive"):
        write_su2(tmp_path / "bad.su2", X, Y)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_ogrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apno.geometry.ogrid'`.

- [ ] **Step 3: Implement `src/apno/geometry/ogrid.py`**

```python
"""The O-grid: one continuous map (xi, eta) -> (x, y), sampled into three nested grids.

The wall (eta = 0) is the section; the outer boundary (eta = 1) is a circle of radius R about
mid-chord. The grid line of constant xi joins the wall point at xi to the circle point at polar
angle -2*pi*xi. This is NOT the spike's construction, which put the far point at the wall point's
angle about mid-chord: that folds the mesh (9,792 inverted cells for t = 0.08, m = 0.04 at 256x97),
measured on 2026-10-01. The stretching is defined on the fine grid, so the coarser grids are exact
subsets of it.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apno.geometry.naca import section, section_s

FINE = (512, 193)
TRAIN = (256, 97)
COARSE = (128, 49)
FAR_RADIUS = 100.0
WALL_SPACING = 2.0e-3  # first radial spacing of the 256x97 grid, chords
N_FINE = FINE[1] - 1


def level_name(shape):
    return f"{shape[0]}x{shape[1]}"


@dataclass(frozen=True)
class Section:
    t: float
    m: float
    p: float = 0.4

    def wall(self, xi, xp=np):
        return section(xi, self.t, self.m, self.p, xp)

    def wall_s(self, s, xp=np):
        return section_s(s, self.t, self.m, self.p, xp)


@dataclass(frozen=True)
class Cylinder:
    """A circle about mid-chord, ordered like the section (clockwise, xi = 0 at (0.5 + r, 0))."""

    radius: float = 0.5

    def wall(self, xi, xp=np):
        th = -2.0 * np.pi * xi
        return 0.5 + self.radius * xp.cos(th), self.radius * xp.sin(th)


def stretch_ratio(radius=FAR_RADIUS, wall_spacing=WALL_SPACING, n_fine=N_FINE):
    """Geometric ratio r of the fine grid, such that its first TWO intervals total wall_spacing."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        first_two = radius * (r * r - 1.0) / (r**n_fine - 1.0)
        lo, hi = (lo, r) if first_two < wall_spacing else (r, hi)
    return 0.5 * (lo + hi)


def stretch(eta, ratio, n_fine=N_FINE, xp=np):
    return (xp.exp(n_fine * eta * math.log(ratio)) - 1.0) / (ratio**n_fine - 1.0)


def map_xy(xi, eta, geom, radius=FAR_RADIUS, ratio=None, xp=np):
    ratio = stretch_ratio(radius) if ratio is None else ratio
    xw, yw = geom.wall(xi, xp)
    th = -2.0 * np.pi * xi
    xf, yf = 0.5 + radius * xp.cos(th), radius * xp.sin(th)
    s = stretch(eta, ratio, xp=xp)
    return xw + s * (xf - xw), yw + s * (yf - yw)


def grid(geom, shape=TRAIN, radius=FAR_RADIUS):
    ni, nj = shape
    xi = np.arange(ni) / ni
    eta = np.arange(nj) / (nj - 1)
    return map_xy(xi[:, None], eta[None, :], geom, radius)


def cell_areas(X, Y):
    """Signed area of quad (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1), periodic in i."""
    x0, y0 = X[:, :-1], Y[:, :-1]
    x1, y1 = np.roll(X, -1, 0)[:, :-1], np.roll(Y, -1, 0)[:, :-1]
    x2, y2 = np.roll(X, -1, 0)[:, 1:], np.roll(Y, -1, 0)[:, 1:]
    x3, y3 = X[:, 1:], Y[:, 1:]
    return 0.5 * (
        (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3)
    )


def write_su2(path, X, Y):
    """Write the grid in SU2 native format. Node id = j*NI + i; markers 'airfoil', 'farfield'."""
    if cell_areas(X, Y).min() <= 0:
        raise ValueError("grid has non-positive cells; refusing to write it")
    ni, nj = X.shape

    def nid(i, j):
        return j * ni + (i % ni)

    out = ["NDIME= 2", f"NELEM= {ni * (nj - 1)}"]
    e = 0
    for j in range(nj - 1):
        for i in range(ni):
            out.append(f"9 {nid(i, j)} {nid(i + 1, j)} {nid(i + 1, j + 1)} {nid(i, j + 1)} {e}")
            e += 1
    out.append(f"NPOIN= {ni * nj}")
    for j in range(nj):
        for i in range(ni):
            out.append(f"{X[i, j]:.17e} {Y[i, j]:.17e} {nid(i, j)}")
    out.append("NMARK= 2")
    for tag, j in (("airfoil", 0), ("farfield", nj - 1)):
        out += [f"MARKER_TAG= {tag}", f"MARKER_ELEMS= {ni}"]
        out += [f"3 {nid(i, j)} {nid(i + 1, j)}" for i in range(ni)]
    Path(path).write_text("\n".join(out) + "\n")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_ogrid.py -v`
Expected: all pass (15 items: the parametrized corners plus 9 single tests).

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/geometry/ogrid.py tests/test_ogrid.py
git commit -m "Geometry: O-grid map at uniform far-field angle, nested grids, SU2 writer"
```

---

### Task 4: SU2 binary restart reader

**Files:**
- Create: `src/apno/su2/__init__.py` (empty), `src/apno/su2/restart.py`
- Test: `tests/test_restart.py`

**Interfaces:**
- Produces:
  - `Restart(names: tuple[str, ...], data: np.ndarray)`, with `.field(name) -> np.ndarray` and `.npoints`.
  - `read_restart(path) -> Restart`; `write_restart(path, names, data)`.
  - `on_grid(restart, X, Y, atol=1e-10) -> dict[str, np.ndarray]`, each array of shape `(NI, NJ)`. It raises `ValueError` if the coordinates disagree with the grid.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from apno.geometry.ogrid import COARSE, Section, grid
from apno.su2.restart import on_grid, read_restart, write_restart

NAMES = ("x", "y", "Density", "Momentum_x", "Momentum_y", "Energy")


def _grid_restart(path, X, Y):
    ni, nj = X.shape
    xs, ys = X.T.reshape(-1), Y.T.reshape(-1)  # node id j*NI + i
    data = np.column_stack([xs, ys, np.ones_like(xs), xs * 2, ys * 3, np.full_like(xs, 7.0)])
    write_restart(path, NAMES, data)
    return data


def test_round_trip(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    data = _grid_restart(tmp_path / "r.dat", X, Y)
    r = read_restart(tmp_path / "r.dat")
    assert r.names == NAMES and r.npoints == X.size
    np.testing.assert_array_equal(r.data, data)
    f = on_grid(r, X, Y)
    np.testing.assert_array_equal(f["Momentum_x"], 2 * X)


def test_shuffled_nodes_fail_the_coordinate_check(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    data = _grid_restart(tmp_path / "r.dat", X, Y)
    write_restart(tmp_path / "s.dat", NAMES, data[np.random.default_rng(0).permutation(len(data))])
    with pytest.raises(ValueError, match="do not match"):
        on_grid(read_restart(tmp_path / "s.dat"), X, Y)


def test_bad_magic_and_truncation(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    _grid_restart(tmp_path / "r.dat", X, Y)
    raw = (tmp_path / "r.dat").read_bytes()
    (tmp_path / "t.dat").write_bytes(raw[:-8])
    with pytest.raises(ValueError, match="size"):
        read_restart(tmp_path / "t.dat")
    (tmp_path / "m.dat").write_bytes(b"\x00" * 4 + raw[4:])
    with pytest.raises(ValueError, match="magic"):
        read_restart(tmp_path / "m.dat")


def test_reads_the_real_spike_restart(spike_dir):
    r = read_restart(spike_dir / "restart.dat")
    assert r.names == NAMES and r.npoints == 24832
    x, y = r.field("x"), r.field("y")
    assert np.hypot(x[0] - 0.5, y[0]) == pytest.approx(0.5, abs=1e-12)  # trailing edge
    assert np.hypot(x[-1] - 0.5, y[-1]) == pytest.approx(50.0, rel=1e-9)  # spike's far field
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_restart.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apno.su2'`.

- [ ] **Step 3: Implement `src/apno/su2/restart.py`**

```python
"""SU2's binary restart file, as written by SU2 8.5.0 (layout verified byte by byte, 2026-10-01).

Header: five little-endian int32 = (535532, n_fields, n_points, 0, 0); then n_fields names, each
33 bytes NUL-padded; then n_points x n_fields float64, point by point. This build writes nothing
after the data, so the file size must match exactly; anything else is refused rather than guessed.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 535532
NAME_BYTES = 33


@dataclass(frozen=True)
class Restart:
    names: tuple
    data: np.ndarray

    @property
    def npoints(self):
        return self.data.shape[0]

    def field(self, name):
        return self.data[:, self.names.index(name)]


def read_restart(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path}: size {len(raw)} is smaller than the header")
    magic, nf, npnt, _, _ = struct.unpack("<5i", raw[:20])
    if magic != MAGIC:
        raise ValueError(f"{path}: not an SU2 binary restart (magic {magic}, expected {MAGIC})")
    offset = 20 + NAME_BYTES * nf
    expected = offset + 8 * nf * npnt
    if len(raw) != expected:
        raise ValueError(f"{path}: size {len(raw)} != expected {expected}")
    names = tuple(
        raw[20 + NAME_BYTES * k : 20 + NAME_BYTES * (k + 1)].split(b"\0")[0].decode()
        for k in range(nf)
    )
    data = np.frombuffer(raw, dtype="<f8", offset=offset).reshape(npnt, nf).copy()
    return Restart(names, data)


def write_restart(path, names, data):
    data = np.ascontiguousarray(data, dtype="<f8")
    head = struct.pack("<5i", MAGIC, len(names), data.shape[0], 0, 0)
    head += b"".join(n.encode().ljust(NAME_BYTES, b"\0") for n in names)
    Path(path).write_bytes(head + data.tobytes())


def on_grid(restart, X, Y, atol=1e-10):
    """Every field as an (NI, NJ) array, after checking each node's coordinates against the grid."""
    ni, nj = X.shape
    if restart.npoints != ni * nj:
        raise ValueError(f"restart has {restart.npoints} points, grid has {ni * nj}")
    out = {n: restart.field(n).reshape(nj, ni).T.copy() for n in restart.names}
    err = max(np.abs(out["x"] - X).max(), np.abs(out["y"] - Y).max())
    if err > atol:
        raise ValueError(f"restart nodes do not match the grid (max coordinate error {err:.3e})")
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_restart.py -v`
Expected: 4 passed. The spike test skips if the spike directory is absent.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/su2 tests/test_restart.py
git commit -m "SU2: binary restart reader with exact-size check and grid coordinate check"
```

---

### Task 5: Convergence rule

**Files:**
- Create: `src/apno/su2/convergence.py`
- Test: `tests/test_convergence.py`

**Interfaces:**
- Produces:
  - `read_history(path) -> dict[str, np.ndarray] | None` (`None` if the file is missing or empty).
  - `Verdict(converged: bool, reason: str, residual_drop: float, cl_range: float, cd_range: float, iterations: int)`, with `.as_dict()`.
  - `judge(history, residual_drop=5.0, window=500, force_tol=1e-6) -> Verdict`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from apno.su2.convergence import judge, read_history


def _hist(n=2000, drop=6.0, cl=lambda it: 0.3 + 0 * it, cd=lambda it: 0.01 + 0 * it):
    it = np.arange(n, dtype=float)
    return {"Inner_Iter": it, "rms[Rho]": -1.0 - drop * it / (n - 1), "CL": cl(it), "CD": cd(it)}


def test_steady_forces_with_stalled_residual_converges():
    h = _hist(drop=6.0)
    h["rms[Rho]"][1000:] = h["rms[Rho]"][1000]  # stalls at a 3-order drop ...
    assert not judge(h).converged
    h = _hist(drop=6.0)
    h["rms[Rho]"][1800:] = h["rms[Rho]"][1800]  # ... stalls after 5.40 orders: fine
    v = judge(h)
    assert v.converged and v.reason == "converged"


def test_oscillating_forces_fail():
    v = judge(_hist(cl=lambda it: 0.3 + 1e-4 * np.sin(it / 10)))
    assert not v.converged and "CL varied" in v.reason


def test_residual_drops_but_forces_drift():
    v = judge(_hist(cd=lambda it: 0.01 + 1e-8 * it))
    assert not v.converged and "CD varied" in v.reason


def test_too_few_iterations():
    v = judge(_hist(n=300))
    assert not v.converged and "window" in v.reason


def test_nan_fails():
    h = _hist()
    h["CL"][-3] = np.nan
    v = judge(h)
    assert not v.converged and "non-finite" in v.reason


def test_missing_and_empty_history(tmp_path):
    assert judge(read_history(tmp_path / "absent.csv")).reason == "no history"
    (tmp_path / "empty.csv").write_text("")
    assert judge(read_history(tmp_path / "empty.csv")).reason == "no history"
    (tmp_path / "nocol.csv").write_text('"Inner_Iter","CL"\n0,0.1\n')
    v = judge(read_history(tmp_path / "nocol.csv"))
    assert not v.converged and "missing column" in v.reason


def test_spike_history_converges_narrowly(spike_dir):
    v = judge(read_history(spike_dir / "history.csv"))
    assert v.converged
    assert v.residual_drop == pytest.approx(6.126, abs=1e-3)
    assert v.cl_range == pytest.approx(8.896e-7, rel=1e-3)
    assert v.iterations == 3999
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_convergence.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/su2/convergence.py`**

```python
"""Convergence is judged by our rule, not by SU2's flag (spec 4.4).

Converged iff the density residual dropped by at least `residual_drop` orders from the first
iteration AND CL and CD each varied by at most `force_tol` over the last `window` iterations. The
spike's coarse case shows why: SU2 said "not converged" (residual stuck 6.1 orders down, against an
absolute -11 target) while CL and CD had been steady to 9e-7 for 500 iterations.
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("Inner_Iter", "rms[Rho]", "CL", "CD")


@dataclass(frozen=True)
class Verdict:
    converged: bool
    reason: str
    residual_drop: float
    cl_range: float
    cd_range: float
    iterations: int

    def as_dict(self):
        return asdict(self)


def read_history(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))
    if len(rows) < 2:
        return None
    head = [h.strip().strip('"').strip() for h in rows[0]]
    body = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
    return {h: body[:, k] for k, h in enumerate(head)}


def judge(history, residual_drop=5.0, window=500, force_tol=1e-6):
    nan = float("nan")
    if not history:
        return Verdict(False, "no history", nan, nan, nan, 0)
    missing = [c for c in COLUMNS if c not in history]
    if missing:
        return Verdict(False, f"history missing column {missing}", nan, nan, nan, 0)
    it, res, cl, cd = (history[c] for c in COLUMNS)
    n_iter = int(it[-1])
    if not all(np.isfinite(a).all() for a in (res, cl, cd)):
        return Verdict(False, "non-finite values in history", nan, nan, nan, n_iter)
    if it[-1] - it[0] < window:
        return Verdict(False, f"only {n_iter} iterations, fewer than the {window}-iteration window",
                       nan, nan, nan, n_iter)
    drop = float(res[0] - res[-1])
    sel = it >= it[-1] - window
    clr, cdr = float(np.ptp(cl[sel])), float(np.ptp(cd[sel]))
    reasons = []
    if drop < residual_drop:
        reasons.append(f"residual dropped {drop:.2f} orders < {residual_drop}")
    if clr > force_tol:
        reasons.append(f"CL varied {clr:.2e} > {force_tol:.0e} over the last {window}")
    if cdr > force_tol:
        reasons.append(f"CD varied {cdr:.2e} > {force_tol:.0e} over the last {window}")
    return Verdict(not reasons, "; ".join(reasons) or "converged", drop, clr, cdr, n_iter)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_convergence.py -v`
Expected: 7 passed (the spike test skips if the spike is absent).

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/su2/convergence.py tests/test_convergence.py
git commit -m "SU2: convergence rule (residual drop and steady forces), failures carry a reason"
```

---

### Task 6: Flow quantities

**Files:**
- Create: `src/apno/physics/__init__.py` (empty), `src/apno/physics/flow.py`
- Test: `tests/test_flow.py`

**Interfaces:**
- Produces (NumPy; all arguments non-dimensional except the inputs to `nondimensionalise`):
  - `freestream(mach) -> (rho_inf, u_inf, p_inf)`, dimensional SI.
  - `nondimensionalise(rho, mx, my, rhoE, mach) -> (rho_h, u_h, v_h, p_h)`.
  - `pressure_coefficient(p_h, mach)`; `local_mach(rho_h, u_h, v_h, p_h, mach)`; `entropy(rho_h, p_h)`.
  - `total_enthalpy_ratio(rho_h, u_h, v_h, p_h, mach)`; `freestream_q(alpha_deg) -> np.ndarray(4)`.
  - `normal_shock(mach) -> (rho2, u2, p2)`, scaled by the upstream state.
  - `isentropic_mach_from_cp(cp, mach)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from apno.constants import GAMMA, P_INF
from apno.physics.flow import (
    entropy, freestream, freestream_q, isentropic_mach_from_cp, local_mach, nondimensionalise,
    normal_shock, pressure_coefficient, total_enthalpy_ratio,
)


def test_free_stream_is_quiet():
    M, a = 0.7, 3.0
    q = freestream_q(a)
    assert pressure_coefficient(q[3], M) == 0.0
    assert entropy(q[0], q[3]) == 0.0
    assert total_enthalpy_ratio(*q, M) == pytest.approx(1.0, abs=1e-15)
    assert local_mach(*q, M) == pytest.approx(M, rel=1e-15)
    assert isentropic_mach_from_cp(0.0, M) == pytest.approx(M, rel=1e-12)


def test_nondimensionalise_free_stream():
    M = 0.5
    rho, U, p = freestream(M)
    a = np.radians(2.0)
    rhoE = p / (GAMMA - 1) + 0.5 * rho * U * U
    out = nondimensionalise(rho, rho * U * np.cos(a), rho * U * np.sin(a), rhoE, M)
    np.testing.assert_allclose(out, [1.0, np.cos(a), np.sin(a), 1.0], rtol=1e-13)
    assert p == P_INF


def test_normal_shock_jumps():
    rho2, u2, p2 = normal_shock(2.0)
    assert rho2 == pytest.approx(8 / 3, rel=1e-12)
    assert p2 == pytest.approx(4.5, rel=1e-12)
    assert u2 == pytest.approx(3 / 8, rel=1e-12)
    assert local_mach(rho2, u2, 0.0, p2, 2.0) == pytest.approx(0.5773502692, rel=1e-9)
    assert total_enthalpy_ratio(rho2, u2, 0.0, p2, 2.0) == pytest.approx(1.0, rel=1e-12)
    assert entropy(rho2, p2) > 0


def test_cp_and_isentropic_mach_round_trip():
    M = 0.75
    p_h = 0.6  # an expansion
    rho_h = p_h ** (1 / GAMMA)
    # isentropic, constant total enthalpy -> speed from H
    speed2 = 1.0 + 2.0 / ((GAMMA - 1) * M**2) * (1.0 - p_h / rho_h)
    m_loc = local_mach(rho_h, np.sqrt(speed2), 0.0, p_h, M)
    assert isentropic_mach_from_cp(pressure_coefficient(p_h, M), M) == pytest.approx(m_loc, 1e-12)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_flow.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/physics/flow.py`**

```python
"""Flow quantities, all from one tested place.

Non-dimensional state everywhere: rho/rho_inf, u/U_inf, v/U_inf, p/p_inf. Then
a^2/U_inf^2 = (p_h / rho_h) / M^2 and H/U_inf^2 = p_h/(rho_h (gamma - 1) M^2) + |u|^2/2.
"""

import numpy as np

from apno.constants import GAMMA, P_INF, R_GAS, T_INF

G = GAMMA


def freestream(mach):
    rho = P_INF / (R_GAS * T_INF)
    return rho, mach * np.sqrt(G * R_GAS * T_INF), P_INF


def nondimensionalise(rho, mx, my, rhoE, mach):
    rho_inf, u_inf, p_inf = freestream(mach)
    p = (G - 1.0) * (rhoE - 0.5 * (mx * mx + my * my) / rho)
    return rho / rho_inf, mx / rho / u_inf, my / rho / u_inf, p / p_inf


def freestream_q(alpha_deg):
    a = np.radians(alpha_deg)
    return np.array([1.0, np.cos(a), np.sin(a), 1.0])


def pressure_coefficient(p_h, mach):
    return (p_h - 1.0) / (0.5 * G * mach * mach)


def local_mach(rho_h, u_h, v_h, p_h, mach):
    return np.sqrt(u_h * u_h + v_h * v_h) * mach / np.sqrt(p_h / rho_h)


def entropy(rho_h, p_h):
    """(s - s_inf)/c_v."""
    return np.log(p_h / rho_h**G)


def total_enthalpy_ratio(rho_h, u_h, v_h, p_h, mach):
    h = p_h / (rho_h * (G - 1.0) * mach * mach) + 0.5 * (u_h * u_h + v_h * v_h)
    return h / (1.0 / ((G - 1.0) * mach * mach) + 0.5)


def normal_shock(mach):
    m2 = mach * mach
    rho2 = (G + 1.0) * m2 / ((G - 1.0) * m2 + 2.0)
    p2 = 1.0 + 2.0 * G / (G + 1.0) * (m2 - 1.0)
    return rho2, 1.0 / rho2, p2


def isentropic_mach_from_cp(cp, mach):
    p_h = 1.0 + 0.5 * G * mach * mach * cp
    t0 = 1.0 + 0.5 * (G - 1.0) * mach * mach
    return np.sqrt(2.0 / (G - 1.0) * (t0 * p_h ** (-(G - 1.0) / G) - 1.0))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_flow.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/physics tests/test_flow.py
git commit -m "Physics: non-dimensional flow quantities, normal shock, isentropic Mach"
```

---

### Task 7: The one force integrator

**Files:**
- Create: `src/apno/score/__init__.py` (empty), `src/apno/score/forces.py`
- Test: `tests/test_forces.py`

**Interfaces:**
- Produces:
  - `vertex_normals(xw, yw) -> (n, 2)`: outward and unnormalised, carrying half of each adjacent edge's length.
  - `wall_forces(xw, yw, cp, alpha_deg, x_ref=0.25, y_ref=0.0) -> (cl, cd, cm)`.

- [ ] **Step 1: Write the failing tests**

```python
import csv

import numpy as np
import pytest

from apno.physics.flow import nondimensionalise, pressure_coefficient
from apno.score.forces import vertex_normals, wall_forces


def _circle(n=256):
    th = -2 * np.pi * np.arange(n) / n
    return 0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th), th


def test_uniform_pressure_gives_no_force():
    x, y, _ = _circle()
    assert np.allclose(wall_forces(x, y, np.full_like(x, 0.7), 3.0), 0.0, atol=1e-14)


def test_potential_cylinder_has_no_lift_or_drag():
    x, y, th = _circle()
    cl, cd, cm = wall_forces(x, y, 1 - 4 * np.sin(th) ** 2, 0.0)
    assert abs(cl) < 1e-13 and abs(cd) < 1e-13


def test_orientation_does_not_matter():
    x, y, th = _circle(64)
    cp = np.cos(th) + 0.3 * np.sin(th)
    a = wall_forces(x, y, cp, 2.0)
    b = wall_forces(x[::-1], y[::-1], cp[::-1], 2.0)
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_normals_point_outward():
    x, y, _ = _circle(64)
    n = vertex_normals(x, y)
    assert np.all((x - 0.5) * n[:, 0] + y * n[:, 1] > 0)


def test_reproduces_su2_on_the_spike(spike_dir):
    rows = list(csv.reader(open(spike_dir / "surface.csv"), skipinitialspace=True))
    head = [h.strip('"') for h in rows[0]]
    d = np.array([[float(v) for v in r] for r in rows[1:] if r])
    d = d[np.argsort(d[:, head.index("PointID")])]
    col = {h: d[:, k] for k, h in enumerate(head)}
    *_, p_h = nondimensionalise(col["Density"], col["Momentum_x"], col["Momentum_y"],
                                col["Energy"], 0.8)
    cl, cd, cm = wall_forces(col["x"], col["y"], pressure_coefficient(p_h, 0.8), 1.25)
    assert cl == pytest.approx(0.2819562904, abs=1e-9)
    assert cd == pytest.approx(0.01969418273, abs=1e-9)
    assert cm == pytest.approx(0.01812806891, abs=1e-9)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_forces.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/score/forces.py`**

```python
"""The one force integrator (spec 8.1), used for SU2, POD, FNO and PINN alike.

Vertex-lumped: each wall node carries half of each adjacent edge's outward normal. On the
2026-10-01 spike this reproduces SU2's CL, CD and CMz to 5e-11; the edge-midpoint trapezoid matches
CL and CD but misses CMz by 1.6e-5. Moments about (x_ref, y_ref), sign as SU2's CMz.
"""

import numpy as np


def vertex_normals(xw, yw):
    P = np.stack([xw, yw], 1)
    e = np.roll(P, -1, 0) - P
    twice_area = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    n_e = np.stack([-e[:, 1], e[:, 0]], 1)  # left normal: outward for a clockwise polygon
    if twice_area > 0:
        n_e = -n_e
    return 0.5 * (n_e + np.roll(n_e, 1, 0))


def wall_forces(xw, yw, cp, alpha_deg, x_ref=0.25, y_ref=0.0):
    f = -np.asarray(cp)[:, None] * vertex_normals(xw, yw)
    a = np.radians(alpha_deg)
    fx, fy = f[:, 0].sum(), f[:, 1].sum()
    cd = fx * np.cos(a) + fy * np.sin(a)
    cl = -fx * np.sin(a) + fy * np.cos(a)
    cm = np.sum((np.asarray(xw) - x_ref) * f[:, 1] - (np.asarray(yw) - y_ref) * f[:, 0])
    return float(cl), float(cd), float(cm)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_forces.py -v`
Expected: 5 passed (the spike test skips if the spike is absent).

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/score tests/test_forces.py
git commit -m "Scoring: vertex-lumped force integrator, reproduces SU2 on the spike"
```

---

### Task 8: SU2 configuration

**Files:**
- Create: `src/apno/su2/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-7)`, a frozen dataclass.
  - `render_config(mach, alpha_deg, settings) -> str`.
  - `VERIFIED_KEYWORDS: frozenset[str]`: keywords checked against the SU2 8.5.0 binary on 2026-10-01.

- [ ] **Step 1: Write the failing tests**

```python
import re
import subprocess

import pytest

from apno.su2.config import VERIFIED_KEYWORDS, Su2Settings, render_config


def _keys(cfg):
    return [ln.split("=")[0].strip() for ln in cfg.splitlines() if "=" in ln and ln[0] != "%"]


def test_every_keyword_is_verified_and_unique():
    keys = _keys(render_config(0.7, 2.5, Su2Settings(mglevel=3)))
    assert len(keys) == len(set(keys))
    assert set(keys) <= VERIFIED_KEYWORDS


def test_values_are_written():
    cfg = render_config(0.7, -1.5, Su2Settings(mglevel=0, iter_cap=1234))
    assert re.search(r"^MACH_NUMBER= 0\.7000000000$", cfg, re.M)
    assert re.search(r"^AOA= -1\.5000000000$", cfg, re.M)
    assert re.search(r"^ITER= 1234$", cfg, re.M)
    assert re.search(r"^MGLEVEL= 0$", cfg, re.M)
    assert "VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )" in cfg


@pytest.mark.su2
def test_keywords_exist_in_the_binary(su2_bin):
    out = subprocess.run(["strings", "-n", "3", str(su2_bin / "SU2_CFD")],
                         capture_output=True, text=True).stdout.splitlines()
    present = set(out)
    # REF_AREA is parsed by this build (the spike's log echoes it) but strings cannot see it whole
    missing = sorted(k for k in VERIFIED_KEYWORDS if k not in present and k != "REF_AREA")
    assert not missing
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/su2/config.py`**

```python
"""The SU2 configuration: the spike's scheme, plus primitive output and a Cauchy stop on forces.

SU2's Cauchy stop is only a STOPPING rule, to avoid running every case to the cap. Whether a case
counts as converged is decided by apno.su2.convergence.judge, never by SU2. Stage S0 checks that the
two agree before the campaign.
"""

from dataclasses import dataclass

from apno.constants import GAMMA, P_INF, R_GAS, T_INF

VERIFIED_KEYWORDS = frozenset("""
SOLVER MATH_PROBLEM RESTART_SOL MACH_NUMBER AOA FREESTREAM_PRESSURE FREESTREAM_TEMPERATURE
GAMMA_VALUE GAS_CONSTANT REF_ORIGIN_MOMENT_X REF_ORIGIN_MOMENT_Y REF_ORIGIN_MOMENT_Z REF_LENGTH
REF_AREA MARKER_EULER MARKER_FAR MARKER_PLOTTING MARKER_MONITORING NUM_METHOD_GRAD CFL_NUMBER
CFL_ADAPT CFL_ADAPT_PARAM ITER LINEAR_SOLVER LINEAR_SOLVER_PREC LINEAR_SOLVER_ERROR
LINEAR_SOLVER_ITER MGLEVEL MGCYCLE CONV_NUM_METHOD_FLOW JST_SENSOR_COEFF TIME_DISCRE_FLOW
CONV_FIELD CONV_CAUCHY_ELEMS CONV_CAUCHY_EPS CONV_STARTITER MESH_FILENAME MESH_FORMAT
SCREEN_OUTPUT SCREEN_WRT_FREQ_INNER HISTORY_OUTPUT HISTORY_WRT_FREQ_INNER CONV_FILENAME
TABULAR_FORMAT OUTPUT_FILES VOLUME_OUTPUT OUTPUT_WRT_FREQ RESTART_FILENAME SURFACE_FILENAME
""".split())


@dataclass(frozen=True)
class Su2Settings:
    mglevel: int = 0
    iter_cap: int = 10000
    cauchy_elems: int = 500
    cauchy_eps: float = 1e-7


def render_config(mach, alpha_deg, settings):
    s = settings
    return f"""% written by apno.su2.config
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {mach:.10f}
AOA= {alpha_deg:.10f}
FREESTREAM_PRESSURE= {P_INF}
FREESTREAM_TEMPERATURE= {T_INF}
GAMMA_VALUE= {GAMMA}
GAS_CONSTANT= {R_GAS}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
REF_LENGTH= 1.0
REF_AREA= 1.0
MARKER_EULER= ( airfoil )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 5.0, 500.0 )
ITER= {s.iter_cap}
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
MGLEVEL= {s.mglevel}
MGCYCLE= W_CYCLE
CONV_NUM_METHOD_FLOW= JST
JST_SENSOR_COEFF= ( 0.5, 0.02 )
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_FIELD= ( LIFT, DRAG )
CONV_CAUCHY_ELEMS= {s.cauchy_elems}
CONV_CAUCHY_EPS= {s.cauchy_eps:.1e}
CONV_STARTITER= {s.cauchy_elems}
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, WALL_TIME, RMS_DENSITY, LIFT, DRAG )
SCREEN_WRT_FREQ_INNER= 100
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
HISTORY_WRT_FREQ_INNER= 1
CONV_FILENAME= history
TABULAR_FORMAT= CSV
OUTPUT_FILES= ( RESTART, SURFACE_CSV )
VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )
OUTPUT_WRT_FREQ= {s.iter_cap}
RESTART_FILENAME= restart.dat
SURFACE_FILENAME= surface
"""
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_config.py -v`, then `SU2_BIN_DIR=$HOME/theLab/bwb-tanker-scratch/bin python -m pytest tests/test_config.py -v -m su2`.
Expected: 2 passed without the variable; with it, the binary test also passes. If it reports a missing keyword, the keyword is spelled wrong for this build. Fix the spelling; never delete the check.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/su2/config.py tests/test_config.py
git commit -m "SU2: configuration rendering with keywords verified against the binary"
```

---

### Task 9: Resumable case driver and campaign

**Files:**
- Create: `src/apno/su2/run.py`, `tests/fake_su2.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `grid`, `write_su2`, `Section`, `level_name` (Task 3); `read_restart`, `on_grid` (Task 4); `read_history`, `judge` (Task 5); `nondimensionalise`, `local_mach` (Task 6); `render_config`, `Su2Settings` (Task 8).
- Produces:
  - `CaseSpec(case_id, mach, alpha_deg, t, m, shape, radius=100.0)`, frozen, with `.level == level_name(shape)` and `.inputs() -> dict`.
  - `case_dir(root, spec) -> Path` (= `root / spec.level / spec.case_id`); `is_done(dir) -> bool`.
  - `run_case(spec, root, su2_bin, settings, judge_kwargs=None) -> dict`, the `result.json` contents.
  - `run_campaign(specs, root, su2_bin, settings_for, workers, judge_kwargs=None) -> list[dict]`, where `settings_for(spec) -> Su2Settings`.
  - `result.json` keys: `case_id, inputs, level, radius, converged, reason, verdict, iterations, cl, cd, cm, max_surface_mach, wall_s, cpu_s, exit_code, su2_version, config_sha256, mesh_sha256, mglevel`.

- [ ] **Step 1: Write the fake solver `tests/fake_su2.py`**

```python
"""A stand-in for SU2_CFD for the driver tests. FAKE_SU2_MODE = ok | crash | nan.

It reads the mesh's nodes in file order and writes a free-stream restart in that order, plus a
history with a 6-order residual drop and constant forces, so the driver's real readers are
exercised.
"""

import math
import os
import re
import struct
import sys
from pathlib import Path


def main():
    cfg = Path(sys.argv[1]).read_text()
    mode = os.environ.get("FAKE_SU2_MODE", "ok")
    print('|   \\__ \\ |_| |/ /   Release 8.5.0 "Harrier"   (fake)')
    if mode == "crash":
        return 1
    mach = float(re.search(r"^MACH_NUMBER=\s*(\S+)", cfg, re.M).group(1))
    alpha = math.radians(float(re.search(r"^AOA=\s*(\S+)", cfg, re.M).group(1)))
    lines = Path("mesh.su2").read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("NPOIN="))
    n = int(lines[k].split("=")[1])
    pts = [tuple(map(float, ln.split()[:2])) for ln in lines[k + 1 : k + 1 + n]]
    g, r_gas, p, t = 1.4, 287.058, 101325.0, 288.15
    rho = p / (r_gas * t)
    u = mach * math.sqrt(g * r_gas * t)
    mx, my = rho * u * math.cos(alpha), rho * u * math.sin(alpha)
    e = p / (g - 1) + 0.5 * rho * u * u
    names = ["x", "y", "Density", "Momentum_x", "Momentum_y", "Energy"]
    head = struct.pack("<5i", 535532, len(names), n, 0, 0)
    head += b"".join(nm.encode().ljust(33, b"\0") for nm in names)
    body = b"".join(struct.pack("<6d", x, y, rho, mx, my, e) for x, y in pts)
    Path("restart.dat").write_bytes(head + body)
    rows = ['"Inner_Iter",    "rms[Rho]"    ,       "CD"       ,       "CL"       ,      "CMz"']
    for it in range(1001):
        cl = "nan" if (mode == "nan" and it > 900) else "0.25"
        rows.append(f"{it}, {-1.0 - 6.0 * it / 1000}, 0.0001, {cl}, 0.01")
    Path("history.csv").write_text("\n".join(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the failing tests `tests/test_run.py`**

```python
import json
import os
import sys
from pathlib import Path

import pytest

from apno.geometry.ogrid import COARSE
from apno.su2.config import Su2Settings
from apno.su2.run import CaseSpec, case_dir, is_done, run_campaign, run_case

FAKE = Path(__file__).with_name("fake_su2.py")
SETTINGS = Su2Settings(iter_cap=1000)
JUDGE = dict(window=200)


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    exe = b / "SU2_CFD"
    exe.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_SU2_MODE", "ok")
    return b


def _spec(k=0, mach=0.5):
    return CaseSpec(f"test-{k:04d}", mach, 2.0, 0.12, 0.02, COARSE)


def test_ok_case(tmp_path, fake_bin):
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["converged"] and res["exit_code"] == 0 and res["su2_version"] == "8.5.0"
    assert res["max_surface_mach"] == pytest.approx(0.5, rel=1e-12)  # free-stream fake
    assert res["cl"] == 0.25 and res["level"] == "128x49"
    assert is_done(case_dir(tmp_path / "runs", _spec()))


def test_done_case_is_not_rerun(tmp_path, fake_bin, monkeypatch):
    first = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    monkeypatch.setenv("FAKE_SU2_MODE", "crash")
    assert run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE) == first


def test_crash_records_failure(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "crash")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not res["converged"] and res["exit_code"] == 1 and "no history" in res["reason"]
    assert res["max_surface_mach"] is None


def test_nan_history_records_failure(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "nan")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not res["converged"] and "non-finite" in res["reason"]


def test_stale_files_rerun(tmp_path, fake_bin):
    d = case_dir(tmp_path / "runs", _spec())
    d.mkdir(parents=True)
    (d / "restart.dat").write_bytes(b"garbage from a killed run")
    (d / "history.csv").write_text("garbage")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["converged"]


def test_tmp_result_is_not_done(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    (d / "result.json.tmp").write_text("{}")
    assert not is_done(d)


def test_campaign(tmp_path, fake_bin):
    specs = [_spec(k, 0.3 + 0.1 * k) for k in range(3)]
    out = run_campaign(specs, tmp_path / "runs", fake_bin, lambda s: SETTINGS, 2, JUDGE)
    assert sorted(r["case_id"] for r in out) == [s.case_id for s in specs]
    assert all(r["converged"] for r in out)
    saved = json.loads((case_dir(tmp_path / "runs", specs[2]) / "result.json").read_text())
    assert saved["inputs"]["mach"] == pytest.approx(0.5)
    assert os.environ["FAKE_SU2_MODE"] == "ok"
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `python -m pytest tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apno.su2.run'`.

- [ ] **Step 4: Implement `src/apno/su2/run.py`**

```python
"""Run SU2 cases, one directory each, resumably.

A case is done iff its result.json exists. result.json is written last and atomically (temporary
file, then rename), so a killed run leaves no result and simply runs again. Stale outputs from a
killed run are deleted before SU2 starts, so they can never be read as the new run's result.
Per-case CPU time comes from os.wait4 on that one child, because RUSAGE_CHILDREN mixes concurrent
cases.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from apno.geometry.ogrid import FAR_RADIUS, Section, grid, level_name, write_su2
from apno.physics.flow import local_mach, nondimensionalise
from apno.su2.config import render_config
from apno.su2.convergence import judge, read_history
from apno.su2.restart import on_grid, read_restart

OUTPUTS = ("restart.dat", "history.csv", "surface.csv", "su2.log", "result.json")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    mach: float
    alpha_deg: float
    t: float
    m: float
    shape: tuple
    radius: float = FAR_RADIUS

    @property
    def level(self):
        return level_name(self.shape)

    def inputs(self):
        return {"mach": self.mach, "alpha_deg": self.alpha_deg, "t": self.t, "m": self.m}


def case_dir(root, spec):
    return Path(root) / spec.level / spec.case_id


def is_done(d):
    return (Path(d) / "result.json").exists()


def _write_json_atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _sha(path_or_text):
    data = path_or_text.read_bytes() if isinstance(path_or_text, Path) else path_or_text.encode()
    return hashlib.sha256(data).hexdigest()


def _max_surface_mach(d, X, Y, mach):
    try:
        f = on_grid(read_restart(d / "restart.dat"), X, Y)
    except (OSError, ValueError):
        return None
    q = nondimensionalise(f["Density"][:, 0], f["Momentum_x"][:, 0], f["Momentum_y"][:, 0],
                          f["Energy"][:, 0], mach)
    return float(local_mach(*q, mach).max())


def run_case(spec, root, su2_bin, settings, judge_kwargs=None):
    d = case_dir(root, spec)
    if is_done(d):
        return json.loads((d / "result.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS + ("result.json.tmp",):
        (d / name).unlink(missing_ok=True)
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    write_su2(d / "mesh.su2", X, Y)
    cfg = render_config(spec.mach, spec.alpha_deg, settings)
    (d / "case.cfg").write_text(cfg)

    t0 = time.perf_counter()
    with open(d / "su2.log", "w") as log:
        proc = subprocess.Popen([str(Path(su2_bin) / "SU2_CFD"), "case.cfg"], cwd=d,
                                stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0

    verdict = judge(read_history(d / "history.csv"), **(judge_kwargs or {}))
    hist = read_history(d / "history.csv") or {}
    last = {k: float(hist[k][-1]) if k in hist else None for k in ("CL", "CD", "CMz")}
    version = re.search(r"Release (\S+)", (d / "su2.log").read_text())
    result = {
        "case_id": spec.case_id, "inputs": spec.inputs(), "level": spec.level,
        "radius": spec.radius, "converged": verdict.converged, "reason": verdict.reason,
        "verdict": verdict.as_dict(), "iterations": verdict.iterations,
        "cl": last["CL"], "cd": last["CD"], "cm": last["CMz"],
        "max_surface_mach": _max_surface_mach(d, X, Y, spec.mach),
        "wall_s": wall, "cpu_s": usage.ru_utime + usage.ru_stime,
        "exit_code": proc.returncode, "su2_version": version.group(1) if version else None,
        "config_sha256": _sha(cfg), "mesh_sha256": _sha(d / "mesh.su2"),
        "mglevel": settings.mglevel,
    }
    _write_json_atomic(d / "result.json", result)
    return result


def run_campaign(specs, root, su2_bin, settings_for, workers, judge_kwargs=None):
    """Run every spec; a driver exception is reported, not recorded as a result, so it reruns."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, s, root, su2_bin, settings_for(s), judge_kwargs): s
                   for s in specs}
        for n, fut in enumerate(as_completed(futures), 1):
            spec = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- reported, case left not-done
                print(f"[{n}/{len(specs)}] {spec.case_id}: DRIVER ERROR {exc!r}", flush=True)
                continue
            print(f"[{n}/{len(specs)}] {spec.case_id}: {res['reason']} "
                  f"({res['iterations']} it, {res['wall_s']:.0f} s)", flush=True)
            results.append(res)
    return results
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_run.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add src/apno/su2/run.py tests/fake_su2.py tests/test_run.py
git commit -m "SU2: resumable case driver with atomic result.json and per-case CPU time"
```

---

### Task 10: The Sobol design and the case lists

**Files:**
- Create: `src/apno/data/__init__.py` (empty), `src/apno/data/sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `CaseSpec` (Task 9); `FINE`, `TRAIN` (Task 3).
- Produces:
  - `BOX`, `EXTRAP_MACH`, `SEEDS`, `PREFIXES = (64, 128, 256, 512, 896)`.
  - `design() -> dict[str, np.ndarray]` mapping each split name to an `(n, 4)` array of `[mach, alpha_deg, t, m]`. The splits are `train` 896, `val` 128, `test_in` 64, `test_ext` 16.
  - `case_specs(split, shape) -> list[CaseSpec]`, with `case_id = f"{split}-{k:04d}"`.
  - `campaign_specs(which) -> list[CaseSpec]`, where `which` is `"train"` (train and val at 256 × 97, plus test_in and test_ext at 256 × 97) or `"fine"` (test_in and test_ext at 512 × 193).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from apno.data.sampling import BOX, EXTRAP_MACH, campaign_specs, case_specs, design


def test_deterministic_and_sized():
    a, b = design(), design()
    assert {k: v.shape[0] for k, v in a.items()} == {"train": 896, "val": 128, "test_in": 64,
                                                     "test_ext": 16}
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_inside_the_box():
    d = design()
    lo = np.array([BOX[k][0] for k in ("mach", "alpha_deg", "t", "m")])
    hi = np.array([BOX[k][1] for k in ("mach", "alpha_deg", "t", "m")])
    for split in ("train", "val", "test_in"):
        assert np.all(d[split] >= lo) and np.all(d[split] <= hi)
    ext = d["test_ext"]
    assert np.all((ext[:, 0] >= EXTRAP_MACH[0]) & (ext[:, 0] <= EXTRAP_MACH[1]))
    assert np.all(ext[:, 1:] >= lo[1:]) and np.all(ext[:, 1:] <= hi[1:])


def test_splits_disjoint():
    d = design()
    rows = np.concatenate(list(d.values()))
    assert len(np.unique(rows.round(12), axis=0)) == len(rows)


def test_prefix_of_64_is_balanced():
    u = (design()["train"][:64] - [0.30, -2.0, 0.08, 0.0]) / [0.5, 8.0, 0.08, 0.04]
    assert np.all(np.abs(u.mean(0) - 0.5) < 0.05)


def test_case_specs():
    specs = case_specs("test_ext", (512, 193))
    assert specs[3].case_id == "test_ext-0003" and specs[3].level == "512x193"
    assert len(campaign_specs("train")) == 896 + 128 + 64 + 16
    assert len(campaign_specs("fine")) == 64 + 16
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/data/sampling.py`**

```python
"""The design, fixed before any model is trained (spec 5.3). Scrambled Sobol, recorded seeds."""

import numpy as np
from scipy.stats import qmc

from apno.geometry.ogrid import FINE, TRAIN
from apno.su2.run import CaseSpec

BOX = {"mach": (0.30, 0.80), "alpha_deg": (-2.0, 6.0), "t": (0.08, 0.16), "m": (0.0, 0.04)}
EXTRAP_MACH = (0.80, 0.85)
SEEDS = {"pool": 20261001, "test_in": 20261002, "test_ext": 20261003}
PREFIXES = (64, 128, 256, 512, 896)
_LO = np.array([BOX[k][0] for k in BOX])
_HI = np.array([BOX[k][1] for k in BOX])


def _sobol(n, seed):
    return qmc.Sobol(d=4, scramble=True, seed=seed).random(n)


def design():
    pool = _LO + _sobol(1024, SEEDS["pool"]) * (_HI - _LO)
    test_in = _LO + _sobol(64, SEEDS["test_in"]) * (_HI - _LO)
    lo, hi = _LO.copy(), _HI.copy()
    lo[0], hi[0] = EXTRAP_MACH
    test_ext = lo + _sobol(16, SEEDS["test_ext"]) * (hi - lo)
    return {"train": pool[:896], "val": pool[896:], "test_in": test_in, "test_ext": test_ext}


def case_specs(split, shape):
    return [CaseSpec(f"{split}-{k:04d}", *map(float, row), shape)
            for k, row in enumerate(design()[split])]


def campaign_specs(which):
    if which == "train":
        return [s for split in ("train", "val", "test_in", "test_ext")
                for s in case_specs(split, TRAIN)]
    if which == "fine":
        return [s for split in ("test_in", "test_ext") for s in case_specs(split, FINE)]
    raise ValueError(which)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/data tests/test_sampling.py
git commit -m "Data: fixed Sobol design, splits and case lists"
```

---

### Task 11: Stage S0, checking SU2's accuracy and cost, then freezing the settings

**Files:**
- Create: `src/apno/settings.py`, `scripts/s0_check.py`, `results/s0/` (output)
- Modify: `docs/findings.md`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: Tasks 3–9.
- Produces:
  - `apno.settings.settings_for(spec) -> Su2Settings`, which uses `FINE_ITER_CAP` for 512 × 193.
  - `CONVERGENCE` (the `judge_kwargs` dict) and `WORKERS: int`.
  - `results/s0/summary.json`.

- [ ] **Step 1: Write the failing test `tests/test_settings.py`**

```python
from apno.geometry.ogrid import FINE, TRAIN
from apno.settings import CONVERGENCE, WORKERS, settings_for
from apno.su2.run import CaseSpec


def test_caps_by_level_and_judge_defaults():
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, TRAIN)).iter_cap == 10000
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, FINE)).iter_cap == 20000
    assert CONVERGENCE == {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}
    assert 1 <= WORKERS <= 8
```

Run: `python -m pytest tests/test_settings.py -v`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Implement `src/apno/settings.py` with the spec's starting values**

```python
"""Campaign settings. Starting values from spec 4.4; stage S0 confirms or revises them, then they
are FROZEN before S1. Changing a value after S1 starts requires a new dated entry in
docs/findings.md and re-running every case it affects.
"""

from dataclasses import replace

from apno.geometry.ogrid import FINE
from apno.su2.config import Su2Settings

BASE = Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-7)
FINE_ITER_CAP = 20000
CONVERGENCE = {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}
WORKERS = 6


def settings_for(spec):
    return replace(BASE, iter_cap=FINE_ITER_CAP) if spec.shape == FINE else BASE
```

Run the test again. Expected: 1 passed.

- [ ] **Step 3: Write `scripts/s0_check.py`**

```python
"""Stage S0: check SU2's accuracy and cost before the campaign (spec 4.6). Writes results/s0/.

Exits 1 if the force integrator fails to reproduce SU2 on any converged case: if the scorer cannot
score SU2 against SU2, it cannot score anything. Usage:
    SU2_BIN_DIR=... python scripts/s0_check.py [--workers 6]
"""

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from apno.geometry.ogrid import COARSE, FINE, TRAIN, Section, grid
from apno.physics.flow import nondimensionalise, pressure_coefficient
from apno.score.forces import wall_forces
from apno.settings import CONVERGENCE, settings_for
from apno.su2.restart import on_grid, read_restart
from apno.su2.run import CaseSpec, case_dir, run_campaign

CASES = {
    "sym": (0.50, 0.0, 0.12, 0.00),
    "sub0": (0.50, 0.0, 0.12, 0.02),
    "sub2": (0.50, 2.0, 0.12, 0.02),
    "super": (0.75, 2.0, 0.12, 0.02),
    "anchor": (0.80, 1.25, 0.12, 0.00),
}
ROOT = Path("runs/s0")
OUT = Path("results/s0")


def specs():
    out = []
    for name, (M, a, t, m) in CASES.items():
        for shape in (COARSE, TRAIN, FINE):
            out.append((CaseSpec(f"{name}-mg0", M, a, t, m, shape), 0))
        out.append((CaseSpec(f"{name}-mg3", M, a, t, m, TRAIN), 3))
    for name in ("sub2", "anchor"):
        M, a, t, m = CASES[name]
        for radius in (50.0, 200.0):
            out.append((CaseSpec(f"{name}-R{radius:.0f}", M, a, t, m, TRAIN, radius), 0))
    return out


def scorer_forces(spec):
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    f = on_grid(read_restart(case_dir(ROOT, spec) / "restart.dat"), X, Y)
    q = nondimensionalise(f["Density"], f["Momentum_x"], f["Momentum_y"], f["Energy"], spec.mach)
    cp = pressure_coefficient(q[3], spec.mach)
    return wall_forces(X[:, 0], Y[:, 0], cp[:, 0], spec.alpha_deg), X, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    su2_bin = Path(os.environ["SU2_BIN_DIR"])
    pairs = specs()
    mg = {s.case_id + s.level: m for s, m in pairs}
    res = run_campaign([s for s, _ in pairs], ROOT, su2_bin,
                       lambda s: replace(settings_for(s), mglevel=mg[s.case_id + s.level]),
                       args.workers, CONVERGENCE)
    by = {(r["case_id"], r["level"]): r for r in res}

    summary = {"runs": res, "scorer_check": [], "mesh": {}, "multigrid": {}, "radius": {},
               "known_answers": {}}
    worst = 0.0
    surf = {}
    for spec, _ in pairs:
        r = by.get((spec.case_id, spec.level))
        if not r or not r["converged"]:
            continue
        (cl, cd, cm), X, cp = scorer_forces(spec)
        d = max(abs(cl - r["cl"]), abs(cd - r["cd"]), abs(cm - r["cm"]))
        worst = max(worst, d)
        summary["scorer_check"].append({"case": spec.case_id, "level": spec.level, "max_diff": d})
        surf[(spec.case_id, spec.level)] = cp[:, 0]

    for name in CASES:
        rows = [by.get((f"{name}-mg0", lv)) for lv in ("128x49", "256x97", "512x193")]
        if not all(r and r["converged"] for r in rows):
            summary["mesh"][name] = "not all three levels converged"
            continue
        entry = {}
        for k in ("cl", "cd", "cm"):
            c, t, f = (r[k] for r in rows)
            ratio = abs(c - t) / abs(t - f) if t != f else math.inf
            entry[k] = {"128x49": c, "256x97": t, "512x193": f, "train_minus_fine": t - f,
                        "observed_order": math.log2(ratio) if 0 < ratio < math.inf else None}
        cp_t, cp_f = surf[(f"{name}-mg0", "256x97")], surf[(f"{name}-mg0", "512x193")][::2]
        entry["cp_train_vs_fine"] = {"l2": float(np.sqrt(np.mean((cp_t - cp_f) ** 2))),
                                     "max": float(np.abs(cp_t - cp_f).max())}
        summary["mesh"][name] = entry
        a, b = by.get((f"{name}-mg0", "256x97")), by.get((f"{name}-mg3", "256x97"))
        if a and b:
            summary["multigrid"][name] = {
                "mg0": {k: a[k] for k in ("wall_s", "cpu_s", "iterations", "converged", "cl")},
                "mg3": {k: b[k] for k in ("wall_s", "cpu_s", "iterations", "converged", "cl")}}
    for name in ("sub2", "anchor"):
        summary["radius"][name] = {
            R: (by.get((f"{name}-R{R}", "256x97")) or {}).get("cl") for R in ("50", "200")}
        summary["radius"][name]["100"] = by[(f"{name}-mg0", "256x97")]["cl"]
    ka = summary["known_answers"]
    ka["sym_abs_cl_fine"] = abs(by[("sym-mg0", "512x193")]["cl"])
    ka["subcritical_cd_counts_fine"] = {n: 1e4 * by[(f"{n}-mg0", "512x193")]["cd"]
                                        for n in ("sym", "sub0", "sub2")}
    M = CASES["sub2"][0]
    d_cl = by[("sub2-mg0", "512x193")]["cl"] - by[("sub0-mg0", "512x193")]["cl"]
    slope = d_cl / math.radians(2)
    ka["lift_slope_ratio_to_prandtl_glauert"] = slope / (2 * math.pi / math.sqrt(1 - M * M))
    ka["anchor_cl_fine"] = by[("anchor-mg0", "512x193")]["cl"]
    ka["vassberg_jameson_2010_cl"] = "read from the paper and recorded in docs/findings.md"
    summary["scorer_worst_abs_diff"] = worst

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2, default=str))
    failed = [f'{r["case_id"]}@{r["level"]}: {r["reason"]}' for r in res if not r["converged"]]
    print("NOT CONVERGED:", failed or "none")
    if worst > 1e-6:
        print(f"SCORER CHECK FAILED: worst difference {worst:.3e} > 1e-6")
        return 1
    print(f"scorer check passed: worst difference {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run S0 for real (about 1–2 hours, mostly the five fine-grid runs)**

Run:
```bash
SU2_BIN_DIR=$HOME/theLab/bwb-tanker-scratch/bin python scripts/s0_check.py --workers 6 2>&1 | tee results/s0/s0.log
echo "exit $?"
```
Expected: `scorer check passed` and exit 0. If the scorer check fails, stop. The difference is a finding about the scorer, to resolve before anything else (spec §8.1), and the 10⁻⁶ tolerance is never loosened. If a case is listed as NOT CONVERGED, read its `su2.log` and `history.csv` under `runs/s0/` before changing any setting.

- [ ] **Step 5: Read Vassberg & Jameson (2010)**

Find the grid-converged CL for NACA 0012 at M 0.8, α 1.25°, inviscid, in "In pursuit of grid convergence for two-dimensional Euler solutions" (J. Aircraft 47(4), 2010). Compare it with `known_answers.anchor_cl_fine`. If the paper cannot be obtained, write that in the findings entry; do not use the ≈0.35 remembered value as if it were read.

- [ ] **Step 6: Freeze the settings, writing the decision before editing the code**

Append a dated entry to `docs/findings.md`, with the command and the measured numbers from `results/s0/summary.json`:
- **Multigrid:** wall time and iterations for MGLEVEL 0 against 3. Choose 3 if it converged every case and cut the median wall time; otherwise choose 0.
- **Convergence rule:** if any S0 case stopped on SU2's Cauchy rule but failed our rule, lower `cauchy_eps` by 10× and rerun those cases. Record the case and both outcomes.
- **Outer radius:** the CL change from 50 to 100 to 200 chords. Keep 100 if |CL(200) − CL(100)| < 1 × 10⁻³; otherwise record the change and keep 100 anyway. Moving the radius changes every mesh, so it is a decision for the author, not this task.
- **The floor:** CL, CD and CM differences between the training and fine grids, and the observed orders.
- **Known answers:** sym |CL|; subcritical CD in counts; the lift-slope ratio; anchor CL against Vassberg & Jameson.
- **Per-case cost:** median wall and CPU time per grid level, and the projected campaign time.

Then edit `BASE` in `src/apno/settings.py` (`mglevel`, `cauchy_eps`) and `WORKERS` to match the entry. Change the module docstring's first line to name the findings entry's date.

- [ ] **Step 7: Verify and commit**

```bash
python -m pytest tests/test_settings.py -q   # update its expected values only if the entry changed them
ruff check . && git add src/apno/settings.py scripts/s0_check.py tests/test_settings.py results/s0 docs/findings.md
git commit -m "S0: SU2 accuracy and cost measured; campaign settings frozen"
```

---

### Task 12: Stage S1, the data campaign

**Files:**
- Create: `scripts/s1_campaign.py`, `results/s1/` (output)
- Modify: `docs/findings.md`

**Interfaces:**
- Consumes: `campaign_specs` (Task 10); `settings_for`, `CONVERGENCE`, `WORKERS` (Task 11); `run_campaign` (Task 9).
- Produces: `runs/su2/<level>/<case_id>/` for every case; `results/s1/campaign_<which>.json` with counts, failures and their reasons, and the summed wall and CPU time.

- [ ] **Step 1: Write `scripts/s1_campaign.py`**

```python
"""Stage S1: the data campaign. Resumable: rerun the same command after any interruption.

    SU2_BIN_DIR=... python scripts/s1_campaign.py train   # 1,104 cases at 256x97
    SU2_BIN_DIR=... python scripts/s1_campaign.py fine    # 80 cases at 512x193
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from apno.data.sampling import campaign_specs
from apno.settings import CONVERGENCE, WORKERS, settings_for
from apno.su2.run import run_campaign

ROOT = Path("runs/su2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("train", "fine"))
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    specs = campaign_specs(args.which)
    res = run_campaign(specs, ROOT, Path(os.environ["SU2_BIN_DIR"]), settings_for, args.workers,
                       CONVERGENCE)
    failures = [{"case_id": r["case_id"], "level": r["level"], "reason": r["reason"],
                 "inputs": r["inputs"]} for r in res if not r["converged"]]
    summary = {
        "which": args.which, "requested": len(specs), "finished": len(res),
        "converged": sum(r["converged"] for r in res), "failures": failures,
        "failure_reasons": Counter(f["reason"].split(" ")[0] for f in failures),
        "wall_s_sum": sum(r["wall_s"] for r in res), "cpu_s_sum": sum(r["cpu_s"] for r in res),
    }
    out = Path("results/s1")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"campaign_{args.which}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    return 0 if summary["finished"] == summary["requested"] else 2


if __name__ == "__main__":
    sys.exit(main())
```

Exit 2 means some cases ended in a driver error. Rerun the same command; finished cases are skipped.

- [ ] **Step 2: Smoke-test it on the fake solver**

```bash
mkdir -p $TMPDIR/fakebin && { echo "#!$(which python)"; cat tests/fake_su2.py; } > $TMPDIR/fakebin/SU2_CFD && chmod +x $TMPDIR/fakebin/SU2_CFD
cd $TMPDIR && rm -rf runs results && SU2_BIN_DIR=$TMPDIR/fakebin python ~/theLab/airfoil-pinn-no/scripts/s1_campaign.py fine --workers 4; cd -
```
Expected: `"finished": 80, "converged": 80`. The fake writes 1,000 iterations with a 6-order residual drop and constant forces, which passes the 500-iteration rule. The point is that the script finishes, resumes (run it twice: the second run is instant) and writes its summary. Delete `$TMPDIR/runs` and `$TMPDIR/results` afterwards.

- [ ] **Step 3: Run the campaign for real, in the background (one to two nights)**

```bash
SU2_BIN_DIR=$HOME/theLab/bwb-tanker-scratch/bin nohup python scripts/s1_campaign.py train > results/s1/train.log 2>&1 &
```
When it exits, run the same command with `fine` and `results/s1/fine.log`. Check memory with `vm_stat` during the first hour. If pages are being swapped out steadily, stop and rerun with `--workers` lowered by 2; the campaign resumes.

- [ ] **Step 4: Record the campaign in `docs/findings.md`**

A dated entry, with both commands: counts converged and failed per level, the failure reasons, total CPU hours, and the median iterations. Where the failures sit in the input box is a result: list the inputs of every failed case.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add scripts/s1_campaign.py results/s1 docs/findings.md
git commit -m "S1: data campaign run; failures mapped with reasons"
```

---

### Task 13: Building the dataset: HDF5, regimes and the showcase rule

**Files:**
- Create: `src/apno/data/build.py`, `scripts/build_dataset.py`
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: `case_specs` (Task 10); `case_dir`, `CaseSpec` (Task 9); `on_grid`, `read_restart` (Task 4); `nondimensionalise` (Task 6); `grid`, `Section` (Task 3).
- Produces:
  - `load_case(root, spec) -> dict | None`: `xy (NI, NJ, 2)`, `q (NI, NJ, 4)`, `inputs (4,)`, `forces (3,)`, `max_surface_mach`, `case_id`. Returns `None` if the case is not converged or has no result.
  - `build_level(specs_by_split, root, out_path, dtype) -> dict` (the failures per split).
  - `regime(max_surface_mach) -> "supercritical" | "subcritical"`.
  - `showcase(case_ids, max_mach) -> list[str]`.
  - The HDF5 layout: `f[split]` holds `xy`, `q`, `inputs`, `forces_su2`, `max_surface_mach` and `case_id` (UTF-8 strings). Files are `data/hdf5/256x97.h5` (all four splits) and `data/hdf5/512x193.h5` (`test_in` and `test_ext`).

- [ ] **Step 1: Write the failing tests**

```python
import json

import h5py
import numpy as np
import pytest

from apno.data.build import build_level, regime, showcase
from apno.geometry.ogrid import COARSE, Section, grid
from apno.physics.flow import freestream
from apno.su2.restart import write_restart
from apno.su2.run import CaseSpec, case_dir


def _fake_case(root, spec, converged=True):
    d = case_dir(root, spec)
    d.mkdir(parents=True)
    X, Y = grid(Section(spec.t, spec.m), spec.shape)
    rho, U, p = freestream(spec.mach)
    a = np.radians(spec.alpha_deg)
    n = X.size
    data = np.column_stack([X.T.ravel(), Y.T.ravel(), np.full(n, rho),
                            np.full(n, rho * U * np.cos(a)), np.full(n, rho * U * np.sin(a)),
                            np.full(n, p / 0.4 + 0.5 * rho * U * U)])
    write_restart(d / "restart.dat", ("x", "y", "Density", "Momentum_x", "Momentum_y", "Energy"),
                  data)
    (d / "result.json").write_text(json.dumps({
        "converged": converged, "reason": "converged" if converged else "CL varied",
        "cl": 0.1, "cd": 0.0, "cm": 0.0, "max_surface_mach": spec.mach}))


def test_build_writes_converged_cases_and_lists_failures(tmp_path):
    specs = {"train": [CaseSpec(f"train-{k:04d}", 0.4 + 0.1 * k, 1.0, 0.12, 0.0, COARSE)
                       for k in range(3)]}
    for k, s in enumerate(specs["train"]):
        _fake_case(tmp_path, s, converged=(k != 1))
    fails = build_level(specs, tmp_path, tmp_path / "out.h5", np.float32)
    assert [f["case_id"] for f in fails["train"]] == ["train-0001"]
    with h5py.File(tmp_path / "out.h5") as f:
        g = f["train"]
        assert g["q"].shape == (2, 128, 49, 4) and g["q"].dtype == np.float32
        assert list(g["case_id"].asstr()[...]) == ["train-0000", "train-0002"]
        np.testing.assert_allclose(g["q"][0, :, :, 0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(g["q"][0, :, :, 1], np.cos(np.radians(1.0)), rtol=1e-6)


def test_regime():
    assert regime(1.0001) == "supercritical" and regime(0.9999) == "subcritical"


def test_showcase_rule_on_64():
    ids = [f"c{k}" for k in range(64)]
    mach = np.linspace(0.5, 1.4, 64)[::-1]  # deliberately unsorted input
    picked = showcase(ids, mach)
    order = [ids[k] for k in np.argsort(mach, kind="stable")]
    assert picked == [order[round(p * 63)] for p in (0.10, 0.30, 0.50, 0.70, 0.85, 0.95)]


def test_showcase_on_survivors():
    ids = [f"c{k}" for k in range(10)]
    picked = showcase(ids, np.arange(10.0))
    assert picked == ["c1", "c3", "c4", "c6", "c8", "c9"]
    with pytest.raises(ValueError, match="at least 6"):
        showcase(ids[:5], np.arange(5.0))
```

Check of `test_showcase_on_survivors`: n = 10, so the indices are round(p · 9) = 0.9→1, 2.7→3, 4.5→4 (Python rounds half to even), 6.3→6, 7.65→8, 8.55→9.

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/data/build.py`**

```python
"""Raw SU2 case directories -> one HDF5 file per grid level, deterministically (spec 5.1-5.3).

Only converged cases are written; every failure is returned with its reason, never silently
dropped. Each case's restart nodes are checked against a freshly generated grid, so a node-ordering
mistake fails loudly instead of corrupting data.
"""

import json

import h5py
import numpy as np

from apno.geometry.ogrid import Section, grid
from apno.physics.flow import nondimensionalise
from apno.su2.restart import on_grid, read_restart
from apno.su2.run import case_dir

SHOWCASE_PERCENTILES = (0.10, 0.30, 0.50, 0.70, 0.85, 0.95)


def regime(max_surface_mach):
    return "supercritical" if max_surface_mach > 1.0 else "subcritical"


def showcase(case_ids, max_mach):
    n = len(case_ids)
    if n < len(SHOWCASE_PERCENTILES):
        raise ValueError(f"need at least 6 surviving test cases, have {n}")
    order = np.argsort(np.asarray(max_mach), kind="stable")
    return [case_ids[order[round(p * (n - 1))]] for p in SHOWCASE_PERCENTILES]


def load_case(root, spec):
    d = case_dir(root, spec)
    if not (d / "result.json").exists():
        return None
    res = json.loads((d / "result.json").read_text())
    if not res["converged"]:
        return None
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    f = on_grid(read_restart(d / "restart.dat"), X, Y)
    q = nondimensionalise(f["Density"], f["Momentum_x"], f["Momentum_y"], f["Energy"], spec.mach)
    return {"case_id": spec.case_id, "xy": np.stack([X, Y], -1), "q": np.stack(q, -1),
            "inputs": np.array([spec.mach, spec.alpha_deg, spec.t, spec.m]),
            "forces": np.array([res["cl"], res["cd"], res["cm"]]),
            "max_surface_mach": res["max_surface_mach"]}


def build_level(specs_by_split, root, out_path, dtype):
    failures = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for split, specs in specs_by_split.items():
            cases, failures[split] = [], []
            for spec in specs:
                c = load_case(root, spec)
                if c is None:
                    rj = case_dir(root, spec) / "result.json"
                    why = json.loads(rj.read_text())["reason"] if rj.exists() else "not run"
                    failures[split].append({"case_id": spec.case_id, "reason": why,
                                            "inputs": spec.inputs()})
                else:
                    cases.append(c)
            g = f.create_group(split)
            g["xy"] = np.stack([c["xy"] for c in cases]).astype(dtype)
            g["q"] = np.stack([c["q"] for c in cases]).astype(dtype)
            g["inputs"] = np.stack([c["inputs"] for c in cases])
            g["forces_su2"] = np.stack([c["forces"] for c in cases])
            g["max_surface_mach"] = np.array([c["max_surface_mach"] for c in cases])
            g.create_dataset("case_id", data=[c["case_id"] for c in cases],
                             dtype=h5py.string_dtype())
    tmp.replace(out_path)
    return failures
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_build.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write `scripts/build_dataset.py` and run it after S1**

```python
"""Build data/hdf5/{256x97,512x193}.h5 from runs/su2 and pick the PINN showcase cases.

Writes results/s1/failures_<level>.json and results/s1/showcase.json.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from apno.data.build import build_level, regime, showcase
from apno.data.sampling import case_specs
from apno.geometry.ogrid import FINE, TRAIN, level_name

ROOT, OUT, RES = Path("runs/su2"), Path("data/hdf5"), Path("results/s1")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    plan = {TRAIN: (("train", "val", "test_in", "test_ext"), np.float32),
            FINE: (("test_in", "test_ext"), np.float64)}
    for shape, (splits, dtype) in plan.items():
        lv = level_name(shape)
        fails = build_level({s: case_specs(s, shape) for s in splits}, ROOT, OUT / f"{lv}.h5",
                            dtype)
        (RES / f"failures_{lv}.json").write_text(json.dumps(fails, indent=2))
        print(lv, {s: len(v) for s, v in fails.items()}, "failures")
    with h5py.File(OUT / "512x193.h5") as f, h5py.File(OUT / "256x97.h5") as t:
        g = f["test_in"]
        ids = list(g["case_id"].asstr()[...])
        # a test case counts only if it converged on BOTH grids (it needs truth and floor)
        both = set(t["test_in"]["case_id"].asstr()[...])
        keep = [k for k, c in enumerate(ids) if c in both]
        mach = g["max_surface_mach"][...][keep]
        ids = [ids[k] for k in keep]
        picked = showcase(ids, mach)
        lookup = dict(zip(ids, mach))
    out = {"rule": "sorted by fine-grid max surface Mach; index round(p*(n-1))",
           "n_survivors": len(ids), "cases": [
               {"case_id": c, "max_surface_mach": float(lookup[c]),
                "regime": regime(lookup[c])} for c in picked]}
    (RES / "showcase.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

Run: `python scripts/build_dataset.py`
Expected: both HDF5 files are written, the failure counts match `results/s1/campaign_*.json`, and the six showcase cases are printed. The supercritical count among them depends on the data. Record it in the findings entry from Task 12.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add src/apno/data/build.py scripts/build_dataset.py tests/test_build.py results/s1
git commit -m "Data: HDF5 builder, regime, rule-based showcase selection"
```

---

### Task 14: Output targets, input features, and the POD baseline

**Files:**
- Create: `src/apno/models/__init__.py` (empty), `src/apno/models/targets.py`, `src/apno/models/pod.py`
- Test: `tests/test_targets.py`, `tests/test_pod.py`

**Interfaces:**
- Produces:
  - `to_targets(q, inputs) -> y`, mapping `q (N, NI, NJ, 4)` to `y (N, 4, NI, NJ)`, and `from_targets(y, inputs) -> q`.
  - `Standardiser.fit(y) -> self`, with `.transform(y)`, `.inverse(y)`, `.state() -> dict` and `Standardiser.from_state(dict)`.
  - `features(xy, inputs) -> (N, 6, NI, NJ)` float32.
  - `PodModel(n_modes, smoothing).fit(inputs, y_std) -> self`, `.predict(inputs) -> y_std`.

- [ ] **Step 1: Write the failing tests**

`tests/test_targets.py`:
```python
import numpy as np

from apno.constants import GAMMA
from apno.models.targets import Standardiser, features, from_targets, to_targets


def _q(n=3, ni=8, nj=5, seed=0):
    rng = np.random.default_rng(seed)
    return 1 + 0.1 * rng.standard_normal((n, ni, nj, 4)), np.column_stack(
        [rng.uniform(0.3, 0.8, n), rng.uniform(-2, 6, n), rng.uniform(.08, .16, n),
         rng.uniform(0, .04, n)])


def test_round_trip_exact():
    q, inp = _q()
    np.testing.assert_allclose(from_targets(to_targets(q, inp), inp), q, atol=1e-13)


def test_scaled_density_matches_cp_at_low_mach():
    M = 0.05
    rho = 1 + 1e-4 * np.random.default_rng(1).standard_normal((1, 4, 3))
    p = rho**GAMMA  # isentropic small perturbation
    q = np.stack([rho, np.ones_like(rho), np.zeros_like(rho), p], -1)
    y = to_targets(q, np.array([[M, 0.0, 0.12, 0.0]]))
    np.testing.assert_allclose(y[:, 3], y[:, 2], rtol=2e-3)


def test_standardiser():
    q, inp = _q(n=20)
    y = to_targets(q, inp)
    s = Standardiser().fit(y)
    z = s.transform(y)
    np.testing.assert_allclose(z.mean(axis=(0, 2, 3)), 0, atol=1e-12)
    np.testing.assert_allclose(Standardiser.from_state(s.state()).inverse(z), y, atol=1e-12)


def test_features_shape_and_ring():
    xy = np.random.default_rng(2).uniform(1, 2, (2, 8, 5, 2))
    f = features(xy, np.array([[0.5, 2.0, 0.12, 0.0], [0.7, -1.0, 0.1, 0.02]]))
    assert f.shape == (2, 6, 8, 5) and f.dtype == np.float32
    np.testing.assert_allclose(f[0, 2, 0], np.linspace(0, 1, 5), atol=1e-7)
    assert np.allclose(f[1, 3], 0.7) and np.allclose(f[1, 4], np.cos(np.radians(-1.0)))
```

`tests/test_pod.py`:
```python
import numpy as np

from apno.models.pod import PodModel


def _family(inp, ni=6, nj=4):
    # fields linear in the inputs: thin-plate splines reproduce degree-1 polynomials exactly
    base = np.random.default_rng(0).standard_normal((4, 4, ni, nj))
    return np.einsum("nk,kcij->ncij", inp, base)


def test_reproduces_training_with_all_modes():
    inp = np.random.default_rng(1).uniform(size=(30, 4))
    y = _family(inp) + 0.01 * np.random.default_rng(2).standard_normal((30, 4, 6, 4))
    m = PodModel(n_modes=29, smoothing=0.0).fit(inp, y)
    np.testing.assert_allclose(m.predict(inp), y, atol=1e-8)


def test_interpolates_a_linear_family():
    rng = np.random.default_rng(3)
    inp, new = rng.uniform(size=(40, 4)), rng.uniform(size=(5, 4))
    m = PodModel(n_modes=4, smoothing=0.0).fit(inp, _family(inp))
    np.testing.assert_allclose(m.predict(new), _family(new), atol=1e-8)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_targets.py tests/test_pod.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/models/targets.py`**

```python
"""What the models predict and what they are given (spec 6.1-6.2).

Targets, each of order 1 across the Mach range: u - cos(alpha), v - sin(alpha), Cp, and the scaled
density perturbation (rho - 1)/(M^2/2), which tends to Cp at low Mach. Features: radially compressed
node position about mid-chord, the ring index j/(NJ-1), M, cos(alpha), sin(alpha).
"""

import numpy as np

from apno.constants import GAMMA


def _ma(inputs):
    M = np.asarray(inputs)[:, 0][:, None, None]
    a = np.radians(np.asarray(inputs)[:, 1])[:, None, None]
    return M, a


def to_targets(q, inputs):
    M, a = _ma(inputs)
    rho, u, v, p = (q[..., k] for k in range(4))
    y = [u - np.cos(a), v - np.sin(a), (p - 1) / (0.5 * GAMMA * M * M), (rho - 1) / (0.5 * M * M)]
    return np.stack(y, 1)


def from_targets(y, inputs):
    M, a = _ma(inputs)
    u = y[:, 0] + np.cos(a)
    v = y[:, 1] + np.sin(a)
    p = 1 + y[:, 2] * 0.5 * GAMMA * M * M
    rho = 1 + y[:, 3] * 0.5 * M * M
    return np.stack([rho, u, v, p], -1)


class Standardiser:
    def fit(self, y):
        self.mean = y.mean(axis=(0, 2, 3), keepdims=True)
        self.std = y.std(axis=(0, 2, 3), keepdims=True)
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

    def state(self):
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.mean = np.array(st["mean"]).reshape(1, -1, 1, 1)
        s.std = np.array(st["std"]).reshape(1, -1, 1, 1)
        return s


def features(xy, inputs):
    n, ni, nj, _ = xy.shape
    dx, dy = xy[..., 0] - 0.5, xy[..., 1]
    r = np.hypot(dx, dy)  # > 0: mid-chord is inside every section
    k = np.log1p(r) / r
    ring = np.broadcast_to(np.linspace(0.0, 1.0, nj), (n, ni, nj))
    M, a = _ma(inputs)
    const = [np.broadcast_to(c, (n, ni, nj)) for c in (M, np.cos(a), np.sin(a))]
    return np.stack([dx * k, dy * k, ring, *const], 1).astype(np.float32)
```

- [ ] **Step 4: Implement `src/apno/models/pod.py`**

```python
"""The POD baseline (spec 6.5): a linear basis of the standardised training fields, with each mode's
weight interpolated over the four inputs by a thin-plate-spline RBF. Possible only because every
case is the same array shape. Inputs are scaled to the unit box first.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator

from apno.data.sampling import BOX

_LO = np.array([BOX[k][0] for k in BOX])
_SPAN = np.array([BOX[k][1] - BOX[k][0] for k in BOX])


class PodModel:
    def __init__(self, n_modes, smoothing=0.0):
        self.n_modes, self.smoothing = n_modes, smoothing

    def fit(self, inputs, y):
        n = y.shape[0]
        self.shape = y.shape[1:]
        Y = y.reshape(n, -1)
        self.mean = Y.mean(0)
        Yc = Y - self.mean
        lam, V = np.linalg.eigh(Yc @ Yc.T)  # snapshot method: n x n, not D x D
        lam, V = lam[::-1], V[:, ::-1]
        k = min(self.n_modes, int(np.sum(lam > 1e-12 * lam[0])))
        self.modes = (Yc.T @ V[:, :k]) / np.sqrt(lam[:k])  # (D, k), orthonormal columns
        coeffs = Yc @ self.modes
        self.rbf = RBFInterpolator((inputs - _LO) / _SPAN, coeffs, kernel="thin_plate_spline",
                                   smoothing=self.smoothing)
        return self

    def predict(self, inputs):
        c = self.rbf((np.asarray(inputs) - _LO) / _SPAN)
        return (c @ self.modes.T + self.mean).reshape(len(inputs), *self.shape)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_targets.py tests/test_pod.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add src/apno/models tests/test_targets.py tests/test_pod.py
git commit -m "Models: output targets, input features, POD baseline"
```

---

### Task 15: The FNO, its data loading and training

**Files:**
- Create: `src/apno/models/fno.py`, `src/apno/models/train_fno.py`, `src/apno/data/predictions.py`
- Test: `tests/test_fno.py`, `tests/test_predictions.py`

**Interfaces:**
- Consumes: `features`, `to_targets`, `from_targets`, `Standardiser` (Task 14).
- Produces:
  - `FNO2d(in_ch=6, out_ch=4, width=32, modes_i=24, modes_j=12, layers=4, pad_fraction=1/6)`.
  - `load_split(h5_path, split) -> dict(xy, q, inputs, case_id)`, which raises `PermissionError` for `test_in` or `test_ext`; and `load_features_only(h5_path, split) -> dict(xy, inputs, case_id)`.
  - `FnoConfig(width=32, modes_i=24, modes_j=12, layers=4, loss="A", wall_weight=10.0, lr=1e-3, weight_decay=1e-4, batch=8, epochs=300, patience=30, seed=0, n_train=896)`.
  - `train(cfg, h5_path, out_dir) -> dict` (with keys `best_val`, `epochs_run`, `wall_s`, `cpu_s` and `checkpoint`); `load_model(checkpoint) -> (model, Standardiser, cfg)`.
  - `predict(model, std, xy, inputs, batch=8) -> (q (N, NI, NJ, 4), seconds_per_case)`.
  - `write_predictions(path, case_ids, q, attrs)` and `read_predictions(path) -> (dict case_id → q, attrs)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_fno.py`:
```python
import h5py
import numpy as np
import pytest
import torch

from apno.models.fno import FNO2d
from apno.models.train_fno import FnoConfig, load_features_only, load_split, predict, train


def test_shapes_at_both_resolutions():
    m = FNO2d(width=8, modes_i=4, modes_j=3, layers=2).eval()
    for ni, nj in ((128, 49), (256, 97)):
        assert m(torch.randn(2, 6, ni, nj)).shape == (2, 4, ni, nj)


def test_roll_equivariance_around_the_airfoil():
    torch.manual_seed(0)
    m = FNO2d(width=8, modes_i=6, modes_j=4, layers=3).double().eval()
    x = torch.randn(1, 6, 32, 17, dtype=torch.float64)
    with torch.no_grad():
        a = m(torch.roll(x, 5, dims=2))
        b = torch.roll(m(x), 5, dims=2)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def _tiny_h5(path, n=16, ni=16, nj=9):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for split in ("train", "val", "test_in"):
            g = f.create_group(split)
            xy = np.stack(np.meshgrid(np.linspace(1, 3, ni), np.linspace(1, 2, nj),
                                      indexing="ij"), -1)
            g["xy"] = np.broadcast_to(xy, (n, ni, nj, 2)).astype(np.float32)
            inp = np.column_stack([rng.uniform(.3, .8, n), rng.uniform(-2, 6, n),
                                   rng.uniform(.08, .16, n), rng.uniform(0, .04, n)])
            g["inputs"] = inp
            q = np.ones((n, ni, nj, 4)) + 0.05 * inp[:, None, None, :1] * xy[None, ..., :1]
            g["q"] = q.astype(np.float32)
            g.create_dataset("case_id", data=[f"{split}-{k}" for k in range(n)],
                             dtype=h5py.string_dtype())


def test_loader_refuses_test_splits(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    with pytest.raises(PermissionError):
        load_split(tmp_path / "d.h5", "test_in")
    feats = load_features_only(tmp_path / "d.h5", "test_in")
    assert "q" not in feats and feats["xy"].shape[0] == 16


def test_training_smoke(tmp_path):
    _tiny_h5(tmp_path / "d.h5")
    cfg = FnoConfig(width=8, modes_i=3, modes_j=2, layers=2, epochs=3, patience=10, n_train=16,
                    batch=4)
    out = train(cfg, tmp_path / "d.h5", tmp_path / "run", subsample=1)
    assert out["epochs_run"] == 3 and np.isfinite(out["best_val"])
    from apno.models.train_fno import load_model
    model, std, _ = load_model(out["checkpoint"])
    f = load_features_only(tmp_path / "d.h5", "test_in")
    q, sec = predict(model, std, f["xy"], f["inputs"])
    assert q.shape == (16, 16, 9, 4) and np.isfinite(q).all() and sec > 0
```

`tests/test_predictions.py`:
```python
import numpy as np

from apno.data.predictions import read_predictions, write_predictions


def test_round_trip(tmp_path):
    q = np.random.default_rng(0).standard_normal((2, 4, 3, 4))
    write_predictions(tmp_path / "p.h5", ["a", "b"], q, {"method": "fno", "seed": 1})
    got, attrs = read_predictions(tmp_path / "p.h5")
    np.testing.assert_array_equal(got["b"], q[1])
    assert attrs["method"] == "fno" and attrs["seed"] == 1
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_fno.py tests/test_predictions.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/models/fno.py`**

```python
"""A Fourier neural operator in the O-grid's index space (spec 6.3).

i (around the airfoil) is periodic, so the FFT along it is exact. j (outward) is not: the field is
zero-padded by (NJ - 1) * pad_fraction rows before the FFT and cropped after, so the padded domain
spans the same eta at every resolution. That is what lets one model run at 128x49 and 256x97.
Truncating the modes smooths jumps: smeared shocks and overshoot are expected and measured, not
fixed.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes_i, modes_j):
        super().__init__()
        self.mi, self.mj = modes_i, modes_j
        scale = 1.0 / (cin * cout)
        self.w_pos = nn.Parameter(scale * torch.randn(cin, cout, modes_i, modes_j,
                                                      dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(cin, cout, modes_i, modes_j,
                                                      dtype=torch.cfloat))

    def forward(self, x):
        b, _, ni, nj = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w_pos.shape[1], ni, nj // 2 + 1, dtype=xf.dtype,
                          device=x.device)
        w_pos, w_neg = self.w_pos.to(xf.dtype), self.w_neg.to(xf.dtype)
        out[:, :, : self.mi, : self.mj] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, : self.mi, : self.mj], w_pos)
        out[:, :, -self.mi :, : self.mj] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.mi :, : self.mj], w_neg)
        return torch.fft.irfft2(out, s=(ni, nj))


class FNO2d(nn.Module):
    def __init__(self, in_ch=6, out_ch=4, width=32, modes_i=24, modes_j=12, layers=4,
                 pad_fraction=1 / 6):
        super().__init__()
        self.pad_fraction = pad_fraction
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes_i, modes_j)
                                      for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, x):
        nj = x.shape[-1]
        pad = round((nj - 1) * self.pad_fraction)
        h = F.pad(self.lift(x), (0, pad))
        for k, (s, w) in enumerate(zip(self.spectral, self.pointwise)):
            h = s(h) + w(h)
            if k < len(self.spectral) - 1:
                h = F.gelu(h)
        return self.proj(h[..., :nj])
```

- [ ] **Step 4: Implement `src/apno/data/predictions.py`**

```python
"""Prediction files: one per (method, configuration, seed), q at 256x97 for the test cases."""

import h5py
import numpy as np


def write_predictions(path, case_ids, q, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["q"] = np.asarray(q, dtype=np.float64)
        f.create_dataset("case_id", data=list(case_ids), dtype=h5py.string_dtype())
        for k, v in attrs.items():
            f.attrs[k] = v


def read_predictions(path):
    with h5py.File(path) as f:
        ids = list(f["case_id"].asstr()[...])
        q = f["q"][...]
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
    return dict(zip(ids, q)), attrs
```

- [ ] **Step 5: Implement `src/apno/models/train_fno.py`**

```python
"""Training and inference for the FNO (spec 6.4).

Leakage guard: `load_split` refuses test splits; test cases are only ever read through
`load_features_only`, which never returns field values. Training runs on the training grid sampled
at every second node (128x49 from 256x97), and evaluation runs at 256x97.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from apno.models.fno import FNO2d
from apno.models.targets import Standardiser, features, from_targets, to_targets

TRAINABLE = ("train", "val")


@dataclass
class FnoConfig:
    width: int = 32
    modes_i: int = 24
    modes_j: int = 12
    layers: int = 4
    loss: str = "A"
    wall_weight: float = 10.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 8
    epochs: int = 300
    patience: int = 30
    seed: int = 0
    n_train: int = 896


def load_split(h5_path, split):
    if split not in TRAINABLE:
        raise PermissionError(f"split {split!r} is not readable for training")
    with h5py.File(h5_path) as f:
        g = f[split]
        return {"xy": g["xy"][...], "q": g["q"][...], "inputs": g["inputs"][...],
                "case_id": list(g["case_id"].asstr()[...])}


def load_features_only(h5_path, split):
    with h5py.File(h5_path) as f:
        g = f[split]
        return {"xy": g["xy"][...], "inputs": g["inputs"][...],
                "case_id": list(g["case_id"].asstr()[...])}


def relative_l2(pred, true, eps=1e-6):
    num = (pred - true).flatten(2).norm(dim=2)
    return (num / (true.flatten(2).norm(dim=2) + eps)).mean()


def loss_fn(pred, true, cfg):
    loss = relative_l2(pred, true)
    if cfg.loss == "B":
        loss = loss + cfg.wall_weight * relative_l2(pred[..., :1], true[..., :1])
    return loss


def _tensors(d, std, sub, n=None):
    sl = slice(None, n)
    x = features(d["xy"][sl, ::sub, ::sub], d["inputs"][sl])
    y = std.transform(to_targets(d["q"][sl, ::sub, ::sub], d["inputs"][sl]))
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def train(cfg, h5_path, out_dir, subsample=2):
    torch.manual_seed(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = load_split(h5_path, "train"), load_split(h5_path, "val")
    std = Standardiser().fit(to_targets(tr["q"][: cfg.n_train], tr["inputs"][: cfg.n_train]))
    xt, yt = _tensors(tr, std, subsample, cfg.n_train)
    xv, yv = _tensors(va, std, subsample)
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    best, best_epoch, log = float("inf"), -1, []
    t0, c0 = time.perf_counter(), time.process_time()
    epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), cfg.batch):
            idx = perm[k : k + cfg.batch]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx], cfg).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(xv[k : k + 32]), yv[k : k + 32], cfg).item()
                                 for k in range(0, len(xv), 32)]))
        log.append({"epoch": epoch, "val": val})
        if val < best:
            best, best_epoch = val, epoch
            torch.save({"model": model.state_dict(), "std": std.state(), "cfg": asdict(cfg)},
                       out_dir / "best.pt")
        elif epoch - best_epoch >= cfg.patience:
            break
    out = {"best_val": best, "best_epoch": best_epoch, "epochs_run": epoch + 1,
           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "checkpoint": str(out_dir / "best.pt"), "cfg": asdict(cfg), "log": log}
    (out_dir / "train.json").write_text(json.dumps(out, indent=2))
    return out


def load_model(checkpoint):
    ck = torch.load(checkpoint, weights_only=True)
    cfg = FnoConfig(**ck["cfg"])
    model = FNO2d(width=cfg.width, modes_i=cfg.modes_i, modes_j=cfg.modes_j,
                  layers=cfg.layers).float()
    model.load_state_dict(ck["model"])
    return model.eval(), Standardiser.from_state(ck["std"]), cfg


def predict(model, std, xy, inputs, batch=8):
    x = torch.from_numpy(features(xy, inputs))
    t0 = time.perf_counter()
    with torch.no_grad():
        z = torch.cat([model(x[k : k + batch]) for k in range(0, len(x), batch)]).numpy()
    sec = (time.perf_counter() - t0) / len(x)
    return from_targets(std.inverse(z.astype(np.float64)), inputs), sec
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_fno.py tests/test_predictions.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
ruff check . && git add src/apno/models src/apno/data/predictions.py tests/test_fno.py tests/test_predictions.py
git commit -m "Models: hand-written FNO, leakage-guarded loading, training and prediction files"
```

---

### Task 16: Stage S2, the operator runs

**Files:**
- Create: `scripts/s2_operator.py`, `results/s2/` (output)
- Modify: `docs/findings.md`

**Interfaces:**
- Consumes: Tasks 14–15.
- Produces:
  - `runs/fno/<tag>/best.pt` and `train.json`.
  - Prediction files `runs/predictions/<method>-n<N>-s<seed>[-<variant>].h5`, with attrs `method`, `n_train`, `seed`, `variant`, `train_wall_s`, `train_cpu_s` and `query_s`.
  - `results/s2/tuning.json`, `results/s2/runs.json`.

The run tags and what each one is:

| Tag | What |
|---|---|
| `tune-<k>` | k = 0…7, the eight tuning configurations at n = 896, loss A, 100 epochs |
| `fno-n896-s{0,1,2}` | the selected configuration, loss A, 300 epochs |
| `fno-n896-s0-lossB` | the selected configuration, loss B |
| `fno-n{64,128,256,512}-s0` | the data-efficiency curve, loss A |
| `pod-n{64,…,896}` | POD with modes and smoothing chosen on validation for each n |

- [ ] **Step 1: Write `scripts/s2_operator.py`**

```python
"""Stage S2: POD and FNO. Subcommands run independently and resume by skipping finished tags.

    python scripts/s2_operator.py tune        # 8 configurations, picks one by validation loss
    python scripts/s2_operator.py final       # 3 seeds at 896, loss B, efficiency prefixes
    python scripts/s2_operator.py pod         # POD at every prefix
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from apno.data.predictions import write_predictions
from apno.data.sampling import PREFIXES
from apno.models.pod import PodModel
from apno.models.targets import Standardiser, from_targets, to_targets
from apno.models.train_fno import (
    FnoConfig, load_features_only, load_model, load_split, predict, train,
)

H5 = Path("data/hdf5/256x97.h5")
RUNS, PRED, RES = Path("runs/fno"), Path("runs/predictions"), Path("results/s2")
TUNING = [FnoConfig(width=w, modes_i=mi, modes_j=mj, epochs=100)
          for w, mi, mj in itertools.product((32, 48), (16, 24), (8, 12))]


def _predict_tests(model, std):
    out = {}
    for split in ("test_in", "test_ext"):
        f = load_features_only(H5, split)
        q, sec = predict(model, std, f["xy"], f["inputs"])
        out[split] = (f["case_id"], q, sec)
    ids = out["test_in"][0] + out["test_ext"][0]
    q = np.concatenate([out["test_in"][1], out["test_ext"][1]])
    return ids, q, float(np.mean([out[s][2] for s in out]))


def _run_fno(tag, cfg, variant=""):
    d = RUNS / tag
    res = (json.loads((d / "train.json").read_text()) if (d / "train.json").exists()
           else train(cfg, H5, d))
    model, std, _ = load_model(res["checkpoint"])
    ids, q, sec = _predict_tests(model, std)
    write_predictions(PRED / f"{tag}.h5", ids, q, {
        "method": "fno", "n_train": cfg.n_train, "seed": cfg.seed, "variant": variant,
        "train_wall_s": res["wall_s"], "train_cpu_s": res["cpu_s"], "query_s": sec})
    return {"tag": tag, "best_val": res["best_val"], "epochs_run": res["epochs_run"],
            "wall_s": res["wall_s"], "cpu_s": res["cpu_s"], "query_s": sec, "cfg": asdict(cfg)}


def tune():
    rows = [_run_fno(f"tune-{k}", c) for k, c in enumerate(TUNING)]
    best = min(rows, key=lambda r: r["best_val"])
    (RES / "tuning.json").write_text(json.dumps({"rows": rows, "selected": best["tag"]}, indent=2))
    print(json.dumps({r["tag"]: r["best_val"] for r in rows}, indent=2), "selected", best["tag"])


def final():
    sel = json.loads((RES / "tuning.json").read_text())["selected"]
    base = replace(TUNING[int(sel.split("-")[1])], epochs=300)
    rows = [_run_fno(f"fno-n896-s{s}", replace(base, seed=s)) for s in (0, 1, 2)]
    rows.append(_run_fno("fno-n896-s0-lossB", replace(base, loss="B"), variant="lossB"))
    rows += [_run_fno(f"fno-n{n}-s0", replace(base, n_train=n)) for n in PREFIXES[:-1]]
    (RES / "runs.json").write_text(json.dumps(rows, indent=2))


def pod():
    tr, va = load_split(H5, "train"), load_split(H5, "val")
    rows = []
    for n in PREFIXES:
        std = Standardiser().fit(to_targets(tr["q"][:n], tr["inputs"][:n]))
        y = std.transform(to_targets(tr["q"][:n], tr["inputs"][:n]))
        yv = to_targets(va["q"], va["inputs"])
        t0, c0 = time.perf_counter(), time.process_time()
        best = None
        for k, s in itertools.product((8, 16, 32, 64, 128, 256), (0.0, 1e-6, 1e-4, 1e-2)):
            if k >= n:
                continue
            m = PodModel(k, s).fit(tr["inputs"][:n], y)
            err = float(np.mean(np.linalg.norm((std.inverse(m.predict(va["inputs"])) - yv)
                                               .reshape(len(yv), 4, -1), axis=2)
                                / np.linalg.norm(yv.reshape(len(yv), 4, -1), axis=2)))
            if best is None or err < best[0]:
                best = (err, k, s, m)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        ids, qs, t_q = [], [], []
        for split in ("test_in", "test_ext"):
            f = load_features_only(H5, split)
            t1 = time.perf_counter()
            qs.append(from_targets(std.inverse(best[3].predict(f["inputs"])), f["inputs"]))
            t_q.append((time.perf_counter() - t1) / len(f["inputs"]))
            ids += f["case_id"]
        write_predictions(PRED / f"pod-n{n}.h5", ids, np.concatenate(qs), {
            "method": "pod", "n_train": n, "seed": 0, "variant": "",
            "train_wall_s": wall, "train_cpu_s": cpu, "query_s": float(np.mean(t_q))})
        rows.append({"n": n, "val_err": best[0], "modes": best[1], "smoothing": best[2],
                     "wall_s": wall})
        print(rows[-1])
    (RES / "pod.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("tune", "final", "pod"))
    RES.mkdir(parents=True, exist_ok=True)
    {"tune": tune, "final": final, "pod": pod}[ap.parse_args().what]()
```

Each `_run_fno` call loads the training arrays again, which costs seconds against hours of training. Accept that rather than caching state across calls.

- [ ] **Step 2: Smoke-test against a small HDF5 before the real runs**

```bash
python - <<'EOF'
import h5py
with h5py.File("data/hdf5/256x97.h5") as f:
    print({k: f[k]["q"].shape for k in f})
EOF
python scripts/s2_operator.py pod
```
Expected: the split sizes print (at most 896, 128, 64 and 16, minus failures). The POD run writes five prediction files and `results/s2/pod.json` within minutes.

- [ ] **Step 3: Run tuning, then the final runs, in the background**

```bash
nohup python scripts/s2_operator.py tune > results/s2/tune.log 2>&1 &
```
When it finishes, run the same with `final`. Expect hours per configuration on the CPU; spec §6.4 estimates about one second per step at full resolution, and training at 128 × 49 is about 4× cheaper. If the Apple GPU was available in Task 1, add `model.to("mps")` and move the tensors in `train` and `predict`. Record the device in the findings entry.

- [ ] **Step 4: Record S2 in `docs/findings.md`**

A dated entry, with the commands:
- the selected configuration and all eight validation losses;
- the epochs run and the CPU hours per run;
- the POD modes and smoothing chosen at each n.

Scores come later, in S4.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add scripts/s2_operator.py results/s2 docs/findings.md
git commit -m "S2: POD and FNO trained; predictions written for the test sets"
```

---

### Task 17: The Euler residual in O-grid coordinates

**Files:**
- Create: `src/apno/physics/euler.py`
- Test: `tests/test_euler.py`

**Interfaces:**
- Consumes: `map_xy`, `stretch_ratio`, `Section` (Task 3).
- Produces:
  - `residual(xi, eta, state_fn, geom, radius=100.0, ratio=None, viscosity=0.0) -> (N, 3)`. This is the physical residual of continuity and the two momentum equations, `div(F, G) − ν ∇²(ρ, ρu, ρv)`, evaluated at the points. `state_fn(xi, eta) -> (rho, u, v, p)` are torch tensors in PINN units: ρ/ρ∞, u/U∞, p/(ρ∞U∞²).
  - `metrics(xi, eta, geom, radius, ratio) -> (x, y, x_xi, x_eta, y_xi, y_eta, J)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import sympy as sp
import torch

from apno.geometry.ogrid import Section
from apno.physics.euler import metrics, residual

GEOM = Section(0.12, 0.03)

@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only; a module-level set_default_dtype leaks into every test."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)



def _pts(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.01 + 0.98 * torch.rand(n, generator=g), 1e-3 + (1 - 1e-3) * torch.rand(n, generator=g)


def test_free_stream_preserved_on_the_curved_grid():
    xi, eta = _pts()

    def state(xi, eta):
        z = 0 * xi + 0 * eta
        return 1 + z, 0.9 + z, 0.1 + z, 2.5 + z

    r = residual(xi, eta, state, GEOM)
    assert r.abs().max() < 1e-9


def _manufactured():
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 10) * sp.sin(x) * sp.cos(y)
    u = 1 + sp.Rational(1, 10) * sp.cos(x) * sp.sin(2 * y)
    v = sp.Rational(1, 10) * sp.sin(y) * sp.cos(x)
    p = 3 + sp.Rational(1, 10) * sp.cos(x + y)
    F = [rho * u, rho * u**2 + p, rho * u * v]
    G = [rho * v, rho * u * v, rho * v**2 + p]
    div = [sp.lambdify((x, y), sp.diff(F[k], x) + sp.diff(G[k], y)) for k in range(3)]
    cons = (rho, rho * u, rho * v)
    lap = [sp.lambdify((x, y), sp.diff(q, x, 2) + sp.diff(q, y, 2)) for q in cons]
    fields = [sp.lambdify((x, y), f, "math") for f in (rho, u, v, p)]
    return div, lap, fields


def test_manufactured_solution():
    div, lap, _ = _manufactured()
    xi, eta = _pts(200, 1)
    eta = eta * 0.05  # keep x, y within a few chords so the fields stay O(1)

    def state(xi, eta):
        x, y, *_ = metrics(xi, eta, GEOM)
        rho = 1 + 0.1 * torch.sin(x) * torch.cos(y)
        u = 1 + 0.1 * torch.cos(x) * torch.sin(2 * y)
        v = 0.1 * torch.sin(y) * torch.cos(x)
        p = 3 + 0.1 * torch.cos(x + y)
        return rho, u, v, p

    nu = 1e-2
    r = residual(xi, eta, state, GEOM, viscosity=nu).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    want = np.stack([div[k](x, y) - nu * lap[k](x, y) for k in range(3)], 1)
    np.testing.assert_allclose(r, want, rtol=1e-8, atol=1e-8)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_euler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/physics/euler.py`**

```python
"""Steady 2D Euler residual in the O-grid's coordinates (xi, eta), by automatic differentiation.

Strong conservation form: div(F, G) = (1/J) [d/dxi (y_eta F - x_eta G) + d/deta (-y_xi F + x_xi G)],
with J = x_xi y_eta - x_eta y_xi. Because the metric terms come from autograd of one smooth map,
the mixed partials cancel exactly and a uniform flow gives zero residual (free-stream preservation).
The optional artificial viscosity subtracts nu * Laplacian of (rho, rho u, rho v), built the same
way.
Only continuity and momentum are solved: total enthalpy is constant and fixes the pressure.
"""

import torch

from apno.geometry.ogrid import FAR_RADIUS, map_xy, stretch_ratio


def _d(f, wrt):
    return torch.autograd.grad(f.sum(), wrt, create_graph=True)[0]


def metrics(xi, eta, geom, radius=FAR_RADIUS, ratio=None):
    if not xi.requires_grad:
        xi = xi.clone().requires_grad_(True)
    if not eta.requires_grad:
        eta = eta.clone().requires_grad_(True)
    ratio = stretch_ratio(radius) if ratio is None else ratio
    x, y = map_xy(xi, eta, geom, radius, ratio, xp=torch)
    x_xi, x_eta, y_xi, y_eta = _d(x, xi), _d(x, eta), _d(y, xi), _d(y, eta)
    return x, y, x_xi, x_eta, y_xi, y_eta, x_xi * y_eta - x_eta * y_xi


def residual(xi, eta, state_fn, geom, radius=FAR_RADIUS, ratio=None, viscosity=0.0):
    xi = xi.clone().requires_grad_(True)
    eta = eta.clone().requires_grad_(True)
    _, _, x_xi, x_eta, y_xi, y_eta, J = metrics(xi, eta, geom, radius, ratio)
    rho, u, v, p = state_fn(xi, eta)
    F = (rho * u, rho * u * u + p, rho * u * v)
    G = (rho * v, rho * u * v, rho * v * v + p)
    out = []
    for f, g, q in zip(F, G, (rho, rho * u, rho * v)):
        div = (_d(y_eta * f - x_eta * g, xi) + _d(-y_xi * f + x_xi * g, eta)) / J
        if viscosity:
            q_xi, q_eta = _d(q, xi), _d(q, eta)
            q_x = (q_xi * y_eta - q_eta * y_xi) / J
            q_y = (-q_xi * x_eta + q_eta * x_xi) / J
            lap = (_d(y_eta * q_x - x_eta * q_y, xi) + _d(-y_xi * q_x + x_xi * q_y, eta)) / J
            div = div - viscosity * lap
        out.append(div)
    return torch.stack(out, -1)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_euler.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/physics/euler.py tests/test_euler.py
git commit -m "Physics: Euler residual in grid coordinates, free-stream preserving, with viscosity"
```

---

### Task 18: The PINN model

**Files:**
- Create: `src/apno/models/pinn.py`
- Test: `tests/test_pinn.py`

**Interfaces:**
- Consumes: `Section`, `Cylinder`, `FAR_RADIUS` (Task 3); `GAMMA`.
- Produces:
  - `PinnCase(mach, alpha_deg, geometry, radius=FAR_RADIUS)`, frozen. It has exactly these four fields and no flow data.
  - `PinnNet(hidden=96, depth=7)`.
  - `wall_normal(xi, geom) -> (nx, ny)`: the outward unit normal at the wall point ξ.
  - `Pinn(case, net)`, with `.state(xi, eta) -> (rho, u, v, p)` in PINN units, `.h_inf`, `.double()`, `.parameters()`.
  - `to_dataset_q(rho, u, v, p, mach) -> (…, 4)`, giving ρ/ρ∞, u/U∞, v/U∞, p/p∞.

- [ ] **Step 1: Write the failing tests**

```python
import dataclasses

import numpy as np
import torch

from apno.constants import GAMMA
from apno.geometry.ogrid import Cylinder, Section
import pytest

from apno.models.pinn import Pinn, PinnCase, PinnNet, to_dataset_q, wall_normal


@pytest.fixture(autouse=True)
def _float64():
    """float64 inside this module only; a module-level set_default_dtype leaks into every test."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pinn(geom=Section(0.12, 0.02)):
    torch.manual_seed(0)
    return Pinn(PinnCase(0.6, 3.0, geom), PinnNet(hidden=16, depth=2))


def test_no_flow_through_the_wall_by_construction():
    p = _pinn()
    xi = torch.linspace(0, 1, 257)[:-1]
    _, u, v, _ = p.state(xi, torch.zeros_like(xi))
    nx, ny = wall_normal(xi, p.case.geometry)
    assert (u * nx + v * ny).abs().max() < 1e-12


def test_total_enthalpy_constant_by_construction():
    p = _pinn()
    xi, eta = torch.rand(100), torch.rand(100)
    rho, u, v, pr = p.state(xi, eta)
    h = GAMMA / (GAMMA - 1) * pr / rho + 0.5 * (u * u + v * v)
    torch.testing.assert_close(h, torch.full_like(h, p.h_inf), rtol=1e-12, atol=1e-12)


def test_normals_point_outward_on_the_cylinder():
    xi = torch.linspace(0, 1, 65)[:-1]
    nx, ny = wall_normal(xi, Cylinder())
    th = -2 * np.pi * xi
    torch.testing.assert_close(nx, torch.cos(th), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ny, torch.sin(th), rtol=1e-12, atol=1e-12)


def test_section_normal_finite_at_trailing_edge_and_matches_finite_differences():
    sec = Section(0.12, 0.03)
    xi = torch.cat([torch.zeros(1), torch.linspace(0.02, 0.98, 49)])
    nx, ny = wall_normal(xi, sec)
    assert torch.isfinite(nx).all() and torch.isfinite(ny).all()
    assert nx[0] > 0  # the trailing-edge normal points aft, out of the section
    h = 1e-6
    xa, ya = sec.wall(xi[1:] - h, xp=torch)
    xb, yb = sec.wall(xi[1:] + h, xp=torch)
    tx, ty = xb - xa, yb - ya
    n = torch.sqrt(tx * tx + ty * ty)
    torch.testing.assert_close(nx[1:], -ty / n, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ny[1:], tx / n, rtol=1e-6, atol=1e-6)


def test_case_carries_no_flow_data():
    assert {f.name for f in dataclasses.fields(PinnCase)} == {"mach", "alpha_deg", "geometry",
                                                              "radius"}


def test_dataset_units():
    M = 0.5
    q = to_dataset_q(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0),
                     torch.tensor(1 / (GAMMA * M * M)), M)
    torch.testing.assert_close(q, torch.tensor([1.0, 1.0, 0.0, 1.0]))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_pinn.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/models/pinn.py`**

```python
"""The PINN (spec 7): one case at a time, from the equations, boundary conditions and inputs only.

Units: rho/rho_inf, u/U_inf, p/(rho_inf U_inf^2). Exact by construction: positive density
(exponential output), constant total enthalpy (pressure from H_inf), and no flow through the wall
(normal component removed with weight (1 - eta)^4: 1 at the wall, 0 at the outer circle).
The network sees (cos 2 pi xi, sin 2 pi xi, eta), so periodicity around the airfoil is built in.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from apno.constants import GAMMA
from apno.geometry.ogrid import FAR_RADIUS

G = GAMMA


@dataclass(frozen=True)
class PinnCase:
    mach: float
    alpha_deg: float
    geometry: object
    radius: float = FAR_RADIUS


class PinnNet(nn.Module):
    def __init__(self, hidden=96, depth=7):
        super().__init__()
        layers, width = [], 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xi, eta):
        f = torch.stack([torch.cos(2 * math.pi * xi), torch.sin(2 * math.pi * xi), eta], -1)
        return self.mlp(f)


def wall_normal(xi, geom):
    """Outward unit normal at wall point xi; differentiable in xi for the residual.

    A section is differentiated in s = cos(pi xi), because d/dxi of the cosine-clustered curve is
    exactly zero at the trailing edge (0/0 there). Since ds/dxi < 0 on (0, 1), the xi-tangent is
    -(x_s, y_s) and its left (outward) normal is (y_s, -x_s). At xi = 0 this is the lower surface's
    side of the trailing-edge corner, which is the named hazard of spec 7.7.
    """
    with torch.enable_grad():
        x = xi if xi.requires_grad else xi.detach().clone().requires_grad_(True)
        if hasattr(geom, "wall_s"):
            s = torch.cos(math.pi * x)
            xw, yw = geom.wall_s(s, xp=torch)
            xs = torch.autograd.grad(xw.sum(), s, create_graph=True)[0]
            ys = torch.autograd.grad(yw.sum(), s, create_graph=True)[0]
            nx, ny = ys, -xs
        else:
            xw, yw = geom.wall(x, xp=torch)
            dx = torch.autograd.grad(xw.sum(), x, create_graph=True)[0]
            dy = torch.autograd.grad(yw.sum(), x, create_graph=True)[0]
            nx, ny = -dy, dx  # left normal of the clockwise wall = outward
    norm = torch.sqrt(nx * nx + ny * ny)
    return nx / norm, ny / norm


def to_dataset_q(rho, u, v, p, mach):
    return torch.stack([rho, u, v, p * G * mach * mach], -1)


class Pinn:
    def __init__(self, case, net):
        self.case, self.net = case, net
        self.h_inf = 1.0 / ((G - 1.0) * case.mach**2) + 0.5
        a = math.radians(case.alpha_deg)
        self.ca, self.sa = math.cos(a), math.sin(a)

    def parameters(self):
        return self.net.parameters()

    def double(self):
        self.net.double()
        return self

    def state(self, xi, eta):
        out = self.net(xi, eta)
        rho = torch.exp(out[..., 0])
        u, v = self.ca + out[..., 1], self.sa + out[..., 2]
        nx, ny = wall_normal(xi, self.case.geometry)
        phi = (1.0 - eta) ** 4
        un = u * nx + v * ny
        u, v = u - phi * un * nx, v - phi * un * ny
        p = (G - 1.0) / G * rho * (self.h_inf - 0.5 * (u * u + v * v))
        return rho, u, v, p
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_pinn.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/models/pinn.py tests/test_pinn.py
git commit -m "Models: PINN with exact wall tangency, constant enthalpy, positive density"
```

---

### Task 19: PINN training and the cylinder warm-up (stage S3a)

**Files:**
- Create: `src/apno/models/train_pinn.py`, `scripts/s3_warmup.py`, `results/s3/` (output)
- Test: `tests/test_train_pinn.py`

**Interfaces:**
- Consumes: `residual` (Task 17); `Pinn`, `PinnCase`, `PinnNet`, `wall_normal`, `to_dataset_q` (Task 18); `grid`, `stretch_ratio` (Task 3).
- Produces:
  - `PinnConfig(hidden=96, depth=7, n_interior=20000, resample_every=1000, n_outer=1000, n_wall=512, adam_steps=30000, lr0=1e-3, lr1=1e-4, lbfgs_iters=5000, w_res=1.0, w_outer=10.0, w_kutta=10.0, viscosity=0.0, kutta=True, zero_circulation=False, seed=0)`.
  - `losses(pinn, pts, cfg, ratio) -> dict[str, Tensor]`, with keys `residual`, `outer`, `kutta`, `circulation` and `total`.
  - `train(case, cfg, log=print) -> (pinn, record)`, where `record` holds `wall_s`, `cpu_s`, `history` and `final`.
  - `evaluate_on_grid(pinn, shape) -> np.ndarray (NI, NJ, 4)`, in dataset units.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from apno.geometry.ogrid import Cylinder, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, evaluate_on_grid, train


def test_tiny_training_reduces_the_loss_and_evaluates():
    cfg = PinnConfig(hidden=16, depth=2, n_interior=256, resample_every=50, n_outer=64,
                     n_wall=64, adam_steps=100, lbfgs_iters=10)
    pinn, rec = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, log=lambda *a: None)
    hist = rec["history"]
    assert hist[-1]["total"] < hist[0]["total"]
    assert np.isfinite(rec["final"]["total"]) and rec["cpu_s"] > 0
    q = evaluate_on_grid(pinn, (32, 9))
    assert q.shape == (32, 9, 4) and np.isfinite(q).all()


def test_zero_circulation_variant_runs():
    cfg = PinnConfig(hidden=8, depth=2, n_interior=128, resample_every=50, n_outer=32, n_wall=32,
                     adam_steps=20, lbfgs_iters=2, kutta=False, zero_circulation=True)
    _, rec = train(PinnCase(0.1, 0.0, Cylinder()), cfg, log=lambda *a: None)
    assert "circulation" in rec["final"] and rec["final"]["kutta"] == 0.0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_train_pinn.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/models/train_pinn.py`**

```python
"""PINN training (spec 7.5): Adam with exponential decay, then L-BFGS in float64.

Points are uniform in (xi, eta), which the map turns into clustering near the wall. The Kutta
condition is a penalty on the velocity at the trailing-edge wall point (xi, eta) = (0, 0); the
cylinder warm-up replaces it with zero circulation, since a cylinder has no sharp edge to fix it.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from apno.geometry.ogrid import stretch_ratio
from apno.models.pinn import Pinn, PinnNet, to_dataset_q
from apno.physics.euler import residual


@dataclass
class PinnConfig:
    hidden: int = 96
    depth: int = 7
    n_interior: int = 20000
    resample_every: int = 1000
    n_outer: int = 1000
    n_wall: int = 512
    adam_steps: int = 30000
    lr0: float = 1e-3
    lr1: float = 1e-4
    lbfgs_iters: int = 5000
    w_res: float = 1.0
    w_outer: float = 10.0
    w_kutta: float = 10.0
    viscosity: float = 0.0
    kutta: bool = True
    zero_circulation: bool = False
    seed: int = 0


def _points(cfg, gen, dtype):
    xi = torch.rand(cfg.n_interior, generator=gen, dtype=dtype)
    eta = 1.0 - torch.rand(cfg.n_interior, generator=gen, dtype=dtype)  # (0, 1]
    xo = torch.rand(cfg.n_outer, generator=gen, dtype=dtype)
    xw = torch.arange(cfg.n_wall, dtype=dtype) / cfg.n_wall
    return {"xi": xi, "eta": eta, "xi_outer": xo, "xi_wall": xw}


def losses(pinn, pts, cfg, ratio):
    case = pinn.case
    r = residual(pts["xi"], pts["eta"], pinn.state, case.geometry, case.radius, ratio,
                 cfg.viscosity)
    out = {"residual": (r * r).sum(-1).mean()}
    xo = pts["xi_outer"]
    rho, u, v, _ = pinn.state(xo, torch.ones_like(xo))
    out["outer"] = ((rho - 1) ** 2 + (u - pinn.ca) ** 2 + (v - pinn.sa) ** 2).mean()
    zero = torch.zeros(1, dtype=xo.dtype)
    if cfg.kutta:
        _, uk, vk, _ = pinn.state(zero, zero)
        out["kutta"] = (uk * uk + vk * vk).sum()
    else:
        out["kutta"] = zero.sum()
    if cfg.zero_circulation:
        xw = pts["xi_wall"].clone().requires_grad_(True)
        _, uw, vw, _ = pinn.state(xw, torch.zeros_like(xw))
        x, y = case.geometry.wall(xw, xp=torch)
        dx = torch.autograd.grad(x.sum(), xw, create_graph=True)[0]
        dy = torch.autograd.grad(y.sum(), xw, create_graph=True)[0]
        gamma = (uw * dx + vw * dy).mean()
        out["circulation"] = gamma * gamma
    else:
        out["circulation"] = zero.sum()
    out["total"] = (cfg.w_res * out["residual"] + cfg.w_outer * out["outer"]
                    + cfg.w_kutta * (out["kutta"] + out["circulation"]))
    return out


def _snapshot(L):
    return {k: float(v.detach()) for k, v in L.items()}


def train(case, cfg, log=print):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ratio = stretch_ratio(case.radius)
    pinn = Pinn(case, PinnNet(cfg.hidden, cfg.depth).float())
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg.lr1 / cfg.lr0) ** (1 / cfg.adam_steps))
    history = []
    t0, c0 = time.perf_counter(), time.process_time()
    pts = None
    for step in range(cfg.adam_steps):
        if step % cfg.resample_every == 0:
            pts = _points(cfg, gen, torch.float32)
        opt.zero_grad()
        L = losses(pinn, pts, cfg, ratio)
        L["total"].backward()
        opt.step()
        sched.step()
        if step % max(1, cfg.adam_steps // 60) == 0 or step == cfg.adam_steps - 1:
            history.append({"step": step, **_snapshot(L)})
            log(f"adam {step}: {history[-1]}")
    pinn.double()
    pts = {k: v.double() for k, v in pts.items()}
    lbfgs = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        lbfgs.zero_grad()
        total = losses(pinn, pts, cfg, ratio)["total"]
        total.backward()
        return total

    lbfgs.step(closure)
    final = _snapshot(losses(pinn, pts, cfg, ratio))
    history.append({"step": "lbfgs", **final})
    log(f"final: {final}")
    rec = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "history": history, "final": final}
    return pinn, rec


def evaluate_on_grid(pinn, shape):
    ni, nj = shape
    xi = (torch.arange(ni, dtype=torch.float64) / ni)[:, None].expand(ni, nj).reshape(-1)
    eta = (torch.arange(nj, dtype=torch.float64) / (nj - 1))[None, :].expand(ni, nj).reshape(-1)
    pinn.double()
    with torch.no_grad():
        rho, u, v, p = pinn.state(xi, eta)
        q = to_dataset_q(rho, u, v, p, pinn.case.mach)
    return q.reshape(ni, nj, 4).detach().numpy()


def circle_cp_error(q, mach):
    """For the warm-up: max |Cp - (1 - 4 sin^2 theta)| on the cylinder's wall row."""
    ni = q.shape[0]
    th = -2 * math.pi * np.arange(ni) / ni
    cp = (q[:, 0, 3] - 1.0) / (0.5 * 1.4 * mach * mach)
    return float(np.abs(cp - (1 - 4 * np.sin(th) ** 2)).max()), cp
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_train_pinn.py -v`
Expected: 2 passed, within a minute or two on the CPU.

- [ ] **Step 5: Write `scripts/s3_warmup.py` and run it**

```python
"""Stage S3a: the cylinder warm-up (spec 7.6). Exits 1 if it fails its pass criterion.

Pass: max |Cp - (1 - 4 sin^2 theta)| <= 0.1 around the cylinder at Mach 0.1, and lift and drag zero
to the scorer's resolution. Writes results/s3/warmup.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from apno.geometry.ogrid import TRAIN, Cylinder, grid
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, circle_cp_error, evaluate_on_grid, train
from apno.score.forces import wall_forces


def main():
    case = PinnCase(0.1, 0.0, Cylinder())
    cfg = PinnConfig(kutta=False, zero_circulation=True)
    pinn, rec = train(case, cfg)
    q = evaluate_on_grid(pinn, TRAIN)
    err, cp = circle_cp_error(q, case.mach)
    X, Y = grid(Cylinder(), TRAIN)
    cl, cd, cm = wall_forces(X[:, 0], Y[:, 0], cp, 0.0)
    passed = err <= 0.1 and abs(cl) < 1e-2 and abs(cd) < 1e-2
    out = {"max_cp_error": err, "cl": cl, "cd": cd, "cm": cm, "passed": passed,
           "wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"], "final_losses": rec["final"]}
    Path("results/s3").mkdir(parents=True, exist_ok=True)
    Path("results/s3/warmup.json").write_text(json.dumps(out, indent=2))
    np.save("runs/warmup_q.npy", q)
    print(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

The warm-up's lift and drag tolerance is 1 × 10⁻². The cylinder's force integrator itself is exact to 10⁻¹³ (Task 7), so anything above that is the PINN.

Run: `mkdir -p runs && nohup python scripts/s3_warmup.py > results/s3/warmup.log 2>&1 &`, then wait for it to exit.
Expected: `"passed": true`. If it fails, do not tune toward the criterion. Record the error, the losses and the time in findings, then use superpowers:systematic-debugging, starting with the residual on the converged network and whether the outer penalty is satisfied. The criterion is fixed.

- [ ] **Step 6: Record and commit**

Add a findings entry with the command, `results/s3/warmup.json`, and the measured CPU time per case. That time replaces the spec's estimate of one to two hours in the cost projection for S3.

```bash
ruff check . && git add src/apno/models/train_pinn.py scripts/s3_warmup.py tests/test_train_pinn.py results/s3 docs/findings.md
git commit -m "S3a: PINN training and the cylinder warm-up against potential flow"
```

---

### Task 20: Stage S3, the PINN showcase runs

**Files:**
- Create: `scripts/s3_pinn.py`
- Modify: `docs/findings.md`

**Interfaces:**
- Consumes: `results/s1/showcase.json` (Task 13); `train`, `evaluate_on_grid`, `PinnConfig` (Task 19); `case_specs` (Task 10); `write_predictions` (Task 15).
- Produces prediction files `runs/predictions/pinn-<variant>-<case_id>.h5`, with attrs `method` (`pinn`), `variant`, `seed` (0), `n_train` (0), `train_wall_s`, `train_cpu_s` and `query_s`; and `results/s3/pinn_runs.json`.

The runs, all fixed before any of them runs:

| Variant | Cases | Settings |
|---|---|---|
| `euler` | all six showcase cases | `kutta=True`, `viscosity=0` |
| `visc1e-3` | supercritical showcase cases | `viscosity=1e-3` |
| `visc3e-4` | the supercritical showcase case with the highest max surface Mach | `viscosity=3e-4` |
| `nokutta` | the first subcritical showcase case in showcase order | `kutta=False` |

- [ ] **Step 1: Write `scripts/s3_pinn.py`**

```python
"""Stage S3: the PINN on the showcase cases (spec 7.3-7.4). Resumes by skipping finished files."""

import json
from pathlib import Path

from apno.data.predictions import write_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, evaluate_on_grid, train

PRED, RES = Path("runs/predictions"), Path("results/s3")


def plan():
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    sup = [c for c in show if c["regime"] == "supercritical"]
    sub = [c for c in show if c["regime"] == "subcritical"]
    runs = [("euler", c["case_id"], PinnConfig()) for c in show]
    runs += [("visc1e-3", c["case_id"], PinnConfig(viscosity=1e-3)) for c in sup]
    if sup:
        top = max(sup, key=lambda c: c["max_surface_mach"])["case_id"]
        runs.append(("visc3e-4", top, PinnConfig(viscosity=3e-4)))
    if sub:
        runs.append(("nokutta", sub[0]["case_id"], PinnConfig(kutta=False)))
    return runs


def main():
    specs = {s.case_id: s for s in case_specs("test_in", TRAIN)}
    RES.mkdir(parents=True, exist_ok=True)
    record_path = RES / "pinn_runs.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for variant, cid, cfg in plan():
        path = PRED / f"pinn-{variant}-{cid}.h5"
        if path.exists():
            continue
        s = specs[cid]
        pinn, rec = train(PinnCase(s.mach, s.alpha_deg, Section(s.t, s.m)), cfg)
        q = evaluate_on_grid(pinn, TRAIN)
        write_predictions(path, [cid], q[None], {
            "method": "pinn", "variant": variant, "seed": cfg.seed, "n_train": 0,
            "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"], "query_s": rec["wall_s"]})
        record[f"{variant}/{cid}"] = {"wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"],
                                      "final": rec["final"], "history": rec["history"]}
        record_path.write_text(json.dumps(record, indent=2))
        print(variant, cid, rec["final"], flush=True)


if __name__ == "__main__":
    main()
```

For the PINN, `query_s` equals its training time, because answering a new case means training on it.

- [ ] **Step 2: Run it in the background**

Run: `nohup python scripts/s3_pinn.py > results/s3/pinn.log 2>&1 &`
Expected: one prediction file per planned run, each at the warm-up's measured cost. Several runs can go in parallel only if memory allows: start a second process with the same command after the first case finishes. Two processes skip each other's finished files, but they could both start the same unfinished case, so give the second process a reversed plan (`plan()[::-1]`) to avoid that.

- [ ] **Step 3: Record and commit**

Add a findings entry: the command, the CPU time per run, the final losses per run, and every run that failed to converge, stated as a result.
```bash
ruff check . && git add scripts/s3_pinn.py results/s3 docs/findings.md
git commit -m "S3: PINN showcase runs (Euler, artificial viscosity, no-Kutta)"
```

---

### Task 21: The metrics

**Files:**
- Create: `src/apno/score/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `wall_forces`, `vertex_normals` (Task 7); `cell_areas` (Task 3); flow functions (Task 6).
- Produces:
  - `node_areas(X, Y) -> (NI, NJ)`.
  - `shocks(xw, cp, mach) -> dict` mapping `lower` and `upper` to `{"x": float, "mach_ahead": float} | None`.
  - `score_case(pred_q, true_q, X, Y, mach, alpha_deg, truth_regime) -> dict`. `pred_q` may be `None`, which gives `{"status": "missing"}`; NaN in it gives `{"status": "non-finite"}`. Otherwise the dict holds `status="ok"` and these keys: `cl_err, cd_err_counts, cm_err, cl_true, cd_true_counts, cm_true, cp_l2, cp_max, near_field_rel_l2, shock, physics`. `shock` holds a per-surface position error, the Mach ahead and the overshoot. `physics` holds `entropy_max, entropy_mean, enthalpy_max_dev, wall_normal_rms` and `mass_flux` (a dict keyed by j).
  - `physics_checks(q, X, Y, mach, near) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import freestream_q
from apno.score.metrics import node_areas, score_case, shocks

X, Y = grid(Section(0.12, 0.0), TRAIN)


def _free(alpha=2.0):
    return np.broadcast_to(freestream_q(alpha), X.shape + (4,)).copy()


def test_node_areas_sum_to_the_annulus():
    total = node_areas(X, Y).sum()
    airfoil = 0.685 * 0.12  # NACA 4-digit area is about 0.685 t c
    assert total == pytest.approx(np.pi * 100.0**2 - airfoil, rel=1e-3)


def test_identical_fields_score_zero():
    q = _free()
    s = score_case(q, q, X, Y, 0.6, 2.0, "subcritical")
    assert s["status"] == "ok"
    assert s["cl_err"] == 0 and s["cd_err_counts"] == 0 and s["cp_max"] == 0
    assert s["near_field_rel_l2"] == 0.0


def test_free_stream_physics_is_clean():
    s = score_case(_free(), _free(), X, Y, 0.6, 2.0, "subcritical")["physics"]
    assert abs(s["entropy_max"]) < 1e-14 and s["enthalpy_max_dev"] < 1e-14
    assert all(abs(v) < 1e-10 for v in s["mass_flux"].values())


def test_shock_found_on_a_synthetic_step():
    x = np.linspace(0, 1, 101)
    cp = np.where(x < 0.6, -1.0 + 0.2 * x, 0.1)  # suction plateau, then a jump at 0.6
    xw = np.concatenate([x[::-1], x[1:-1]])  # lower TE->LE then upper LE->TE
    cpw = np.concatenate([np.full(101, -0.2), cp[1:-1]])
    out = shocks(xw, cpw, 0.75)
    assert out["upper"]["x"] == pytest.approx(0.595, abs=0.006)
    assert out["lower"] is None


def test_missing_and_nonfinite_predictions():
    q = _free()
    assert score_case(None, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "missing"}
    bad = q.copy()
    bad[3, 4, 1] = np.nan
    assert score_case(bad, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "non-finite"}
```

In `test_shock_found_on_a_synthetic_step`, the wall array has 200 nodes (index 100 is the leading edge) and the lower surface's Cp is flat at −0.2, which is subsonic at M = 0.75 (isentropic M ≈ 0.86 < 1), so no shock is reported there. On the upper surface the plateau Cp ≈ −0.88 is supersonic, and the jump sits between x = 0.59 and 0.60.

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/score/metrics.py`**

```python
"""What is scored (spec 8.2). One function per quantity; score_case runs them all on one case.

The near field is nodes within 2.5 chords of mid-chord, weighted by dual-cell area. Drag is in
counts and never a percentage. Shocks are located from wall pressure, per surface, only where that
surface goes supersonic. Physics checks run on every contestant, SU2 included.
"""

import numpy as np

from apno.geometry.ogrid import cell_areas
from apno.physics.flow import (
    entropy, freestream_q, isentropic_mach_from_cp, pressure_coefficient, total_enthalpy_ratio,
)
from apno.score.forces import vertex_normals, wall_forces

NEAR_RADIUS = 2.5
FLUX_LINES = (8, 24, 48, 72)
OVERSHOOT_WINDOW = 0.05


def node_areas(X, Y):
    A = cell_areas(X, Y)
    w = np.zeros_like(X)
    w[:, :-1] += A / 4 + np.roll(A, 1, 0) / 4
    w[:, 1:] += A / 4 + np.roll(A, 1, 0) / 4
    return w


def _surfaces(ni):
    le = ni // 2
    return {"lower": np.arange(le, -1, -1), "upper": np.r_[np.arange(le, ni), 0]}  # LE -> TE


def shocks(xw, cp, mach):
    out = {}
    for name, idx in _surfaces(len(xw)).items():
        x, c = xw[idx], cp[idx]
        m_loc = isentropic_mach_from_cp(c, mach)
        if not np.nanmax(m_loc) > 1.0:
            out[name] = None
            continue
        k0 = int(np.argmin(c))
        dx = np.diff(x[k0:])
        slope = np.where(dx > 0, np.diff(c[k0:]) / np.where(dx > 0, dx, 1), -np.inf)
        k = k0 + int(np.argmax(slope))
        out[name] = {"x": float(0.5 * (x[k] + x[k + 1])), "mach_ahead": float(m_loc[k])}
    return out


def _mass_flux(q, X, Y, j):
    n = vertex_normals(X[:, j], Y[:, j])
    rho, u, v = q[:, j, 0], q[:, j, 1], q[:, j, 2]
    return float(np.sum(rho * (u * n[:, 0] + v * n[:, 1])))


def physics_checks(q, X, Y, mach, near):
    w = node_areas(X, Y)[near]
    s = entropy(q[..., 0], q[..., 3])[near]
    h = total_enthalpy_ratio(*(q[..., k] for k in range(4)), mach)[near]
    n = vertex_normals(X[:, 0], Y[:, 0])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    un = q[:, 0, 1] * n[:, 0] + q[:, 0, 2] * n[:, 1]
    return {"entropy_max": float(np.max(s)), "entropy_mean": float(np.sum(w * s) / np.sum(w)),
            "enthalpy_max_dev": float(np.max(np.abs(h - 1))),
            "wall_normal_rms": float(np.sqrt(np.mean(un * un))),
            "mass_flux": {j: _mass_flux(q, X, Y, j) for j in FLUX_LINES}}


def score_case(pred_q, true_q, X, Y, mach, alpha_deg, truth_regime):
    if pred_q is None:
        return {"status": "missing"}
    if not np.isfinite(pred_q).all():
        return {"status": "non-finite"}
    xw, yw = X[:, 0], Y[:, 0]
    cp_p = pressure_coefficient(pred_q[:, 0, 3], mach)
    cp_t = pressure_coefficient(true_q[:, 0, 3], mach)
    fp, ft = wall_forces(xw, yw, cp_p, alpha_deg), wall_forces(xw, yw, cp_t, alpha_deg)
    near = np.hypot(X - 0.5, Y) <= NEAR_RADIUS
    w = node_areas(X, Y)[near][:, None]
    q_inf = freestream_q(alpha_deg)
    dp, dt = pred_q[near] - q_inf, true_q[near] - q_inf
    den = np.sqrt(np.sum(w * dt * dt))
    out = {"status": "ok",
           "cl_err": fp[0] - ft[0], "cd_err_counts": 1e4 * (fp[1] - ft[1]), "cm_err": fp[2] - ft[2],
           "cl_true": ft[0], "cd_true_counts": 1e4 * ft[1], "cm_true": ft[2],
           "cp_l2": float(np.sqrt(np.mean((cp_p - cp_t) ** 2))),
           "cp_max": float(np.abs(cp_p - cp_t).max()),
           "near_field_rel_l2": float(np.sqrt(np.sum(w * (dp - dt) ** 2)) / den) if den else 0.0,
           "physics": physics_checks(pred_q, X, Y, mach, near)}
    if truth_regime == "supercritical":
        sp, st = shocks(xw, cp_p, mach), shocks(xw, cp_t, mach)
        res = {}
        for surf, t in st.items():
            if t is None:
                continue
            idx = _surfaces(len(xw))[surf]
            win = idx[np.abs(xw[idx] - t["x"]) <= OVERSHOOT_WINDOW]
            p = sp[surf]
            res[surf] = {"x_true": t["x"], "x_err": None if p is None else p["x"] - t["x"],
                         "missed": p is None, "mach_ahead_true": t["mach_ahead"],
                         "mach_ahead_err": None if p is None else p["mach_ahead"] - t["mach_ahead"],
                         "overshoot": float(np.abs(cp_p[win] - cp_t[win]).max())}
        out["shock"] = res
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 5 passed. If `test_node_areas_sum_to_the_annulus` misses at rel 10⁻³, print the two numbers before changing anything. The node-area construction must sum exactly to the cell areas, so the difference should come only from the polygon approximating the circle and the airfoil area estimate.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/apno/score/metrics.py tests/test_metrics.py
git commit -m "Scoring: forces, surface, near-field, shocks and physics checks per case"
```

---

### Task 22: Stage S4, scoring every contestant, plots and the cost record

**Files:**
- Create: `src/apno/score/report.py`, `scripts/s4_score.py`, `results/s4/` (output)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `score_case` (Task 21); `read_predictions` (Task 15); the HDF5 files (Task 13); `results/s1/campaign_train.json` and the `result.json` files for costs.
- Produces:
  - `aggregate(rows) -> list[dict]`. Each input row is `{method, variant, n_train, seed, split, regime, case_id, score}`; the output gives the mean, median and max of each scalar metric per (method, variant, n_train, seed, split, regime), plus counts of `missing` and `non-finite`.
  - `break_even(costs, n_max) -> dict[str, np.ndarray]`.
  - `results/s4/scores.json`, `scores.md`, `break_even.png`, `data_efficiency.png`, `showcase_<case_id>.png` and `costs.json`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from apno.score.report import aggregate, break_even


def _row(method, val, status="ok"):
    score = {"status": status}
    if status == "ok":
        score.update({"cl_err": val, "near_field_rel_l2": abs(val)})
    return {"method": method, "variant": "", "n_train": 896, "seed": 0, "split": "test_in",
            "regime": "subcritical", "case_id": "x", "score": score}


def test_aggregate_counts_failures_and_keeps_signs():
    out = aggregate([_row("fno", -0.2), _row("fno", 0.1), _row("fno", 0, "non-finite")])
    (g,) = out
    assert g["n_ok"] == 2 and g["n_non-finite"] == 1
    assert np.isclose(g["cl_err_mean"], -0.05) and np.isclose(g["cl_err_absmax"], 0.2)


def test_break_even():
    costs = {"su2": {"fixed": 0.0, "per": 10.0}, "fno": {"fixed": 1000.0, "per": 0.01}}
    curves = break_even(costs, 200)
    assert curves["su2"][100] == 1000.0
    assert np.argmax(curves["fno"] < curves["su2"]) == 101
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/apno/score/report.py`**

```python
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
```

- [ ] **Step 4: Implement `scripts/s4_score.py`**

```python
"""Stage S4: score every prediction file, the floor and SU2 itself; write tables, plots, costs."""

import json
from pathlib import Path

import h5py
import numpy as np

from apno.data.build import regime
from apno.data.predictions import read_predictions
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import pressure_coefficient
from apno.score.metrics import score_case
from apno.score.report import (
    aggregate, break_even, markdown_table, plot_break_even, plot_efficiency, plot_showcase,
)

OUT = Path("results/s4")


def _group(path, split):
    with h5py.File(path) as f:
        g = f[split]
        return dict(zip(g["case_id"].asstr()[...], g["q"][...])), dict(
            zip(g["case_id"].asstr()[...], g["max_surface_mach"][...]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    truth, floor, regimes = {}, {}, {}
    for split in ("test_in", "test_ext"):
        t, mach = _group("data/hdf5/512x193.h5", split)
        fl, _ = _group("data/hdf5/256x97.h5", split)
        for cid in t:
            if cid in fl:
                truth[cid], floor[cid] = t[cid][::2, ::2], fl[cid]
                regimes[cid] = regime(mach[cid])
    contestants = [({"method": "su2-256x97", "variant": "floor", "n_train": 0, "seed": 0}, floor)]
    for p in sorted(Path("runs/predictions").glob("*.h5")):
        q, attrs = read_predictions(p)
        contestants.append((attrs, q))
    rows = []
    for attrs, preds in contestants:
        scope = list(preds) if attrs["method"] == "pinn" else list(truth)
        for cid in scope:
            if cid not in truth:
                continue
            s = specs[cid]
            X, Y = grid(Section(s.t, s.m), TRAIN)
            rows.append({"method": attrs["method"], "variant": attrs.get("variant", ""),
                         "n_train": int(attrs.get("n_train", 0)), "seed": int(attrs.get("seed", 0)),
                         "split": cid.split("-")[0], "regime": regimes[cid], "case_id": cid,
                         "score": score_case(preds.get(cid), truth[cid], X, Y, s.mach,
                                             s.alpha_deg, regimes[cid])})
    # SU2's own physics checks on the truth itself (spec 8.2: every column including SU2)
    for cid, q in truth.items():
        s = specs[cid]
        X, Y = grid(Section(s.t, s.m), TRAIN)
        rows.append({"method": "su2-512x193", "variant": "truth", "n_train": 0, "seed": 0,
                     "split": cid.split("-")[0], "regime": regimes[cid], "case_id": cid,
                     "score": score_case(q, q, X, Y, s.mach, s.alpha_deg, regimes[cid])})
    groups = aggregate(rows)
    (OUT / "scores.json").write_text(json.dumps({"rows": rows, "groups": groups}, indent=2,
                                                default=float))
    (OUT / "scores.md").write_text(markdown_table(groups) + "\n")

    # costs, in CPU seconds
    results = [json.loads(p.read_text()) for p in Path("runs/su2/256x97").glob("*/result.json")]
    per_su2 = float(np.median([r["cpu_s"] for r in results if r["case_id"].startswith("test")]))
    campaign = {n: sum(r["cpu_s"] for r in results
                       if r["case_id"].startswith("val")
                       or (r["case_id"].startswith("train") and int(r["case_id"][-4:]) < n))
                for n in (64, 128, 256, 512, 896)}
    costs = {"su2": {"fixed": 0.0, "per": per_su2}}
    for attrs, _ in contestants[1:]:
        if attrs["method"] in ("fno", "pod") and attrs["n_train"] == 896 and attrs["seed"] == 0 \
                and not attrs.get("variant"):
            costs[attrs["method"]] = {"fixed": campaign[896] + attrs["train_cpu_s"],
                                      "per": attrs["query_s"]}
    pinn = [a["train_cpu_s"] for a, _ in contestants if a["method"] == "pinn"
            and a.get("variant") == "euler"]
    if pinn:
        costs["pinn"] = {"fixed": 0.0, "per": float(np.mean(pinn))}
    (OUT / "costs.json").write_text(json.dumps({"costs": costs, "campaign_cpu_s": campaign},
                                               indent=2))
    plot_break_even(break_even(costs, 5000), OUT / "break_even.png")

    # data efficiency on test_in
    pts = {}
    for method in ("fno", "pod"):
        gs = sorted((g for g in groups if g["method"] == method and g["split"] == "test_in"
                     and g["seed"] == 0 and not g["variant"]), key=lambda g: g["n_train"])
        by_n = {}
        for g in gs:
            by_n.setdefault(g["n_train"], []).append((g["n_ok"], g.get("near_field_rel_l2_mean")))
        n = sorted(by_n)
        err = [sum(k * e for k, e in by_n[m] if e is not None) / sum(k for k, _ in by_n[m])
               for m in n]
        pts[method] = (n, err)
    fl = [r["score"]["near_field_rel_l2"] for r in rows
          if r["method"] == "su2-256x97" and r["split"] == "test_in"]
    plot_efficiency(pts, float(np.mean(fl)), OUT / "data_efficiency.png")

    # showcase wall pressure
    show = json.loads(Path("results/s1/showcase.json").read_text())["cases"]
    for c in show:
        cid = c["case_id"]
        s = specs[cid]
        X, _ = grid(Section(s.t, s.m), TRAIN)
        curves = {"truth": pressure_coefficient(truth[cid][:, 0, 3], s.mach),
                  "su2 256x97": pressure_coefficient(floor[cid][:, 0, 3], s.mach)}
        for attrs, preds in contestants[1:]:
            if cid in preds and (attrs["method"] == "pinn" or
                                 (attrs["n_train"] == 896 and attrs["seed"] == 0)):
                label = f"{attrs['method']} {attrs.get('variant', '')}".strip()
                curves[label] = pressure_coefficient(preds[cid][:, 0, 3], s.mach)
        plot_showcase(X[:, 0], curves, f"{cid}  max M {c['max_surface_mach']:.2f}",
                      OUT / f"showcase_{cid}.png")
    print((OUT / "scores.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests, then S4**

Run: `python -m pytest tests/test_report.py -v`, then `python scripts/s4_score.py`.
Expected: 2 passed. The script prints the table and writes `results/s4/` in full: `scores.json`, `scores.md`, `costs.json`, `break_even.png`, `data_efficiency.png` and six `showcase_*.png`.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add src/apno/score/report.py scripts/s4_score.py tests/test_report.py results/s4
git commit -m "S4: head-to-head scores, cost record, break-even, data efficiency, showcase plots"
```

---

### Task 23: The findings write-up and the predictions checked

**Files:**
- Modify: `docs/findings.md`

- [ ] **Step 1: Check each prediction against the measurement**

Append a dated S4 entry. Under each of the three predictions recorded in Task 1, write **held**, **failed** or **partly**, with the figure that decides it and the command it came from:
1. **PINN against SU2 on cost per case:** the ratio from `results/s4/costs.json`.
2. **FNO break-even:** the n at which `fno` crosses `su2` in the break-even curve (from `break_even(costs, 5000)` in `costs.json`), and the data-efficiency curve's distance from the floor at each n.
3. **No Kutta condition:** the `nokutta` run's `cl_err` against the `euler` run's on the same case, with both final residual losses from `results/s3/pinn_runs.json`.

Never edit the prediction text itself.

- [ ] **Step 2: Write the results, split by regime and set, with the floor beside every error**

From `results/s4/scores.md`:
- for each method, the near-field error, the CL error, the CD error in counts and the Cp error, for in-range and extrapolation, subcritical and supercritical;
- the FNO's spread across its three seeds;
- loss A against loss B;
- the shock position errors and overshoots;
- the physics checks on SU2 itself.

State every failure as a result: missed shocks, expansion shocks (an entropy decrease across a compression), non-finite predictions and PINN runs that did not converge.

- [ ] **Step 3: Commit**

```bash
git add docs/findings.md && git commit -m "Findings: head-to-head results and the three predictions checked"
```

---

## Self-review against the spec (done while writing this plan)

- **§4.1 design →** Task 10. **§4.2 geometry →** Tasks 2–3, with amendment 1. **§4.3 settings →** Task 8. **§4.4 convergence →** Task 5, with the thresholds frozen in Task 11. **§4.5 driver →** Task 9. **§4.6 S0 →** Task 11, including every listed check and the scorer check. **§4.7 cost →** measured in Tasks 11–12.
- **§5 →** Task 13 (amendment 4 for the layout). Leakage is guarded in Tasks 15 and 18; failures are listed in Tasks 12–13.
- **§6 →** Tasks 14–16 (amendment 3 for padding), including the 8-configuration budget, loss A against B, 128 × 49 training, 256 × 97 evaluation and the seeds.
- **§7 →** Tasks 17–20: the enthalpy reduction, grid coordinates, the exact constraints, the Kutta condition, artificial viscosity at 10⁻³ and 3 × 10⁻⁴, the no-Kutta run, the cylinder warm-up and its pass criterion.
- **§8 →** Tasks 21–23, plus the scorer check in Task 11 (amendment 2). The predictions are recorded in Task 1, before any model runs.
- **§9.3 tests →** grid (3), reader (4), convergence (5), physics (6), force integrator (7), Euler residual including the manufactured solution (17), FNO equivariance and both resolutions (15), PINN constraints (18), no leakage (15, 18).
- **§10 open items →** multigrid, thresholds, caps and radius (Task 11); Vassberg & Jameson (Task 11, Step 5); MPS (Task 1, Step 4). **The Wassing, Langer & Bekemeyer reference** is still to be verified before S3: add its full citation to the findings entry of Task 19, or record that it could not be found.
- **Deliberately not built (§9.4 extensions):** the Cartesian FNO, the parametric PINN, the pressure-tap PINN, PINO, and camber position as an input.
