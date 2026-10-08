# Stage S5 Implementation Plan: an Euler PINN with localised artificial viscosity

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This plan is executed by a multi-agent Workflow** (§ Execution below): implementers do not commit; an integrator commits each task after its review passes.

**Goal:** Build stage S5 — a Wassing-Langer-Bekemeyer-style Euler PINN (`euler_av`) with localised
artificial viscosity, its training, the gate metric, and the scripts that run the SU2 reference, the
gate, the ablation and the step-3 comparison and score them — tested, but not yet run.

**Architecture:** Library code in `src/apno` (viscosity algebra, the PINN model and losses, its
training loop, the gate metric, S5 case and scoring helpers), three stage scripts in `scripts/`, and
one test file per module. Everything reuses the existing grid map, the strong-conservation divergence,
the wall normal, the node areas, the S4 scorer and the SU2 driver unchanged.

**Tech Stack:** Python 3.11, PyTorch (MPS float32 for training, CPU float64 for evaluation), NumPy,
SciPy-free, h5py, Matplotlib (triangulation for the gate metric), SymPy (tests), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-07-s5-euler-pinn-av-design.md` (read it with this plan; the
main spec is `docs/superpowers/specs/2026-10-01-airfoil-pinn-vs-neural-operator-design.md`).

## Global Constraints

- Environment `airfoil-ml` (Python 3.11): `export PATH=/Users/esousa/miniforge3/envs/airfoil-ml/bin:$PATH`; run every command from the repository root `/Users/esousa/theLab/airfoil-pinn-no`.
- Lint: `ruff check .` (rules E, F, W; line length 100). Tests: `python -m pytest -q`; every task ends with both passing.
- Node ordering: node id `j * NI + i`, arrays indexed `[i, j]`; `i = 0` is the trailing edge, lower surface first; `j = 0` is the wall.
- Dataset units: ρ/ρ∞, u/U∞, v/U∞, p/p∞. The S5 PINN's internal scaling: ρ∞ = 1, p∞ = 1, (u∞, v∞) = √γ M∞ (cos α, sin α), coordinates in chords.
- S5 prediction files go to `runs/predictions_s5/` only — **never** `runs/predictions/` (S4's scorer reads every file there). `results/s4/` is never written; `scripts/s4_score.py` is not run.
- S5 records go to `results/s5/`; no record entry is ever overwritten; a finished configuration or run is skipped on a re-run.
- Values pinned by the design: ν = 1.9·10⁻³; eigenvalue floor ε = 0.25; λ_res = 2·10⁴; batch 5,000; 40 epochs; L-BFGS max_iter 1,000, max_eval 1,250, tolerance_grad 1e-6, tolerance_change 1e-10, history 1,000, strong Wolfe; 128 Fourier frequencies, σ = 1; 5 hidden layers of 128; adaptive tanh n = 10, ω₀ = 0.1; sensor k⁽⁰⁾_stag = 3, k⁽¹⁾_stag = 5, reading A (k⁽¹⁾_shock = 4, k⁽⁰⁾_shock = 0.03), reading B (k⁽⁰⁾_shock = 4, k⁽¹⁾_shock = 0.03); sensor floor 10⁻⁴; gate RMAE ≤ 1 % for Cp and Mach; lattice 401×401 on x ∈ [−0.5, 1.5], y ∈ [−1, 1]; budget 72 h; at most two shards.
- The viscous term enters the residual as **∇·F − D** (the dissipative sign, design §2.4).
- SU2 only through `SU2_BIN_DIR`; tests use the fake SU2 (`fake_bin` fixture, `tests/fake_su2.py`), never real SU2.
- Commit messages: `S5: <what>` and the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A non-physical predicted state during training** (raw primitive outputs start near 0, so ρ ≤ 0 or p ≤ 0 occurs early): the sound speed in the sensor and in |A| must stay finite (floored), not turn the whole loss into NaN. Pinned in Task 2 (`test_non_physical_states_give_finite_viscosity`).
2. **A batch where the sensor never fires** (subsonic flow, or the initial near-uniform network): the viscous term must be exactly zero and the residual equal the inviscid one, with no error on an empty active set. Pinned in Task 2 (`test_empty_batch_gives_empty_term`) and Task 5 (`test_uniform_flow_has_zero_residual_on_the_curved_grid[matrix]`).
3. **The duplicate column at ξ = 1** used by the periodicity term must be the same physical points as ξ = 0, or L_lr compares different points. Pinned in Task 5 (`test_the_duplicate_column_is_the_same_physical_points`).
4. **Re-running the gate after an interruption** must not re-train a configuration already recorded (failed or passed) and must continue with the next one; a decided gate must not re-run at all. Pinned in Task 7 (`test_the_gate_resumes_after_recorded_failures`, `test_a_decided_gate_does_not_run_again`).
5. **A comparison case whose truth or prediction file is missing** must be reported as a status, not crash the scorer. Pinned in Task 7 (`test_missing_truth_is_reported_not_raised`) and Task 8 (`test_missing_truth_or_file_is_a_status`, `test_main_writes_its_outputs_when_nothing_has_run`).

---

## Execution (multi-agent Workflow)

Task 1 (the predictions and the dated design corrections) is **done** (commit `102d6bb` on branch
`s5`). The remaining tasks run in dependency waves; tasks in one wave own disjoint files:

| wave | tasks | depends on |
|---|---|---|
| A | 2 (viscosity), 3 (gate metric), 4 (validation cases + SU2 script) | — |
| B | 5 (the PINN model and losses) | 2 |
| C | 6 (training) | 5 |
| D | 7 (S5 helpers + PINN stage script) | 3, 4, 6 |
| E | 8 (S5 scoring script) | 3, 7 |
| F | 9 (docs + full verification) | all |

Implementers follow each task's steps (TDD), run the task's tests, and **do not commit**. After a
task's reviewers pass it, the integrator commits that task's files with the task's message, in task
order. Work happens on branch `s5`.

## File structure

| file | responsibility |
|---|---|
| `src/apno/physics/viscosity.py` (new) | sensors, closed-form \|A\|, the viscous term D — pure algebra on tensors |
| `src/apno/score/rmae.py` (new) | the gate metric: lattice, triangulation on the grid's cells, RMAE of Cp and Mach |
| `src/apno/data/sampling.py` (modify) | `S5_VALIDATION`, `s5_validation_specs` |
| `scripts/s5_su2.py` (new) | SU2 on the validation case at both levels, fallback order, `results/s5/su2.json` |
| `src/apno/models/pinn_euler_av.py` (new) | `AvConfig`, scaling, the network, `EulerAvPinn` (state, residual, viscous term), `losses` |
| `src/apno/models/train_pinn_av.py` (new) | residual nodes, batches, the L-BFGS loop and the Adam-then-L-BFGS variant, timing, projection, evaluation |
| `src/apno/score/s5.py` (new) | S5 case lists, case construction, truth loading, the run scores (gate metric, collapse, cylinder) — shared by both S5 scripts |
| `scripts/s5_pinn.py` (new) | gate and steps phases: plans, preflight and budget, sequential gate, sharded steps, records |
| `scripts/s5_score.py` (new) | ablation verdicts, the step-3 comparison, costs, tables and plots in `results/s5/` |
| `tests/test_viscosity.py`, `tests/test_rmae.py`, `tests/test_s5_su2.py`, `tests/test_pinn_euler_av.py`, `tests/test_train_pinn_av.py`, `tests/test_s5.py`, `tests/test_s5_pinn.py`, `tests/test_s5_score.py` (new) | one per module |
| `CLAUDE.md`, `README.md` (modify) | the S5 stage rows |

(`src/apno/score/s5.py` refines design §9: the truth loader and the run scores are needed by both
S5 scripts, so they live in the library rather than in a script.)

---

### Task 1: Predictions and dated design corrections — DONE

Committed as `102d6bb`: the 2026-10-07 findings entry with P4–P7, and three dated corrections in the
design (the dissipative sign ∇·F − D; stalled outer iterations as the recorded proxy for line-search
failures; the upper bound counted in max_eval closures). Nothing to do.

---

### Task 2: Artificial viscosity — sensors, closed-form |A|, the viscous term

**Files:**
- Create: `src/apno/physics/viscosity.py`
- Test: `tests/test_viscosity.py`

**Interfaces:**
- Consumes: `apno.constants.GAMMA`.
- Produces:
  - `EIG_FLOOR = 0.25`, `STATE_FLOOR = 1e-6`, `SELECT = (1.0, 0.0, 0.0, 1.0)`
  - `SensorConstants(k0_shock, k1_shock, k0_stag=3.0, k1_stag=5.0)` (frozen dataclass); `READING_A`, `READING_B`, `READINGS = {"A": ..., "B": ...}`
  - `sound_speed(rho, p, gamma=GAMMA) -> Tensor`
  - `shock_sensor(u, v, p_x, p_y, k) -> Tensor`; `stagnation_sensor(grad_mx, grad_my, k) -> Tensor`
  - `sensor(u, v, p_x, p_y, mx_x, mx_y, my_x, my_y, k) -> Tensor` (detached)
  - `flux_jacobian_parts(rho, u, v, p, kx, ky, gamma=GAMMA) -> (M (n,4,4), q (n,), c (n,))`
  - `abs_jacobian(rho, u, v, p, kx, ky, eps=EIG_FLOOR, gamma=GAMMA) -> (n,4,4)`
  - `viscous_term(w_xx, w_yy, rho, u, v, p, s, nu, form="matrix", eps=EIG_FLOOR, gamma=GAMMA) -> (n,4)`

- [ ] **Step 1: Write the failing tests** — `tests/test_viscosity.py`:

```python
import numpy as np
import pytest
import torch

from apno.physics.viscosity import (
    READING_A, READING_B, abs_jacobian, flux_jacobian_parts, sensor, shock_sensor,
    stagnation_sensor, viscous_term,
)

G = 1.4


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _states(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    rho = 0.5 + torch.rand(n, generator=g)
    p = 0.5 + torch.rand(n, generator=g)
    u = 2 * torch.rand(n, generator=g) - 1
    v = 2 * torch.rand(n, generator=g) - 1
    return rho, u, v, p


def _flux(w, kx, ky):
    r, m, n, e = w
    u, v = m / r, n / r
    p = (G - 1) * (e - 0.5 * r * (u * u + v * v))
    q = u * kx + v * ky
    return torch.stack([r * q, m * q + p * kx, n * q + p * ky, (e + p) * q])


@pytest.mark.parametrize("kx,ky", [(1.0, 0.0), (0.0, 1.0)])
def test_flux_jacobian_matches_autograd(kx, ky):
    rho, u, v, p = _states()
    m, q, _ = flux_jacobian_parts(rho, u, v, p, kx, ky)
    for k in range(len(rho)):
        e = p[k] / (G - 1) + 0.5 * rho[k] * (u[k] ** 2 + v[k] ** 2)
        w = torch.stack([rho[k], rho[k] * u[k], rho[k] * v[k], e])
        want = torch.autograd.functional.jacobian(lambda x: _flux(x, kx, ky), w)
        torch.testing.assert_close(q[k] * torch.eye(4) + m[k], want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("eps", [0.0, 0.25])
@pytest.mark.parametrize("kx,ky", [(1.0, 0.0), (0.0, 1.0)])
def test_abs_jacobian_equals_eigendecomposition(kx, ky, eps):
    rho, u, v, p = _states()
    got = abs_jacobian(rho, u, v, p, kx, ky, eps=eps).numpy()
    m, q, _ = flux_jacobian_parts(rho, u, v, p, kx, ky)
    for k in range(len(rho)):
        lam, t = np.linalg.eig((q[k] * torch.eye(4) + m[k]).numpy())
        want = (t @ np.diag(np.maximum(np.abs(lam), eps)) @ np.linalg.inv(t)).real
        np.testing.assert_allclose(got[k], want, rtol=1e-9, atol=1e-11)


def test_limiter_floors_every_eigenvalue():
    rho, u, v, p = _states()
    u = torch.zeros_like(u)  # q = 0 along x: the double eigenvalue u is 0 without the floor
    lam = np.linalg.eigvals(abs_jacobian(rho, u, v, p, 1.0, 0.0, eps=0.25).numpy())
    assert np.all(lam.real >= 0.25 - 1e-9)


def test_shock_sensor_positive_only_for_pressure_rising_along_the_flow():
    u, v = torch.ones(3), torch.zeros(3)
    s = shock_sensor(u, v, torch.tensor([2.0, -2.0, 0.0]), torch.zeros(3), READING_A)
    assert s[0] > 0.9 and s[1] == 0 and s[2] == 0


def test_stagnation_sensor_switches_off_at_large_momentum_gradients():
    s = stagnation_sensor(torch.tensor([0.0, 10.0]), torch.tensor([0.0, 10.0]), READING_A)
    assert s[0] > 0.99 and s[1] < 1e-6


def test_sensor_is_zero_on_uniform_flow_and_carries_no_gradient():
    u = torch.full((5,), 0.8, requires_grad=True)
    z = torch.zeros(5)
    s = sensor(u, z, z, z, z, z, z, z, READING_A)
    assert torch.all(s == 0) and not s.requires_grad


def test_readings_are_the_two_reads_of_table_vii():
    assert (READING_A.k1_shock, READING_A.k0_shock) == (4.0, 0.03)
    assert (READING_B.k1_shock, READING_B.k0_shock) == (0.03, 4.0)
    assert (READING_A.k0_stag, READING_A.k1_stag) == (3.0, 5.0)


@pytest.mark.parametrize("form", ["matrix", "scalar"])
def test_viscous_term_acts_on_density_and_energy_rows_only(form):
    rho, u, v, p = _states(8)
    g = torch.Generator().manual_seed(3)
    wxx, wyy = torch.randn(8, 4, generator=g), torch.randn(8, 4, generator=g)
    d = viscous_term(wxx, wyy, rho, u, v, p, torch.ones(8), 1.9e-3, form)
    assert torch.all(d[:, 1:3] == 0) and torch.any(d[:, 0] != 0) and torch.any(d[:, 3] != 0)
    assert torch.all(viscous_term(wxx, wyy, rho, u, v, p, torch.zeros(8), 1.9e-3, form) == 0)


def test_scalar_form_is_nu_s_spectral_radius_laplacian():
    rho, u, v, p = _states(6)
    s = torch.full((6,), 0.5)
    lam = torch.sqrt(1.4 * p / rho) + torch.sqrt(u * u + v * v)
    d = viscous_term(torch.ones(6, 4), 2 * torch.ones(6, 4), rho, u, v, p, s, 1e-3, "scalar")
    torch.testing.assert_close(d[:, 0], 1e-3 * 0.5 * lam * 3.0)


def test_unknown_form_is_refused():
    rho, u, v, p = _states(2)
    with pytest.raises(ValueError, match="unknown viscosity form"):
        viscous_term(torch.ones(2, 4), torch.ones(2, 4), rho, u, v, p, torch.ones(2), 1e-3, "x")


def test_non_physical_states_give_finite_viscosity():
    rho, p = torch.tensor([0.0, -0.5, 1.0]), torch.tensor([-1.0, 0.0, 1e-9])
    z = torch.zeros(3)
    d = viscous_term(torch.ones(3, 4), torch.ones(3, 4), rho, z, z, p, torch.ones(3), 1e-3)
    assert torch.isfinite(d).all()


def test_empty_batch_gives_empty_term():
    z = torch.zeros(0)
    d = viscous_term(torch.zeros(0, 4), torch.zeros(0, 4), z, z, z, z, z, 1e-3, "matrix")
    assert d.shape == (0, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viscosity.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'apno.physics.viscosity'`.

- [ ] **Step 3: Write the implementation** — `src/apno/physics/viscosity.py`:

```python
"""Localised artificial viscosity for the S5 Euler PINN (S5 design §2.4; Wassing, Langer &
Bekemeyer 2025, §II D-E and Appendix C).

Every function takes torch tensors of shape (n,) per variable, in the S5 PINN's scaling
(rho_inf = p_inf = 1, |u_inf| = sqrt(gamma) M_inf). The conserved state is W = (rho, rho u, rho v,
rho E). The viscous term D is dissipative: the S5 residual is div F - D (their Eq. 13 prints + D,
which with their Eq. 14 would be anti-dissipative; read as a sign typo, design §2.4).

|A| for the flux Jacobian A = dF_k/dW in the direction (kx, ky) is closed-form and exact:
A = q I + M with M = r_a l_b + (gamma - 1) r_b beta, of rank two with eigenvalues {-c, 0, 0, c}, so
M^3 = c^2 M and any function of A fixed by its values on the spectrum {q - c, q, q, q + c} is a
quadratic in M: |A| = |l3| I + (|l1| - |l2|)/(2c) M + ((|l1| + |l2|)/2 - |l3|)/c^2 M^2, with
l1 = q + c, l2 = q - c, l3 = q and each |l| floored at EIG_FLOOR (their limiter). This equals
T|Lambda|T^-1 to round-off. rho and p are floored at STATE_FLOOR inside c and H, so a non-physical
state (early in training) gives a finite viscosity rather than NaN.
"""

from dataclasses import dataclass

import torch

from apno.constants import GAMMA

EIG_FLOOR = 0.25  # their Appendix C limiter on |lambda|
STATE_FLOOR = 1e-6  # floor on rho and p inside c and H
SELECT = (1.0, 0.0, 0.0, 1.0)  # I_{rho, rho E}: viscosity on the density and energy equations


@dataclass(frozen=True)
class SensorConstants:
    k0_shock: float  # threshold
    k1_shock: float  # steepness
    k0_stag: float = 3.0
    k1_stag: float = 5.0


READING_A = SensorConstants(k0_shock=0.03, k1_shock=4.0)  # steepness 4, as in their Fig. 3
READING_B = SensorConstants(k0_shock=4.0, k1_shock=0.03)  # their Table VII, read literally
READINGS = {"A": READING_A, "B": READING_B}


def sound_speed(rho, p, gamma=GAMMA):
    return torch.sqrt(gamma * p.clamp_min(STATE_FLOOR) / rho.clamp_min(STATE_FLOOR))


def shock_sensor(u, v, p_x, p_y, k):
    """tanh(max(0, k1 (q/|q|).grad p - k0)): positive where pressure rises along the flow."""
    speed = torch.sqrt(u * u + v * v).clamp_min(STATE_FLOOR)
    along = (u * p_x + v * p_y) / speed
    return torch.tanh(torch.clamp(k.k1_shock * along - k.k0_shock, min=0.0))


def stagnation_sensor(grad_mx, grad_my, k):
    """sig(k1 (k0 - (|grad rho u| + |grad rho v|))) from the two gradient magnitudes: near 0 where
    the momentum gradients are large (stagnation points), near 1 elsewhere."""
    return torch.sigmoid(k.k1_stag * (k.k0_stag - (grad_mx + grad_my)))


def sensor(u, v, p_x, p_y, mx_x, mx_y, my_x, my_y, k):
    """s = s_shock s_stag, detached: the PINN must not move the shock to lower its loss."""
    with torch.no_grad():
        s = shock_sensor(u, v, p_x, p_y, k) * stagnation_sensor(
            torch.hypot(mx_x, mx_y), torch.hypot(my_x, my_y), k)
    return s.detach()


def flux_jacobian_parts(rho, u, v, p, kx, ky, gamma=GAMMA):
    """(M, q, c) with A = q I + M the Jacobian dF_k/dW of the Euler flux in direction (kx, ky)."""
    q = u * kx + v * ky
    c = sound_speed(rho, p, gamma)
    h = gamma / (gamma - 1) * p / rho.clamp_min(STATE_FLOOR) + 0.5 * (u * u + v * v)
    one, zero = torch.ones_like(rho), torch.zeros_like(rho)
    r_a = torch.stack([one, u, v, h], -1)
    r_b = torch.stack([zero, kx * one, ky * one, q], -1)
    l_b = torch.stack([-q, kx * one, ky * one, zero], -1)
    beta = torch.stack([0.5 * (u * u + v * v), -u, -v, one], -1)
    m = r_a[..., :, None] * l_b[..., None, :] + (gamma - 1) * r_b[..., :, None] * beta[..., None, :]
    return m, q, c


def abs_jacobian(rho, u, v, p, kx, ky, eps=EIG_FLOOR, gamma=GAMMA):
    """|A| = T|Lambda|T^-1 of dF_k/dW, each |eigenvalue| floored at eps; shape (n, 4, 4)."""
    m, q, c = flux_jacobian_parts(rho, u, v, p, kx, ky, gamma)
    l1 = (q + c).abs().clamp_min(eps)
    l2 = (q - c).abs().clamp_min(eps)
    l3 = q.abs().clamp_min(eps)
    eye = torch.eye(4, dtype=rho.dtype, device=rho.device).expand(m.shape)
    a = (0.5 * (l1 - l2) / c)[:, None, None]
    b = ((0.5 * (l1 + l2) - l3) / (c * c))[:, None, None]
    return l3[:, None, None] * eye + a * m + b * (m @ m)


def viscous_term(w_xx, w_yy, rho, u, v, p, s, nu, form="matrix", eps=EIG_FLOOR, gamma=GAMMA):
    """D = mu_x W_xx + mu_y W_yy on the density and energy rows only; shape (n, 4).

    w_xx, w_yy: (n, 4) second derivatives of the conserved state in x and y; s: (n,) the sensor.
    form "matrix" (their model V): mu_x = nu s I |A|, mu_y = nu s I |B|; "scalar" (their model IV):
    mu = nu s I (c + |u|). The momentum rows of D are exactly zero.
    """
    select = torch.tensor(SELECT, dtype=rho.dtype, device=rho.device)
    if form == "matrix":
        ax = abs_jacobian(rho, u, v, p, 1.0, 0.0, eps, gamma)
        by = abs_jacobian(rho, u, v, p, 0.0, 1.0, eps, gamma)
        d = (ax @ w_xx[..., None])[..., 0] + (by @ w_yy[..., None])[..., 0]
    elif form == "scalar":
        lam = sound_speed(rho, p, gamma) + torch.sqrt(u * u + v * v)
        d = lam[:, None] * (w_xx + w_yy)
    else:
        raise ValueError(f"unknown viscosity form {form!r}; expected 'matrix' or 'scalar'")
    return nu * s[:, None] * select * d
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viscosity.py -q && ruff check src/apno/physics/viscosity.py tests/test_viscosity.py`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/physics/viscosity.py tests/test_viscosity.py
git commit -m "S5: artificial viscosity: sensors, closed-form |A| with the limiter, the viscous term" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The gate metric — RMAE of Cp and Mach on the lattice

**Files:**
- Create: `src/apno/score/rmae.py`
- Test: `tests/test_rmae.py`

**Interfaces:**
- Consumes: `apno.physics.flow.pressure_coefficient(p_h, mach)`, `local_mach(rho_h, u_h, v_h, p_h, mach)`.
- Produces:
  - `BOX = ((-0.5, 1.5), (-1.0, 1.0))`, `N_LATTICE = 401`
  - `triangulation(X, Y) -> matplotlib.tri.Triangulation` (the grid's own cells, two triangles per quad, periodic in i; vertex id `i * nj + j`)
  - `lattice(X, Y, n=N_LATTICE, box=BOX) -> ndarray (m, 2)` (points outside the section)
  - `on_lattice(field, tri, pts) -> ndarray (m,)` (NaN outside every cell)
  - `rmae(pred, true, X, Y, n=N_LATTICE) -> float` (`inf` if the prediction is non-finite where the truth is finite)
  - `rmae_cp_mach(pred_q, true_q, X, Y, mach, n=N_LATTICE) -> {"cp": float, "mach": float}`

- [ ] **Step 1: Write the failing tests** — `tests/test_rmae.py`:

```python
import numpy as np
import pytest

from apno.geometry.ogrid import Section, grid
from apno.score.rmae import lattice, on_lattice, rmae, rmae_cp_mach, triangulation

X, Y = grid(Section(0.12, 0.0), (64, 25))  # a small grid keeps these fast


def _field():
    return np.sin(X) + 0.5 * np.cos(2 * Y)


def test_identical_fields_have_zero_error():
    f = _field()
    assert rmae(f, f, X, Y, n=101) == 0.0


def test_a_constant_offset_gives_the_offset_over_the_span():
    f = _field()
    ft = on_lattice(f, triangulation(X, Y), lattice(X, Y, 101))
    span = np.nanmax(ft) - np.nanmin(ft)
    assert rmae(f + 0.01, f, X, Y, n=101) == pytest.approx(0.01 / span, rel=1e-9)


def test_lattice_points_inside_the_section_are_dropped():
    pts = lattice(X, Y, 101)
    assert len(pts) < 101 * 101
    on_chord = (np.abs(pts[:, 1]) < 1e-9) & (pts[:, 0] > 0.05) & (pts[:, 0] < 0.95)
    assert not on_chord.any()


def test_a_non_finite_prediction_fails_the_metric():
    f = _field()
    g = f.copy()
    g[10, :] = np.nan  # a whole radial line: the lattice certainly meets it
    assert rmae(g, f, X, Y, n=101) == float("inf")


def test_cp_and_mach_from_dataset_q():
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1.0
    q[..., 1] = 1.0 + 0.05 * np.cos(Y)
    q[..., 3] = 1.0 + 0.1 * np.sin(X)
    assert rmae_cp_mach(q, q, X, Y, 0.7, n=51) == {"cp": 0.0, "mach": 0.0}
    p = q.copy()
    p[..., 3] += 0.01
    assert rmae_cp_mach(p, q, X, Y, 0.7, n=51)["cp"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rmae.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'apno.score.rmae'`.

- [ ] **Step 3: Write the implementation** — `src/apno/score/rmae.py`:

```python
"""The S5 gate metric (S5 design §3.3): relative mean absolute error on a uniform lattice, as in
Wassing, Langer & Bekemeyer (2025), Eq. 28.

Their box (-1, 1)^2 about the mid-chord of a section centred at the origin becomes x in [-0.5, 1.5],
y in [-1, 1] for this study's chord [0, 1]. Lattice points inside the section are dropped; both
fields are interpolated linearly on the grid's own cells, each quad (i, j)-(i+1, j)-(i+1, j+1)-
(i, j+1), periodic in i, split into two triangles, so nothing is interpolated across the body.
RMAE(f) = mean |f_hat - f| / (max f - min f), with the max and min of the truth over the lattice;
a prediction that is non-finite anywhere the truth is finite scores inf.
"""

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.tri import LinearTriInterpolator, Triangulation

from apno.physics.flow import local_mach, pressure_coefficient

BOX = ((-0.5, 1.5), (-1.0, 1.0))
N_LATTICE = 401


def triangulation(X, Y):
    ni, nj = X.shape
    i, j = np.meshgrid(np.arange(ni), np.arange(nj - 1), indexing="ij")
    a = i * nj + j
    b = ((i + 1) % ni) * nj + j
    tri = np.concatenate([np.stack([a, b, b + 1], -1).reshape(-1, 3),
                          np.stack([a, b + 1, a + 1], -1).reshape(-1, 3)])
    return Triangulation(X.ravel(), Y.ravel(), tri)


def lattice(X, Y, n=N_LATTICE, box=BOX):
    """(m, 2) points of the uniform n x n lattice over box that lie outside the section."""
    xs, ys = np.meshgrid(np.linspace(*box[0], n), np.linspace(*box[1], n), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    inside = MplPath(np.stack([X[:, 0], Y[:, 0]], 1)).contains_points(pts)
    return pts[~inside]


def on_lattice(field, tri, pts):
    """field (ni, nj) interpolated linearly at pts; NaN where a point is outside every cell."""
    v = LinearTriInterpolator(tri, np.asarray(field, dtype=float).ravel())(pts[:, 0], pts[:, 1])
    return np.ma.filled(v.astype(float), np.nan)


def rmae(pred, true, X, Y, n=N_LATTICE):
    tri, pts = triangulation(X, Y), lattice(X, Y, n)
    fp, ft = on_lattice(pred, tri, pts), on_lattice(true, tri, pts)
    ok = np.isfinite(ft)
    if not np.all(np.isfinite(fp[ok])):
        return float("inf")
    span = ft[ok].max() - ft[ok].min()
    return float(np.mean(np.abs(fp[ok] - ft[ok])) / span)


def rmae_cp_mach(pred_q, true_q, X, Y, mach, n=N_LATTICE):
    """{"cp": RMAE of Cp, "mach": RMAE of local Mach}; q (ni, nj, 4) in the dataset's units."""
    cp = [pressure_coefficient(q[..., 3], mach) for q in (pred_q, true_q)]
    with np.errstate(invalid="ignore"):  # a non-physical prediction gives NaN Mach: scored inf
        m = [local_mach(q[..., 0], q[..., 1], q[..., 2], q[..., 3], mach) for q in (pred_q, true_q)]
    return {"cp": rmae(cp[0], cp[1], X, Y, n), "mach": rmae(m[0], m[1], X, Y, n)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rmae.py -q && ruff check src/apno/score/rmae.py tests/test_rmae.py`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/score/rmae.py tests/test_rmae.py
git commit -m "S5: the gate metric: RMAE of Cp and Mach on the lattice, interpolated on the grid's cells" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The validation cases and the SU2 reference script

**Files:**
- Modify: `src/apno/data/sampling.py` (append after `prefix_indices`)
- Create: `scripts/s5_su2.py`
- Test: `tests/test_s5_su2.py`

**Interfaces:**
- Consumes: `apno.su2.run.CaseSpec`, `run_campaign(specs, root, su2_bin, settings_for, workers, judge_kwargs)`, `apno.settings.check_frozen`, `settings_for`, `CONVERGENCE`, `apno.geometry.ogrid.TRAIN`, `FINE`.
- Produces:
  - `S5_VALIDATION = (("B", 0.72, 3.0), ("C", 0.75, 2.0), ("A", 0.70, 4.0))`
  - `s5_validation_specs(shape) -> list[CaseSpec]` with ids `s5-val-B`, `s5-val-C`, `s5-val-A`, t 0.12, m 0.0
  - `results/s5/su2.json`: `{"tried": [{"case_id", "inputs", "levels": {level: {...}}, "converged_both"}], "selected": case_id | null}`; SU2 cases in `runs/su2/<level>/s5-val-<tag>/`.

- [ ] **Step 1: Write the failing tests** — `tests/test_s5_su2.py`:

```python
import importlib.util
import json
from pathlib import Path

from apno.data.sampling import s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_su2.py"


def _load():
    spec = importlib.util.spec_from_file_location("s5_su2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validation_specs_are_naca0012_in_fallback_order():
    for shape in (TRAIN, FINE):
        got = [(s.case_id, s.mach, s.alpha_deg, s.t, s.m, s.shape)
               for s in s5_validation_specs(shape)]
        assert got == [("s5-val-B", 0.72, 3.0, 0.12, 0.0, shape),
                       ("s5-val-C", 0.75, 2.0, 0.12, 0.0, shape),
                       ("s5-val-A", 0.70, 4.0, 0.12, 0.0, shape)]


def test_case_b_is_selected_when_it_converges(fake_bin, tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-B"
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B"]
    assert set(out["tried"][0]["levels"]) == {"256x97", "512x193"}
    assert Path("runs/su2/512x193/s5-val-B/result.json").exists()


def test_falls_back_to_case_c_when_b_does_not_finish(fake_bin, tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    monkeypatch.setenv("FAKE_SU2_KILL", "s5-val-B")  # killed at both levels: no result
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-C"
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B", "s5-val-C"]
    assert out["tried"][0]["converged_both"] is False


def test_exit_2_when_no_case_converges(fake_bin, tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    monkeypatch.setenv("FAKE_SU2_MODE", "nan")
    assert mod.main([]) == 2
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] is None and len(out["tried"]) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_s5_su2.py -q`
Expected: collection error, `ImportError: cannot import name 's5_validation_specs'`.

- [ ] **Step 3: Write the implementation**

Append to `src/apno/data/sampling.py` (after `prefix_indices`; `CaseSpec` is already imported there):

```python
# S5 design §3.1: Wassing, Langer & Bekemeyer's NACA 0012 cases B, C, A, in the fallback order
S5_VALIDATION = (("B", 0.72, 3.0), ("C", 0.75, 2.0), ("A", 0.70, 4.0))


def s5_validation_specs(shape):
    """The S5 validation cases (NACA 0012: t 0.12, m 0) at `shape`, in the fallback order."""
    return [CaseSpec(f"s5-val-{tag}", mach, alpha, 0.12, 0.0, shape)
            for tag, mach, alpha in S5_VALIDATION]
```

Create `scripts/s5_su2.py`:

```python
"""Stage S5, the SU2 reference (S5 design §3.1-3.2): the validation case at 256x97 and 512x193.

    SU2_BIN_DIR=... python scripts/s5_su2.py [--workers 2]

Runs Wassing, Langer & Bekemeyer's case B (NACA 0012, M 0.72, alpha 3 deg) at both levels with the
frozen S0 settings and the convergence rule, into runs/su2/<level>/s5-val-B/; if either level does
not converge (or finish), case C, then case A: the fallback order fixed in the design. Writes
results/s5/su2.json with every case tried and the selected one (regenerated from the resumable runs
on every call). Exit 0 when a case is selected, 2 when none converged at both levels.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from apno import settings
from apno.data.sampling import s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN
from apno.settings import CONVERGENCE, settings_for
from apno.su2.run import run_campaign

ROOT, RES = Path("runs/su2"), Path("results/s5")
KEEP = ("converged", "reason", "iterations", "cl", "cd", "cm", "wall_s", "cpu_s",
        "max_surface_mach")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    settings.check_frozen()
    su2_bin = Path(os.environ["SU2_BIN_DIR"])
    tried, selected = [], None
    for coarse, fine in zip(s5_validation_specs(TRAIN), s5_validation_specs(FINE)):
        res = run_campaign([coarse, fine], ROOT, su2_bin, settings_for, args.workers, CONVERGENCE)
        levels = {r["level"]: {k: r[k] for k in KEEP} for r in res}
        ok = len(levels) == 2 and all(v["converged"] for v in levels.values())
        tried.append({"case_id": coarse.case_id, "inputs": coarse.inputs(), "levels": levels,
                      "converged_both": ok})
        if ok:
            selected = coarse.case_id
            break
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "su2.json").write_text(json.dumps({"tried": tried, "selected": selected}, indent=2))
    print(json.dumps({"tried": [t["case_id"] for t in tried], "selected": selected}), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_s5_su2.py tests/test_sampling.py -q && ruff check scripts/s5_su2.py src/apno/data/sampling.py tests/test_s5_su2.py`
Expected: all pass (the fake-SU2 tests write 512x193 meshes and take tens of seconds); `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/data/sampling.py scripts/s5_su2.py tests/test_s5_su2.py
git commit -m "S5: the validation cases and the SU2 reference script with its fallback order" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The PINN — configuration, scaling, network, state, residual and losses

**Files:**
- Create: `src/apno/models/pinn_euler_av.py`
- Test: `tests/test_pinn_euler_av.py`

**Interfaces:**
- Consumes: Task 2's `READINGS`, `sensor`, `viscous_term`; `apno.physics.euler.metrics(xi, eta, geom, radius, ratio)` (returns `x, y, x_xi, x_eta, y_xi, y_eta, J`), `divergence(f, g, q, xi, eta, terms, viscosity=0.0)`; `apno.models.pinn.wall_normal(xi, geom)`, `PinnCase(mach, alpha_deg, geometry, radius=100.0)`; `apno.geometry.ogrid.FINE`, `stretch_ratio`.
- Produces:
  - `AvConfig` (frozen dataclass, fields below), `config_from_dict(d) -> AvConfig`
  - `freestream(mach, alpha_deg) -> (1.0, u_inf, v_inf, 1.0)`; `to_dataset_q(rho, u, v, p, mach) -> Tensor (..., 4)`; `from_dataset_q(q, mach) -> (rho, u, v, p)`
  - `conserved(rho, u, v, p) -> 4-tuple`; `fluxes(rho, u, v, p) -> (F 4-tuple, G 4-tuple)`; `physical_grad(f, xi, eta, terms) -> (f_x, f_y)`
  - `FourierEmbedding`, `AdaptiveTanh`, `EulerAvNet(cfg)` (`.embed.phi` buffer when encoding is "fourier")
  - `EulerAvPinn(case, cfg)`: attributes `case, cfg, net, ratio, w_inf, k, viscous_everywhere`; methods `parameters()`, `primitive(xi, eta) -> (rho, u, v, p)` (S5 scaling), `residual(xi, eta) -> (n, 4)` ((n, 3) for closure "enthalpy"), `viscous(xi, eta, terms, u, v, p, w) -> (n, 4)`
  - `losses(pinn, batch, bnd, cfg) -> {"residual", "far", "wall", "periodic", "total"}` with `batch = {"xi", "eta", "w"}` and `bnd = {"xi_ring", "eta_lr"}`

- [ ] **Step 1: Write the failing tests** — `tests/test_pinn_euler_av.py`:

```python
import dataclasses
import json
import math

import numpy as np
import pytest
import sympy as sp
import torch

import apno.models.pinn_euler_av as mod
from apno.geometry.ogrid import Section, map_xy, stretch_ratio
from apno.models.pinn import PinnCase, wall_normal
from apno.models.pinn_euler_av import (
    AvConfig, EulerAvNet, EulerAvPinn, config_from_dict, freestream, from_dataset_q, losses,
    to_dataset_q,
)
from apno.physics.euler import metrics

GEOM = Section(0.12, 0.03)
CASE = PinnCase(0.72, 3.0, GEOM)


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _pts(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.01 + 0.98 * torch.rand(n, generator=g), 1e-3 + (1 - 1e-3) * torch.rand(n, generator=g)


def test_freestream_and_dataset_units_round_trip():
    prim = tuple(torch.tensor(x) for x in freestream(0.72, 3.0))
    q = to_dataset_q(*prim, 0.72)
    a = math.radians(3.0)
    torch.testing.assert_close(q, torch.tensor([1.0, math.cos(a), math.sin(a), 1.0]))
    for got, want in zip(from_dataset_q(q, 0.72), prim):
        torch.testing.assert_close(got, want)


def test_network_size_and_shapes():
    net = EulerAvNet(AvConfig())
    assert sum(p.numel() for p in net.parameters()) == 100_109
    xi, eta = _pts(7)
    assert net(xi, eta).shape == (7, 4)
    assert EulerAvNet(AvConfig(closure="enthalpy"))(xi, eta).shape == (7, 3)
    assert EulerAvNet(AvConfig(encoding="periodic"))(xi, eta).shape == (7, 4)


def test_the_embedding_is_fixed_by_the_seed():
    a, b, c = (EulerAvNet(AvConfig(seed=s)).embed.phi for s in (0, 0, 1))
    assert torch.equal(a, b) and not torch.equal(a, c) and a.shape == (128, 2)


def test_config_round_trips_through_json():
    cfg = AvConfig(sensor="B", te_exclusion=0.02)
    assert config_from_dict(json.loads(json.dumps(dataclasses.asdict(cfg)))) == cfg


def test_unknown_options_are_refused():
    with pytest.raises(ValueError):
        EulerAvPinn(CASE, AvConfig(encoding="raw"))
    with pytest.raises(ValueError):
        EulerAvPinn(CASE, AvConfig(closure="potential"))


def _const(values):
    def state(xi, eta):
        z = 0 * xi + 0 * eta
        return tuple(z + v for v in values)
    return state


@pytest.mark.parametrize("viscosity", ["none", "matrix", "scalar"])
def test_uniform_flow_has_zero_residual_on_the_curved_grid(viscosity, monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity=viscosity))
    monkeypatch.setattr(pinn, "primitive", _const(pinn.w_inf))
    xi, eta = _pts()
    r = pinn.residual(xi, eta)
    assert r.shape == (300, 4) and r.abs().max() < 1e-9


def _manufactured():
    x, y = sp.symbols("x y")
    rho = 1 + sp.Rational(1, 10) * sp.sin(x) * sp.cos(y)
    u = 1 + sp.Rational(1, 10) * sp.cos(x) * sp.sin(2 * y)
    v = sp.Rational(1, 10) * sp.sin(y) * sp.cos(x)
    p = 3 + sp.Rational(1, 10) * sp.cos(x + y)
    e = p / sp.Rational(2, 5) + rho * (u**2 + v**2) / 2  # gamma - 1 = 0.4
    F = [rho * u, rho * u**2 + p, rho * u * v, u * (e + p)]
    G = [rho * v, rho * u * v, rho * v**2 + p, v * (e + p)]
    div = [sp.lambdify((x, y), sp.diff(F[k], x) + sp.diff(G[k], y)) for k in range(4)]
    lap_rho = sp.lambdify((x, y), sp.diff(rho, x, 2) + sp.diff(rho, y, 2))
    lam = sp.lambdify((x, y), sp.sqrt(sp.Rational(7, 5) * p / rho) + sp.sqrt(u**2 + v**2))
    return div, lap_rho, lam


def _manufactured_state(xi, eta):
    x, y, *_ = metrics(xi, eta, GEOM)
    rho = 1 + 0.1 * torch.sin(x) * torch.cos(y)
    u = 1 + 0.1 * torch.cos(x) * torch.sin(2 * y)
    v = 0.1 * torch.sin(y) * torch.cos(x)
    p = 3 + 0.1 * torch.cos(x + y)
    return rho, u, v, p


def test_manufactured_solution_through_all_four_equations(monkeypatch):
    div, _, _ = _manufactured()
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    monkeypatch.setattr(pinn, "primitive", _manufactured_state)
    xi, eta = _pts(200, 1)
    eta = eta * 0.05  # keep x, y within a few chords so the fields stay O(1)
    r = pinn.residual(xi, eta).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    want = np.stack([div[k](x, y) for k in range(4)], 1)
    np.testing.assert_allclose(r, want, rtol=1e-8, atol=1e-8)


def test_scalar_viscosity_through_the_map_matches_sympy(monkeypatch):
    div, lap_rho, lam = _manufactured()
    nu = 1e-2
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="scalar", nu=nu))
    monkeypatch.setattr(pinn, "primitive", _manufactured_state)
    monkeypatch.setattr(mod, "sensor", lambda u, *a: torch.ones_like(u))
    xi, eta = _pts(150, 2)
    eta = eta * 0.05
    r = pinn.residual(xi, eta).detach().numpy()
    x, y, *_ = (t.detach().numpy() for t in metrics(xi, eta, GEOM))
    np.testing.assert_allclose(r[:, 0], div[0](x, y) - nu * lam(x, y) * lap_rho(x, y),
                               rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(r[:, 1:3], np.stack([div[1](x, y), div[2](x, y)], 1),
                               rtol=1e-8, atol=1e-8)


def test_far_field_and_periodic_terms_vanish_at_free_stream(monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    monkeypatch.setattr(pinn, "primitive", _const(pinn.w_inf))
    xi, eta = _pts(10)
    bnd = {"xi_ring": torch.arange(64) / 64, "eta_lr": torch.arange(25) / 24}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(10)}, bnd, pinn.cfg)
    assert L["far"] < 1e-28 and L["periodic"] < 1e-28 and L["residual"] < 1e-16


def test_the_wall_term_vanishes_for_tangent_flow(monkeypatch):
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))

    def tangent(xi, eta):
        nx, ny = wall_normal(xi, GEOM)
        return torch.ones_like(xi), -ny, nx, torch.ones_like(xi)

    monkeypatch.setattr(pinn, "primitive", tangent)
    xi, eta = _pts(10)
    bnd = {"xi_ring": torch.arange(64) / 64, "eta_lr": torch.arange(25) / 24}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(10)}, bnd, pinn.cfg)
    assert L["wall"] < 1e-28


def test_the_total_weights_the_residual_by_lambda_res():
    pinn = EulerAvPinn(CASE, AvConfig(viscosity="none"))
    xi, eta = _pts(20)
    bnd = {"xi_ring": torch.arange(16) / 16, "eta_lr": torch.arange(9) / 8}
    L = losses(pinn, {"xi": xi, "eta": eta, "w": torch.ones(20)}, bnd, pinn.cfg)
    want = 2e4 * L["residual"] + L["far"] + L["wall"] + L["periodic"]
    torch.testing.assert_close(L["total"], want)


def test_the_duplicate_column_is_the_same_physical_points():
    eta = torch.linspace(0, 1, 25)
    ratio = stretch_ratio(100.0)
    x0, y0 = map_xy(torch.zeros(25), eta, GEOM, 100.0, ratio, xp=torch)
    x1, y1 = map_xy(torch.ones(25), eta, GEOM, 100.0, ratio, xp=torch)
    torch.testing.assert_close(x0, x1, atol=1e-12, rtol=0)
    torch.testing.assert_close(y0, y1, atol=1e-12, rtol=0)


def test_the_enthalpy_closure_is_tangent_on_the_wall_and_has_three_rows():
    pinn = EulerAvPinn(CASE, AvConfig(closure="enthalpy", viscosity="none"))
    xr = torch.arange(64) / 64
    _, u, v, _ = pinn.primitive(xr, torch.zeros(64))
    nx, ny = wall_normal(xr, GEOM)
    assert (u * nx + v * ny).abs().max() < 1e-12
    xi, eta = _pts(20)
    assert pinn.residual(xi, eta).shape == (20, 3)


def test_the_pinn_carries_no_flow_field_data():
    pinn = EulerAvPinn(CASE, AvConfig())
    assert set(vars(pinn)) == {"case", "cfg", "net", "ratio", "w_inf", "k", "viscous_everywhere"}
    assert set(vars(CASE)) == {"mach", "alpha_deg", "geometry", "radius"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pinn_euler_av.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'apno.models.pinn_euler_av'`.

- [ ] **Step 3: Write the implementation** — `src/apno/models/pinn_euler_av.py`:

```python
"""The S5 Euler PINN with localised artificial viscosity (S5 design §2; Wassing, Langer &
Bekemeyer 2025, arXiv:2408.17364).

The network outputs the primitive variables (rho, u, v, p) in their scaling: rho_inf = p_inf = 1,
(u_inf, v_inf) = sqrt(gamma) M_inf (cos alpha, sin alpha), coordinates in chords, so c_inf =
sqrt(gamma) and their viscosity factor and sensor constants keep their meaning. `to_dataset_q`
converts to the dataset's units exactly: rho and p unchanged, u / U_inf = u / (sqrt(gamma) M_inf).

Inputs are the collocation grid's index coordinates (NI xi, (NJ - 1) eta) through a random
Fourier-feature embedding (n_freq frequencies from N(0, sigma^2), sin and cos), then `depth`
weight-normalised layers of `hidden` adaptive-tanh units, tanh(n omega_k x) with n = act_scale and
omega_k from omega0 (their scale factor is not stated; n = 10, omega0 = 0.1 follows Jagtap et al.).
Ablation variants (design §4): encoding "periodic" feeds (cos 2 pi xi, sin 2 pi xi, eta) (A4);
closure "enthalpy" is this study's Euler closure (A2): outputs ln rho and the velocity's departure
from free stream, pressure from constant total enthalpy, wall tangency exact by the (1 - eta)^4
projection, and the energy equation, then an identity, dropped from the residual.

The residual is the steady Euler system in strong conservation form through the O-grid's metric
terms (`apno.physics.euler.divergence`), minus the dissipative viscous term D of
`apno.physics.viscosity` (design §2.4). D is evaluated only where the sensor exceeds
cfg.sensor_floor, on a re-evaluation of the state at those points; `viscous_everywhere` (set by the
preflight's timing) evaluates it at every point, the cost's upper bound.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from apno.constants import GAMMA
from apno.geometry.ogrid import FINE, stretch_ratio
from apno.models.pinn import wall_normal
from apno.physics.euler import divergence, metrics
from apno.physics.viscosity import READINGS, sensor, viscous_term

G = GAMMA
ENCODINGS = ("fourier", "periodic")
CLOSURES = ("primitive", "enthalpy")
VISCOSITIES = ("matrix", "scalar", "none")


@dataclass(frozen=True)
class AvConfig:
    # network (design §2.2)
    hidden: int = 128
    depth: int = 5
    n_freq: int = 128
    sigma: float = 1.0
    act_scale: float = 10.0
    omega0: float = 0.1
    encoding: str = "fourier"  # A4: "periodic"
    closure: str = "primitive"  # A2: "enthalpy"
    # loss and collocation (design §2.3)
    weighting: str = "volume"  # A1: "uniform"
    lambda_res: float = 2e4  # A1: 1.0
    w_boundary: float = 1.0  # A1: 10.0
    te_exclusion: float = 0.0  # T4: 0.02 chord
    collocation: str = "nodes"  # A5: "random"
    grid_shape: tuple = FINE
    # artificial viscosity (design §2.4)
    viscosity: str = "matrix"  # T3: "scalar"; "none" switches D off
    nu: float = 1.9e-3
    sensor: str = "A"  # T2: "B"
    sensor_floor: float = 1e-4
    # optimiser (design §2.5; A3)
    optimizer: str = "lbfgs"  # A3: "adam_lbfgs"
    batch: int = 5000
    epochs: int = 40
    lbfgs_max_iter: int = 1000
    lbfgs_max_eval: int = 1250
    lbfgs_tol_grad: float = 1e-6
    lbfgs_tol_change: float = 1e-10
    lbfgs_history: int = 1000
    adam_steps: int = 30000
    adam_lr0: float = 1e-3
    adam_lr1: float = 1e-4
    final_lbfgs_iters: int = 5000
    final_lbfgs_history: int = 50
    seed: int = 0
    device: str = "auto"  # training; evaluation is always CPU float64


def config_from_dict(d):
    """The inverse of dataclasses.asdict through JSON (which turns the grid shape into a list)."""
    return AvConfig(**{**d, "grid_shape": tuple(d["grid_shape"])})


def freestream(mach, alpha_deg):
    a, s = math.radians(alpha_deg), math.sqrt(G) * mach
    return 1.0, s * math.cos(a), s * math.sin(a), 1.0


def to_dataset_q(rho, u, v, p, mach):
    s = math.sqrt(G) * mach
    return torch.stack([rho, u / s, v / s, p], -1)


def from_dataset_q(q, mach):
    s = math.sqrt(G) * mach
    return q[..., 0], q[..., 1] * s, q[..., 2] * s, q[..., 3]


def conserved(rho, u, v, p):
    return rho, rho * u, rho * v, p / (G - 1) + 0.5 * rho * (u * u + v * v)


def fluxes(rho, u, v, p):
    e = conserved(rho, u, v, p)[3]
    return ((rho * u, rho * u * u + p, rho * u * v, u * (e + p)),
            (rho * v, rho * u * v, rho * v * v + p, v * (e + p)))


def _grad(f, wrt):
    g = torch.autograd.grad(f.sum(), wrt, create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(wrt) if g is None else g


def physical_grad(f, xi, eta, terms):
    """(df/dx, df/dy) by the chain rule through the map's metric terms (x_xi, x_eta, y_xi, y_eta,
    J), differentiable again."""
    x_xi, x_eta, y_xi, y_eta, J = terms
    f_xi, f_eta = _grad(f, xi), _grad(f, eta)
    return (f_xi * y_eta - f_eta * y_xi) / J, (-f_xi * x_eta + f_eta * x_xi) / J


class FourierEmbedding(nn.Module):
    def __init__(self, n_freq, sigma, generator):
        super().__init__()
        self.register_buffer("phi", sigma * torch.randn(n_freq, 2, generator=generator))

    def forward(self, r):
        z = r @ self.phi.T
        return torch.stack([torch.sin(z), torch.cos(z)], -1).flatten(-2)


class AdaptiveTanh(nn.Module):
    def __init__(self, scale, omega0):
        super().__init__()
        self.scale = scale
        self.omega = nn.Parameter(torch.tensor(float(omega0)))

    def forward(self, x):
        return torch.tanh(self.scale * self.omega * x)


class EulerAvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.ni, self.nj1 = cfg.grid_shape[0], cfg.grid_shape[1] - 1
        if cfg.encoding == "fourier":
            self.embed, width = FourierEmbedding(cfg.n_freq, cfg.sigma, gen), 2 * cfg.n_freq
        else:
            self.embed, width = None, 3
        layers = []
        for _ in range(cfg.depth):
            layers += [weight_norm(nn.Linear(width, cfg.hidden)),
                       AdaptiveTanh(cfg.act_scale, cfg.omega0)]
            width = cfg.hidden
        layers.append(weight_norm(nn.Linear(width, 3 if cfg.closure == "enthalpy" else 4)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xi, eta):
        if self.embed is None:
            f = torch.stack([torch.cos(2 * math.pi * xi), torch.sin(2 * math.pi * xi), eta], -1)
        else:
            f = self.embed(torch.stack([self.ni * xi, self.nj1 * eta], -1))
        return self.mlp(f)


class EulerAvPinn:
    formulation = "euler_av"

    def __init__(self, case, cfg):
        if (cfg.encoding not in ENCODINGS or cfg.closure not in CLOSURES
                or cfg.viscosity not in VISCOSITIES):
            raise ValueError(f"unknown option in {cfg}")
        self.case, self.cfg = case, cfg
        self.net = EulerAvNet(cfg)
        self.ratio = stretch_ratio(case.radius)
        self.w_inf = freestream(case.mach, case.alpha_deg)
        self.k = READINGS[cfg.sensor]
        self.viscous_everywhere = False

    def parameters(self):
        return self.net.parameters()

    def primitive(self, xi, eta):
        """(rho, u, v, p) in the S5 scaling at (xi, eta)."""
        out = self.net(xi, eta)
        if self.cfg.closure == "primitive":
            return out.unbind(-1)
        rho = torch.exp(out[..., 0])
        u, v = self.w_inf[1] + out[..., 1], self.w_inf[2] + out[..., 2]
        nx, ny = wall_normal(xi, self.case.geometry)
        blend = (1.0 - eta) ** 4
        un = u * nx + v * ny
        u, v = u - blend * un * nx, v - blend * un * ny
        h_inf = G / (G - 1) + 0.5 * G * self.case.mach ** 2  # rho_inf = p_inf = 1
        return rho, u, v, (G - 1) / G * rho * (h_inf - 0.5 * (u * u + v * v))

    def _terms(self, xi, eta):
        return metrics(xi, eta, self.case.geometry, self.case.radius, self.ratio)[2:]

    def residual(self, xi, eta):
        """div F - D at (xi, eta): (n, 4), or (n, 3) for the enthalpy closure."""
        xi = xi.detach().clone().requires_grad_(True)
        eta = eta.detach().clone().requires_grad_(True)
        terms = self._terms(xi, eta)
        rho, u, v, p = self.primitive(xi, eta)
        w = conserved(rho, u, v, p)
        f, g = fluxes(rho, u, v, p)
        rows = 3 if self.cfg.closure == "enthalpy" else 4
        r = torch.stack([divergence(f[k], g[k], w[k], xi, eta, terms) for k in range(rows)], -1)
        if self.cfg.viscosity == "none":
            return r
        return r - self.viscous(xi, eta, terms, u, v, p, w)[:, :rows]

    def viscous(self, xi, eta, terms, u, v, p, w):
        """D at every point, (n, 4): zero where the sensor is at or below cfg.sensor_floor."""
        p_x, p_y = physical_grad(p, xi, eta, terms)
        mx_x, mx_y = physical_grad(w[1], xi, eta, terms)
        my_x, my_y = physical_grad(w[2], xi, eta, terms)
        s = sensor(u, v, p_x, p_y, mx_x, mx_y, my_x, my_y, self.k)
        on = torch.ones_like(s, dtype=torch.bool) if self.viscous_everywhere else (
            s > self.cfg.sensor_floor)
        idx = on.nonzero(as_tuple=True)[0]
        d = torch.zeros(len(s), 4, dtype=s.dtype, device=s.device)
        if len(idx) == 0:
            return d
        xm = xi.detach()[idx].clone().requires_grad_(True)
        em = eta.detach()[idx].clone().requires_grad_(True)
        tm = self._terms(xm, em)
        rho_m, u_m, v_m, p_m = self.primitive(xm, em)
        first = [physical_grad(c, xm, em, tm) for c in conserved(rho_m, u_m, v_m, p_m)]
        w_xx = torch.stack([physical_grad(cx, xm, em, tm)[0] for cx, _ in first], -1)
        w_yy = torch.stack([physical_grad(cy, xm, em, tm)[1] for _, cy in first], -1)
        dm = viscous_term(w_xx, w_yy, rho_m, u_m, v_m, p_m, s[idx], self.cfg.nu,
                          self.cfg.viscosity)
        return d.index_put((idx,), dm)


def losses(pinn, batch, bnd, cfg):
    """The S5 loss terms (design §2.3) and the weighted total.

    batch: {"xi", "eta", "w"}, residual points and their weights (mean 1 over the residual set);
    bnd: {"xi_ring", "eta_lr"}: the wall (eta = 0) and outer (eta = 1) rows at xi_ring, and the
    periodic pairs (xi = 0, xi = 1) at eta_lr. Squared norms are summed over the components.
    """
    r = pinn.residual(batch["xi"], batch["eta"])
    res = (batch["w"] * (r * r).sum(-1)).mean()
    xr = bnd["xi_ring"]
    one, zero = torch.ones_like(xr), torch.zeros_like(xr)
    far = sum((a - b) ** 2 for a, b in zip(pinn.primitive(xr, one), pinn.w_inf)).mean()
    _, uw, vw, _ = pinn.primitive(xr, zero)
    nx, ny = wall_normal(xr, pinn.case.geometry)
    wall = ((uw * nx + vw * ny) ** 2).mean()
    el = bnd["eta_lr"]
    left, right = pinn.primitive(torch.zeros_like(el), el), pinn.primitive(torch.ones_like(el), el)
    periodic = sum((a - b) ** 2 for a, b in zip(left, right)).mean()
    total = cfg.lambda_res * res + cfg.w_boundary * (far + wall + periodic)
    return {"residual": res, "far": far, "wall": wall, "periodic": periodic, "total": total}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pinn_euler_av.py -q && ruff check src/apno/models/pinn_euler_av.py tests/test_pinn_euler_av.py`
Expected: all pass; `All checks passed!` If `test_network_size_and_shapes` reports a count other than 100,109, report it as a deviation rather than editing the number: it was measured with this exact architecture (embedding buffer excluded; weight-normalised layers carry g and v; five ω).

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/models/pinn_euler_av.py tests/test_pinn_euler_av.py
git commit -m "S5: the euler_av PINN: Fourier features, adaptive tanh, full Euler residual minus localised viscosity" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Training — residual nodes, the L-BFGS loop, the A3 and A5 variants, timing, evaluation

**Files:**
- Create: `src/apno/models/train_pinn_av.py`
- Test: `tests/test_train_pinn_av.py`

**Interfaces:**
- Consumes: Task 5's `AvConfig`, `EulerAvPinn(case, cfg)`, `losses(pinn, batch, bnd, cfg)`, `to_dataset_q`; `apno.models.train_pinn.resolve_device(name)`; `apno.geometry.ogrid.Section`, `grid`, `map_xy`, `stretch_ratio`; `apno.physics.euler.metrics`; `apno.score.metrics.node_areas(X, Y)`.
- Produces:
  - `NonFiniteLoss(RuntimeError)`
  - `residual_nodes(case, cfg) -> {"xi", "eta", "w"}` (float64 CPU tensors)
  - `boundary_points(cfg) -> {"xi_ring", "eta_lr"}`
  - `random_batch(case, cfg, n, gen, ratio) -> {"xi", "eta", "w"}`
  - `make_pinn(case, cfg, device="cpu", dtype=torch.float32) -> EulerAvPinn`
  - `train(case, cfg, log=print) -> (pinn, rec)`; `rec` keys: `formulation, device, history, stalled, func_evals, diverged, reason, final, wall_s, cpu_s`
  - `time_closure(case, cfg, n=3) -> {"device": str, "closure_s": float}`
  - `projected_upper_s(cfg, n_nodes, closure_s) -> float`
  - `evaluate_on_grid(pinn, shape) -> ndarray (ni, nj, 4)` in the dataset's units

- [ ] **Step 1: Write the failing tests** — `tests/test_train_pinn_av.py`:

```python
import math

import numpy as np
import pytest
import torch

import apno.models.train_pinn_av as tr
from apno.geometry.ogrid import Section, grid
from apno.models.pinn import PinnCase
from apno.models.pinn_euler_av import AvConfig
from apno.models.train_pinn_av import (
    boundary_points, evaluate_on_grid, projected_upper_s, residual_nodes, time_closure, train,
)

CASE = PinnCase(0.72, 3.0, Section(0.12, 0.0))
TINY = AvConfig(hidden=8, depth=2, n_freq=4, grid_shape=(16, 9), batch=40, epochs=1,
                lbfgs_max_iter=2, lbfgs_max_eval=3, lbfgs_history=5, device="cpu")


def test_residual_nodes_drop_the_trailing_edge_node_and_normalise_the_weights():
    nodes = residual_nodes(CASE, TINY)
    assert len(nodes["xi"]) == 16 * 9 - 1
    assert not ((nodes["xi"] == 0) & (nodes["eta"] == 0)).any()
    assert nodes["w"].mean().item() == pytest.approx(1.0)
    assert nodes["xi"].dtype == torch.float64


def test_uniform_weighting_and_the_trailing_edge_exclusion():
    from dataclasses import replace
    assert torch.all(residual_nodes(CASE, replace(TINY, weighting="uniform"))["w"] == 1)
    cfg = replace(TINY, grid_shape=(64, 25), te_exclusion=0.02)
    nodes = residual_nodes(CASE, cfg)
    X, Y = grid(CASE.geometry, (64, 25))
    i = np.rint(nodes["xi"].numpy() * 64).astype(int)
    j = np.rint(nodes["eta"].numpy() * 24).astype(int)
    assert np.all(np.hypot(X[i, j] - 1.0, Y[i, j]) >= 0.02)
    assert len(nodes["xi"]) < 64 * 25 - 1


def test_boundary_points_cover_the_ring_and_the_periodic_pairs():
    b = boundary_points(TINY)
    assert torch.equal(b["xi_ring"], torch.arange(16, dtype=torch.float64) / 16)
    assert torch.equal(b["eta_lr"], torch.arange(9, dtype=torch.float64) / 8)


def test_train_runs_and_records_every_outer_iteration():
    pinn, rec = train(CASE, TINY, log=lambda *a: None)
    assert not rec["diverged"] and rec["device"] == "cpu"
    assert len(rec["history"]) == math.ceil((16 * 9 - 1) / 40)
    assert rec["func_evals"] > 0 and rec["wall_s"] > 0 and "total" in rec["final"]
    q = evaluate_on_grid(pinn, (8, 5))
    assert q.shape == (8, 5, 4) and np.isfinite(q).all()


def test_the_adam_then_lbfgs_variant():
    from dataclasses import replace
    cfg = replace(TINY, optimizer="adam_lbfgs", adam_steps=3, final_lbfgs_iters=2,
                  final_lbfgs_history=3)
    _, rec = train(CASE, cfg, log=lambda *a: None)
    assert not rec["diverged"] and rec["history"][-1]["epoch"] == "final"


def test_the_random_collocation_variant():
    from dataclasses import replace
    _, rec = train(CASE, replace(TINY, collocation="random"), log=lambda *a: None)
    assert not rec["diverged"] and len(rec["history"]) == math.ceil((16 * 9 - 1) / 40)


def test_a_non_finite_loss_stops_the_run(monkeypatch):
    def nan_losses(*a, **k):
        t = torch.tensor(float("nan"), requires_grad=True)
        return {"residual": t, "far": t, "wall": t, "periodic": t, "total": t}

    monkeypatch.setattr(tr, "losses", nan_losses)
    _, rec = train(CASE, TINY, log=lambda *a: None)
    assert rec["diverged"] and "non-finite" in rec["reason"]


def test_time_closure_and_the_upper_bound():
    t = time_closure(CASE, TINY, n=1)
    assert t["device"] == "cpu" and t["closure_s"] > 0
    full = AvConfig()
    assert projected_upper_s(full, 98_815, 0.01) == pytest.approx(40 * 20 * 1250 * 0.01)
    from dataclasses import replace
    adam = replace(full, optimizer="adam_lbfgs")
    assert projected_upper_s(adam, 98_815, 0.01) == pytest.approx((30000 + 6250) * 0.01)


def test_evaluation_lands_exactly_on_the_collocation_nodes():
    xi = torch.arange(8, dtype=torch.float64) / 8  # the 8x5 scoring nodes of a 16x9 grid
    assert torch.equal(16 * xi, 2 * torch.arange(8, dtype=torch.float64))
    eta = torch.arange(5, dtype=torch.float64) / 4
    assert torch.equal(8 * eta, 2 * torch.arange(5, dtype=torch.float64))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_train_pinn_av.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'apno.models.train_pinn_av'`.

- [ ] **Step 3: Write the implementation** — `src/apno/models/train_pinn_av.py`:

```python
"""Training the S5 Euler PINN (S5 design §2.3, §2.5, §4 A3 and A5, §7.1).

L-BFGS only, in mini-batches (their §II H): an epoch is one pass over the residual nodes in batches
of cfg.batch drawn without replacement; each batch gets a fresh torch.optim.LBFGS, reset after its
outer iteration because the batch changes the objective; the boundary terms are evaluated in full
at every closure. optimizer "adam_lbfgs" (A3) is this study's schedule instead: Adam over the same
mini-batches, the learning rate decaying exponentially from adam_lr0 to adam_lr1, then L-BFGS on one
fixed batch. collocation "random" (A5) draws every batch afresh, uniform in (xi, eta), weighted by
J normalised to mean 1 over the batch (the continuous counterpart of the dual-cell area).

Residual nodes: every node of cfg.grid_shape except a section's trailing-edge wall node (J = 0 there
under the cosine clustering) and, with te_exclusion > 0, the nodes within that many chords of
(1, 0); weights are the dual-cell areas normalised to mean 1 ("volume") or 1 ("uniform"). Training
runs in float32 on cfg.device ("auto": the Apple GPU when MPS is available); evaluation is in
float64 on the CPU. A non-finite loss stops the run (rec["diverged"], with the reason); an outer
iteration whose loss did not fall is counted in rec["stalled"].
"""

import math
import time

import numpy as np
import torch

from apno.geometry.ogrid import Section, grid, map_xy, stretch_ratio
from apno.models.pinn_euler_av import EulerAvPinn, losses, to_dataset_q
from apno.models.train_pinn import resolve_device
from apno.physics.euler import metrics
from apno.score.metrics import node_areas


class NonFiniteLoss(RuntimeError):
    pass


def residual_nodes(case, cfg):
    """{"xi", "eta", "w"}: the residual nodes as float64 CPU tensors (design §2.3, §2.6)."""
    ni, nj = cfg.grid_shape
    X, Y = grid(case.geometry, cfg.grid_shape, case.radius)
    i, j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    keep = np.ones((ni, nj), dtype=bool)
    if isinstance(case.geometry, Section):
        keep[0, 0] = False  # J = 0 at the trailing-edge wall node
        if cfg.te_exclusion > 0:
            keep &= np.hypot(X - 1.0, Y) >= cfg.te_exclusion
    w = node_areas(X, Y)[keep] if cfg.weighting == "volume" else np.ones(int(keep.sum()))
    return {"xi": torch.tensor(i[keep] / ni, dtype=torch.float64),
            "eta": torch.tensor(j[keep] / (nj - 1), dtype=torch.float64),
            "w": torch.tensor(w / w.mean(), dtype=torch.float64)}


def boundary_points(cfg):
    ni, nj = cfg.grid_shape
    return {"xi_ring": torch.arange(ni, dtype=torch.float64) / ni,
            "eta_lr": torch.arange(nj, dtype=torch.float64) / (nj - 1)}


def random_batch(case, cfg, n, gen, ratio):
    """n points uniform in (xi, eta), eta in (0, 1], weighted by J (mean 1) or 1 (A5)."""
    xi = torch.rand(n, generator=gen, dtype=torch.float64)
    eta = 1.0 - torch.rand(n, generator=gen, dtype=torch.float64)
    if isinstance(case.geometry, Section) and cfg.te_exclusion > 0:
        x, y = map_xy(xi, eta, case.geometry, case.radius, ratio, xp=torch)
        keep = torch.hypot(x - 1.0, y) >= cfg.te_exclusion
        xi, eta = xi[keep], eta[keep]
    if cfg.weighting == "volume":
        J = metrics(xi, eta, case.geometry, case.radius, ratio)[-1].detach()
        w = J / J.mean()
    else:
        w = torch.ones_like(xi)
    return {"xi": xi, "eta": eta, "w": w}


def make_pinn(case, cfg, device="cpu", dtype=torch.float32):
    pinn = EulerAvPinn(case, cfg)
    pinn.net.to(device=device, dtype=dtype)
    return pinn


def _to(d, device, dtype):
    return {k: v.to(device=device, dtype=dtype) for k, v in d.items()}


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()


def _snapshot(L):
    return {k: float(v.detach()) for k, v in L.items()}


def train(case, cfg, log=print):
    device, dtype = resolve_device(cfg.device), torch.float32
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    pinn = make_pinn(case, cfg, device, dtype)
    ratio = stretch_ratio(case.radius)
    nodes = residual_nodes(case, cfg)
    bnd = _to(boundary_points(cfg), device, dtype)
    n = len(nodes["xi"])
    rec = {"formulation": "euler_av", "device": device, "history": [], "stalled": 0,
           "func_evals": 0, "diverged": False, "reason": None, "final": {}}

    def batches():
        perm = torch.randperm(n, generator=gen)
        for k in range(0, n, cfg.batch):
            if cfg.collocation == "random":
                b = random_batch(case, cfg, min(cfg.batch, n - k), gen, ratio)
            else:
                b = {key: v[perm[k:k + cfg.batch]] for key, v in nodes.items()}
            yield _to(b, device, dtype)

    def objective(batch):
        L = losses(pinn, batch, bnd, cfg)
        if not torch.isfinite(L["total"]):
            raise NonFiniteLoss(f"non-finite loss: {_snapshot(L)}")
        return L

    def lbfgs_step(batch, max_iter, max_eval, history):
        opt = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_eval,
                                tolerance_grad=cfg.lbfgs_tol_grad,
                                tolerance_change=cfg.lbfgs_tol_change, history_size=history,
                                line_search_fn="strong_wolfe")
        evals = [0]

        def closure():
            opt.zero_grad()
            total = objective(batch)["total"]
            total.backward()
            evals[0] += 1
            return total

        before = float(opt.step(closure))
        return before, objective(batch), evals[0]

    def record(epoch, b, before, after, evals):
        snap = _snapshot(after)
        rec["history"].append({"epoch": epoch, "batch": b, "loss_before": before,
                               "func_evals": evals, **snap})
        rec["func_evals"] += evals
        rec["stalled"] += int(not snap["total"] < before)
        rec["final"] = snap
        log(f"{epoch}/{b}: {before:.4g} -> {snap['total']:.4g} ({evals} evals)")

    t0, c0 = time.perf_counter(), time.process_time()
    try:
        if cfg.optimizer == "lbfgs":
            for epoch in range(cfg.epochs):
                for b, batch in enumerate(batches()):
                    record(epoch, b, *lbfgs_step(batch, cfg.lbfgs_max_iter, cfg.lbfgs_max_eval,
                                                 cfg.lbfgs_history))
        elif cfg.optimizer == "adam_lbfgs":
            opt = torch.optim.Adam(pinn.parameters(), lr=cfg.adam_lr0)
            gamma = (cfg.adam_lr1 / cfg.adam_lr0) ** (1 / cfg.adam_steps)
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
            step, fixed = 0, None
            while step < cfg.adam_steps:
                for batch in batches():
                    fixed = batch if fixed is None else fixed
                    if step >= cfg.adam_steps:
                        break
                    opt.zero_grad()
                    L = objective(batch)
                    L["total"].backward()
                    opt.step()
                    sched.step()
                    step += 1
                    if step % 500 == 0 or step == cfg.adam_steps:
                        rec["history"].append({"epoch": "adam", "step": step, **_snapshot(L)})
            record("final", 0, *lbfgs_step(fixed, cfg.final_lbfgs_iters,
                                           math.ceil(1.25 * cfg.final_lbfgs_iters),
                                           cfg.final_lbfgs_history))
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    except NonFiniteLoss as exc:
        rec["diverged"], rec["reason"] = True, str(exc)
        log(f"diverged: {exc}")
    _sync(device)
    rec["wall_s"], rec["cpu_s"] = time.perf_counter() - t0, time.process_time() - c0
    return pinn, rec


def time_closure(case, cfg, n=3):
    """Median wall seconds of one closure (loss and backward) on one batch of cfg.batch residual
    nodes, on cfg.device, from a fresh network, with the viscous term forced on at every point (the
    cost's upper bound); the first call is a warm-up and is not counted."""
    device = resolve_device(cfg.device)
    pinn = make_pinn(case, cfg, device)
    pinn.viscous_everywhere = True
    gen = torch.Generator().manual_seed(cfg.seed)
    nodes = residual_nodes(case, cfg)
    idx = torch.randperm(len(nodes["xi"]), generator=gen)[:cfg.batch]
    batch = _to({k: v[idx] for k, v in nodes.items()}, device, torch.float32)
    bnd = _to(boundary_points(cfg), device, torch.float32)
    times = []
    for _ in range(n + 1):
        _sync(device)
        t = time.perf_counter()
        for p in pinn.parameters():
            p.grad = None
        losses(pinn, batch, bnd, cfg)["total"].backward()
        _sync(device)
        times.append(time.perf_counter() - t)
    return {"device": device, "closure_s": float(np.median(times[1:]))}


def projected_upper_s(cfg, n_nodes, closure_s):
    """A run's upper-bound wall seconds (design §7.1): every L-BFGS outer iteration at max_eval
    closures; for A3, every Adam step plus 1.25 closures per final L-BFGS iteration."""
    if cfg.optimizer == "adam_lbfgs":
        return (cfg.adam_steps + math.ceil(1.25 * cfg.final_lbfgs_iters)) * closure_s
    return cfg.epochs * math.ceil(n_nodes / cfg.batch) * cfg.lbfgs_max_eval * closure_s


def evaluate_on_grid(pinn, shape):
    """q (ni, nj, 4) in the dataset's units at the nodes of `shape`, on the CPU in float64. For the
    512x193 collocation grid and shape 256x97 these are exactly every second collocation node."""
    ni, nj = shape
    xi = (torch.arange(ni, dtype=torch.float64) / ni)[:, None].expand(ni, nj).reshape(-1)
    eta = (torch.arange(nj, dtype=torch.float64) / (nj - 1))[None, :].expand(ni, nj).reshape(-1)
    pinn.net.to("cpu").double()
    with torch.no_grad():
        q = to_dataset_q(*pinn.primitive(xi, eta), pinn.case.mach)
    return q.reshape(ni, nj, 4).numpy()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_train_pinn_av.py -q && ruff check src/apno/models/train_pinn_av.py tests/test_train_pinn_av.py`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/models/train_pinn_av.py tests/test_train_pinn_av.py
git commit -m "S5: training the euler_av PINN: mini-batch L-BFGS, the A3 and A5 variants, timing, evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: S5 helpers and the PINN stage script (gate and steps)

**Files:**
- Create: `src/apno/score/s5.py`
- Create: `scripts/s5_pinn.py`
- Test: `tests/test_s5.py`, `tests/test_s5_pinn.py`

**Interfaces:**
- Consumes: Task 3's `rmae_cp_mach`; Task 4's `s5_validation_specs`; Task 5's `AvConfig`, `config_from_dict`; Task 6's `train`, `evaluate_on_grid`, `time_closure`, `projected_upper_s`, `residual_nodes`; `apno.data.build.load_case(root, spec)` (returns `{"q", "max_surface_mach", ...}` or `None`); `apno.data.predictions.write_predictions(path, case_ids, q, attrs)`; `apno.data.sampling.case_specs`; `apno.models.train_pinn.circle_cp_error(q, mach)`; `apno.score.forces.wall_forces`; `apno.score.metrics.score_case(pred_q, true_q, X, Y, mach, alpha_deg, truth_regime, flux_lines=FLUX_LINES)`.
- Produces (`apno.score.s5`): `GATE_RMAE = 0.01`, `STEP3_IN`, `STEP3_EXT`, `CYLINDER = "cylinder"`, `ABLATION_AIRFOIL = "test_in-0022"`, `all_specs() -> {case_id: CaseSpec}`, `pinn_case(case_id, specs) -> PinnCase`, `truth_and_floor(case_id) -> (truth (256,97,4), floor (256,97,4), max_surface_mach) | (None, None, None)`, `wall_speed_max(q) -> float`, `airfoil_score(q, case_id, specs) -> dict`, `cylinder_score(q) -> dict`, `score_run(q, case_id, specs) -> dict`. Every score dict has `status`, `passed`, `collapsed`.
- Produces (`scripts/s5_pinn.py`): `gate_configs()`, `steps_plan(r, validation_id)`, `shard`, `parse_shard`, `prediction_path(name)`, `run_one(name, case_id, cfg, specs)`, `timings(runs, specs, preflight_file)`, `reprojected_hours(cfg, closure_s, recipe)`, `main(argv)`; records `results/s5/{su2,gate,recipe,preflight_gate,preflight_steps{suffix},runs{suffix}}.json`; prediction files `runs/predictions_s5/pinn-euler_av-<name>.h5`. Run-record entries: `case_id, cfg, git_commit, git_dirty, wall_s, cpu_s, device, func_evals, stalled, diverged, reason, final, history, score`.

- [ ] **Step 1: Write the failing tests**

`tests/test_s5.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

import apno.score.s5 as s5
from apno.geometry.ogrid import TRAIN, Cylinder, Section, grid

REPO = Path(__file__).parents[1]
DATA = REPO / "data" / "hdf5"


def _truth(case_id="s5-val-B"):
    s = s5.all_specs()[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    a = np.radians(s.alpha_deg)
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1 + 0.05 * np.sin(X)
    q[..., 1] = np.cos(a) + 0.1 * np.cos(Y)
    q[..., 2] = np.sin(a)
    q[..., 3] = 1 + 0.1 * np.cos(X) * np.exp(-Y ** 2)
    return q


def test_all_specs_hold_the_validation_and_comparison_cases():
    specs = s5.all_specs()
    want = {"s5-val-B", "s5-val-C", "s5-val-A", s5.ABLATION_AIRFOIL, *s5.STEP3_IN, *s5.STEP3_EXT}
    assert want <= set(specs)


def test_step3_in_range_cases_are_the_supercritical_showcase():
    show = json.loads((REPO / "results/s1/showcase.json").read_text())["cases"]
    assert tuple(c["case_id"] for c in show if c["regime"] == "supercritical") == s5.STEP3_IN


@pytest.mark.skipif(not (DATA / "512x193.h5").exists(), reason="dataset not built (data/hdf5)")
def test_step3_extrapolation_cases_follow_the_showcase_rule():
    import h5py
    with h5py.File(DATA / "512x193.h5") as f, h5py.File(DATA / "256x97.h5") as g:
        ids = list(f["test_ext"]["case_id"].asstr()[...])
        mm = f["test_ext"]["max_surface_mach"][...]
        both = set(g["test_ext"]["case_id"].asstr()[...])
    keep = [k for k, c in enumerate(ids) if c in both]
    ids, mm = [ids[k] for k in keep], mm[keep]
    order = np.argsort(mm, kind="stable")
    picked = tuple(ids[order[round(p * (len(ids) - 1))]] for p in (0.10, 0.50, 0.90))
    assert picked == s5.STEP3_EXT


def test_pinn_case_builds_sections_and_the_cylinder():
    specs = s5.all_specs()
    c = s5.pinn_case("s5-val-B", specs)
    assert (c.mach, c.alpha_deg, c.geometry) == (0.72, 3.0, Section(0.12, 0.0))
    cyl = s5.pinn_case(s5.CYLINDER, specs)
    assert (cyl.mach, cyl.alpha_deg, cyl.geometry) == (0.1, 0.0, Cylinder())


def test_airfoil_score_passes_the_truth_and_flags_a_collapse(monkeypatch):
    q = _truth()
    monkeypatch.setattr(s5, "truth_and_floor", lambda cid: (q, q.copy(), 0.9))
    ok = s5.airfoil_score(q, "s5-val-B", s5.all_specs())
    assert ok["status"] == "ok" and ok["passed"] and not ok["collapsed"]
    assert ok["rmae"] == {"cp": 0.0, "mach": 0.0} and ok["regime"] == "subcritical"
    rest = q.copy()
    rest[..., 1:3] = 0
    bad = s5.airfoil_score(rest, "s5-val-B", s5.all_specs())
    assert bad["collapsed"] and not bad["passed"]


def test_missing_truth_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(s5, "truth_and_floor", lambda cid: (None, None, None))
    got = s5.airfoil_score(np.zeros((256, 97, 4)), "test_ext-0005", s5.all_specs())
    assert got == {"status": "missing truth", "passed": False, "collapsed": None}


def test_cylinder_score_on_potential_flow():
    X, _ = grid(Cylinder(), TRAIN)
    th = -2 * np.pi * np.arange(256) / 256
    q = np.zeros(X.shape + (4,))
    q[..., 0], q[..., 1], q[..., 3] = 1.0, 1.0, 1.0
    q[:, 0, 3] = 1 + (1 - 4 * np.sin(th) ** 2) * 0.5 * 1.4 * 0.1 ** 2
    q[:, 0, 1], q[:, 0, 2] = 2 * np.abs(np.sin(th)), 0.0
    sc = s5.cylinder_score(q)
    assert sc["passed"] and not sc["collapsed"] and sc["max_cp_error"] < 1e-9
    q[:, 0, 1] = 0.0
    assert s5.cylinder_score(q)["collapsed"]
```

`tests/test_s5_pinn.py`:

```python
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from apno.models.pinn_euler_av import AvConfig

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_pinn.py"


def _diff(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    return {k: v for k, v in db.items() if da[k] != v}


@pytest.fixture
def s5(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s5_pinn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    Path("results/s5").mkdir(parents=True)
    Path("results/s5/su2.json").write_text(json.dumps({"tried": [], "selected": "s5-val-B"}))
    mod.trained, mod.verdicts, mod.closure_s = [], [], 1e-6

    def fake_train(case, cfg, log=print):
        mod.trained.append(cfg)
        return object(), {"wall_s": 100.0, "cpu_s": 50.0, "device": "cpu", "func_evals": 10,
                          "stalled": 0, "diverged": False, "reason": None,
                          "final": {"total": 1.0}, "history": []}

    def fake_score(q, case_id, specs):
        ok = mod.verdicts.pop(0) if mod.verdicts else False
        return {"status": "ok", "passed": ok, "collapsed": False}

    monkeypatch.setattr(mod, "train", fake_train)
    monkeypatch.setattr(mod, "evaluate_on_grid", lambda pinn, shape: np.ones(shape + (4,)))
    monkeypatch.setattr(mod, "score_run", fake_score)
    monkeypatch.setattr(mod, "time_closure",
                        lambda case, cfg: {"device": "cpu", "closure_s": mod.closure_s})
    return mod


def _recipe(wall_s=3600.0, closure_s=0.1):
    Path("results/s5/recipe.json").write_text(json.dumps({
        "config": "T1", "cfg": dataclasses.asdict(AvConfig()), "validation_case": "s5-val-B",
        "gate_run": {"wall_s": wall_s, "closure_s": closure_s}}))


def test_gate_configs_change_one_thing_each(s5):
    base = AvConfig()
    got = {n: _diff(base, c) for n, c in s5.gate_configs()}
    assert list(got) == ["T1", "T2", "T3", "T4"]
    assert got == {"T1": {}, "T2": {"sensor": "B"}, "T3": {"viscosity": "scalar"},
                   "T4": {"te_exclusion": 0.02}}


def test_steps_plan_reverts_one_ingredient_per_arm(s5):
    r = AvConfig(sensor="B")
    plan = s5.steps_plan(r, "s5-val-B")
    names = ["A1", "A2", "A3", "A4", "A5", "A6a", "A6b", *s5.STEP3_IN, *s5.STEP3_EXT]
    assert [n for n, _, _ in plan] == names
    d = {n: _diff(r, c) for n, _, c in plan}
    assert d["A1"] == {"weighting": "uniform", "lambda_res": 1.0, "w_boundary": 10.0}
    assert d["A2"] == {"closure": "enthalpy"} and d["A3"] == {"optimizer": "adam_lbfgs"}
    assert d["A4"] == {"encoding": "periodic"} and d["A5"] == {"collocation": "random"}
    assert all(d[n] == {} for n in names[5:])
    cases = {n: c for n, c, _ in plan}
    assert cases["A1"] == "s5-val-B" and cases["A6a"] == "cylinder"
    assert cases["A6b"] == "test_in-0022" and cases["test_ext-0006"] == "test_ext-0006"


def test_the_gate_stops_at_the_first_pass(s5):
    s5.verdicts = [False, True]
    assert s5.main(["gate"]) == 0
    assert len(s5.trained) == 2
    recipe = json.loads(Path("results/s5/recipe.json").read_text())
    assert recipe["config"] == "T2" and recipe["cfg"]["sensor"] == "B"
    assert recipe["gate_run"] == {"wall_s": 100.0, "closure_s": 1e-6}
    gate = json.loads(Path("results/s5/gate.json").read_text())
    assert list(gate["configs"]) == ["T1", "T2"] and gate["validation_case"] == "s5-val-B"
    assert Path("runs/predictions_s5/pinn-euler_av-T1.h5").exists()
    assert not Path("runs/predictions").exists()


def test_the_gate_resumes_after_recorded_failures(s5):
    entry = {"score": {"status": "ok", "passed": False, "collapsed": False},
             "cfg": dataclasses.asdict(AvConfig()), "wall_s": 1.0}
    Path("results/s5/gate.json").write_text(json.dumps(
        {"validation_case": "s5-val-B", "configs": {"T1": entry}}))
    s5.verdicts = [True]
    assert s5.main(["gate"]) == 0
    assert len(s5.trained) == 1 and s5.trained[0].sensor == "B"
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] == "T2"


def test_a_decided_gate_does_not_run_again(s5):
    _recipe()
    assert s5.main(["gate"]) == 0 and s5.trained == []


def test_no_pass_records_a_null_recipe_and_the_steps_stop(s5):
    s5.verdicts = [False] * 4
    assert s5.main(["gate"]) == 0 and len(s5.trained) == 4
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] is None
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 4


def test_the_gate_refuses_past_its_budget(s5):
    s5.closure_s = 1.0
    assert s5.main(["gate"]) == 3 and s5.trained == []
    pre = json.loads(Path("results/s5/preflight_gate.json").read_text())
    assert pre["hours"]["T1"] == pytest.approx(40 * 20 * 1250 / 3600)
    assert pre["within_budget"] is False


def test_the_gate_needs_a_validation_case(s5):
    Path("results/s5/su2.json").write_text(json.dumps({"tried": [], "selected": None}))
    assert s5.main(["gate"]) == 2 and s5.trained == []


def test_steps_reproject_from_the_gate_run_and_write_only_under_predictions_s5(s5):
    _recipe(wall_s=3600.0, closure_s=0.1)
    s5.closure_s = 0.2
    assert s5.main(["steps", "--preflight-only"]) == 0 and s5.trained == []
    pre = json.loads(Path("results/s5/preflight_steps.json").read_text())
    assert pre["hours"]["A1"] == pytest.approx(2.0)
    assert pre["hours"]["A3"] == pytest.approx((30000 + 6250) * 0.2 / 3600)
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 13
    record = json.loads(Path("results/s5/runs.json").read_text())
    assert list(record) == [n for n, _, _ in s5.steps_plan(AvConfig(), "s5-val-B")]
    assert record["A6a"]["case_id"] == "cylinder" and record["A1"]["score"]["status"] == "ok"
    assert all(p.name.startswith("pinn-euler_av-") for p in Path("runs/predictions_s5").iterdir())
    assert not Path("runs/predictions").exists()


def test_steps_shard_by_plan_index(s5):
    _recipe()
    assert s5.main(["steps", "--shard", "1/2"]) == 0 and len(s5.trained) == 6
    record = json.loads(Path("results/s5/runs_shard1of2.json").read_text())
    assert list(record) == ["A2", "A4", "A6a", s5.STEP3_IN[0], s5.STEP3_IN[2], s5.STEP3_EXT[1]]


def test_steps_need_a_recipe(s5):
    assert s5.main(["steps"]) == 2 and s5.trained == []


def test_a_diverged_run_is_recorded_without_a_prediction_file(s5, monkeypatch):
    def diverging(case, cfg, log=print):
        s5.trained.append(cfg)
        return object(), {"wall_s": 5.0, "cpu_s": 5.0, "device": "cpu", "func_evals": 1,
                          "stalled": 0, "diverged": True, "reason": "non-finite loss",
                          "final": {}, "history": []}

    monkeypatch.setattr(s5, "train", diverging)
    assert s5.main(["gate"]) == 0
    gate = json.loads(Path("results/s5/gate.json").read_text())
    assert gate["configs"]["T1"]["score"] == {"status": "diverged", "passed": False,
                                              "collapsed": None}
    assert not Path("runs/predictions_s5/pinn-euler_av-T1.h5").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_s5.py tests/test_s5_pinn.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'apno.score.s5'`.

- [ ] **Step 3: Write the implementation**

`src/apno/score/s5.py`:

```python
"""Stage S5's cases, truth and run scores, shared by scripts/s5_pinn.py and scripts/s5_score.py
(S5 design §3.3, §4, §5).

A run is scored at the 256x97 nodes against the truth (SU2 at 512x193 sampled every second node)
by the gate metric (RMAE of Cp and of local Mach, `apno.score.rmae`), the S4 scores
(`score_case`) and the collapse check: its largest wall speed below half the truth's, the
fluid-at-rest signature. The cylinder (A6a) is scored by the S3a criterion against potential flow,
whose wall speed peaks at 2 U_inf. Every score dict carries status, passed and collapsed.
"""

from pathlib import Path

import h5py
import numpy as np

from apno.data.build import load_case
from apno.data.sampling import case_specs, s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN, Cylinder, Section, grid
from apno.models.pinn import PinnCase
from apno.models.train_pinn import circle_cp_error
from apno.score.forces import wall_forces
from apno.score.metrics import score_case
from apno.score.rmae import rmae_cp_mach

GATE_RMAE = 0.01
STEP3_IN = ("test_in-0004", "test_in-0031", "test_in-0040")  # the supercritical showcase
STEP3_EXT = ("test_ext-0005", "test_ext-0009", "test_ext-0006")  # S1's rule, p = 0.1, 0.5, 0.9
CYLINDER = "cylinder"
ABLATION_AIRFOIL = "test_in-0022"
SU2_ROOT = Path("runs/su2")
H5 = {"truth": Path("data/hdf5/512x193.h5"), "floor": Path("data/hdf5/256x97.h5")}


def all_specs():
    specs = {s.case_id: s for split in ("test_in", "test_ext") for s in case_specs(split, TRAIN)}
    specs.update({s.case_id: s for s in s5_validation_specs(TRAIN)})
    return specs


def pinn_case(case_id, specs):
    if case_id == CYLINDER:
        return PinnCase(0.1, 0.0, Cylinder())
    s = specs[case_id]
    return PinnCase(s.mach, s.alpha_deg, Section(s.t, s.m))


def truth_and_floor(case_id):
    """(truth at the 256x97 nodes, the 256x97 floor, the truth's max surface Mach), or
    (None, None, None) when either is missing (not run, not converged, not in the dataset)."""
    if case_id.startswith("s5-val-"):
        fine = {s.case_id: s for s in s5_validation_specs(FINE)}[case_id]
        coarse = {s.case_id: s for s in s5_validation_specs(TRAIN)}[case_id]
        t, f = load_case(SU2_ROOT, fine), load_case(SU2_ROOT, coarse)
        if t is None or f is None:
            return None, None, None
        return t["q"][::2, ::2], f["q"], float(t["max_surface_mach"])
    split = case_id.split("-")[0]
    got = {}
    for name, path in H5.items():
        if not path.exists():
            return None, None, None
        with h5py.File(path) as h:
            ids = list(h[split]["case_id"].asstr()[...])
            if case_id not in ids:
                return None, None, None
            k = ids.index(case_id)
            got[name] = (h[split]["q"][k], float(h[split]["max_surface_mach"][k]))
    return got["truth"][0][::2, ::2], got["floor"][0], got["truth"][1]


def wall_speed_max(q):
    return float(np.sqrt(q[:, 0, 1] ** 2 + q[:, 0, 2] ** 2).max())


def airfoil_score(q, case_id, specs):
    truth, floor, mmax = truth_and_floor(case_id)
    if truth is None:
        return {"status": "missing truth", "passed": False, "collapsed": None}
    s = specs[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    r = rmae_cp_mach(q, truth, X, Y, s.mach)
    regime = "supercritical" if mmax > 1.0 else "subcritical"
    return {"status": "ok", "rmae": r, "floor_rmae": rmae_cp_mach(floor, truth, X, Y, s.mach),
            "passed": r["cp"] <= GATE_RMAE and r["mach"] <= GATE_RMAE,
            "collapsed": wall_speed_max(q) < 0.5 * wall_speed_max(truth), "regime": regime,
            "s4": score_case(q, truth, X, Y, s.mach, s.alpha_deg, regime)}


def cylinder_score(q):
    err, cp = circle_cp_error(q, 0.1)
    X, Y = grid(Cylinder(), TRAIN)
    cl, cd, cm = wall_forces(X[:, 0], Y[:, 0], cp, 0.0)
    return {"status": "ok", "max_cp_error": err, "cl": cl, "cd": cd, "cm": cm,
            "passed": bool(err <= 0.1 and abs(cl) < 1e-2 and abs(cd) < 1e-2),
            "collapsed": wall_speed_max(q) < 0.5 * 2.0}


def score_run(q, case_id, specs):
    return cylinder_score(q) if case_id == CYLINDER else airfoil_score(q, case_id, specs)
```

`scripts/s5_pinn.py`:

```python
"""Stage S5: the Euler PINN with localised artificial viscosity (S5 design §3-§5, §7).

    python scripts/s5_pinn.py gate  [--budget-hours 72] [--preflight-only] [--preflight-file PATH]
    python scripts/s5_pinn.py steps [--budget-hours 72] [--preflight-only] [--preflight-file PATH]
                                    [--shard K/N]

gate: the tuning configurations T1-T4 (design §3.4) on the validation case selected by
scripts/s5_su2.py (results/s5/su2.json), in order; the first that passes the gate (RMAE of Cp and of
local Mach each <= 1 % against SU2 at 512x193) is recipe R, written to results/s5/recipe.json. A
configuration already in results/s5/gate.json is not re-run; once recipe.json exists the gate is
decided and does not run again. If none passes, recipe.json records config null: S5 stops after
step 1 and the author decides.

steps: the ablation arms A1-A6b (design §4) and the step-3 cases (§5), each a variant of R, sharded
by plan index K mod N, recorded in results/s5/runs{suffix}.json.

Before either phase trains, each unfinished run is timed for one closure on its own case
(`time_closure`, viscous term forced on everywhere). The gate budgets the upper bound of every
outer iteration at max_eval closures; the steps re-project each run from R's measured gate run,
scaled by the ratio of timed closures (A3 keeps its upper bound). A phase REFUSES TO START (exit 3)
when its unfinished runs' projection exceeds --budget-hours; the check is per phase (design §7.1).
Prediction files go to runs/predictions_s5/pinn-euler_av-<name>.h5, never runs/predictions/.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

from apno.data.predictions import write_predictions
from apno.geometry.ogrid import TRAIN
from apno.models.pinn_euler_av import AvConfig, config_from_dict
from apno.models.train_pinn_av import (
    evaluate_on_grid, projected_upper_s, residual_nodes, time_closure, train,
)
from apno.score.s5 import (
    ABLATION_AIRFOIL, CYLINDER, STEP3_EXT, STEP3_IN, all_specs, pinn_case, score_run,
)

REPO = Path(__file__).resolve().parents[1]
RES, PRED5 = Path("results/s5"), Path("runs/predictions_s5")
BUDGET_HOURS = 72.0


def gate_configs():
    base, rep = AvConfig(), dataclasses.replace
    return [("T1", base), ("T2", rep(base, sensor="B")), ("T3", rep(base, viscosity="scalar")),
            ("T4", rep(base, te_exclusion=0.02))]


def steps_plan(r, validation_id):
    """[(name, case_id, cfg)] in priority order (design §4, §5)."""
    rep = dataclasses.replace
    runs = [("A1", validation_id, rep(r, weighting="uniform", lambda_res=1.0, w_boundary=10.0)),
            ("A2", validation_id, rep(r, closure="enthalpy")),
            ("A3", validation_id, rep(r, optimizer="adam_lbfgs")),
            ("A4", validation_id, rep(r, encoding="periodic")),
            ("A5", validation_id, rep(r, collocation="random")),
            ("A6a", CYLINDER, r), ("A6b", ABLATION_AIRFOIL, r)]
    return runs + [(cid, cid, r) for cid in STEP3_IN + STEP3_EXT]


def shard(runs, k, n):
    return [x for i, x in enumerate(runs) if i % n == k]


def parse_shard(text):
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard wants K/N, got {text!r}") from None
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard needs 0 <= K < N, got {text!r}")
    return k, n


def code_version():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def prediction_path(name):
    return PRED5 / f"pinn-euler_av-{name}.h5"


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def run_one(name, case_id, cfg, specs):
    """Train, evaluate, write the prediction file (unless diverged), score: the record entry."""
    pinn, rec = train(pinn_case(case_id, specs), cfg)
    keep = ("wall_s", "cpu_s", "device", "func_evals", "stalled", "diverged", "reason", "final",
            "history")
    entry = {"case_id": case_id, "cfg": dataclasses.asdict(cfg), **code_version(),
             **{k: rec[k] for k in keep}}
    if rec["diverged"]:
        entry["score"] = {"status": "diverged", "passed": False, "collapsed": None}
        return entry
    q = evaluate_on_grid(pinn, TRAIN)
    path = prediction_path(name)
    write_predictions(path, [case_id], q[None], {
        "method": "pinn", "tag": path.stem, "variant": f"euler_av-{name}",
        "formulation": "euler_av", "seed": cfg.seed, "n_train": 0,
        "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"], "query_s": rec["wall_s"],
        "train_device": rec["device"]})
    entry["score"] = score_run(q, case_id, specs)
    return entry


def timings(runs, specs, preflight_file=None):
    """{name: {"device", "closure_s"}} per run, timed on its own case or reused by name."""
    held = json.loads(Path(preflight_file).read_text())["timing_s"] if preflight_file else {}
    out = {}
    for name, cid, cfg in runs:
        out[name] = held.get(name) or time_closure(pinn_case(cid, specs), cfg)
        print(f"preflight {name} on {cid}: {out[name]}", flush=True)
    return out


def reprojected_hours(cfg, closure_s, recipe):
    if cfg.optimizer == "adam_lbfgs":
        return projected_upper_s(cfg, 0, closure_s) / 3600
    g = recipe["gate_run"]
    return g["wall_s"] * closure_s / g["closure_s"] / 3600


def _budget(path, timing, hours, budget, note):
    total = sum(hours.values())
    _write(path, {"timing_s": timing, "hours": hours, "projected_total_hours": total,
                  "budget_hours": budget, "within_budget": total <= budget, "note": note})
    print(f"projected {total:.1f} h for {len(hours)} runs against {budget:.1f} h", flush=True)
    if total > budget:
        print("REFUSING TO START: the projection exceeds the budget. Raise --budget-hours "
              "deliberately, on record, or cut runs as a recorded decision.", flush=True)
        return False
    return True


def run_gate(args, specs):
    su2_path = RES / "su2.json"
    vid = json.loads(su2_path.read_text()).get("selected") if su2_path.exists() else None
    if not vid:
        print("no validation case: run scripts/s5_su2.py first (results/s5/su2.json)", flush=True)
        return 2
    if (RES / "recipe.json").exists():
        print("results/s5/recipe.json exists: the gate is decided", flush=True)
        return 0
    gate_path = RES / "gate.json"
    gate = (json.loads(gate_path.read_text()) if gate_path.exists()
            else {"validation_case": vid, "configs": {}})
    configs = gate_configs()
    todo = [(n, vid, c) for n, c in configs if n not in gate["configs"]]
    timing = timings(todo, specs, args.preflight_file)
    n_nodes = len(residual_nodes(pinn_case(vid, specs), AvConfig())["xi"])
    hours = {n: projected_upper_s(c, n_nodes, timing[n]["closure_s"]) / 3600 for n, _, c in todo}
    if not _budget(RES / "preflight_gate.json", timing, hours, args.budget_hours,
                   "upper bound: every outer iteration at max_eval closures, viscous term on"):
        return 3
    if args.preflight_only:
        return 0
    for name, cfg in configs:
        if name not in gate["configs"]:
            gate["configs"][name] = run_one(name, vid, cfg, specs)
            _write(gate_path, gate)
        entry = gate["configs"][name]
        if entry["score"].get("passed"):
            closure = (timing.get(name) or time_closure(pinn_case(vid, specs), cfg))["closure_s"]
            _write(RES / "recipe.json", {"config": name, "cfg": entry["cfg"],
                                         "validation_case": vid,
                                         "gate_run": {"wall_s": entry["wall_s"],
                                                      "closure_s": closure}})
            print(f"{name} passed the gate: it is recipe R", flush=True)
            return 0
    _write(RES / "recipe.json", {"config": None, "cfg": None, "validation_case": vid,
                                 "gate_run": None})
    print("no configuration passed the gate: S5 stops after step 1", flush=True)
    return 0


def run_steps(args, specs):
    rp = RES / "recipe.json"
    if not rp.exists():
        print("no results/s5/recipe.json: run the gate phase first", flush=True)
        return 2
    recipe = json.loads(rp.read_text())
    if recipe["config"] is None:
        print("the gate failed: S5 stopped after step 1", flush=True)
        return 0
    runs = shard(steps_plan(config_from_dict(recipe["cfg"]), recipe["validation_case"]),
                 *args.shard)
    k, n = args.shard
    suffix = "" if n == 1 else f"_shard{k}of{n}"
    rec_path = RES / f"runs{suffix}.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    todo = [x for x in runs if x[0] not in record]
    timing = timings(todo, specs, args.preflight_file)
    hours = {name: reprojected_hours(cfg, timing[name]["closure_s"], recipe)
             for name, _, cfg in todo}
    if not _budget(RES / f"preflight_steps{suffix}.json", timing, hours, args.budget_hours,
                   "re-projected from R's measured gate run; A3 at its upper bound"):
        return 3
    if args.preflight_only:
        return 0
    for name, cid, cfg in runs:
        if name in record:
            continue
        record[name] = run_one(name, cid, cfg, specs)
        _write(rec_path, record)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("gate", "steps"))
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--preflight-file", type=Path, default=None, metavar="PATH")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="K/N")
    args = ap.parse_args(argv)
    if args.phase == "gate" and args.shard != (0, 1):
        ap.error("the gate is sequential: --shard applies to the steps phase only")
    RES.mkdir(parents=True, exist_ok=True)
    specs = all_specs()
    return run_gate(args, specs) if args.phase == "gate" else run_steps(args, specs)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_s5.py tests/test_s5_pinn.py -q && ruff check src/apno/score/s5.py scripts/s5_pinn.py tests/test_s5.py tests/test_s5_pinn.py`
Expected: all pass; `All checks passed!` (`test_step3_extrapolation_cases_follow_the_showcase_rule` runs here because `data/hdf5/` exists on this machine.)

- [ ] **Step 5: Commit** (integrator)

```bash
git add src/apno/score/s5.py scripts/s5_pinn.py tests/test_s5.py tests/test_s5_pinn.py
git commit -m "S5: the PINN stage script: sequential gate, sharded ablation and comparison runs, per-phase budget" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The S5 scoring script

**Files:**
- Create: `scripts/s5_score.py`
- Test: `tests/test_s5_score.py`

**Interfaces:**
- Consumes: Task 7's `apno.score.s5` (`STEP3_IN`, `STEP3_EXT`, `all_specs`, `truth_and_floor`) and record formats (`results/s5/gate.json`, `recipe.json`, `runs*.json` entries with `score`, `diverged`, `wall_s`, `cpu_s`, `device`); Task 3's `rmae_cp_mach`; `apno.data.predictions.read_predictions(path, dataset)`, `datasets(path)`; `apno.score.metrics.score_case`, `FLUX_LINES`; `apno.score.report.plot_showcase(xw, curves, title, path)`; `apno.physics.flow.pressure_coefficient`; `results/s4/costs.json` (`costs.su2.per_cpu`, `per_wall`; `costs.fno.per`).
- Produces: `results/s5/ablation.json`, `scores.json`, `scores.md`, `costs.json`, `showcase_s5_<case>.png`; functions `verdict(entry)`, `ablation(runs)`, `method_q(method, case_id)`, `score_column(method, case_id, specs)`, `group_ratios(table)`, `degradation(groups)`, `costs(runs, s4_costs)`, `markdown(...)`, `main(argv)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_s5_score.py`:

```python
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_score.py"


@pytest.fixture
def sc():
    spec = importlib.util.spec_from_file_location("s5_score", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verdict_rules(sc):
    ok = {"status": "ok", "passed": True, "collapsed": False}
    assert sc.verdict(None) == "not run"
    assert sc.verdict({"diverged": True, "score": {"status": "diverged"}}) == "needed"
    assert sc.verdict({"diverged": False, "score": ok}) == "not needed"
    assert sc.verdict({"diverged": False, "score": {**ok, "collapsed": True}}) == "needed"
    assert sc.verdict({"diverged": False, "score": {**ok, "passed": False}}) == "needed"
    assert sc.verdict({"diverged": False, "score": {"status": "missing truth"}}) == "needed"


def test_ablation_lists_every_arm(sc):
    runs = {"A1": {"diverged": False, "score": {"status": "ok", "passed": True,
                                               "collapsed": False}}}
    got = sc.ablation(runs)
    assert list(got) == ["A1", "A2", "A3", "A4", "A5", "A6a", "A6b"]
    assert got["A1"]["verdict"] == "not needed" and got["A2"]["verdict"] == "not run"


def _table(sc):
    def row(nf):
        return {"status": "ok", "near_field_rel_l2": nf}

    table = {}
    for c in sc.STEP3_IN:
        table[c] = {"floor": row(0.01), "fno-256": row(0.02), "fno-128": {"status": "missing"},
                    "pod": row(0.03), "pinn-pure": row(0.5), "euler_av": row(0.05)}
    for c in sc.STEP3_EXT:
        table[c] = {"floor": row(0.05), "fno-256": row(0.2), "fno-128": {"status": "missing"},
                    "pod": row(0.5), "pinn-pure": {"status": "missing"}, "euler_av": row(0.1)}
    return table


def test_group_ratios_and_degradation(sc):
    g = sc.group_ratios(_table(sc))
    assert g["fno-256"]["in"] == pytest.approx(2.0) and g["fno-256"]["ext"] == pytest.approx(4.0)
    assert g["euler_av"]["in"] == pytest.approx(5.0) and g["euler_av"]["ext"] == pytest.approx(2.0)
    assert g["fno-128"]["in"] is None and g["fno-128"]["n_in"] == 0
    assert g["pinn-pure"]["n_ext"] == 0
    d = sc.degradation(g)
    assert d["fno-256"] == pytest.approx(2.0) and d["euler_av"] == pytest.approx(0.4)
    assert d["fno-128"] is None and "pinn-pure" not in d


def test_missing_truth_or_file_is_a_status(sc, monkeypatch):
    specs = sc.all_specs()
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (None, None, None))
    assert sc.score_column("euler_av", "test_ext-0005", specs) == {"status": "missing truth"}
    q = np.ones((256, 97, 4))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (q, q, 1.2))
    monkeypatch.setattr(sc, "method_q", lambda m, cid: None)
    assert sc.score_column("euler_av", "test_ext-0005", specs) == {"status": "missing"}


def test_costs_against_su2(sc):
    runs = {c: {"wall_s": 3600.0, "cpu_s": 1800.0, "device": "mps"}
            for c in sc.STEP3_IN + sc.STEP3_EXT}
    s4 = {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}
    c = sc.costs(runs, s4)
    assert c["euler_av"]["per_wall"] == 3600.0 and c["euler_av"]["n_runs"] == 6
    assert c["ratio_to_su2_cpu"] == {"wall": pytest.approx(18.0), "cpu": pytest.approx(9.0)}
    assert sc.costs({}, s4)["euler_av"] is None


def test_main_writes_its_outputs_when_nothing_has_run(sc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("results/s4").mkdir(parents=True)
    Path("results/s4/costs.json").write_text(json.dumps(
        {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (None, None, None))
    assert sc.main([]) == 0
    scores = json.loads(Path("results/s5/scores.json").read_text())
    assert scores["table"]["test_ext-0005"]["euler_av"] == {"status": "missing truth"}
    assert Path("results/s5/scores.md").exists() and Path("results/s5/ablation.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_s5_score.py -q`
Expected: errors, `FileNotFoundError` (or `ModuleNotFoundError`) for `scripts/s5_score.py`.

- [ ] **Step 3: Write the implementation** — `scripts/s5_score.py`:

```python
"""Stage S5 scores (S5 design §3.5, §4, §5.3-5.4): the gate table, the ablation verdicts, the
step-3 comparison and the costs, into results/s5/.

    python scripts/s5_score.py

Reads the S5 records (results/s5/), the S5 prediction files (runs/predictions_s5/) and, for the
comparison columns, the S4 prediction files (runs/predictions/, read only) and the truth.
results/s4 is read for SU2's and the FNO's costs and never written. A column whose prediction file
or truth is missing is reported as a status and left out of the group means, with the counts kept.

Ablation verdict (design §4): "needed" if the arm failed its pass criterion, collapsed, diverged or
could not be scored; "not needed" if it passed; "not run" if it has no record. The comparison
(§5.3): per case and method the S4 scores and the gate's RMAE; per method the mean near-field ratio
to the floor over the in-range and over the extrapolation cases; degradation = the extrapolation
mean over the in-range mean, for the FNO, POD and the new PINN.
"""

import json
import sys
from pathlib import Path

import numpy as np

from apno.data.predictions import datasets, read_predictions
from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import pressure_coefficient
from apno.score.metrics import FLUX_LINES, score_case
from apno.score.report import plot_showcase
from apno.score.rmae import rmae_cp_mach
from apno.score.s5 import STEP3_EXT, STEP3_IN, all_specs, truth_and_floor

RES, PRED5, PRED = Path("results/s5"), Path("runs/predictions_s5"), Path("runs/predictions")
ARMS = ("A1", "A2", "A3", "A4", "A5", "A6a", "A6b")
METHODS = ("floor", "fno-256", "fno-128", "pod", "pinn-pure", "euler_av")
DEGRADED = ("fno-256", "fno-128", "pod", "euler_av")


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_runs():
    out = {}
    for p in sorted(RES.glob("runs*.json")):
        out.update(json.loads(p.read_text()))
    return out


def verdict(entry):
    if entry is None:
        return "not run"
    sc = entry.get("score") or {}
    if entry.get("diverged") or sc.get("status") != "ok":
        return "needed"
    return "needed" if (not sc.get("passed") or sc.get("collapsed")) else "not needed"


def ablation(runs):
    return {a: {"verdict": verdict(runs.get(a)), "score": (runs.get(a) or {}).get("score")}
            for a in ARMS}


def method_q(method, case_id):
    """q of a comparison column for one case, or None if its file or dataset is missing."""
    files = {"euler_av": (PRED5 / f"pinn-euler_av-{case_id}.h5", "q"),
             "fno-256": (PRED / "fno-n896-s0.h5", "q"),
             "fno-128": (PRED / "fno-n896-s0.h5", "q_128x49"),
             "pod": (PRED / "pod-n896.h5", "q"),
             "pinn-pure": (PRED / f"pinn-pure-{case_id}.h5", "q")}
    path, ds = files[method]
    if not path.exists() or ds not in datasets(path):
        return None
    return read_predictions(path, ds)[0].get(case_id)


def score_column(method, case_id, specs):
    truth, floor, mmax = truth_and_floor(case_id)
    if truth is None:
        return {"status": "missing truth"}
    q = floor if method == "floor" else method_q(method, case_id)
    if q is None:
        return {"status": "missing"}
    s = specs[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    t, flux = truth, FLUX_LINES
    if method == "fno-128":
        X, Y, t, flux = X[::2, ::2], Y[::2, ::2], truth[::2, ::2], tuple(j // 2 for j in FLUX_LINES)
    regime = "supercritical" if mmax > 1.0 else "subcritical"
    out = score_case(q, t, X, Y, s.mach, s.alpha_deg, regime, flux)
    if out["status"] == "ok":
        out["rmae"] = rmae_cp_mach(q, t, X, Y, s.mach)
    return out


def group_ratios(table):
    out = {}
    for m in METHODS[1:]:
        out[m] = {}
        for grp, cases in (("in", STEP3_IN), ("ext", STEP3_EXT)):
            r = [table[c][m]["near_field_rel_l2"] / table[c]["floor"]["near_field_rel_l2"]
                 for c in cases if table[c][m].get("status") == "ok"
                 and table[c]["floor"].get("status") == "ok"]
            out[m][grp] = float(np.mean(r)) if r else None
            out[m][f"n_{grp}"] = len(r)
    return out


def degradation(groups):
    return {m: (groups[m]["ext"] / groups[m]["in"]
                if groups[m]["in"] and groups[m]["ext"] is not None else None)
            for m in DEGRADED}


def costs(runs, s4_costs):
    su2 = s4_costs["costs"]["su2"]
    step3 = [runs[c] for c in STEP3_IN + STEP3_EXT if c in runs]
    if not step3:
        return {"euler_av": None, "su2": su2}
    per_wall = float(np.mean([r["wall_s"] for r in step3]))
    per_cpu = float(np.mean([r["cpu_s"] for r in step3]))
    return {"euler_av": {"n_runs": len(step3), "per_wall": per_wall, "per_cpu": per_cpu,
                         "devices": sorted({r["device"] for r in step3})},
            "su2": {"per_wall": su2["per_wall"], "per_cpu": su2["per_cpu"]},
            "fno_per_query": s4_costs["costs"]["fno"]["per"],
            "ratio_to_su2_cpu": {"wall": per_wall / su2["per_cpu"],
                                 "cpu": per_cpu / su2["per_cpu"]}}


def _fmt(x, f="{:.4g}"):
    return "—" if x is None else f.format(x)


def markdown(gate, abl, table, groups, degr, cost):
    out = ["## Gate (design §3)", "", "| config | passed | RMAE Cp | RMAE Mach | collapsed |",
           "|---|---|---|---|---|"]
    for name, e in ((gate or {}).get("configs") or {}).items():
        sc = e.get("score") or {}
        r = sc.get("rmae") or {}
        out.append(f"| {name} | {sc.get('passed')} | {_fmt(r.get('cp'))} | "
                   f"{_fmt(r.get('mach'))} | {sc.get('collapsed')} |")
    out += ["", "## Ablation (design §4)", "", "| arm | verdict |", "|---|---|"]
    out += [f"| {a} | {v['verdict']} |" for a, v in abl.items()]
    out += ["", "## Comparison (design §5): near field, and ratio to the floor", "",
            "| case | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for c, row in table.items():
        fl = row["floor"].get("near_field_rel_l2") if row["floor"].get("status") == "ok" else None
        cells = []
        for m in METHODS:
            nf = row[m].get("near_field_rel_l2") if row[m].get("status") == "ok" else None
            ratio = nf / fl if nf is not None and fl else None
            cells.append(row[m].get("status") if nf is None
                         else f"{nf:.4f} ({_fmt(ratio, '{:.1f}')}x)")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out += ["", "| method | in-range ratio (n) | extrapolation ratio (n) | degradation |",
            "|---|---|---|---|"]
    for m, g in groups.items():
        out.append(f"| {m} | {_fmt(g['in'])} ({g['n_in']}) | {_fmt(g['ext'])} ({g['n_ext']}) | "
                   f"{_fmt(degr.get(m))} |")
    out += ["", "## Costs (design §5.4)", "", "```", json.dumps(cost, indent=2), "```", ""]
    return "\n".join(out)


def plots(specs, cases):
    for c in cases:
        truth, floor, _ = truth_and_floor(c)
        if truth is None:
            continue
        s = specs[c]
        X, _ = grid(Section(s.t, s.m), TRAIN)
        curves = {"truth": pressure_coefficient(truth[:, 0, 3], s.mach),
                  "su2 256x97": pressure_coefficient(floor[:, 0, 3], s.mach)}
        for m in ("fno-256", "pod", "pinn-pure", "euler_av"):
            q = method_q(m, c)
            if q is not None:
                curves[m] = pressure_coefficient(q[:, 0, 3], s.mach)
        plot_showcase(X[:, 0], curves, c, RES / f"showcase_s5_{c}.png")


def main(argv=None):
    RES.mkdir(parents=True, exist_ok=True)
    specs = all_specs()
    gate, runs = _load(RES / "gate.json"), load_runs()
    abl = ablation(runs)
    table = {c: {m: score_column(m, c, specs) for m in METHODS} for c in STEP3_IN + STEP3_EXT}
    groups = group_ratios(table)
    degr = degradation(groups)
    cost = costs(runs, json.loads(Path("results/s4/costs.json").read_text()))
    (RES / "ablation.json").write_text(json.dumps(abl, indent=2, default=float))
    (RES / "scores.json").write_text(json.dumps(
        {"table": table, "groups": groups, "degradation": degr}, indent=2, default=float))
    (RES / "costs.json").write_text(json.dumps(cost, indent=2, default=float))
    (RES / "scores.md").write_text(markdown(gate, abl, table, groups, degr, cost))
    plots(specs, STEP3_IN + STEP3_EXT)
    print((RES / "scores.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_s5_score.py -q && ruff check scripts/s5_score.py tests/test_s5_score.py`
Expected: all pass; `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add scripts/s5_score.py tests/test_s5_score.py
git commit -m "S5: the scoring script: ablation verdicts, the comparison with degradation, costs, plots" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Docs and full verification

**Files:**
- Modify: `CLAUDE.md` (the stage-scripts table and the architecture bullets)
- Modify: `README.md` (the "Running the pipeline" table)

**Interfaces:**
- Consumes: the commands of Tasks 4, 7 and 8.
- Produces: documentation only.

- [ ] **Step 1: Add the S5 rows to CLAUDE.md's stage-scripts table** (after the S4 row):

```markdown
| S5 | `SU2_BIN_DIR=... python scripts/s5_su2.py` | `runs/su2/<level>/s5-val-*/`, `results/s5/su2.json` |
| S5 | `python scripts/s5_pinn.py gate` (then `steps --shard K/N`) | `runs/predictions_s5/`, `results/s5/` |
| S5 | `python scripts/s5_score.py` | `results/s5/` |
```

- [ ] **Step 2: Add one architecture bullet to CLAUDE.md** (after the **PINN** bullet):

```markdown
- **S5 PINN** (`models/pinn_euler_av.py`, `train_pinn_av.py`, `physics/viscosity.py`): an Euler
  PINN built to Wassing, Langer & Bekemeyer's recipe — primitive outputs, Fourier features of the
  512x193 grid indices, collocation at its nodes, mini-batch L-BFGS, and artificial viscosity
  (closed-form |A|) localised by a shock times a stagnation sensor. Its residual is div F − D (the
  dissipative sign). The S5 design is `docs/superpowers/specs/2026-10-07-s5-euler-pinn-av-design.md`;
  S5 prediction files go to `runs/predictions_s5/`, never `runs/predictions/` (S4's scorer reads
  every file there).
```

- [ ] **Step 3: Add the S5 rows to README.md's "Running the pipeline" table** (after the S4 row):

```markdown
| S5 | SU2 reference for the validation case | `SU2_BIN_DIR=... python scripts/s5_su2.py` |
| S5 | Euler PINN: gate, then ablation and comparison | `python scripts/s5_pinn.py gate`, then `steps` |
| S5 | Score S5 against the FNO and the floor | `python scripts/s5_score.py` |
```

- [ ] **Step 4: Run the full verification**

Run: `python -m pytest -q && ruff check .`
Expected: every test passes (the previous 196 plus the new S5 tests; the one SU2 test still skips), `All checks passed!`

- [ ] **Step 5: Commit** (integrator)

```bash
git add CLAUDE.md README.md
git commit -m "S5: the stage commands in CLAUDE.md and README" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review against the design

- **§1 purpose and decisions** → this plan's header and Task 7's plans (gate order, tuning budget, per-phase budget).
- **§2.1 state and scaling** → Task 5 (`freestream`, `to_dataset_q`, `from_dataset_q`). **§2.2 network** → Task 5 (`EulerAvNet`, count pinned). **§2.3 collocation and loss** → Task 5 (`losses`) and Task 6 (`residual_nodes`, batches, `boundary_points`). **§2.4 viscosity** → Task 2, wired in Task 5 (`viscous`). **§2.5 optimiser** → Task 6. **§2.6 departures** → Tasks 5–6 (TE node, boundary terms in full, n and ω₀).
- **§3.1–3.2 validation case and reference** → Task 4. **§3.3 gate metric** → Task 3, used in Task 7 (`airfoil_score`). **§3.4 tuning budget** → Task 7 (`gate_configs`, sequential `run_gate`). **§3.5 records** → Task 7 (`gate.json`, `recipe.json`, `runs/predictions_s5/`).
- **§4 ablation** → Task 7 (`steps_plan`, scored by `score_run`), verdicts in Task 8.
- **§5 comparison** → Task 7 (runs), Task 8 (`score_column`, `group_ratios`, `degradation`, `costs`, plots).
- **§6 predictions** → Task 1 (done).
- **§7 budget, order, failures** → Task 6 (`time_closure`, `projected_upper_s`, `NonFiniteLoss`, stalled), Task 7 (per-phase `_budget`, re-projection, shards, skip-finished resume, refuse to re-run a decided gate), Task 4 (fallback order).
- **§8 tests** → each listed test is in the owning task; the Review Focus lines are pinned as named there.
- **§9 files** → as listed, plus `src/apno/score/s5.py` (shared helpers; a refinement noted under File structure).

---

## Implementation notes (2026-10-08): where the code departs from this plan's text

The plan's code blocks are kept as written; the reviews of the multi-agent run changed the code in
these places, each with a regression test (commits on branch `s5`):

- **Task 2** (a813c61, d174ef0): H is built from the floored sound speed (`h = c*c/(gamma-1) + ...`),
  so M's spectrum stays {-c, 0, 0, c} and |A| stays dissipative at non-physical states; c² is also
  floored at `C2_FLOOR = 1e-3`, which keeps float32 |A| dissipative; the scalar form's |u| radicand is
  floored, so its gradient is finite at zero velocity.
- **Task 3** (4f966dd): each quad is split along the diagonal that gives two positively oriented
  triangles (folded quads are left out); a non-finite truth raises; a non-physical prediction scores
  inf.
- **Task 4** (a711e87, d18dcfe): a level with no result (SU2 killed, a driver error) is resumed, not
  fallen back from; a recorded failure at either level falls back; exit codes 0 selected, 1 unfinished
  (re-run), 2 none converged, 3 the runs disagree with the selection on record, 4 cannot run
  (environment, another driver); su2.json is written atomically and gains an `unfinished` key; a
  selection on record is never changed. The plan's test `test_falls_back_to_case_c_when_b_does_not_finish`
  was replaced: a kill is not a result (spec §4.5).
- **Task 5** (1446e4d, 114620c): every string option of `AvConfig` is checked at construction
  (`__post_init__`, ValueError naming the field); added tests: the matrix viscosity through the map,
  the energy row, boundary terms that tell their rows and columns apart.
- **Task 6** (6dd9413, 27ccdb7): a divergence's reason names where it happened, and its outer
  iteration's closures are counted; the default log flushes; `time_closure` also returns
  `lbfgs_iter_s` (the measured two-loop recursion at the full history), and
  `projected_upper_s(cfg, n_nodes, closure_s, lbfgs_iter_s=0.0)` charges it per iteration.
- **Task 7** (ebfb980, fce42ab, 04bc30b): resume across shard layouts (any `runs*.json` counts); a
  prediction file without a record is scored, never retrained; the truth is checked before training
  (exit 2); a scoring failure is kept and re-scored (exit 4); records and prediction files are
  written atomically; entries record `argv`, `eval_device` and the code version taken before
  training; a gate decided in gate.json without recipe.json is finished at once, without timing; the
  projections include the L-BFGS overhead; tests of the real truth loader.
- **Task 8** (6de4b58, 6bdbbdf): per-run costs and the break-even plot; |CL| and |CD| errors beside
  the near-field ratios; every arm's losses and costs; a diverged step-3 run is a status; degradation
  only on all six cases; S4's "by construction" labels; the ablation caveat in ablation.json; the
  gate's status column; an RMAE table; escaped table cells; P7's ratio on S4's cost rule.
- **Task 9** (9bbb2d5): CLAUDE.md and README rows carry the scripts' real exit codes.

Before any gate run, see `docs/findings.md`, 2026-10-08: the recipe as built is trapped in a uniform
field in its first outer iteration, and the gate's upper bound exceeds its budget.
