# Airfoil family: a PINN and a neural operator, head-to-head — design

**Date:** 2026-10-01
**Status:** designed and approved section by section in conversation; this written spec awaits
review before any implementation planning.
**Purpose:** the author's own learning. Not a course, not a deliverable of any other study. Depth
over portability.

---

## 1. Goal and decisions

Learn how physics-informed neural networks (PINNs) and neural operators work by training both on
one aerodynamic problem and scoring them against the same truth, with SU2 generating the data.

| Decision | Choice (the author's) |
|---|---|
| What to come away with | **both methods, head-to-head** on the same case and the same SU2 truth |
| Case | **a 2D airfoil family** in inviscid compressible flow (the Euler equations) |
| Audience | **the author's own learning**, run on this Mac |
| How a flow field is laid out as data | **one structured O-grid for everything** |

**Alternatives considered and set aside**, kept so the reasons are not lost:
- *Unsteady laminar cylinder wake* — the canonical PINN case, but not aircraft aerodynamics, likely
  an hour or more per case here, and a neural operator on it becomes an autoregressive time-stepper.
- *3D wing (ONERA M6)* — SU2's flagship case, but tens of minutes per case even on the tutorial mesh:
  a capstone, not a training set.
- *DeepONet on unstructured meshes* — any geometry and the cleanest conceptual link to the PINN, but
  a different point set per case complicates batching and scoring, and it tends to trail FNO-type
  models on fields with shocks.
- *Cartesian box with the airfoil masked out* — the textbook FNO setup, but it staircases the wall and
  puts interpolation error exactly where pressure and forces are computed. Kept only as an optional
  side experiment (§9.4).

**Defaults stated in conversation and not objected to:** a standalone local git repository at
`~/theLab/airfoil-pinn-no`, separate from any other project; both models written from scratch in
PyTorch (no DeepXDE, no `neuraloperator`), because seeing the machinery is the point; training on the
CPU, with the Apple GPU used if PyTorch's MPS backend works outside the agent's sandbox.

---

## 2. What is already measured on this machine

A throwaway SU2 spike ran on 2026-10-01 (`~/theLab/bwb-tanker-study/cache/su2_spike/`, gitignored
there): inviscid NACA 0012, Mach 0.8, angle of attack 1.25°, an algebraic O-grid from its own
`make_mesh.py`, JST scheme, no multigrid (`MG level: 0` in its log).

| Grid | Nodes | Iterations | s per iteration | CL | CD | Residual, log10 rms(ρ) |
|---|---|---|---|---|---|---|
| 256 × 97 | 24,832 | 4,000 (the cap) | ~0.06 | 0.281956 | 0.019694 | −7.55; SU2's −11 criterion not met, forces steady to six digits |
| 512 × 193 | 98,816 | 8,235 | ~0.28 | 0.308234 | 0.020336 | −8.01; converged at its −8 criterion |

- **The transonic case is not grid-converged at this size.** Lift rose 9.3 % on one doubling of the
  mesh in each direction and is still below the ≈0.35 that Vassberg & Jameson (2010) report for this
  case. That value is **quoted from memory and not yet verified** (§10).
- **SU2 8.5.0** ("Harrier") is at `~/theLab/bwb-tanker-scratch/bin/`. It is an x86_64 build, so it
  runs under Rosetta, one serial process per case, and the endpoint security agent did not block it.
- **SU2's binary restart file layout**, verified byte by byte on the spike's file: five int32 values
  (535532, the field count, the point count, 0, 0), then one 33-byte NUL-padded name per field, then
  the point count × field count float64 values, point by point. 218 + 24,832 × 6 × 8 = 1,192,154
  bytes, which is the file's exact size. The spike ran dimensional and wrote only x, y and the four
  conserved variables.
- **Option names verified in this build:** `MGLEVEL`, `MGCYCLE`, `CONV_FIELD`, `CONV_CAUCHY_ELEMS`,
  `CONV_CAUCHY_EPS`, `VOLUME_OUTPUT`, `PRIMITIVE`, `Pressure_Coefficient`, `REF_DIMENSIONALIZATION`,
  and `SLOPE_LIMITER_FLOW` — the older `LIMITER_FLOW` does not exist.
- **Machine:** 12 cores; memory is the tight resource (about 6 GB sat in the memory compressor at the
  time of writing); 391 GB of disk free.
- **PyTorch:** both arm64 installs (2.5.1 in the `xpoc-m2` env, 2.10.0 in base miniforge) are built
  with MPS support, but report it unavailable from inside the agent's sandbox — most likely the
  sandbox denying Metal, not verified. On the CPU, one FNO-style spectral layer (batch 8, width 32,
  a 256 × 97 grid, 32 × 12 modes) took about 70 ms forward.

---

## 3. The pipeline

```
NACA 4-digit shape + Mach + angle of attack
  → O-grid generator (Python, fixed size, coarse and fine grids that nest exactly)
  → SU2 Euler runs, 6–8 at a time, one directory per case
  → reader + convergence rule → HDF5 dataset
  → FNO (trains on the whole family)   |   PINN (trains on each test case, equations only)
  → one scorer: same nodes, one force integrator, truth = fine-grid SU2, plus a cost record
```

---

## 4. Generating the data (stages S0 and S1)

### 4.1 What varies

- NACA 4-digit sections, chord 1, closed trailing edge (x⁴ coefficient −0.1036, as in the spike).
- **Varied:** Mach M ∈ [0.30, 0.80]; angle of attack α ∈ [−2°, 6°]; thickness t ∈ [0.08, 0.16];
  maximum camber m ∈ [0, 0.04]. **Fixed:** camber position p = 0.4.
- **Training pool:** the first 1,024 points of a scrambled Sobol sequence over that box, seed fixed and
  recorded.

### 4.2 Geometry and mesh

- **The section is a continuous curve.** With β ∈ [0, 2π), the camber-line station is
  x_c = ½(1 + cos β); the thickness is applied normal to the camber line; β ∈ [0, π] is the lower
  surface (trailing edge to leading edge) and β ∈ [π, 2π) the upper (leading edge back to the
  trailing edge) — the spike's ordering.
- **The O-grid is a continuous map** (ξ, η) ∈ [0, 1) × [0, 1] → (x, y), with β = 2πξ (trailing edge
  at ξ = 0) and η the outward coordinate (wall at 0, outer boundary at 1):
  x(ξ, η) = x_w(ξ) + s(η) · (x_f(ξ) − x_w(ξ)), where x_w is the wall point, x_f the point on a circle
  of radius 100 chords about mid-chord at polar angle −2πξ, and s(η) a geometric stretching.
  (Amended 2026-10-01 by the implementation plan: the spike's "same angle about mid-chord"
  construction folds the mesh — 9,792 inverted cells at t = 0.08, m = 0.04 — and the uniform angle
  has no inverted cell at any corner of the box.) **The mesh generator samples this map and the PINN uses it
  continuously, so one function defines the geometry for both.**
- **Nested grids by construction:** the fine grid is 512 × 193 (ξ_i = i/512, η_j = j/192); the
  training grid, 256 × 97, is every second fine node in both directions; the coarse grid, 128 × 49,
  every fourth. The stretching is defined on the fine grid so that the training grid's first wall
  spacing is 2.0 × 10⁻³ chords along the nominal 100-chord radial distance, the spike's value.
- **Outer boundary at 100 chords** (the spike used 50), because a lifting airfoil's disturbance decays
  slowly and a near boundary biases lift. S0 measures the sensitivity (§4.6).
- One mesh per case, since the shape differs case to case; generation takes seconds. Meshes are
  written in SU2's native format with node id j · NI + i, so every output maps straight back onto an
  (i, j) array.

### 4.3 SU2 settings

- Compressible Euler; the spike's JST scheme and coefficients (0.5, 0.02); implicit Euler
  pseudo-time stepping with CFL adaptation; FGMRES with ILU; weighted-least-squares gradients. This
  is the configuration that ran cleanly here.
- **Multigrid:** S0 times 0 and 3 levels and fixes the choice.
- **Free stream:** dimensional (p∞ = 101,325 Pa, T∞ = 288.15 K), as in the spike, and made
  non-dimensional in the reader, in one tested place (§5.2).
- **Outputs:** `OUTPUT_FILES= (RESTART, SURFACE_CSV)`, `VOLUME_OUTPUT= (COORDINATES, SOLUTION,
  PRIMITIVE)`, and the history CSV.
- The reader computes primitive variables itself from the conserved ones, so nothing depends on which
  fields SU2 chooses to write; where SU2 also wrote primitives, the reader checks they agree.

### 4.4 Convergence is judged by our rule, not by SU2's flag

- Read from the history file. A case is **converged** if and only if (a) the density residual has
  dropped by at least 5 orders of magnitude from its first iteration, **and** (b) over the last 500
  iterations both CL and CD have varied by no more than 1 × 10⁻⁶ (0.01 drag counts).
- Iteration caps: 10,000 on the training grid, 20,000 on the fine grid.
- **These are starting values.** S0 confirms or revises them on its own cases, and then they are
  frozen before S1 starts.
- The reason for the rule: on the spike's coarse grid SU2 reported "not converged" (residual stuck at
  −7.55 against a −11 target) while lift and drag had been steady to six digits for hundreds of
  iterations.
- A case that fails is recorded with which criterion failed and by how much. It is never silently
  dropped; where SU2 itself struggles is a result.

### 4.5 Running the campaign

- A Python driver runs 6–8 SU2 processes at once. Memory, not cores, is the limit, so the driver's
  concurrency is a setting and `vm_stat` is checked before raising it.
- Each case gets its own directory: config, mesh, log, history, restart and surface files, and a
  `result.json` **written last and atomically** (temporary file, then rename). It records the four
  inputs, the grid level, the verdict and its reason, the iteration count, the residual drop, SU2's
  CL / CD / CM, wall and CPU time, SU2's version, the config and mesh hashes, the generator version,
  the exit code and the maximum surface Mach number.
- **A case is done if and only if its `result.json` exists.** The campaign resumes after any
  interruption, and a killed run leaves no `result.json`, so it simply runs again.

### 4.6 Stage S0 — check SU2's accuracy and cost before the campaign

On four cases — a symmetric section at α = 0, a subcritical lifting case, a supercritical case, and
the NACA 0012 anchor at Mach 0.8 and 1.25°:

1. **Speed:** time each case with and without multigrid; this fixes the settings and the budget.
2. **Mesh error:** solve on all three nested grids. This gives the observed order of accuracy and the
   training grid's own error in CL, CD and surface pressure — **the floor** used in §8.
3. **Outer boundary:** compare 50, 100 and 200 chords on CL, and confirm or change 100.
4. **Known answers:**
   - a symmetric section at α = 0 gives |CL| at round-off level (the grid is mirror-symmetric);
   - subcritical cases give near-zero drag in inviscid flow, so whatever drag remains is numerical
     error, reported in counts;
   - the lift-curve slope roughly follows thin-airfoil theory with the Prandtl–Glauert factor
     (thickness raises it a little, so this check is approximate);
   - the NACA 0012 anchor is compared with Vassberg & Jameson's grid-converged value, once that value
     has been read from the paper.
5. **Scorer check** (§8.1) on SU2's own solutions.
6. **Convergence-rule thresholds** confirmed or revised, then frozen.

S0 ends with a dated entry in `docs/findings.md` that records the numbers and the frozen campaign
settings.

### 4.7 Cost and disk

Estimates from the spike's timings; S0 replaces them with measurements.

- 1,024 training-grid runs at up to 4 minutes each: about 8–9 hours on 8 processes without multigrid,
  probably much less with it.
- 80 test cases on the fine grid at about 40 minutes each: about 7 hours on 8 processes. The same 80
  on the training grid add about 40 minutes.
- Disk: about 6 GB of raw output (meshes, histories, restart files) and about 1 GB of HDF5.

---

## 5. The dataset and the test sets (stage S1)

### 5.1 Two layers

- **Raw layer, the evidence:** every case directory as SU2 left it, plus its `result.json`.
- **Training layer, derived:** one HDF5 file per grid level, rebuilt deterministically from the raw
  layer by `data/build.py` and never edited by hand.

### 5.2 What the HDF5 files hold

| Dataset | Shape | Content |
|---|---|---|
| `xy` | N × NI × NJ × 2 | node coordinates (they differ per case, because the shape does) |
| `q` | N × NI × NJ × 4 | ρ/ρ∞, u/U∞, v/U∞, p/p∞ |
| `inputs` | N × 4 | M, α in degrees, t, m |
| `forces_su2` | N × 3 | SU2's CL, CD, and CM about the quarter chord |
| `verdict`, `case_id`, `max_surface_mach`, `split` | N | provenance and labels |

- Training-grid files are float32; the fine-grid truth files are float64, because they feed the
  scorer's 10⁻⁶ check. Amended by the plan: each file holds one group per split and only converged cases; failures are listed in results/s1/failures_<level>.json.
- **Derived quantities come from one tested function** in `physics/flow.py` and are never stored
  twice: the pressure coefficient, local Mach number, entropy (s − s∞)/c_v = ln[(p/p∞)/(ρ/ρ∞)^γ], and
  total enthalpy. Entropy earns its place: in shock-free inviscid flow it must stay at its free-stream
  value, so any rise is numerical error, and that check applies to SU2, the FNO and the PINN alike.

### 5.3 The sets, all fixed before any model is trained

| Set | Cases | Grid | Purpose |
|---|---|---|---|
| Training | Sobol points 1–896 | 256 × 97 | fit the FNO and the POD baseline |
| Validation | Sobol points 897–1,024 | 256 × 97 | early stopping and tuning only, never final scores |
| Test, in range | 64 Sobol points from a different seed | 512 × 193 **and** 256 × 97 | interpolation score, and the floor |
| Test, extrapolation | 16 Sobol points with M ∈ [0.80, 0.85], other inputs in range | 512 × 193 **and** 256 × 97 | shocks stronger than anything in training |
| PINN showcase | 6 of the in-range test cases | — | the direct head-to-head |

- **Data-efficiency curve:** the first 64, 128, 256 and 512 Sobol points are each balanced designs on
  their own, so the FNO and the POD baseline are retrained on those prefixes and on the full 896 with
  no new SU2 runs. That gives test error against the number of SU2 cases paid for.
- **Showcase cases are picked by rule:** sort the n surviving in-range test cases by maximum surface
  Mach number (from the fine-grid truth) and take the cases at 0-based sorted index round(p · (n − 1))
  for p = 0.10, 0.30, 0.50, 0.70, 0.85 and 0.95. Nobody picks flattering cases.
- **Regime:** a case is *supercritical* if its maximum surface Mach number exceeds 1 (fine-grid
  truth), which for these sections means a shock, and *subcritical* otherwise. Every score is
  reported for the two regimes separately.
- **No leakage:** the FNO and the POD baseline never see a test case; the physics-only PINN gets a
  test case's inputs and grid, never SU2's field values; only the scorer reads the fine-grid truth.
- **A failed test case is reported, not replaced:** a test case that fails the convergence rule leaves
  its set and is listed with the reason. No substitute is chosen.

---

## 6. The neural operator: an FNO on the O-grid (stage S2)

### 6.1 Inputs, six channels per node

- The node position, radially compressed about mid-chord: (x − 0.5, y) · ln(1 + r)/r, with r the
  distance from (0.5, 0). The 100-chord outer boundary maps to a radius of ln 101 ≈ 4.6; the airfoil
  undergoes a smooth, invertible distortion and keeps all its shape information.
- The ring index j/(NJ − 1): 0 at the wall, 1 at the outer boundary.
- M, cos α and sin α, the same value at every node.

The shape reaches the model only through the node positions, so it learns a map from geometry and
free stream to flow field, not a curve fit through four numbers.

### 6.2 Outputs, four channels, each of order 1 across the Mach range

- u/U∞ − cos α and v/U∞ − sin α;
- the pressure coefficient Cp = (p/p∞ − 1)/(½γM²);
- the scaled density perturbation (ρ/ρ∞ − 1)/(½M²), which reduces to Cp at low Mach.

Raw p/p∞ − 1 shrinks like M², so without this scaling the low-Mach cases would barely register in the
loss. Each output channel is then standardised over the training set.

### 6.3 Architecture, hand-written (starting sizes)

- A pointwise layer lifts the 6 inputs to 32 channels; four Fourier layers; a pointwise projection
  32 → 128 → 4.
- Each Fourier layer: a real FFT over (i, j); keep the lowest 24 wavenumbers of each sign around the
  airfoil and the lowest 12 radially; multiply by learned complex weights; inverse FFT; add a
  pointwise linear path; GELU activation.
- Around the airfoil (i) the FFT is exact, because the grid is periodic. Radially (j) the field is
  zero-padded by (NJ − 1)/6 rows (8 at 128 × 49) before the FFT and cropped after, so the padded
  domain spans the same η at every resolution.
- Truncating the modes smooths jumps, so smeared shocks and overshoot are expected; §8 measures them.
- Hyperparameters are tuned on the validation set within a budget of at most eight configurations,
  listed before tuning starts.

### 6.4 Training

- **Loss A:** the per-sample relative L2 error of each output channel, averaged over channels.
  **Loss B:** loss A plus the same term evaluated on the wall row (j = 0), weighted 10, a value fixed
  in advance. Comparing A and B is part of the lesson: field error is dominated by the large outer
  region, while forces come only from wall pressure.
- AdamW, learning rate 1 × 10⁻³ with cosine decay, batch 8, at most 300 epochs, early stopping on the
  validation loss with a patience of 30 epochs.
- **Trained on the 128 × 49 subgrid** (the 256 × 97 solutions sampled at every second node, about 4×
  cheaper) and **evaluated at 256 × 97**, which tests the claim that FNOs carry over to finer grids.
- Seeds: three at the full 896 training cases; one per step of the data-efficiency curve.
- Hardware: about one second per step at full resolution on the CPU, from the measured layer time;
  the Apple GPU if MPS works in the author's terminal.

### 6.5 The POD baseline

- Proper orthogonal decomposition of the standardised output fields over the training set, done in
  index space — possible because every case is the same array shape.
- The number of modes is chosen on the validation set. Each mode's weight is interpolated over the
  four inputs with a thin-plate-spline radial basis function (`scipy.interpolate.RBFInterpolator`),
  its smoothing chosen on validation.
- It is about 30 lines and a strong simple model at this data size, so an operator that cannot beat it
  is not earning its complexity. Its known weak spot is exactly where the FNO should win: shocks that
  move with the inputs are hard for any fixed linear basis.

---

## 7. The PINN (stage S3)

### 7.1 The problem it solves

- Steady compressible Euler on the same domain as SU2 (wall to the 100-chord circle), one test case at
  a time, from the equations, the boundary conditions and that case's inputs only.
- Non-dimensional variables ρ/ρ∞, u/U∞ and p/(ρ∞U∞²); γ = 1.4.

### 7.2 Two exact reductions

- **Constant total enthalpy.** In steady inviscid flow from a uniform free stream, total enthalpy is
  the same everywhere, including through shocks. The network outputs density and velocity, and the
  pressure follows algebraically: p = (γ − 1)/γ · ρ (H∞ − ½|u|²), with H∞ = 1/((γ − 1)M²) + ½ in these
  units. The residuals are continuity and the two momentum equations; the energy equation drops out
  with no physics lost.
- **The O-grid's own coordinates.** The network takes (cos 2πξ, sin 2πξ, η) through the same map as the
  mesh (§4.2). The residual is written in strong conservation form in (ξ, η), with the map's metric
  terms from automatic differentiation.

### 7.3 Network and constraints

- A fully connected network, 6–8 hidden layers of 64–128 units, tanh activations; outputs ln ρ and a
  raw velocity.
- **Wall tangency, exact by construction:** u ← u − φ(η) (u · n̂_w(ξ)) n̂_w(ξ), with n̂_w the wall
  normal at the same ξ and φ(η) = (1 − η)⁴ — 1 at the wall, 0 at the outer boundary, and invertible
  for η > 0, so nothing is lost away from the wall.
- **Positive density** by construction (exponential of the output).
- **Outer boundary:** a soft penalty towards free stream on the η = 1 circle, at the same radius as
  SU2.
- **Kutta condition:** a soft penalty driving the velocity to zero at the trailing-edge point, the wall
  point (ξ, η) = (0, 0), since a finite-angle trailing edge is a stagnation point. Inviscid flow past a wall has a family of
  solutions, one for each circulation and so one for each lift value; SU2 picks the physical one
  through its numerical dissipation at the sharp edge, and a PINN has no such mechanism. One run is
  made without the penalty, deliberately (§8.5, prediction 3).

### 7.4 Shocks

- The Euler equations also allow unphysical expansion shocks, and only an entropy condition rules them
  out. SU2's dissipation supplies one; a bare PINN residual does not.
- Supercritical showcase cases therefore run twice: pure Euler, and with a Laplacian artificial
  viscosity on the continuity and momentum residuals, ν = 1 × 10⁻³ (in units of chord × U∞), fixed
  before the run. This is reported as **a change to the equations** — the counterpart of the
  dissipation SU2's scheme adds implicitly. One sensitivity run at ν = 3 × 10⁻⁴ is made on the
  strongest-shock case.
- Precedent, from memory and to be verified (§10): Wassing, Langer & Bekemeyer (DLR, around 2023–24)
  trained PINNs on transonic NACA 0012 flow with artificial viscosity.

### 7.5 Training

- Training points: 20,000 drawn uniformly in (ξ, η) and redrawn every 1,000 steps (uniform in these
  coordinates means clustered near the wall physically); 1,000 points on η = 1; the trailing-edge point
  for the Kutta condition.
- Adam, learning rate 1 × 10⁻³ decaying to 1 × 10⁻⁴ over 30,000 steps, then L-BFGS for up to 5,000
  iterations in float64 on the CPU.
- Loss weights fixed first — residuals 1, outer boundary 10, Kutta 10 — with gradient-norm balancing
  (Wang, Teng & Perdikaris, 2021) as a variant.
- Estimated one to two hours per case on the CPU; the warm-up measures it.

### 7.6 Warm-up, before any airfoil

- A circular cylinder (radius 0.5) at Mach 0.1 on the same O-grid machinery, with a zero-circulation
  penalty in place of the Kutta condition, since a cylinder has no sharp edge to fix its circulation.
- **Pass:** the largest |Cp − (1 − 4 sin²θ)| around the cylinder is at most 0.1 (compressibility alone
  shifts the shoulder value by about 0.05 at this Mach), and lift and drag are zero to the scorer's
  resolution.

### 7.7 Named hazards

- **The trailing edge.** The wall normal jumps at the corner, so neither the map nor the tangency
  projection is smooth across the line leaving the trailing edge, and the network has to absorb that
  right where the Kutta condition acts. The trailing-edge error is watched specifically.
- **Circulation is only weakly fixed by the outer boundary** at 100 chords; the Kutta condition
  carries it.

---

## 8. Head-to-head scoring (stage S4)

### 8.1 Contestants, truth and the one force integrator

- **Columns:** SU2 on the training grid (the floor); the POD baseline; the FNO; the PINN; the PINN with
  artificial viscosity (supercritical cases only).
- **Truth:** SU2 on the fine grid, read at the training grid's nodes, which is exact because the grids
  nest. The PINN is evaluated at those same nodes.
- **One force integrator** (`score/forces.py`): vertex-lumped integration of (p − p∞) over the wall (each node carries half of each adjacent edge's outward normal; measured to reproduce SU2 on the spike to 5 × 10⁻¹¹, where the edge-midpoint trapezoid misses CMz by 1.6 × 10⁻⁵), reference length 1, moments about (0.25, 0). **Before it scores anything, it must reproduce
  SU2's own reported CL, CD and CM on SU2's own solutions to within 1 × 10⁻⁶ in each.** They should agree
  to round-off if SU2 integrates nodal pressure over the same edges; if they do not, the difference is
  a finding about the scorer, resolved before any scoring and never by loosening the tolerance.

### 8.2 What is scored

- **Forces:** absolute errors in CL and CM, and in CD in counts (one count = 10⁻⁴), each shown beside
  the true value. No percentages of quantities near zero: subcritical inviscid drag is essentially
  zero, and some cases have nearly zero lift.
- **Surface:** the pressure-coefficient error along the wall, as an L2 value and a maximum, with plots
  for the showcase cases.
- **Near field:** the relative L2 error of the perturbation from free stream over nodes within 2.5
  chords of mid-chord, weighted by dual-cell area. Error over the whole 100-chord domain would mostly
  score the far field.
- **Shocks** (supercritical cases), from wall pressure by a fixed rule: on each surface whose maximum
  Mach number exceeds 1, the shock position is the x/c of the steepest pressure rise downstream of that
  surface's suction peak; the Mach number just
  ahead of it comes from the isentropic relation; the overshoot is the largest |ΔCp| within 0.05 chord
  of the true shock. (Amended 2026-10-01 by the final review: the steepest rise is taken only over
  segments whose upstream node is supersonic, and the leading-edge node, shared by both surfaces,
  counts for the upper surface only. Without the restriction the rule finds the trailing-edge
  recompression into the closing stagnation point on every real solution — S0 attempt 2's M 0.75 case gave an
  upper shock at x/c = 0.99993 with a subsonic Mach 0.41 ahead at 256 × 97, because the last
  interval of the cosine-clustered wall is about 10⁻⁴ chord — and with it the same restarts give
  x/c = 0.675 (Mach 1.22 ahead), and the NACA 0012 anchor gives 0.627 upper and 0.349 lower.)
- **Physics checks**, applied to every column including SU2 itself: entropy rise in subcritical cases
  (maximum and area-weighted mean over the near field); |H/H∞ − 1|; the RMS velocity through the wall;
  net mass flux through the closed grid lines j = 8, 24, 48 and 72 of the training grid, from just off
  the wall to well into the far field. A check that a model satisfies
  by construction — total enthalpy and wall tangency for the PINN — is reported as such, not counted
  as a pass.
- **Cost**, measured on this machine as wall time and CPU time: SU2 per case; the FNO and the POD
  baseline once (the campaign plus training) and per query; the PINN per case.

### 8.3 Plots

- **Break-even:** cumulative cost against the number of new cases answered, for each method, with
  "just run SU2" drawn as a line.
- **Data efficiency:** FNO and POD test error against the number of training cases (64 → 896), with
  the floor drawn as a line.
- **Showcase:** wall pressure from every column on one plot, one plot per showcase case.

### 8.4 Reporting rules

- Every score is split by regime and by test set.
- Every training run is reported, not only the best seed.
- Failures are results: wrong lift from the wrong circulation, an expansion shock, FNO ringing, a PINN
  that never converges.

### 8.5 Predictions, recorded before any model runs

1. For a single new case, SU2 beats the PINN on cost by one to two orders of magnitude. A PINN's
   strengths are inverse problems and sparse data, which is why the pressure-tap extension (§9.4)
   matters.
2. The FNO beats "just run SU2" only after roughly as many new cases as it took to train it, about a
   thousand, unless the data-efficiency curve shows far fewer training cases are enough.
3. Without the Kutta condition, the PINN's residuals will look acceptable while its lift is wrong.

---

## 9. Project layout, stages and tests

### 9.1 Repository

- A local git repository at `~/theLab/airfoil-pinn-no`, not pushed anywhere. Its first commit is this
  spec.
- Its own conda environment, `airfoil-ml`: PyTorch, NumPy, SciPy, h5py, Matplotlib, SymPy (for the
  manufactured-solution test), pytest and ruff.
- SU2 is found through a `SU2_BIN_DIR` environment variable pointing at the existing binaries; tests
  that need SU2 skip when it is unset.

```
src/apno/
  geometry/   naca.py (continuous section), ogrid.py ((ξ,η)→(x,y) map, nested grids, .su2 writer)
  su2/        config.py, run.py (resumable campaign), restart.py (binary reader), convergence.py
  physics/    flow.py (Cp, Mach, entropy, enthalpy), euler.py (residual in grid coordinates)
  data/       build.py (raw → HDF5, sets, showcase rule)
  models/     pod.py, fno.py, pinn.py
  score/      forces.py (the one force integrator), metrics.py, report.py
scripts/      one entry point per stage
tests/
docs/         this spec, findings.md (dated predictions and results)
results/      score tables, plots, run records       — committed
data/, runs/  raw SU2 cases, HDF5, model checkpoints — not committed (about 7 GB, regenerable)
```

The small results are committed from day one. Results that live only in an untracked folder can be
lost, and evidence that cannot be found later is not evidence.

### 9.2 Stages

| Stage | What | Done when |
|---|---|---|
| S0 Accuracy and cost check | mesh, SU2 settings, reader, convergence rule, scorer; mesh and outer-boundary studies; known answers; timing | a findings entry records the numbers and freezes the campaign settings |
| S1 Data campaign | 1,024 training cases, plus 80 test cases on both grids (one to two nights) | every case recorded, failures mapped, HDF5 files built |
| S2 Operator | POD baseline, FNO, data-efficiency curve, seeds | test scores for both models |
| S3 PINN | cylinder warm-up → subcritical showcase cases → supercritical cases (pure Euler and with viscosity), plus the run without the Kutta condition | scores for all six showcase cases |
| S4 Head-to-head | the scorer run on every column, the plots, the predictions checked | a findings write-up |

**What can run in parallel:** S1 is mostly waiting on SU2, so the S2 code and the whole S3 warm-up can
be built while it runs; the warm-up needs no SU2 data at all. S2 needs S1's data before it can train
properly, and S4 needs everything.

### 9.3 Tests — every tool meets a known answer before it is trusted

- **Grid:** every cell has positive area; coarse nodes are exactly every other fine node; the wall row
  lies on the section; the sampled map equals the written mesh.
- **SU2 reader:** reads the spike's real restart file (fields, counts, magic number); a file with
  shuffled nodes fails the coordinate check against the mesh.
- **Convergence rule:** made-up histories — steady forces with a stalled residual, oscillating forces,
  a residual that drops while the forces still drift.
- **Physics functions:** free stream gives Cp = 0, no entropy rise and H = H∞; a normal shock gives the
  textbook jumps with total enthalpy unchanged.
- **Force integrator:** uniform pressure gives zero force; the potential-flow cylinder gives zero lift
  and drag.
- **Euler residual:** uniform flow on the curved grid gives zero to round-off (the classic
  free-stream-preservation check of metric terms); a manufactured solution matches its analytic
  residual, computed in (x, y) with SymPy.
- **FNO:** rotating the input around the airfoil rotates the output by the same amount; one trained
  model runs at both grid resolutions.
- **PINN:** flow through the wall and changes in total enthalpy are zero by construction.
- **No leakage:** the FNO's data loader cannot load a test case, and the PINN's input object carries
  no flow-field data.

### 9.4 Extensions, deliberately later

- The Cartesian-box FNO side experiment.
- A single PINN covering a range of Mach and α.
- A PINN that also uses a few surface "pressure taps" from SU2.
- An FNO trained with the Euler residual added to its loss (a physics-informed neural operator).
- Camber position as a fifth input.

---

## 10. Open items, each with a defined way to resolve it

| Item | Resolved by |
|---|---|
| Multigrid levels; convergence-rule thresholds; iteration caps | S0 measurement, then frozen before S1 |
| Outer-boundary distance (100 chords) | S0's outer-boundary study |
| Vassberg & Jameson (2010) lift for NACA 0012 at Mach 0.8, 1.25° (≈0.35, from memory) | read from the paper before S0's known-answer check |
| The Wassing, Langer & Bekemeyer reference | verified before S3 |
| Whether the Apple GPU (MPS) is usable | one command in the author's own terminal |

## 11. Values pinned while writing this spec

These were not discussed in conversation; each is a starting value chosen so that nothing in this
spec is left open. Review them.

- Convergence rule: a 5-order residual drop; CL and CD steady to 1 × 10⁻⁶ over the last 500
  iterations; caps of 10,000 and 20,000 iterations (§4.4).
- FNO: radial compression ln(1 + r)/r about mid-chord; 8 rows of radial padding; a 32 → 128 → 4
  projection; AdamW at 1 × 10⁻³ with cosine decay, batch 8, at most 300 epochs, patience 30; wall
  weight 10 in loss B; a tuning budget of eight configurations (§6).
- POD: a thin-plate-spline RBF for the mode weights (§6.5).
- PINN: tangency blend φ(η) = (1 − η)⁴; 20,000 training points redrawn every 1,000 steps; 30,000 Adam
  steps then up to 5,000 L-BFGS iterations; loss weights 1 / 10 / 10; artificial viscosity
  1 × 10⁻³ with one run at 3 × 10⁻⁴ (§7).
- Warm-up pass criterion: largest |ΔCp| ≤ 0.1 around the cylinder (§7.6).
- Showcase selection: sorted index round(p · (n − 1)) for the six percentiles (§5.3).
- Scoring: the near field is within 2.5 chords of mid-chord; the mass-flux check uses grid lines
  j = 8, 24, 48 and 72; the shock-location and overshoot rules; the integrator's 1 × 10⁻⁶ agreement
  with SU2 (§8).
- Prediction 3, about the Kutta condition, written as a prediction here for the first time (§8.5).
