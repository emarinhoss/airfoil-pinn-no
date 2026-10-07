# Stage S5: an Euler PINN with localised artificial viscosity, validated, ablated and tested out of range — design

**Date:** 2026-10-07
**Status:** designed and approved section by section in conversation (2026-10-07); this written
design awaits the author's review before implementation planning.
**Relation to the main spec** (`2026-10-01-airfoil-pinn-vs-neural-operator-design.md`): a new
stage, S5, after S4. S0–S4, their records and the full-potential PINN are unchanged; nothing in
`results/s0`–`results/s4` is re-run or rewritten. Scoring, truth and floor are S4's.

---

## 1. Purpose and decisions

The S4 comparison sets a strong neural operator against a weak PINN (26–65x the floor, lift 64–101 %
low on five of six cases), while Wassing, Langer & Bekemeyer (arXiv:2408.17364v3, 2025;
`docs/findings.md`, 2026-10-07) report an Euler PINN within ~0.3–0.4 % Cp error on transonic NACA
0012 cases. Until this study's PINN is representative, an FNO-against-PINN conclusion mostly measures
this study's PINN. S5 therefore:

1. **validates** a PINN built to their recipe against SU2 on one of their published cases;
2. **ablates** their ingredients, one at a time, to find which of them rescues this study's Euler
   PINN from the fluid-at-rest collapse of 2026-10-04;
3. **compares** the validated PINN with the FNO on in-range and extrapolation cases, where the FNO is
   weakest (3.6x the floor, CD errors to 181 counts, biased).

| Decision (the author's, 2026-10-07) | Choice |
|---|---|
| Compute budget | time every planned run first; default cap **72 h**, enforced by the preflight; case and arm counts set once the cost is measured |
| Step 3 scope | the **three in-range supercritical showcase cases plus extrapolation cases** chosen by rule |
| If the validation gate fails | a **fixed tuning budget of four configurations** (§3.4), tried in order on the validation case only; if none passes, S5 stops after step 1, the failure is a result, and the author decides |
| Approach | **reproduce, then subtract**: build the full recipe, gate it, then revert one ingredient at a time |

**Approaches considered and set aside:** *build up from this study's Euler PINN* (add their
ingredients one at a time; the full recipe is the last arm) — reuses the most code, but single
additions to a collapsing model will likely all still collapse and teach nothing, and the gate is
reached last; *reproduce only* (gate, then step 3) — cheapest, but leaves the 2026-10-04 question
open.

---

## 2. The new formulation, `euler_av`

A third PINN formulation beside `potential` and `euler`, in its own modules. Faithful to Wassing et
al. (their §II and Table VII) except where §2.6 says otherwise.

### 2.1 State and scaling

- The network outputs the primitive variables (ρ, u, v, p); the residual is the full steady 2-D
  Euler system **including the energy equation**, in conserved form W = (ρ, ρu, ρv, ρE),
  E = p/((γ−1)ρ) + ½|u|².
- Their non-dimensionalisation: ρ∞ = 1, p∞ = 1, u∞ = √γ M∞ cos α, v∞ = √γ M∞ sin α (so c∞ = √γ),
  coordinates in chords. Their viscosity factor and sensor constants therefore keep their meaning.
- Conversion to the dataset's units is exact: ρ/ρ∞ and p/p∞ unchanged, u/U∞ = u / (√γ M∞), likewise
  v. Prediction files hold dataset units, so the scorer is unchanged.

### 2.2 Network

- **Inputs:** the grid indices r = (i, j) of the collocation grid (i ∈ [0, 512], j ∈ [0, 192]),
  through a random Fourier-feature embedding: 128 frequencies φ ∈ R^{128×2} drawn from N(0, 1²),
  features (sin φr, cos φr) → 256 values (their Eq. 25).
- **Trainable layers:** 5 hidden layers of 128 units, each a weight-normalised affine map
  (w = g z/‖z‖) followed by the layer-wise adaptive activation tanh(n ω_k x), ω_k trainable per layer,
  **n = 10 with ω_k initialised at 0.1** (their scale factor is not stated; this follows Jagtap et
  al., so n ω_k = 1 at the start); a final weight-normalised affine map to 4 outputs, no activation.
- The seed (0 unless stated) fixes the frequencies, the initialisation and the batch order.

### 2.3 Collocation and loss

- **Collocation:** the nodes of the 512x193 grid (98,816 points; theirs 400x200), plus a duplicate
  column at i = 512 holding the same physical points as i = 0, for the periodicity term. The 256x97
  scoring nodes are every second node, so predictions need no interpolation.
- **Loss** (their Eqs. 8, 12, 13): L = λ_res L_res + L_∞ + L_af + L_lr, with
  - L_res = (1/N) Σ_n w_n |R_n − D_n|², the residual R of the four equations minus the viscous term
    D (§2.4; the dissipative sign — see the note there), over the nodes of the current mini-batch; w_n is the node's dual-cell area normalised to
    mean 1 over the grid (`apno.score.metrics.node_areas`), the volume weighting they adopt;
    **λ_res = 2·10⁴**;
  - L_∞ = mean over the outer row (j = 192) of |q̂ − w∞|², in primitive variables;
  - L_af = mean over the wall row (j = 0) of (u·n̂_w)², n̂_w from `apno.models.pinn.wall_normal`;
  - L_lr = mean over j of |q̂(0, j) − q̂(512, j)|².
- **Batches:** each mini-batch is 5,000 nodes drawn without replacement from the residual nodes; the
  three boundary terms are evaluated on all their nodes at every closure (their batching of the
  boundary terms is not stated; declared here). One epoch is one pass over the residual nodes
  (20 batches).
- **Derivatives:** the metric terms come from automatic differentiation of the analytic O-grid map,
  and the divergence uses the strong conservation form of `apno.physics.euler.divergence` — the same
  quantity as their ∂F/∂x + ∂G/∂y, by the metric identities. Second derivatives for D are taken in
  (x, y) by automatic differentiation through the map.

### 2.4 Artificial viscosity

- D(W) = μ_x ∂²W/∂x² + μ_y ∂²W/∂y² (their Eq. 14), entering the residual as ∇·F − D. *(Corrected
  2026-10-07, while writing the plan: the first version wrote R + D after their Eq. 13, which with
  their Eq. 14 is anti-dissipative; ∇·F − D is the dissipative steady form ∇·F = ∇·(μ∇W) and is read
  as their intent.)* D is active on the **density and energy equations
  only** (the selector I_{ρ,ρE} = diag(1, 0, 0, 1)), with **ν = 1.9·10⁻³**:
  - **matrix form** (their model V, the default): μ_x = ν s I_{ρ,ρE} |A|, μ_y = ν s I_{ρ,ρE} |B|, where
    |A| = T|Λ|T⁻¹ of the flux Jacobian ∂F_x/∂W (likewise B), from the closed form of Swanson & Turkel
    (1992), with each eigenvalue (u + c, u − c, u, u) limited to |λ̃| = max(|λ|, ε), **ε = 0.25** (their
    Appendix C);
  - **scalar form** (their model IV, tuning configuration T3): μ_x = μ_y = ν s I_{ρ,ρE} (c + |u|).
- **Sensor** s = s_shock · s_stag, **detached** from the computational graph (their §II H):
  - s_shock = tanh(max(0, k⁽¹⁾_shock (u/|u|)·∇p − k⁽⁰⁾_shock)) — positive where pressure rises along
    the flow;
  - s_stag = sig(k⁽¹⁾_stag (k⁽⁰⁾_stag − (|∇(ρu)| + |∇(ρv)|))), **k⁽⁰⁾_stag = 3, k⁽¹⁾_stag = 5** — near 0
    at stagnation points, where the momentum gradients are large.
- **The shock-sensor constants are ambiguous in the source:** their Table VII lists k⁽⁰⁾_shock = 4
  and a second row mislabelled "k⁽¹⁾_stag" = 0.03. **Reading A:** steepness k⁽¹⁾_shock = 4, threshold
  k⁽⁰⁾_shock = 0.03 (matching their Fig. 3's steepness of 4 and their remark that the active region
  is large). **Reading B:** the table literally, threshold k⁽⁰⁾_shock = 4, steepness k⁽¹⁾_shock = 0.03.
  Both are tuning configurations (§3.4).
- As in their implementation, D is computed only at nodes where s > 10⁻⁴ and set to 0 elsewhere.

### 2.5 Optimiser

- **L-BFGS only** (their §II H, Table VII), in mini-batches: for each batch a fresh
  `torch.optim.LBFGS(lr=1, max_iter=1000, max_eval=1250, tolerance_grad=1e-6, tolerance_change=1e-10,
  history_size=1000, line_search_fn="strong_wolfe")` — reset after each outer iteration, because the
  batch changes the objective; **40 epochs**.
- Device: float32 on the Apple GPU (MPS) when available, else the CPU (their precision is not
  stated). Evaluation at the grid nodes is in float64 on the CPU, as for the existing PINNs. Both
  devices are recorded.

### 2.6 Departures from their recipe, declared

1. Metric terms by automatic differentiation of this study's analytic map, not finite differences
   on an elliptic grid; this study's O-grid (algebraic, 100-chord circle), not theirs, because the
   truth lives on it.
2. The collocation grid is 512x193 nodes, not 400x200.
3. The trailing-edge wall node, where J = 0 on this grid (x_ξ = y_ξ = 0 under cosine clustering), is
   left out of L_res; it is the only node removed (T4 adds the 0.02-chord exclusion).
4. The boundary terms are evaluated in full at every closure (§2.3).
5. n = 10, ω₀ = 0.1 for the adaptive activation (§2.2).

---

## 3. Step 1: the validation gate

### 3.1 The validation case

- **Their case B: NACA 0012 (t = 0.12, m = 0), M∞ = 0.72, α = 3°.** It is in none of this study's
  train, validation or test sets, so tuning on it leaks nothing into any test score.
- If SU2 fails the frozen convergence rule on it at either level, the fallback order, fixed now, is
  their case C (0.75, 2°), then case A (0.70, 4°). Their case D (0.78, 3°) is excluded: their own
  method reaches only ~3 % there.

### 3.2 The reference

- SU2 runs the case with the frozen S0 settings (`apno.settings`) at 256x97 and 512x193, through
  `apno.su2.run` and the convergence rule, into `runs/su2/<level>/s5-val-<tag>/`.
- As in S4: **truth** = 512x193 sampled at every second node; **floor** = the 256x97 solution scored
  against it.

### 3.3 The gate metric

- **RMAE**, their metric: on a uniform 401x401 lattice over the box x ∈ [−0.5, 1.5], y ∈ [−1, 1]
  (their (−1, 1)² box about mid-chord, shifted to this study's chord), excluding lattice points
  inside the section; both fields interpolated linearly on the 256x97 grid's own cells (each quad
  split into two triangles); RMAE(f) = mean |f̂ − f| / (max f − min f), with max and min of the truth
  over the lattice. Computed for **Cp and local Mach number**.
- **Pass:** RMAE(Cp) ≤ 1.0 % **and** RMAE(M) ≤ 1.0 %, against SU2 at 512x193 (they report 0.3 %
  against their own reference; the margin allows for a different reference solver and section).
- **Also reported, not gated:** the S4 scores on the case (`apno.score.metrics.score_case`: near field,
  CL, CD, CM, Cp RMS and maximum, shocks, physics checks), and the floor's RMAE on the same lattice.

### 3.4 The tuning budget

Four configurations, tried **in this order** with seed 0; the **first that passes is recipe R**; none
is chosen as best among several:

| | configuration |
|---|---|
| T1 | §2 as written, sensor reading A |
| T2 | T1 with sensor reading B |
| T3 | T1 with the scalar viscosity (their model IV) in place of the matrix form |
| T4 | T1 with the 0.02-chord trailing-edge exclusion of this study's potential PINN (`exclude_trailing_edge`) applied to L_res — the degenerate trailing edge of this grid dominated the potential PINN's residual on 2026-10-05 |

If none passes, S5 stops after step 1; all four are recorded as results and the author decides.

### 3.5 Records and the stated limitation

- `results/s5/gate.json`: per configuration, the metrics, final losses, function-evaluation counts,
  stalled outer iterations (whose loss did not fall; PyTorch's L-BFGS does not report line-search
  failures, so this is the measurable proxy), wall and CPU time, devices, the git commit and the command line.
- Prediction files go to **`runs/predictions_s5/`**, never `runs/predictions/`: S4's scorer reads
  every file there, so mixing them would change any re-run of S4.
- **Limitation:** one seed per configuration (they averaged four), because of cost.

---

## 4. Step 2: the ablation

Each arm takes recipe R and reverts **exactly one ingredient** to this study's Euler PINN (spec §7.1–
7.5); everything else stays R, seed 0, gate metric of §3.3. Arms A1–A5 run on the validation case.
In priority order, should the preflight force cuts:

| arm | reverts | to this study's choice |
|---|---|---|
| A1 | residual weighting | unweighted mean residual, weight 1; boundary weights 10 — in place of volume weighting and λ_res = 2·10⁴ |
| A2 | state construction | outputs ln ρ and the velocity perturbation (in R's scaling); pressure from constant total enthalpy; wall tangency exact by the (1 − η)⁴ projection; the energy residual drops out as an identity, so the viscosity acts on continuity only and L_af is identically 0 |
| A3 | optimiser | 30,000 Adam steps (1e-3 decaying exponentially to 1e-4, on the same mini-batches), then L-BFGS (≤ 5,000 iterations, history 50, strong Wolfe) on one fixed batch |
| A4 | input encoding | (cos 2πξ, sin 2πξ, η) in place of Fourier features of the indices; periodicity becomes exact (L_lr stays, identically 0) |
| A5 | collocation | random points uniform in (ξ, η), a fresh draw of 5,000 for each L-BFGS outer iteration (the counterpart of this study's 20,000 points redrawn every 1,000 steps), in place of the fixed nodes; the embedding takes the continuous index coordinates (512ξ, 192η); the residual weight is J(ξ, η) normalised to mean 1, the continuous counterpart of the dual-cell area |
| A6a | Mach range: cylinder | R unchanged on the S3a cylinder warm-up (radius 0.5, M∞ 0.1), where the Euler PINN first collapsed; pass = the S3a criterion (max \|ΔCp\| ≤ 0.1 against 1 − 4 sin²θ; CL and CD zero to the scorer's resolution) |
| A6b | Mach range: airfoil | R unchanged on `test_in-0022` (M∞ 0.36), where it collapsed on an airfoil; gate metric against that case's existing truth — a measurement only, nothing is tuned on a test case |

**Verdict per arm:**
- **needed** if the arm fails its pass criterion **or collapses** — its largest wall speed is below
  half the truth's (SU2's for an airfoil, potential flow's 2 U∞ for the cylinder: the fluid-at-rest
  signature); **not needed** if it passes.
- Every arm also reports RMAE, near field, CL, the collapse check, losses and costs.

**What it can and cannot say:**
- "Not needed" means not needed **given the rest of R**: a one-at-a-time ablation misses interactions
  (two ingredients that each suffice both read as not needed). Stated wherever a verdict appears.
- The 2026-10-04 question is answered by which arms collapse: A1 or A2 → this study's weighting or
  closure; A6a or A6b → the Mach range; none → a combination, which this design does not isolate.
- One seed per arm.

**Records:** `results/s5/ablation.json`. The arms are independent once R exists and may run as two
shards.

---

## 5. Step 3: the comparison

### 5.1 Cases

- **In range, supercritical:** the three showcase cases `test_in-0004`, `test_in-0031`,
  `test_in-0040` (maximum surface Mach 1.21, 1.38, 1.61; `results/s1/showcase.json`).
- **Extrapolation:** S1's showcase rule (`apno.data.build.showcase`'s sort and rounding) on the 14
  extrapolation cases scored in S4 — sorted by the truth's maximum surface Mach, sorted index
  round(p(n − 1)) for **p = 0.10, 0.50, 0.90**:

  | case | M∞ | α | t | m | max surface Mach |
  |---|---|---|---|---|---|
  | `test_ext-0005` | 0.801 | −1.58° | 0.143 | 0.006 | 1.43 |
  | `test_ext-0009` | 0.822 | −0.65° | 0.132 | 0.033 | 1.64 |
  | `test_ext-0006` | 0.824 | 3.78° | 0.114 | 0.026 | 1.78 |

- **Extension:** if the preflight shows room under the cap, the author may add extrapolation cases
  by the same rule with more percentiles; the decision is recorded before launch.

### 5.2 Model

Recipe R exactly as it passed the gate, seed 0; no tuning on any test case.

### 5.3 Scoring

- `apno.score.metrics.score_case` at 256x97 against the same truth: near field, CL, CD, CM, Cp RMS and
  maximum, shocks (position, missed, overshoot), physics checks (labelled "by construction" where an
  identity holds) — plus the gate's RMAE of Cp and Mach, for comparison with Wassing et al.
- **Beside it, on the same six cases:** the floor; the FNO `fno-n896-s0` at 256x97 and at 128x49;
  POD `pod-n896`; the full-potential PINN `pure` on the three in-range cases. They are read from the
  existing prediction files and scored by the same code; `results/s4` is not touched.
- **Group figures:** per method, the mean ratio to the floor of the near-field error over the three
  in-range and over the three extrapolation cases (CL and CD errors reported beside, not gated).
- **Degradation** = a method's mean extrapolation ratio to the floor divided by its mean in-range
  ratio, on the near-field error, on these six cases, for the FNO, POD and the new PINN — the like-for-like measure of whether the PINN holds up where the FNO does not.
  The FNO's degradation is computed on the same six cases, not from its whole-set S4 figures.

### 5.4 Costs and records

- Per run: wall and CPU time, the device of each phase; GPU work costed at wall time, CPU work at CPU
  time (S4's rule); beside SU2 per case and the FNO per query, with the new PINN's break-even line.
- `results/s5/scores.json`, `scores.md`, `costs.json`, and one wall-Cp plot per case with every
  method.

---

## 6. Predictions, proposed

Recorded as a dated entry in `docs/findings.md` **after the author's review of this design and before
any S5 run**, and never edited afterwards; the author may change them before they are recorded.

- **P4.** Recipe R passes the gate within the four configurations.
- **P5.** Reverting the residual weighting (A1) breaks R; reverting the input encoding (A4) or the
  collocation (A5) does not. R on the cylinder at M 0.1 (A6a) fails or collapses: the Mach range is
  part of why this study's Euler PINN failed.
- **P6.** On the three extrapolation cases the new PINN degrades less than the FNO (a smaller
  degradation factor, §5.3), but its mean in-range near-field ratio to the floor stays above the
  FNO's at 256x97.
- **P7.** Per case, the new PINN still costs at least 10x SU2's CPU time.

---

## 7. Budget, order and failures

### 7.1 Preflight and budget

- Before any PINN run, every planned run — T1–T4, A1–A6b, the six step-3 cases — is timed on its own
  case for a few L-BFGS iterations (Adam steps for A3), and its wall time projected as an **upper
  bound**: epochs × batches × max_eval × measured seconds per closure — for R, 40 × 20 × 1,250 =
  1,000,000 closures, max_eval being PyTorch's hard limit per outer iteration — because L-BFGS often
  stops well before it; the closure is timed with the viscous term forced on everywhere; the bound will be large, and
  is replaced by the re-projection from R's measured run below.
- Written to `results/s5/preflight.json`; the stage **refuses to start (exit 3)** when the projection
  for the unfinished runs **of the current phase** exceeds `--budget-hours` (default **72**). The check
  applies per phase: before the gate, to T1–T4; after the gate, to steps 2–3, on the re-projection
  below. (Applied to the whole plan at the start, the upper bounds of 17 runs would refuse it
  outright.) `--preflight-file` reuses an
  uncontended preflight's timings, as in S3.
- After the gate, the steps 2–3 projection is **re-made from R's measured run** (its actual iteration
  counts and wall time) before they launch.

### 7.2 Order

1. SU2 on the validation case (§3.2).
2. The gate, sequentially, T1 → T4 (§3.4).
3. The re-projection of steps 2–3 (§7.1).
4. The ablation and step 3, at most **two shards** at once (three thrash this 16 GB machine; S3).

### 7.3 Failures

- A non-finite loss stops the run; it is recorded as **diverged** with the outer iteration, the batch
  and the reason, and writes no prediction file; it counts as a gate failure, or as "needed" in the
  ablation.
- An outer iteration whose loss does not fall is counted as **stalled** and training moves to the
  next batch; the count is recorded.
- An interrupted run restarts from scratch; finished prediction files are skipped (S3's resume rule).
- No record is overwritten; file names carry the formulation (`pinn-euler_av-<arm or case>.h5`).
- SU2 non-convergence on the validation case: the fallback order of §3.1.

---

## 8. Tests, written before the code

- Scaling: their units ↔ the dataset's, both ways; free stream maps to free stream.
- Residual: uniform free stream gives zero on all four equations, to round-off in float64, on the
  curved grid, with and without viscosity.
- Energy equation: a manufactured solution through the full residual matches its analytic value,
  computed in (x, y) with SymPy (as `tests/test_euler.py` does for the existing residual).
- Matrix viscosity: the closed-form |A| and |B| equal T|Λ|T⁻¹ from a numerical eigendecomposition on
  random states (limiter off); with the limiter, no eigenvalue falls below 0.25.
- Sensors: s = 0 on uniform flow; s_shock > 0 where pressure rises along the flow and 0 where it
  falls; s_stag ≈ 0 at a stagnation state; no gradient flows through s.
- Network: output shapes; the parameter count pinned; the embedding identical for a fixed seed.
- Boundary terms: L_∞ = 0 at free stream, L_af = 0 for tangent flow, L_lr = 0 for a periodic field.
- RMAE: 0 for identical fields; the known value for a constant offset; lattice points inside the
  section excluded.
- No leakage: the S5 PINN's input object carries no flow-field data.
- Preflight: refuses a plan over budget (as `tests/test_s3_preflight.py`).
- Records: S5 prediction files are written under `runs/predictions_s5/`, never `runs/predictions/`.

---

## 9. Files

| file | content |
|---|---|
| `src/apno/models/pinn_euler_av.py` | Fourier embedding, adaptive tanh, weight-normalised layers; the state in their scaling; conversion to dataset units; the loss terms |
| `src/apno/physics/viscosity.py` | the sensors; scalar and matrix viscosity; closed-form \|A\|, \|B\| with the limiter; the viscous term D |
| `src/apno/models/train_pinn_av.py` | the mini-batch L-BFGS loop and the A3 Adam-then-L-BFGS variant; timing for the preflight; evaluation at the nodes |
| `src/apno/score/rmae.py` | the lattice, the interpolation on the grid's cells, RMAE |
| `src/apno/data/sampling.py` | the validation case and its fallbacks, as `CaseSpec`s |
| `scripts/s5_su2.py` | SU2 on the validation case at both levels |
| `scripts/s5_pinn.py` | preflight, gate, ablation and step-3 runs; `--shard K/N`, `--budget-hours`, `--preflight-file` |
| `scripts/s5_score.py` | the gate, ablation and comparison tables, costs and plots into `results/s5/` |
| `tests/test_pinn_euler_av.py`, `tests/test_viscosity.py`, `tests/test_rmae.py`, `tests/test_s5.py` | §8 |

Existing modules are reused unchanged: the grid map (`apno.geometry.ogrid`), `divergence`
(`apno.physics.euler`), `wall_normal` (`apno.models.pinn`), `node_areas` and `score_case`
(`apno.score.metrics`), `exclude_trailing_edge` (`apno.models.train_pinn`), the SU2 driver.

---

## 10. Out of scope (candidates after S5)

- A parametric PINN over a slice of inputs against the FNO on the same slice.
- Re-running the SU2 anchor at 1024x385 or with a 200-chord far field, to quantify the truth's gap
  to Vassberg & Jameson.
- A PINN given a few SU2 surface pressures (spec §9.4).
- An FNO with the Euler residual in its loss (spec §9.4).
