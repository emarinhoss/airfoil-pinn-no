# Findings

Dated entries. Every figure carries the command that produced it. Predictions are recorded before
the run that tests them and are never edited afterwards; outcomes are appended below them.

## 2026-10-01 — predictions (spec §8.5), recorded before any model runs

1. For a single new case, SU2 beats the PINN on cost by one to two orders of magnitude.
2. The FNO beats "just run SU2" only after roughly as many new cases as it took to train it
   (about a thousand), unless the data-efficiency curve shows far fewer training cases suffice.
3. Without the Kutta condition, the PINN's residuals look acceptable while its lift is wrong.

## 2026-10-01 — environment

- Apple GPU (MPS) available in the author's terminal: True (measured by the controller's shell,
  torch 2.13.0, `torch.backends.mps.is_available()`).
- Measured while planning (from the spike in ~/theLab/bwb-tanker-study/cache/su2_spike):
  the far-point-at-same-angle O-grid folds (9,792 inverted cells at t = 0.08, m = 0.04);
  vertex-lumped force integration reproduces SU2's CL/CD/CMz to 5e-11.

## 2026-10-01 — final review fix wave: measurements and decisions

- **Shock locator (spec 8.2 amended).** The steepest-rise rule found the trailing-edge
  recompression on every real supercritical solution. On S0 attempt 2's restarts
  (`runs/s0_attempt2_eps1e-8`), `shocks()` before the fix: super-mg0 upper x/c = 0.99851 / 0.99993
  / 0.99991 at 128x49 / 256x97 / 512x193 with Mach 0.47 / 0.41 / 0.42 ahead; anchor both surfaces at
  0.9997-0.99998. After restricting the search to supersonic-upstream segments: super upper
  0.681 / 0.675 / 0.678 (Mach ahead 1.13 / 1.22 / 1.21); anchor upper 0.633 / 0.627 / 0.630 and
  lower 0.343 / 0.349 / 0.352. Command, per (name, M, alpha, t, m, level):
  `python -c "from apno.geometry.ogrid import *; from apno.su2.restart import *; from apno.physics.flow import *; from apno.score.metrics import shocks; X,Y=grid(Section(0.12,0.02),TRAIN); f=on_grid(read_restart('runs/s0_attempt2_eps1e-8/256x97/super-mg0/restart.dat'),X,Y); q=nondimensionalise(f['Density'],f['Momentum_x'],f['Momentum_y'],f['Energy'],0.75); print(shocks(X[:,0],pressure_coefficient(q[3][:,0],0.75),0.75))"`
- **FNO device (spec 6.4).** Training defaults to the Apple GPU when MPS is available. FNO2d at the
  spec sizes, batch 8 at 128x49, one AdamW step: CPU 0.19 s, MPS 0.058 s (command:
  `python -c "import torch,time; from apno.models.fno import FNO2d; d='mps'; m=FNO2d().to(d); x=torch.randn(8,6,128,49,device=d); o=torch.optim.AdamW(m.parameters()); [(o.zero_grad(),m(x).square().mean().backward(),o.step()) for _ in range(2)]; torch.mps.synchronize(); t=time.perf_counter(); [(o.zero_grad(),m(x).square().mean().backward(),o.step()) for _ in range(3)]; torch.mps.synchronize(); print((time.perf_counter()-t)/3)"`,
  `d='cpu'` and no synchronize for the CPU). `process_time()` cannot see GPU work, so S4 costs
  GPU training at WALL time and CPU training at CPU time; `costs.json` records the basis per method.
- **PINN cost (spec 7.5's estimate does not hold for the viscous runs).** The viscous flux is now
  folded into the conservative divergence (same operator, one derivative pass). Measured on
  test_in-0000 at the spec sizes (20,000 points, 7x96 tanh), wall seconds:
  Euler 0.31 per Adam step and 0.76 per float64 closure (3.6 h projected per run), visc1e-3 2.12
  and 7.26 (27.8 h), gradnorm 0.35 and 0.64 plus 0.45 per balancing update (3.8 h). Command:
  `python -c "from apno.data.sampling import case_specs; from apno.geometry.ogrid import TRAIN, Section; from apno.models.pinn import PinnCase; from apno.models.train_pinn import *; s=case_specs('test_in',TRAIN)[0]; c=PinnCase(s.mach,s.alpha_deg,Section(s.t,s.m)); cfg=PinnConfig(viscosity=1e-3); r=time_steps(c,cfg,2,1); print(r, projected_s(cfg,r)/3600)"`.
  `scripts/s3_pinn.py` now times every planned variant first, writes `results/s3/preflight.json`
  and refuses to start past `--budget-hours` (default 48). With two to four viscous runs the
  projection is about 90-140 h, so S3 needs a recorded decision (fewer points, steps or viscous
  runs, or a larger budget) before it launches.
- **Gradient-norm balancing (spec 7.5's variant)** is implemented as
  `PinnConfig(grad_balance=True)` and S3 plans one such run, on the median-Mach showcase case.

## 2026-10-01 — stage S0: SU2 accuracy and cost, and the frozen campaign settings

Five cases (name: M, alpha deg, t, m) — sym 0.50, 0, 0.12, 0; sub0 0.50, 0, 0.12, 0.02; sub2 0.50,
2, 0.12, 0.02; super 0.75, 2, 0.12, 0.02; anchor 0.80, 1.25, 0.12, 0 (NACA 0012) — at 128x49,
256x97 and 512x193 with MGLEVEL 0, plus MGLEVEL 3 at 256x97 and outer radius 50 and 200 chords
for sub2 and anchor at 256x97: 24 runs per attempt, SU2 8.5.0, 6 workers.

**How each attempt was run.** `scripts/s0_check.py` reads `apno.settings.BASE`, which still held
the spec's starting `cauchy_eps = 1e-7`. Attempts 2 and 3 therefore ran with a process-local
override, never committed: `python -c` doing
`apno.settings.BASE = dataclasses.replace(BASE, cauchy_eps=...)` and then
`runpy.run_path('scripts/s0_check.py', run_name='__main__')` with `--workers 6`. Attempt 3's
launcher (`runs/run_s0_attempt3.sh`, not committed; `runs/` is ignored) was:

```bash
cd /Users/esousa/theLab/airfoil-pinn-no
export PATH=/Users/esousa/miniforge3/envs/airfoil-ml/bin:$PATH
export SU2_BIN_DIR=/Users/esousa/theLab/bwb-tanker-scratch/bin
python -c "
import dataclasses, runpy, sys
import apno.settings as st
st.BASE = dataclasses.replace(st.BASE, cauchy_eps=1e-9)
print('S0 attempt 3 with', st.BASE, flush=True)
sys.argv = ['s0_check.py', '--workers', '6']
runpy.run_path('scripts/s0_check.py', run_name='__main__')
" > results/s0/s0.log 2>&1
echo "S0_EXIT=$?" >> results/s0/s0.log
```

Those overrides are now replaced by the frozen values below, so a plain
`SU2_BIN_DIR=... python scripts/s0_check.py --workers 6` reproduces attempt 3's settings.

**The three attempts (convergence rule, Step 6).** Our rule (`CONVERGENCE`): residual drop at least
5 orders, and CL and CD each varying by at most 1e-6 over the last 500 iterations.

- Attempt 1, `cauchy_eps` 1e-7 (`results/s0/attempt1_eps1e-7.log`, `_summary.json`): **0 of 24
  converged by our rule.** Every run exited 0 on SU2's Cauchy stop, at 736 to 3,735 iterations,
  with residual drop 5.13 to 11.52 orders (so the residual criterion passed everywhere) but CL
  varying 1.4e-6 to 2.85e-5 and/or CD 1.0e-6 to 2.1e-6 over the window: the stop fired while the
  500-iteration window still held the start-up transient. S0_EXIT=1.
- Attempt 2, `cauchy_eps` 1e-8 (`results/s0/attempt2_eps1e-8_partial.log`): killed by the
  controller harness's background time limit with 23 of 24 done (anchor-mg0 at 512x193 never
  finished; no summary was written). **9 of 23 converged**; the 14 failures were just over the
  line, CL varying 1.01e-6 to 2.84e-6 over 500 iterations.
- Attempt 3, `cauchy_eps` 1e-9 (`results/s0/attempt3_eps1e-9_first_pass.log`, `_summary.json`):
  first pass 22 of 24. The two failures, sym-mg0 and sub0-mg0 at 256x97 (886 and 1,029
  iterations), have `exit_code` -2 in the summary: SIGINT, an external interruption, not a
  convergence result. The old driver recorded them as failures; the fix wave (f47a776) made a case
  with no result leave S0 incomplete, and the resume on that driver reran only those two (1,561 and
  1,565 iterations, both converged). **`results/s0/s0.log` and `results/s0/summary.json` are the
  final S0 record: 24 of 24 converged, `NOT CONVERGED: none`, scorer check worst difference
  4.936e-11 (< 1e-6), S0_EXIT=0.** The 22 first-pass `result.json` files predate the driver's
  `settings` field; the two reruns carry `cauchy_eps: 1e-09`, and `s0.log`'s first line records
  `Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-09)`.

Why 1e-9 and not 1e-8: SU2's Cauchy stop is relative (the fix wave's finding, now
`apno.settings.consistency_errors`): at the stop CL may still vary by about
`cauchy_elems * cauchy_eps * |CL|` over the window. The failures fit it: sub2 (CL 0.570) varied
2.82e-5 at 1e-7 and 2.84e-6 at 1e-8 against 500 x eps x 0.570 = 2.85e-5 and 2.85e-6. At 1e-9 the
bound at the design's highest-lift corner is 1e-9 x 500 x CL_BOUND (1.86) = 9.3e-7 < force_tol
1e-6, so `consistency_errors()` is empty — **with a 7 % margin. Known risk:** a high-lift S1 case
can still stop on SU2's rule with CL varying a little over 1e-6; it then fails our rule and is
recorded as a failure in S1's summary, not silently dropped. S1's summary also records the
measured max |CL| beside CL_BOUND and warns if it exceeds it.

**Iteration caps** unchanged (10,000 / 20,000 at 512x193): S0 never hit a cap; the most was 5,210
iterations (anchor-mg0 at 512x193).

**Multigrid** (`summary.json` "multigrid", 256x97; wall s / CPU s / iterations, MGLEVEL 0 vs 3):

| case | MG 0 | MG 3 |
| --- | --- | --- |
| sym | 95 / 94 / 1,561 | 224 / 179 / 1,113 |
| sub0 | 94 / 93 / 1,565 | 240 / 204 / 1,270 |
| sub2 | 133 / 127 / 1,579 | 212 / 205 / 1,254 |
| super | 209 / 199 / 2,480 | 343 / 283 / 1,829 |
| anchor | 280 / 238 / 2,910 | 320 / 309 / 2,044 |

MGLEVEL 3 converged every case and agreed with MGLEVEL 0 on CL to 3e-9 (super, anchor) or better,
and took fewer iterations, but cost more wall and CPU time on all five (median wall 239.8 s
against 132.6 s). The plan's rule (choose 3 only if it also cuts the median wall time) gives
**MGLEVEL 0**.

**Outer radius** (`summary.json` "radius", CL at 50 / 100 / 200 chords, 256x97):
sub2 0.5679008 / 0.5704280 / 0.5717173; anchor 0.3137152 / 0.3173633 / 0.3194246.
|CL(200) − CL(100)| = **1.29e-3 (sub2) and 2.06e-3 (anchor), both ABOVE the plan's 1e-3
threshold** (and 50 → 100 moved them 2.53e-3 and 3.65e-3). Recorded, and the radius is **kept at
100 chords** as the plan directs: moving it changes every mesh and is the author's decision, not
S0's. Consequence: CL in the dataset carries a far-field truncation of the order of 1e-3 to a few
1e-3 relative to an unbounded domain; it is the same for every method compared, since all use the
same meshes.

**The floor** (training grid 256x97 minus fine grid 512x193, `summary.json` "mesh"; CD in counts =
1e4 x CD; surface Cp compared at the fine grid's every second wall point):

| case | ΔCL | ΔCD (counts) | ΔCM | Cp L2 | Cp max |
| --- | --- | --- | --- | --- | --- |
| sym | 1.2e-14 | −0.13 | 1.5e-15 | 0.0346 | 0.301 |
| sub0 | −2.45e-3 | −0.03 | −5.80e-4 | 0.0340 | 0.300 |
| sub2 | −3.36e-3 | +0.12 | −7.23e-4 | 0.0337 | 0.295 |
| super | −1.19e-2 | −8.92 | −3.78e-3 | 0.0328 | 0.429 |
| anchor | −8.67e-3 | −3.53 | −2.35e-3 | 0.0335 | 0.320 |

Observed orders (log2 of |coarse − train| / |train − fine|), CL / CD / CM: sym −8.33 / 4.77 /
−1.14 (CL and CM are round-off, ~1e-14); sub0 0.02 / 7.07 / −0.15; sub2 −0.84 / 5.32 / −2.23;
super −1.68 / −0.84 / −2.80; anchor −0.63 / 0.88 / −1.20. **For CL they are negative or
non-monotone on every lifting case**: sub2, super and anchor have the 256→512 change larger than
the 128→256 change, and sub0's CL goes 0.28787 / 0.28538 / 0.28783 (down then up; its 0.02 comes
from a non-monotone sequence). CD is non-monotone on super and anchor only (sym and sub0 are monotone decreasing, consistent with their positive orders). **The three grids
are not in the asymptotic range, and Richardson extrapolation does not apply**; the floor is
reported as the plain train-minus-fine difference above, not as an extrapolated error.

**Known answers** (`summary.json` "known_answers", 512x193):
- sym |CL| = 2.1e-15 (zero to round-off, as symmetry requires).
- Subcritical CD (inviscid, so ideally zero): sym −0.74, sub0 −0.21, sub2 +1.23 counts.
- Lift-slope ratio, (CL(sub2) − CL(sub0)) / 2° against 2π/√(1 − M²) at M 0.5: **1.129** (above
  Prandtl-Glauert, as a 12 %-thick section's slope is expected to be; recorded, not judged).
- Anchor (NACA 0012, M 0.8, α 1.25°) CL at 512x193: **anchor_cl_fine = 0.326031618**.
- Vassberg & Jameson (2010), "In pursuit of grid convergence for two-dimensional Euler solutions",
  J. Aircraft 47(4):1152-1166, DOI 10.2514/1.46737 (earlier AIAA 2009-4114): the DOI resolves via
  Crossref, but the paper is paywalled at AIAA ARC, no open copy was found, and web search is not
  available in this environment. **The paper was not read, so no comparison is made**; a
  remembered value is not used as if it had been read. The comparison stays open.

**Per-case cost** (MGLEVEL 0, radius 100, the final `runs/s0/*/*/result.json`, five cases per
level, 6 concurrent workers; median (mean) seconds):

| level | wall | CPU |
| --- | --- | --- |
| 128x49 | 23.7 (27.2) | 18.0 (20.7) |
| 256x97 | 132.6 (162.3) | 127.0 (150.4) |
| 512x193 | 1,248.6 (1,389.7) | 1,124.2 (1,290.3) |

Command: `python -c "import json,glob,statistics as st; R=[json.load(open(p)) for p in glob.glob('runs/s0/*/*/result.json')]; [print(lv, st.median(w:=[r['wall_s'] for r in R if r['level']==lv and r['mglevel']==0 and r['radius']==100.0]), st.mean(w), st.median(c:=[r['cpu_s'] for r in R if r['level']==lv and r['mglevel']==0 and r['radius']==100.0]), st.mean(c)) for lv in ('128x49','256x97','512x193')]"`.

Projected S1 time with 8 workers, wall: 1,104 cases at 256x97, 1,104 x 132.6 / 8 = **5.1 h** at
the median (6.2 h at the mean); 80 cases at 512x193, 80 x 1,248.6 / 8 = **3.5 h** (3.9 h at the
mean); about 9-10 h in all. The transonic cases cost about twice the subcritical ones, so the
campaign's Mach mix sets where in that range it lands, and S0's wall times were measured at 6
concurrent processes, not 8.

**Workers: 8.** The machine has 12 cores; the endpoint-security agents use about 1.5 of them; SU2's
resident memory was measured at about 100 MB per 256x97 process (controller ruling).

**Frozen** (`src/apno/settings.py`, this commit): `BASE = Su2Settings(mglevel=0, iter_cap=10000,
cauchy_elems=500, cauchy_eps=1e-9)`, `FINE_ITER_CAP = 20000`,
`CONVERGENCE = {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}` (unchanged),
`WORKERS = 8`, `FROZEN = True`; `consistency_errors()` is empty and `check_frozen()` passes.

## 2026-10-02 — S3 execution: the PINN's Adam phase on the Apple GPU, and sharded runs

- **Decision (controller, spec 1 and 7.5 unchanged).** Adam runs in float32 on
  `PinnConfig.device` (`"auto"`: MPS when `torch.backends.mps.is_available()`, else CPU); L-BFGS
  and `evaluate_on_grid` run on the CPU in float64, because MPS has no float64. Collocation points
  are drawn from the CPU generator and then moved, so a seed gives the same points on either device.
  Before this change the PINN could not run on MPS at all: the Kutta point in `losses` was created
  on the default device ("Tensor for argument input is on cpu but expected on mps").
- **The controller's probe** (a throwaway script, 20,000 points, float32, one Adam step = `losses`
  + backward + `Adam.step`), measured while the S1 SU2 campaign occupied 8 CPU cores:

  | step | CPU, idle machine (fix wave) | CPU, under S1 load | MPS (GPU idle) |
  |---|---|---|---|
  | Euler Adam step | 0.31 s | 2.4 s | 0.18 s |
  | viscous (nu=1e-3) Adam step | 2.12 s | 11 s | 0.65-0.73 s |

- **Re-measured with the code's own `time_steps`** on test_in-0000 at the spec sizes (20,000
  points, 7x96 tanh), `n_adam=2`, `n_closure=1`, wall seconds per Adam step on the stated device and
  per float64 L-BFGS closure on the CPU, with the projected hours per run from `projected_s`.
  **Every CPU figure, including every L-BFGS closure, was taken while the S1 SU2 campaign occupied
  8 cores (load average about 30), so they are upper bounds;** the MPS Adam figures are not
  inflated by that load.

  | Adam device | variant | Adam step | closure (CPU) | projected per run |
  |---|---|---|---|---|
  | mps | Euler | 0.169 s | 3.36 s | 6.1 h |
  | mps | visc1e-3 | 0.631 s | 16.3 s | 27.9 h |
  | cpu | Euler | 2.19 s | 3.37 s | 23.0 h |
  | cpu | visc1e-3 | 8.01 s | 16.5 s | 89.7 h |

  Command:
  `python -c "from apno.data.sampling import case_specs; from apno.geometry.ogrid import TRAIN, Section; from apno.models.pinn import PinnCase; from apno.models.train_pinn import *; s=case_specs('test_in',TRAIN)[0]; c=PinnCase(s.mach,s.alpha_deg,Section(s.t,s.m)); print(s.case_id); [print(d, nu, r, projected_s(cfg,r)/3600, flush=True) for d in ('mps','cpu') for nu in (0.0,1e-3) for cfg in [PinnConfig(viscosity=nu, device=d)] for r in [time_steps(c,cfg,2,1)]]"`.
  Output:

  ```
  test_in-0000
  mps 0.0 {'adam_device': 'mps', 'lbfgs_device': 'cpu', 'adam_s': 0.16867806250229478, 'closure_s': 3.362479874980636} 6.075761458325562
  mps 0.001 {'adam_device': 'mps', 'lbfgs_device': 'cpu', 'adam_s': 0.6313174165115925, 'closure_s': 16.27935420902213} 27.87119265012734
  cpu 0.0 {'adam_device': 'cpu', 'lbfgs_device': 'cpu', 'adam_s': 2.1937446249939967, 'closure_s': 3.3733409580017906} 22.966400983285794
  cpu 0.001 {'adam_device': 'cpu', 'lbfgs_device': 'cpu', 'adam_s': 8.008924750014558, 'closure_s': 16.50974183299695} 89.6712365737282
  ```

  The MPS Adam steps reproduce the probe (0.17 against 0.18 s Euler, 0.63 against 0.65-0.73 s
  viscous). Under this load the CPU L-BFGS phase now dominates an MPS run: 5,000 closures are
  16,800 s of the Euler run's 21,900 s and 81,400 s of the viscous run's 100,300 s. Combining the
  MPS Adam steps with the fix wave's idle-machine closures (0.76 s Euler, 7.26 s viscous) is
  arithmetic, not a measurement: 2.5 h and 15.3 h per run. The preflight measures the load present
  when it runs, so S3's projection should be taken after S1 finishes.
- **Sharding.** `scripts/s3_pinn.py --shard K/N` runs the plan entries whose index in `plan()`
  order is K mod N (the order is unchanged); each shard budgets only its own unfinished runs and
  writes `results/s3/pinn_runs_shard{K}of{N}.json` and `preflight_shard{K}of{N}.json` (`0/1`, the
  default, keeps the unsharded names). The preflight and the run record name the device of each
  phase; prediction files carry `train_device`, `adam_device` and `lbfgs_device`.

## 2026-10-04 — the Euler PINN converges to fluid at rest

- **The S3a warm-up failed** (`python scripts/s3_warmup.py`, then the Euler PINN of spec 7.1-7.7,
  Adam on MPS and L-BFGS on the CPU, 7.65 h wall; `results/s3/warmup.json`, exit 1): max|ΔCp| 3.94
  against the 0.1 bound, CL 7.8e-7 and CD 1.8e-5, final residual 1.7e-5. Read off the saved field
  (`runs/warmup_q.npy`): the PINN converged to **fluid at rest around the cylinder** — wall speed
  0.004 where potential flow gives 2.0, Cp 0.937 everywhere (stagnation), free stream reached only at
  the outer boundary. Mass and momentum are conserved to 1e-5 for r < 2 but violated in the far
  field: closed-line mass flux −0.13 and x-momentum flux −0.42 at r 25-50.
- **Five hypotheses, each refuted** by a throwaway probe (scripts in `runs/warmup_probe*.py`, logs
  in `runs/probe*.log`; not committed). Each trained the cylinder warm-up configuration (or the
  showcase airfoil) with Adam for 6,000-8,000 steps from seed 0 and printed the shoulder wall speed
  (node i = 64 of 256x97, potential flow 2.0) and max|ΔCp| every 1,000 steps:
  1. *Residual weighting* — the mean of |div|² over points uniform in (ξ, η) weights the domain by
     1/J, so the residual weighted by J^p: p = 0 (as specified) collapses by step 1000 (shoulder
     0.001, max|ΔCp| 3.02, `probe_p0_cpu.log`); p = 0.5 collapses (0.002 and 3.92 at step 8000);
     p = 1 does not collapse but never leaves its initial state (shoulder 0.987, max|ΔCp| 3.35,
     unweighted residual 9.6). Weighting moves where the network fails, not whether.
  2. *Low-Mach stiffness* (at M 0.1 the momentum residual is 71x more sensitive to density than to
     velocity) — an exact reparameterisation, the network outputting the pressure perturbation in
     dynamic-pressure units: J⁰ collapses (shoulder 0.004, max|ΔCp| 3.27 at step 6000,
     `probe2_cylP0.log`); J¹ stays near its initial state (0.998, 3.46).
  3. *Far-field size* — a 10-chord outer boundary collapses (shoulder 0.022, max|ΔCp| 3.81 at step
     6000, `probe3_cylR10.log`).
  4. *Rotational solutions* — a homentropic closure collapses on the cylinder (0.003, 4.00,
     `probe3_cylH.log`) and on the airfoil (wall speed 0.009, Cp_min 1.03 against SU2's −0.94,
     `probe3_foilH.log`).
  5. *The real cases* — the unmodified PINN on showcase case test_in-0022 collapses too (wall speed
     max 0.001-0.005 from step 1000; Cp_min −0.58 against SU2's −0.94 at step 4000,
     `probe2_foil0.log`): the failure is S3's, not the warm-up's.
- **The decisive probe** (`runs/warmup_probe4.py`, `runs/probe4.log`): (a) the analytic
  incompressible potential flow through the Euler residual has mean square 8.3e-8 at M 0.01 and
  8.2e-4 at M 0.1 (it is not the compressible solution), above the rest state's 6e-5 to 2e-4; (b) a
  network supervised-fitted to it for 4,000 Adam steps (shoulder 1.995, max|ΔCp| 0.030 — it would
  pass the warm-up) has PINN loss 0.31 (residual 3.10e-1, outer 3.8e-3), about 1000x the rest
  state's ~1e-4, because small fit errors are large derivative errors; (c) 1,000 PINN-only Adam
  steps from that fit drive it away from the answer (residual 5.85e-3, shoulder 1.237, max|ΔCp|
  1.812).
- **Root cause.** Under the strong-form Euler residual with a soft far-field condition, fluid at rest
  satisfies the equations and the wall condition exactly and violates only the weakly enforced far
  field, and every near-correct flow costs more than it: descent leaves even the true solution. It
  is the formulation, not a parameter. For the full-potential equation in subcritical flow the
  residual loss is (near-)convex in the potential, with no rest attractor, and a Dirichlet far field
  makes rest inadmissible.
- **The author's decision (2026-10-04): "Full-potential PINN".** Spec §7.8 records the formulation
  (φ = free stream + Γξ + ψ·(1 − η), learned Γ, continuity div(ρ∇φ) = 0 with isentropic Bernoulli
  density, soft Neumann wall, exact Dirichlet far field, Kutta on ∂φ/∂ξ at the trailing edge; exact
  for shock-free inviscid flow, an isentropic approximation at shocks). The Euler PINN stays runnable
  as this negative result: `PinnConfig(formulation="euler")`, `--formulation euler` on
  `scripts/s3_warmup.py` and `scripts/s3_pinn.py`.
- **The analytic field through the new residual** (`apno.physics.potential.residual`, 20,000 points
  uniform in (ξ, η), seed 1, incompressible cylinder potential with Bernoulli density): mean square
  2.6e-8 at M 0.01 and 2.6e-4 at M 0.1, with zero wall violation. Command:
  `python -c "import torch; torch.set_default_dtype(torch.float64); from apno.geometry.ogrid import *; from apno.models.pinn_potential import state_from_potential as s; from apno.physics.potential import residual; g=Cylinder(); R=stretch_ratio(100.0); G=torch.Generator().manual_seed(1); xi=torch.rand(20000,generator=G); eta=1-torch.rand(20000,generator=G); phi=lambda a,b,x,y: (x-0.5)+0.25*(x-0.5)/((x-0.5)**2+y*y); [print(M, float((residual(xi,eta,lambda a,b: s(phi,a,b,g,100.0,R,M),g,100.0,R)**2).mean())) for M in (0.01,0.1)]"`.
- **Probe of the potential formulation (evidence, not a pass criterion).** The cylinder warm-up
  configuration (`PinnConfig(kutta=False, zero_circulation=True, device="cpu")`: spec sizes, Γ fixed
  at 0, the first 3,000 steps of the 30,000-step Adam schedule) on the **CPU** (the GPU was running
  S2's FNO training), seed 0, evaluating a copy of the network on the 256x97 grid every 500 steps
  with `evaluate_on_grid` and `circle_cp_error` (load average 17-53; 2,678 s wall, 0.89 s per step):

  | Adam step | shoulder speed (2.0) | max\|ΔCp\| | residual | wall | total |
  |---|---|---|---|---|---|
  | 0 | 1.017 | 2.968 | 1.56e+0 | 5.03e-1 | 6.58e+0 |
  | 500 | 2.305 | 1.658 | 6.08e-1 | 1.51e-2 | 7.60e-1 |
  | 1000 | 2.299 | 1.362 | 4.89e-1 | 2.03e-2 | 6.92e-1 |
  | 1500 | 2.512 | 2.300 | 5.92e-1 | 7.59e-3 | 6.68e-1 |
  | 2000 | 2.323 | 1.367 | 4.96e-1 | 1.96e-2 | 6.92e-1 |
  | 2500 | 2.613 | 2.753 | 4.32e-1 | 6.34e-3 | 4.95e-1 |
  | 3000 | 2.671 | 3.041 | 3.94e-1 | 5.26e-3 | 4.47e-1 |

  It does **not** collapse toward rest: the shoulder speed leaves 1.0 within 500 steps and stays at
  2.3-2.7. It is not converged either: it overshoots 2.0 (the max|ΔCp| is the shoulder's, and grows
  from step 2000), and its residual, 0.39 at step 3000, is ~1500x the analytic field's 2.6e-4 at
  this Mach, so the descent has not yet reached the minimum near the right answer. Whether the full
  30,000 Adam steps and L-BFGS reach it is what the warm-up (run by the controller) measures. The
  probe was a script outside the repository: it repeated `train`'s Adam loop (`make_pinn`,
  `_points`, `losses`, the same Adam and exponential decay) and printed
  `step k: shoulder ... max|dCp| ... residual ... wall ... total ...` every 500 steps.

## 2026-10-05 — stage S2: POD and FNO trained, predictions written

Commands: `bash runs/run_s2_pod.sh` (`python scripts/s2_operator.py pod`), then `bash runs/run_s2_fno.sh`
(`python scripts/s2_operator.py tune`, then `final`). Exit codes 0 for all three; logs in
`results/s2/{pod,tune,final}.log`. Figures below are read from `results/s2/tuning.json`,
`results/s2/runs.json`, `results/s2/pod.json` and `runs/fno/*/train.json`. **No test scores here**;
they come in S4.

**Hardware.** The FNO trained on the Apple GPU (`device: mps` in every `train.json`). Its "wall"
figures are therefore **GPU wall time on this Mac**, shared with other load; the CPU seconds are the
host process's CPU time while it drove the GPU, not a CPU-training cost. POD ran on the CPU.

**Tuning (8 configurations, n = 896, loss A, 100 epochs, layers 4, lr 1e-3, batch 8).** Selected:
**`tune-6`, width 48, modes 24 x 8**, validation relative L2 0.02214 (best epoch 96).

| Tag | width | modes (i x j) | best val | epochs | wall s | CPU s |
|---|---|---|---|---|---|---|
| tune-0 | 32 | 16 x 8 | 0.03154 | 100 | 419 | 216 |
| tune-1 | 32 | 16 x 12 | 0.03083 | 100 | 443 | 232 |
| tune-2 | 32 | 24 x 8 | 0.03101 | 100 | 443 | 224 |
| tune-3 | 32 | 24 x 12 | 0.03100 | 100 | 488 | 226 |
| tune-4 | 48 | 16 x 8 | 0.02290 | 100 | 658 | 255 |
| tune-5 | 48 | 16 x 12 | 0.02225 | 100 | 763 | 309 |
| **tune-6** | 48 | 24 x 8 | **0.02214** | 100 | 793 | 302 |
| tune-7 | 48 | 24 x 12 | 0.02345 | 100 | 859 | 315 |

Width separates the configurations (0.031 at 32 against 0.022-0.023 at 48); the mode counts within a
width differ by less than the seed-to-seed spread seen below (0.0118-0.0127 at 300 epochs), so the
choice of `tune-6` over `tune-5` (0.5 % apart) is not a strong one.

**Final runs (selected configuration, 300 epochs, patience 30).** Validation is relative L2 on the
standardised-target validation split.

| Tag | n | loss | best val | best epoch | epochs run | wall s | CPU s |
|---|---|---|---|---|---|---|---|
| fno-n896-s0 | 896 | A | 0.012605 | 278 | 300 | 2160 | 849 |
| fno-n896-s1 | 896 | A | 0.012744 | 293 | 300 | 2242 | 927 |
| fno-n896-s2 | 896 | A | 0.011775 | 294 | 300 | 2237 | 926 |
| fno-n896-s0-lossB | 896 | B | 0.46170 | 50 | 81 | 605 | 250 |
| fno-n64-s0 | 64 | A | 0.08493 | 297 | 300 | 257 | 72 |
| fno-n128-s0 | 128 | A | 0.05065 | 288 | 300 | 421 | 140 |
| fno-n256-s0 | 256 | A | 0.03320 | 284 | 300 | 737 | 278 |
| fno-n512-s0 | 512 | A | 0.017842 | 274 | 300 | 1384 | 562 |

The loss-B validation loss is on a different scale from loss A and must not be compared with it
directly; it stopped early by patience at epoch 81. The n = 64...512 runs and the seeds still improve
at or near their final epoch (best epochs 274-297 of 300), so none is clearly converged at 300 epochs.

**128 x 49 to 256 x 97 transfer error.** Recorded as `transfer_val_rel_l2` in every row of
`results/s2/runs.json` and `results/s2/tuning.json` (not in `runs/fno/*/train.json`): the relative L2
error of the FNO trained at 128 x 49 and evaluated on the 256 x 97 validation fields. Produced by
`python -c "import json; [print(x['tag'], x['transfer_val_rel_l2']) for x in json.load(open('results/s2/runs.json'))]"`
(and the same over `json.load(open('results/s2/tuning.json'))['rows']`).

| run | transfer_val_rel_l2 |
|---|---|
| fno-n896-s0 / s1 / s2 | 0.0348 / 0.0355 / 0.0379 |
| fno-n896-s0-lossB | 0.0595 |
| fno-n64 / n128 / n256 / n512 (s0) | 0.0443 / 0.0476 / 0.0474 / 0.0431 |
| tune-0 ... tune-7 | 0.0637 / 0.0535 / 0.0538 / 0.0458 / 0.0527 / 0.0456 / 0.0436 / 0.0558 |

**POD (`pod.json`; modes chosen from 8...256, smoothing from 0, 1e-6, 1e-4, 1e-2, on validation).**

| n | cases | modes | smoothing | val rel L2 | wall s | CPU s |
|---|---|---|---|---|---|---|
| 64 | 63 | 32 | 0.01 | 0.1135 | 4.8 | 7.2 |
| 128 | 126 | 64 | 0.01 | 0.0929 | 8.5 | 14.3 |
| 256 | 252 | 128 | 0.01 | 0.0814 | 22.3 | 49.0 |
| 512 | 502 | 256 | 0.01 | 0.0711 | 54.8 | 171.4 |
| 896 | 881 | 256 | 0.01 | 0.0661 | 98.4 | 351.3 |

The chosen smoothing is the largest value in the grid at every n, and the chosen mode count is the
largest allowed (256) at n = 512 and 896, so both selections sit on the edge of their search grids
and the POD baseline may be under-tuned. The FNO and POD validation errors are computed by different code paths in different units of account, so they are not compared here; S4 scores both on the test sets with one scorer.

Prediction files for both test sets were written to `runs/predictions/` (not committed).

## 2026-10-05 — the potential PINN's trailing edge, and S3 without the viscous runs

Author's decision of 2026-10-05: "Fix + non-viscous runs". Spec §7.9 records the amendment; the
probes are `runs/probe_te_exclusion.py`, `runs/probe_equal_kutta.py` and their logs (not committed).

**The trailing-edge singularity.** On showcase case test_in-0022 (M 0.3605, α 2.19°, NACA t 0.140,
m 0.0022) the §7.8 PINN's residual at initialisation was ~1e5, the top 10 of 20,000 collocation
points carrying 99.9 % of it, all within ξ < 0.0014 and η < 0.05 of the trailing edge (TE), where
J ~ 1e-3; on the cylinder the top-10 share is 1.7 %. Two causes. (a) The circulation term Γξ: the
cosine-clustered map has x_ξ = y_ξ = 0 at the TE, so |∇ξ| ~ 1/√r there; the Γ-only residual is 1.2e4
with Θ = ξ and 8e-3 with the quarter-chord vortex angle Θ_v. Measured through the map at
ξ ∈ {1e-4, 1e-3}, η ∈ {0, 1e-4}: |∇ξ| = 11,191 / 4,882 / 993 / 891, against |∇Θ_v| = 0.2122 at all
four (the analytic 1/(2π · 0.75)). (b) The network term's second derivatives are singular at the sharp
corner whatever its input features, as the true corner flow is. The S3 shards started on the §7.8
code thrashed (residual 150-1000, wall loss 5-110 after 11,000 Adam steps) and were stopped
(`results/s3/shard*_stopped_te_singularity.log`).

**The fix that trains.** Θ_v = 1 − ((atan2(y, x − 0.25)) mod 2π)/(2π) for the circulation term, plus
dropping the interior points within 0.02 chord of the TE from the residual: in the probes the
initial residual is ~8 (cylinder-like), training does not collapse, and the wall Cp L2 against SU2 is
0.33-0.35 with Cp_min −0.78 to −0.95 (SU2 −0.942) by 4,000-5,000 Adam steps
(`runs/probe_teexcl_trig.log`, `runs/probe_eqk_excl02_d*.log`). A 0.005-chord exclusion does not
train (initial residual 566; suction peak lost).

**The Kutta condition as first implemented was degenerate.** (∂φ/∂ξ)² at the TE point never
constrains Γ, because x_ξ = y_ξ = 0 there and the free-stream and vortex terms contribute nothing.
It is replaced by the equal-speed form (|u|²(ξ = δ) − |u|²(ξ = 1 − δ))² on the wall, δ = 0.055
(≈ 0.03 chord upstream of the TE, just outside the exclusion), an approximation to the TE limit.
**The circulation finding:** the equal-speed condition is satisfied to ~1e-6 in every probe, yet Γ
stays ~0.02, where SU2's CL 0.319 implies Γ = CL/2 ≈ 0.16. The PINN's lift is about 8× too small.
That is a result to report, not a defect these changes fix.

**Evidence probe on the production code** (first 3,000 Adam steps of the S3 `pure` run on
test_in-0022: `PinnConfig()` defaults, the production lr schedule, resampling and loss, float32 on the
Apple GPU at 0.19 s per step; wall Cp of the CPU float64 evaluation against SU2's 512 × 193 truth
sampled `[::2, ::2]`, wall row). Points dropped by the exclusion at the resamples at 0 / 1000 / 2000 /
3000: 225 / 197 / 212 / 211 of 20,000.

| Adam step | residual | wall | kutta | Γ | wall Cp L2 | max abs ΔCp | Cp_min (SU2 −0.942) |
|---|---|---|---|---|---|---|---|
| 0 | 9.12 | 0.901 | 7.3e-2 | 0.0000 | 0.918 | 10.60 | −9.410 |
| 1000 | 0.0991 | 0.0601 | 1.7e-5 | 0.0123 | 0.410 | 1.193 | −0.158 |
| 2000 | 4.26 | 0.0556 | 2.8e-3 | 0.0147 | 0.450 | 1.771 | −0.777 |
| 3000 | 0.0777 | 0.0360 | 2.3e-6 | 0.0177 | 0.387 | 1.163 | −0.773 |

The losses are the step's own batch, taken just after a resample (so the 2000-step residual is the
first look at fresh points). No collapse to rest (the suction peak reaches −0.77 by step 2000), Kutta
is met to ~1e-6, and Γ grows slowly (0.0177 at 3,000 steps, against ~0.16 implied by SU2), as in the
probes.

**S3 plan, by the author's scope decision.** The six showcase cases without artificial viscosity
(the base variant renamed `euler` → `pure`), `gradnorm` and `nokutta`: 8 runs. The viscous variants
are not planned (selectable with `--viscous`). `--preflight-file PATH` reuses an uncontended
preflight's timings, because three shards preflighting at once measured one variant at 0.83 / 1.14 /
29.3 s per Adam step against 0.75 s uncontended and refused on that noise.
