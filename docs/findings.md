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

## 2026-10-06 — stage S1: the data campaign (run 2026-10-01 to 2026-10-04)

Commands (`runs/run_s1.sh`, train pass then fine pass, both exit 0; logs `results/s1/{train,fine}.log`):
`SU2_BIN_DIR=$HOME/theLab/bwb-tanker-scratch/bin python scripts/s1_campaign.py train` and the same with `fine`.
Summaries: `results/s1/campaign_{train,fine}.json`. Dataset: `python scripts/build_dataset.py`
(`results/s1/build.log`, `failures_{256x97,512x193}.json`, `showcase.json`; HDF5 under `data/hdf5/`).

| Pass | Level | Requested | Converged | Failed | Summed CPU | Summed wall |
|---|---|---|---|---|---|---|
| train | 256x97 | 1,104 | 1,084 | 20 (19 residual, 1 CL) | 69.0 h | 79.1 h (sum over workers) |
| fine | 512x193 | 80 | 75 | 5 (4 residual, 1 CL) | 42.3 h | 44.1 h (sum over workers) |

Per case, medians over all finished cases (`runs/su2/*/*/result.json`): 256x97 took 2,057 iterations,
227.5 s wall and 198.6 s CPU; 512x193 took 3,976.5 iterations, 1,554 s wall and 1,501.7 s CPU.
The 256x97 pass covers all four splits (train 896, val 128, test_in 64, test_ext 16); the 512x193 pass
covers the two test splits only (64 and 16).

**Failure reasons.** Failing means the convergence rule was not met, not that the solver crashed: a
"residual" failure is a residual drop below 5 orders (1.16 to 4.85 orders at 256x97, 1.25 to 2.87 at
512x193), always with CL and CD still moving over the last 500 iterations; the two "CL" failures
(train-0286 at 256x97, test_ext-0008 at 512x193) dropped the residual enough but CL varied by 1.55e-05
and 3.98e-06 against the 1e-06 limit. Every failed case has a zero driver exit (no driver errors).

**Inputs of every failed case** (Mach, alpha deg, t, m):

| Case | Level | Mach | alpha | t | m |
|---|---|---|---|---|---|
| train-0022 | 256x97 | 0.7475 | 2.613 | 0.1401 | 0.0357 |
| train-0102 | 256x97 | 0.7740 | 5.512 | 0.1455 | 0.0180 |
| train-0150 | 256x97 | 0.7514 | 5.943 | 0.1368 | 0.0322 |
| train-0246 | 256x97 | 0.7629 | 4.227 | 0.1561 | 0.0276 |
| train-0281 | 256x97 | 0.7696 | 0.779 | 0.1530 | 0.0393 |
| train-0286 | 256x97 | 0.7329 | 3.240 | 0.1334 | 0.0311 |
| train-0326 | 256x97 | 0.7947 | 4.789 | 0.1415 | 0.0314 |
| train-0374 | 256x97 | 0.7673 | 5.742 | 0.1177 | 0.0379 |
| train-0454 | 256x97 | 0.7989 | 3.655 | 0.1354 | 0.0354 |
| train-0470 | 256x97 | 0.7434 | 5.450 | 0.1520 | 0.0018 |
| train-0622 | 256x97 | 0.6973 | 5.255 | 0.1475 | 0.0252 |
| train-0694 | 256x97 | 0.7561 | 5.857 | 0.1438 | 0.0207 |
| train-0726 | 256x97 | 0.7421 | 4.066 | 0.1233 | 0.0391 |
| train-0774 | 256x97 | 0.7848 | 5.171 | 0.1586 | 0.0395 |
| train-0870 | 256x97 | 0.7764 | 4.879 | 0.1392 | 0.0204 |
| val-0094 | 256x97 | 0.7274 | 5.693 | 0.1501 | 0.0340 |
| val-0102 | 256x97 | 0.7723 | 3.552 | 0.1427 | 0.0263 |
| test_in-0024 | 256x97 | 0.7335 | 4.609 | 0.1471 | 0.0246 |
| test_in-0032 | 256x97 | 0.7811 | 2.720 | 0.1527 | 0.0394 |
| test_ext-0014 | 256x97 | 0.8176 | 4.321 | 0.1595 | 0.0388 |
| test_in-0024 | 512x193 | 0.7335 | 4.609 | 0.1471 | 0.0246 |
| test_in-0032 | 512x193 | 0.7811 | 2.720 | 0.1527 | 0.0394 |
| test_in-0048 | 512x193 | 0.7534 | 4.307 | 0.1368 | 0.0091 |
| test_ext-0008 | 512x193 | 0.8288 | 5.617 | 0.1189 | 0.0091 |
| test_ext-0014 | 512x193 | 0.8176 | 4.321 | 0.1595 | 0.0388 |

Where they sit: all 25 rows have Mach 0.697 to 0.829 (23 of 25 at 0.73 or above) and alpha 0.78 to 5.94
degrees (24 of 25 at 2.5 degrees or more), so they are in the transonic, higher-lift part of the box:
the shock-dominated cases where the Euler solver limit-cycles. The 256x97 failures of the test splits
recur at 512x193 (test_in-0024, test_in-0032, test_ext-0014); the fine grid adds test_in-0048 and
test_ext-0008. Failed cases are excluded from the HDF5 files, not patched.

**Dataset (HDF5).** `256x97.h5`: train 881, val 126, test_in 62, test_ext 15 (1,084 cases).
`512x193.h5`: test_in 61, test_ext 14 (75 cases). The showcase takes the 61 test_in cases that
survive at both levels, sorted by fine-grid maximum surface Mach, index round(p(n-1)):
test_in-0022 (0.512, subcritical), test_in-0021 (0.684, subcritical), test_in-0043 (0.847,
subcritical), test_in-0004 (1.206, supercritical), test_in-0031 (1.377, supercritical),
test_in-0040 (1.610, supercritical). Three subcritical, three supercritical (`results/s1/showcase.json`).

**Machine sleeps.** The machine entered sleep repeatedly during the campaign; `pmset` shows "Entering
Sleep state due to 'Dark Wake Thermal Emergency'" (lid closed / dark wake), at least on 2026-10-02 and
again twice before 05:36 on the same night. Throughput fell from about 120 to about 19 cases per hour
overnight while per-case wall time was unchanged (median 197 s at that point), so the sleeps paused the
campaign rather than slowing the solver. The mitigation was `caffeinate -i -s -m -w <run_s1 pid>`,
which keeps the machine from idle or system sleep; awake throughput was then about 90 cases per hour.
The campaign is resumable, so the pauses cost time only.

## 2026-10-06 — stage S3: the PINN showcase runs (potential formulation, non-viscous plan)

**Command.** `bash runs/run_s3.sh`, i.e. two shards of
`python -u scripts/s3_pinn.py --shard k/2 --preflight-file results/s3/preflight.json` (k = 0, 1;
`OMP_NUM_THREADS=6`). Logs `results/s3/shard{0,1}.log`, run records
`results/s3/pinn_runs_shard{0,1}of2.json`, preflight `results/s3/preflight.json` (35.2 h serial
projected against a 48 h budget), predictions `runs/predictions/pinn-<variant>-<case>.h5` (8 files).
Both shards exited 0. Formulation `potential` (full potential, the 2026-10-04/05 decisions); Adam on
the Apple GPU (`mps`), L-BFGS on `cpu`, for all 8 runs. Scores come in S4; none are computed here.

| Run (variant / case) | Shard | Wall h | CPU h | Final residual | Final wall | Final kutta | Final total |
|---|---|---|---|---|---|---|---|
| pure / test_in-0022 | 0 | 5.45 | 9.74 | 2.25e-3 | 1.32e-4 | 1.2e-7 | 3.58e-3 |
| pure / test_in-0043 | 0 | 4.97 | 9.23 | 1.10e-3 | 4.9e-5 | 8.8e-8 | 1.59e-3 |
| pure / test_in-0031 | 0 | 6.47 | 11.57 | 5.31e-3 | 4.18e-4 | 4.5e-7 | 9.50e-3 |
| gradnorm / test_in-0004 | 0 | 4.96 | 7.20 | 9.11 | 9.4e-3 | 1.9e-13 | 82.2 |
| pure / test_in-0021 | 1 | 5.33 | 9.27 | 4.91e-3 | 3.9e-4 | 5.9e-6 | 8.92e-3 |
| pure / test_in-0004 | 1 | 5.01 | 9.40 | 2.98e-3 | 2.2e-4 | 1.2e-7 | 5.15e-3 |
| pure / test_in-0040 | 1 | 6.44 | 11.56 | 2.38e-2 | 2.1e-3 | 5.7e-6 | 4.51e-2 |
| nokutta / test_in-0022 | 1 | 5.40 | 10.08 | 1.82e-3 | 7.8e-5 | 0 | 2.61e-3 |

Values are the `final` and `wall_s`/`cpu_s` fields of the run records (per-shard wall times overlap:
the two shards ran concurrently). The learned Gamma is NOT recorded in the run records or logs; it
is recovered from the saved predictions in S4, not stated here. The circulation term's loss is 0.0 in
every run (as recorded). `pure` and `nokutta` use fixed weights 10 / 10; `nokutta` sets the Kutta
loss to exactly 0.

**Failure-mode result: gradnorm's loss weights ran away.** On test_in-0004 the gradient-norm balancing
drove `w_kutta` from 1.06e4 at Adam step 0 (fixed runs: 10) to 1.9e6 by step 1,500 and a peak of
7.0e10 at step 16,500; it was still 5.8e9 at step 29,999 (`w_wall` 7.75e3 there). The Kutta loss
had fallen to 3.6e-15 and the weights kept growing to compensate, so the weighted total stayed in the
hundreds to thousands (719 at step 29,999). The Adam phase ended with residual 0.024; L-BFGS then
ended at residual 9.11 (total 82.2), i.e. the quasi-Newton phase, run against those weights,
made the PDE residual about 380 times worse than where Adam left it. The result is a run whose
equation is effectively unenforced: compare `pure` on the same case at 2.98e-3. Reported as a result
of the method; nothing was tuned. Command: `grep -n "w_kutta" results/s3/shard0.log` (the log's
`final:` line is the L-BFGS end state).

**nokutta** (test_in-0022, `kutta=False`): final residual 1.82e-3, wall 7.8e-5, total 2.61e-3 — the
same case under `pure` ended at 2.25e-3 / 1.3e-4 / 3.58e-3. The PDE and wall losses alone converge
equally well, so the loss does not reveal whether the circulation is pinned; that is for the S4 lift
comparison.

**Attempts that did not run to completion (kept under explicit names in `results/s3`).**
1. `*_stopped_te_singularity.log` (3 shards, §7.8 code): residual 150-1000 after 11,000 Adam steps
   on the trailing-edge singularity; stopped; fixed as in the 2026-10-05 entry.
2. `*_refused_budget60.log` and `preflight_shard*of3_concurrent_refused.json`: three shards
   preflighting at once measured contended timings (e.g. one variant at 29 s per Adam step, projected
   289.8 h for the shard against 60 h), so the budget guard refused with exit 3. Resolved by
   `--preflight-file` on an uncontended preflight.
3. Memory: 3 shards thrash a 16 GB machine, so the final run used 2 shards (`runs/run_s3.sh`).
4. `preflight_12run_task25code.*`: a preflight of the earlier 12-run plan (with the viscous
   variants) on the Task 25 code, superseded by the 8-run plan.
Earlier warm-ups: `warmup.json` (potential formulation, PASSED) and `warmup_euler_2026-10-02.*` (the
Euler PINN's failure to fluid at rest).

**Dropped.** The four viscous runs (`visc1e-3` x3, `visc3e-4`) were dropped by the author's decision
of 2026-10-05 (projected 119-271 h for one viscous run); they are not part of S3 and are selectable
with `--viscous`.

## 2026-10-06 — stage S4: head-to-head results, and the three predictions checked

Command: `python scripts/s4_score.py` (commit e06ff06), which wrote `results/s4/scores.md`,
`scores.json`, `costs.json`, `break_even.png`, `data_efficiency.png` and six `showcase_*.png`. Every
figure below is read from those files; the per-case figures come from `scores.json`'s `rows`, e.g.
`python -c "import json; [print(r['tag'], r['case_id'], r['score']['cl_err'], r['score']['cl_true']) for r in json.load(open('results/s4/scores.json'))['rows'] if r['method']=='pinn']"`.

**Four things the brief for this stage predates, stated before any result.**

1. **The PINN is the full-potential formulation** (author's decision 2026-10-04), not the Euler PINN
   of spec §7.1-7.7, which converges to fluid at rest (the five refuted hypotheses and the decisive
   probe are in the 2026-10-04 entry). It is exact only for shock-free inviscid flow and is an
   isentropic approximation at shocks, so on supercritical cases it is solving a different equation
   from SU2.
2. **Its airfoil runs use the trailing-edge fixes of 2026-10-05** (quarter-chord vortex angle,
   0.02-chord residual exclusion, equal-speed Kutta condition), **and its lift is under-predicted**
   because none of the Kutta formulations tried constrains the circulation (2026-10-05 entry). That
   entry's "about 8x" on test_in-0022 was Γ ≈ 0.02 against CL/2 ≈ 0.16 at 3,000 Adam steps; at the end
   of the run S4 measures CL 0.0732 against SU2's 0.3192, **4.4x too small**. Both are measurements,
   at different points of the run.
3. **The four viscous runs were dropped** (author's decision 2026-10-05). Prediction 3 (no Kutta) is
   still testable; the spec's comparison of the pure and artificially-viscous PINN at shocks (§7.4) is
   not made, so nothing here says whether viscosity fixes the PINN's shocks.
4. **The costs mix hardware.** SU2 is CPU time on this Mac (one process per case, measured with 8
   concurrent workers). The FNO's training is GPU wall time on this Mac's Apple GPU (MPS). The PINN is
   wall time of Adam on MPS plus L-BFGS on the CPU, two runs at once. POD is CPU time. Every
   break-even or ratio below inherits that mix.

**Test sets.** test_in (62 cases at 256x97, 61 scored: 37 subcritical, 24 supercritical) and
test_ext (14 scored, **all supercritical**: the extrapolation set has no subcritical case, so no
subcritical-extrapolation figure exists). Truth is SU2 at 512x193 sampled every second node; **the
floor** is SU2's own 256x97 solution scored against that truth. No contestant has a missing or
non-finite prediction on any case (`n_missing` and `n_non-finite` are 0 in every row of scores.md).

### The three predictions

**Prediction 1 — "For a single new case, SU2 beats the PINN on cost by one to two orders of
magnitude." HELD.** `costs.json`: PINN per case 20,201.3 s (mean wall time of the six `pure` runs,
5.61 h) against SU2 per case 209.87 s (median CPU time of the 256x97 test cases): **96.3x**, 1.98
orders. On CPU time instead (mean 36,464 s per `pure` run, from `results/s3/pinn_runs_shard*of2.json`)
it is 174x, 2.24 orders, just outside the band; the prediction holds on the basis S4 costs it at, and
the mixed hardware of point 4 above is what separates the two figures. The ratio is for an answer of
very different quality: the PINN's near-field error is 0.52-0.63 against the floor's 0.008-0.024
(table below). Command:
`python -c "import json; c=json.load(open('results/s4/costs.json'))['costs']; print(c['pinn']['per']/c['su2']['per'])"`.

**Prediction 2 — "The FNO beats 'just run SU2' only after roughly as many new cases as it took to
train it, about a thousand, unless the data-efficiency curve shows far fewer training cases are
enough." HELD.** From `break_even(costs, 5000)` on `costs.json`, the FNO line first falls below
SU2's at **n = 1,126 new cases** (fixed cost 236,237 s = campaign 229,211 s + 8 tuning runs 4,867 s +
final training 2,160 s; per query 0.0536 s). The campaign alone is 1,092 SU2-case-equivalents, so the
break-even is the campaign plus about 34 cases' worth of model search and training; on the
final-fit-only fixed cost it is 1,103. POD breaks even at 1,094 (1,092 on its final fit only). The
GPU-wall part of the FNO's fixed cost is 3.0 % of it, so the hardware mix of point 4 barely moves this
result. Command:
`python -c "import json,numpy as np; from apno.score.report import break_even; c=json.load(open('results/s4/costs.json'))['costs']; b=break_even(c,5000); print({k:int(np.argmax(b[k]<b['su2'])) for k in ('fno','pod')})"`.

The "unless" clause does not fire. The data-efficiency curve (`data_efficiency.png`; mean near-field
relative L2 over the 61 test_in cases at 256x97, both regimes pooled; floor 0.01432) stays well above
the floor at every n and is still falling at 896:

| n (train cases paid) | FNO (seed 0) | FNO / floor | POD | POD / floor | campaign, SU2-case-equivalents |
|---|---|---|---|---|---|
| 64 | 0.1088 | 7.6x | 0.1379 | 9.6x | 197 |
| 128 | 0.0858 | 6.0x | 0.1044 | 7.3x | 257 |
| 256 | 0.0698 | 4.9x | 0.0883 | 6.2x | 380 |
| 512 | 0.0561 | 3.9x | 0.0753 | 5.3x | 690 |
| 896 | 0.0483 | 3.4x | 0.0677 | 4.7x | 1,092 |

(The campaign column is `campaign_cpu_s[n] / su2.per`; every n pays for all 128 validation cases.) A
smaller training set would break even sooner — about 199 cases at n = 64, which is arithmetic on
these figures, not a measured curve — but at 7.6x the floor rather than 3.4x. Command for the table (prints the mean per tag and the case count):
`python -c "import json,collections as c; R=json.load(open('results/s4/scores.json'))['rows']; d=c.defaultdict(list); [d[r['tag']].append(r['score']['near_field_rel_l2']) for r in R if r['grid']=='256x97' and r['split']=='test_in']; [print(t, sum(v)/len(v), len(v)) for t,v in sorted(d.items())]"`.

**Prediction 3 — "Without the Kutta condition, the PINN's residuals will look acceptable while its
lift is wrong." PARTLY.** On test_in-0022 (subcritical, SU2 CL 0.3192): `nokutta` final residual loss
1.82e-3 (total 2.61e-3), CL error −0.2667 (CL 0.0526); `pure`, the same case with the Kutta
condition, final residual loss 2.25e-3 (total 3.58e-3), CL error −0.2461 (CL 0.0732). Residual losses
from `results/s3/pinn_runs_shard{0,1}of2.json` (`final`), CL errors from `scores.json`. The part that
held: without the Kutta condition the residual is as small as with it (smaller, in fact) and the lift
is wrong, 84 % low, so the loss does not reveal the circulation. The part that failed: the prediction
implies the Kutta condition is what gets the lift right, and here it does not — with it the lift is
77 % low, a difference of 0.021 in CL against an error of 0.25. On this case the equal-speed Kutta
condition, met to ~1e-7, makes no material difference to the circulation.

### Results by regime and set, with the floor beside every error

Near field = area-weighted relative L2 of the state within 2.5 chords (mean over cases); CL and CD
are the largest absolute error over the group (CD in counts); Cp = RMS wall-Cp error (mean over
cases). FNO rows are seed 0, n = 896, loss A, scored at 256x97 (the evaluation grid) and at 128x49
(its training grid, against the truth at every fourth fine node). The floor exists only at 256x97,
so a 128x49 row is beside it on a different comparison grid. PINN rows are single cases (3 per
regime, test_in only).

| set / regime | method | near field | CL max | CD max (counts) | Cp RMS |
|---|---|---|---|---|---|
| test_in subcritical (37) | **floor** | **0.0080** | **0.0073** | **6.9** | **0.0357** |
| | FNO 256x97 | 0.0423 | 0.0312 | 31.3 | 0.0358 |
| | FNO 128x49 | 0.0102 | 0.0095 | 7.5 | 0.0334 |
| | POD | 0.0168 | 0.0097 | 14.2 | 0.0371 |
| | PINN pure (3) | 0.5155 | 0.2461 | 41.1 | 0.1603 |
| test_in supercritical (24) | **floor** | **0.0241** | **0.0140** | **17.2** | **0.0341** |
| | FNO 256x97 | 0.0575 | 0.0431 | 40.4 | 0.0521 |
| | FNO 128x49 | 0.0374 | 0.0314 | 46.6 | 0.0470 |
| | POD | 0.1462 | 0.0750 | 122.2 | 0.1190 |
| | PINN pure (3) | 0.6273 | 1.4028 | 776.5 | 0.4045 |
| test_ext supercritical (14) | **floor** | **0.0651** | **0.0247** | **32.4** | **0.0598** |
| | FNO 256x97 | 0.2318 | 0.1847 | 180.6 | 0.2038 |
| | FNO 128x49 | 0.2320 | 0.1876 | 178.3 | 0.1966 |
| | POD | 0.5151 | 0.3483 | 530.2 | 0.3512 |

What this says:
- **In range and subcritical the FNO at its own 128x49 resolution is near the floor** (near field
  1.3x, CL 1.3x, CD 1.1x), **but evaluated at 256x97 it is 5.3x the floor** and POD beats it there on the near field, CL and CD
  (not on wall-Cp RMS, 0.0371 against 0.0358). The resolution transfer costs the FNO more than any other single factor in range.
- **Supercritical, in range, the FNO beats POD by 2.5x** on the near field and 3x on CD at 256x97,
  and is 2.4x the floor.
- **Extrapolation is where every surrogate fails**: the FNO is 3.6x the floor on the near field and
  its CD error reaches 181 counts (5.6x the floor's 32); POD is 7.9x and 530 counts. The FNO's CD
  errors on test_ext are biased, not scattered: median −78 counts at 256x97.
- **The PINN is not competitive on any column**: 0.52-0.63 near field, 64x and 26x the floor.

**The PINN per case (all eight runs, test_in, 256x97).** CL true → predicted, and the error:

| run | regime | SU2 CL | PINN CL | CL error | CD error (counts) | near field |
|---|---|---|---|---|---|---|
| pure / 0022 | sub | 0.3192 | 0.0732 | −0.2461 | +41.1 | 0.559 |
| nokutta / 0022 | sub | 0.3192 | 0.0526 | −0.2667 | +30.0 | 0.569 |
| pure / 0021 | sub | 0.0922 | −0.0012 | −0.0934 | −5.6 | 0.629 |
| pure / 0043 | sub | 0.2799 | 0.1008 | −0.1790 | +15.1 | 0.359 |
| pure / 0004 | super | 1.0139 | 1.0662 | +0.0523 | +88.6 | 0.218 |
| pure / 0031 | super | 0.2949 | 0.0007 | −0.2942 | −9.9 | 0.584 |
| pure / 0040 | super | 1.3879 | −0.0149 | −1.4028 | −776.5 | 1.080 |
| gradnorm / 0004 | super | 1.0139 | −0.3272 | −1.3411 | +724.8 | 1.273 |

Five of the six `pure` runs under-predict lift, three of them to roughly zero (0021, 0031, 0040);
test_in-0004 is the exception, CL 5 % high, with a CD error of 89 counts. The floor's CL errors on
these cases are 0.0003-0.0086.

**The FNO's three seeds (n = 896, loss A), range over seeds 0 / 1 / 2.**

| set / regime | grid | near field | CL max | CD max (counts) | shocks missed |
|---|---|---|---|---|---|
| test_in sub | 256x97 | 0.0420-0.0437 | 0.0281-0.0312 | 31.3-36.1 | — |
| test_in super | 256x97 | 0.0575-0.0613 | 0.0431-0.0451 | 33.8-40.4 | 1-2 of 27 |
| test_ext super | 256x97 | 0.2318-0.2554 | 0.1733-0.2425 | 168.9-180.6 | 0 of 21 |
| test_in sub | 128x49 | 0.0101-0.0103 | 0.0080-0.0095 | 7.5-11.2 | — |
| test_in super | 128x49 | 0.0374-0.0398 | 0.0277-0.0343 | 32.5-46.6 | 0-1 of 27 |
| test_ext super | 128x49 | 0.2320-0.2482 | 0.1804-0.2415 | 168.8-186.0 | 0 of 21 |

The near-field spread is a few per cent of its value; the extrapolation CL maximum varies by 40 %
between seeds, so a single-seed extrapolation CL figure is not reliable to better than that.

**Loss A against loss B (n = 896, seed 0).** Loss B (A plus the wall row weighted 10) is **worse than
loss A on the near field, CL and CD in every set and regime at both grids**, and most at the
supercritical lift: test_in
supercritical CL max 0.2489 against 0.0431 (256x97) and 0.2633 against 0.0314 (128x49), CD max 256
against 40 counts; near field 0.1660 against 0.0575 (supercritical), 0.0694 against 0.0423
(subcritical) and 0.4506 against 0.2318 (test_ext) at 256x97. Its wall-Cp RMS is marginally better
in range subcritical (0.0337 against 0.0358 at 256x97, 0.0333 against 0.0334 at 128x49) and worse
in every other set and regime. Loss B stopped early by
patience at epoch 81 (S2 entry); the comparison is of the runs as made, not of converged runs.

**Shocks** (scores.md, "Shocks"; 27 true shocks on the 24 test_in supercritical cases, 21 on the 14
test_ext cases; position error in chords; "overshoot" is the largest |ΔCp| within 0.05 chord of the
true shock):

| set | method | missed | median / max \|Δx\| | median / max overshoot |
|---|---|---|---|---|
| test_in | **floor** | **0** | **0 / 0.0186** | **0.19 / 0.54** |
| test_in | FNO 256x97 (s0) | 2 | 0 / 0.0577 | 0.42 / 1.05 |
| test_in | FNO 128x49 (s0) | 1 | 0 / 0.0692 | 0.29 / 0.92 |
| test_in | POD | 0 | 0.0610 / 0.1348 | 0.65 / 1.34 |
| test_in | FNO loss B 256x97 | 2 | 0.0123 / 0.0682 | 0.59 / 1.29 |
| test_in | PINN pure (3 cases, 4 shocks) | 1 | 0.0459 / 0.1812 | 0.74 / 1.67 |
| test_in | PINN gradnorm (1 case, 1 shock) | 1 | — | 2.33 |
| test_ext | **floor** | **0** | **0.0078 / 0.0122** | **0.69 / 1.00** |
| test_ext | FNO 256x97 (s0) | 0 | 0.0419 / 0.2131 | 1.13 / 1.58 |
| test_ext | POD | 0 | 0.1059 / 0.2499 | 1.28 / 1.60 |

The FNO places in-range shocks at the floor's median accuracy (0 chords) but misses 1-2 of 27 and
smears them (overshoot median twice the floor's); out of range its position errors reach 0.21 chord.
POD never misses but places them 0.06-0.11 chord off at the median. At n = 64 the FNO misses up to 3
of 27 in range. The PINN misses one of its four shocks, and the gradnorm run misses its only one.

**Physics checks on SU2 itself** (the truth row, `su2-512x193`, sampled onto the 256x97 nodes; and
the floor):
- Entropy, ideally zero in subcritical inviscid flow: truth near-field maximum median 0.0197 (max
  0.0489) over the 37 subcritical cases, the floor 0.0328 (0.0645) — **SU2 generates spurious entropy,
  and more of it on the coarser grid**. Its area-weighted mean is 3.9e-7 (truth) and 1.7e-6 (floor),
  and non-negative on every case at both grids. Supercritical maxima, which include real shock
  entropy: 0.042 median in range and 0.133 out of range (truth).
- Total enthalpy |H/H∞ − 1|: at most 1.3e-6 (subcritical) and 1.2e-5 (supercritical), both grids.
- Wall-normal velocity RMS: 6.4e-9 to 9.8e-9 for the floor on its own grid; 2.8e-5 to 3.7e-5 median
  (5.2e-3 worst, supercritical in range) for the truth, which is evaluated with the 256x97 grid's
  wall normals rather than its own.
- Net mass flux through the closed lines j = 8, 24, 48, 72: truth medians 1.6e-8 to 1.2e-4, worst
  7.2e-4; floor worst 1.3e-3 (test_ext, j = 48).
- No non-positive density or pressure anywhere, either grid.
These are the scale against which the contestants' checks below are read.

### Failures, stated as results

- **Non-positive density or pressure in the near field (test_ext):** POD on 2, 5, 5, 6 and **7 of 14**
  cases at n = 64, 128, 256, 512, 896 — it gets worse with more data; the FNO on 4-5 of 14 at n = 64
  and 128, and on none at n ≥ 256. Where these occur the enthalpy deviation explodes (POD n = 512
  worst 5,770; FNO n = 64 worst 543 at 256x97). No method has one in range.
- **Entropy decrease:** the subcritical near-field mean entropy is **below free stream** on 9-24 of 37
  cases for the FNO (9 for the n = 896 seed-0 run; 16 and 19 for seeds 1 and 2) and 10-23 of 37 for
  POD, against 0 of 37 for SU2 at either grid. Command (counts per tag; tags with no case are absent):
  `python -c "import json,collections as c; R=json.load(open('results/s4/scores.json'))['rows']; d=c.Counter(r['tag'] for r in R if r['regime']=='subcritical' and r['grid']=='256x97' and (r['score'].get('physics') or {}).get('entropy_mean',0)<0); print(sorted(d.items()))"`. That is an unphysical entropy
  decrease somewhere in the field, but it is **not an expansion-shock test**: the scorer has no
  check for an entropy decrease across a compression, so whether any surrogate predicts an expansion
  shock is **not measured**. For the potential PINN entropy is identically constant, so an expansion
  shock (which isentropic potential flow admits) could not be detected by any entropy check at all.
- **Missed shocks:** FNO up to 3 of 27 in range (n = 64), 1-2 at n = 896; PINN 1 of 4 (`pure`) and 1
  of 1 (`gradnorm`); POD and the floor none.
- **Wrong lift from the wrong circulation:** the PINN's CL is 64-101 % low on five of six `pure` runs
  (table above).
- **PINN wall tangency and mass**, which the potential formulation enforces only softly: wall-normal
  RMS 0.011 median (subcritical) and 0.020 (supercritical) against SU2's 1e-8; mass-flux medians at
  j = 48 and 72 of 0.019-0.069 (`pure`), one to two orders above the FNO at n = 896 (4e-4 to 7e-3
  at 256x97) and three to five above SU2's truth.
- **A PINN run that did not converge:** `gradnorm` on test_in-0004 ended with residual 9.11 after
  L-BFGS made it about 380x worse than Adam left it (S3 entry), and scores CL −0.33 against 1.01,
  near field 1.27. The six `pure` runs and `nokutta` ended with residual losses 1.1e-3 to 2.4e-2; no
  convergence threshold was fixed for the PINN in advance, so they are reported, not judged.
- **FNO ringing** (spec §8.4) has no metric in the scorer and is not measured; the showcase plots
  (`results/s4/showcase_*.png`) are the only view of it.
- **Not measured at all:** the viscous PINN (dropped), and the Vassberg & Jameson comparison (still
  open from S0).
