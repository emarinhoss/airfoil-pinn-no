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
