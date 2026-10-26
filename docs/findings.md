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
  remembered value is not used as if it had been read. The comparison stays open. *(2026-10-07: the
  paper has been read and the comparison is made in the 2026-10-07 entry: the 512x193 anchor CL is
  6.4-8.5 % below the paper's grid-converged range, 0.3482-0.3562.)*

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
  *Correction (2026-10-06, second final review):* the paths cited above now hold the potential
  formulation's warm-up PASS (2026-10-05; see the 2026-10-06 S3a entry at the end). This Euler
  record is `results/s3/warmup_euler_2026-10-02.json` and `results/s3/warmup_euler_2026-10-02.log`
  (exit 1, `max_cp_error` 3.937, `wall_s` 27,550), with its field in
  `runs/warmup_q_euler_2026-10-02.npy` (renamed by the controller before the potential run). The
  command that runs this formulation is now `python scripts/s3_warmup.py --formulation euler`; it
  writes `warmup_euler.json` and `runs/warmup_q_euler.npy`, and refuses to overwrite either.
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
  is the formulation, not a parameter. *(Qualified 2026-10-07, after reading Wassing, Langer &
  Bekemeyer: they train an Euler PINN with a strong-form residual and a soft far field to within
  1 % of a finite-volume reference on transonic NACA 0012 cases, so this holds for the configuration
  and the five probes tested here, not for Euler PINNs in general; the differences are listed in
  the 2026-10-07 entry, and which of them matters is not known.)* For the full-potential equation in subcritical flow the
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

Width separates the configurations (0.031 at 32 against 0.022-0.023 at 48). Within a width the mode
counts spread the best validation loss by 0.0007 at width 32 (0.03083-0.03154) and 0.0013 at width 48
(0.02214-0.02345); the seed-to-seed spread of the final runs below is 0.00097 (0.011775-0.012744), but
at 300 epochs, and no seed spread was measured at the tuning's 100 epochs. So the mode count moves the
loss by about as much as a change of seed might, and the choice of `tune-6` over `tune-5` (0.5 %
apart) is not a strong one. *(Corrected 2026-10-06, second final review: this said the mode counts
"differ by less than the seed-to-seed spread", which is false at width 48.)* Command:
`python -c "import json; R=json.load(open('results/s2/tuning.json'))['rows']; F=json.load(open('results/s2/runs.json')); [print(w, min(v:=[r['best_val'] for r in R if r['tag'] in t]), max(v), max(v)-min(v)) for w,t in ((32,('tune-0','tune-1','tune-2','tune-3')),(48,('tune-4','tune-5','tune-6','tune-7')))]; s=[r['best_val'] for r in F if r['tag'] in ('fno-n896-s0','fno-n896-s1','fno-n896-s2')]; print('seeds', min(s), max(s), max(s)-min(s))"`.

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

*(Amended 2026-10-26: these transfers are at the design's padding rule of the time, 17 rows at
256 x 97. Under the amended rule (16 rows) the final runs' transfers are 0.0207 / 0.0206 / 0.0215,
0.0243 (loss B) and 0.0255 / 0.0271 / 0.0268 / 0.0254, and `runs.json` keeps these as
`transfer_val_rel_l2_rows17`. The tuning runs' stay at 17 rows; see the 2026-10-26 entry on the amended padding rule.)*

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
corner whatever its input features, as the true corner flow is. Of the three S3 shards started on
the §7.8 code (`results/s3/shard*_stopped_te_singularity.log`), two trained and thrashed and were
stopped: on shard 0 (base variant, test_in-0022) the residual stayed between 67 and 995 and the wall
loss between 4.7 and 122 from Adam step 500 to step 12,000, the last logged (residual 98.0); on
shard 2 (test_in-0043) between 26 and 472 and 4.2 and 386 to step 8,500 (residual 165). Shard 1 never
trained: its preflight projected 1,202.8 h against a 1,000 h budget and it refused, exit 3. Command:
`python -c "import ast; [print(f, d['step'], round(d['residual'],1), round(d['wall'],2)) for f in ('shard0','shard2') for l in open(f'results/s3/{f}_stopped_te_singularity.log') if l.startswith('adam') for d in [ast.literal_eval(l.split(': ',1)[1])]]"`. *(Corrected 2026-10-06, second final review: this said "residual 150-1000, wall loss
5-110 after 11,000 Adam steps" for all three shards.)*

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

**Evidence probe on the production code** (`python scripts/probes/probe_task26.py`, committed
2026-10-06 from the scratch copy it ran from, on the working tree that became commit 8023caf; its
output, from which this table is read, is `results/s3/probe_task26.log`): first 3,000 Adam steps of
the S3 `pure` run on test_in-0022 (`PinnConfig()` defaults, the production lr schedule, resampling
and loss, float32 on the Apple GPU at 0.19 s per step; wall Cp of the CPU float64 evaluation against SU2's 512 × 193 truth
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

## 2026-10-06 — stage S1: the data campaign (run 2026-10-01 to 2026-10-02)

*(Corrected 2026-10-06, second final review: the heading said "to 2026-10-04". The first and last
`result.json` were written 2026-10-01 22:37:51 and 2026-10-02 20:52:41 (local time):
`python -c "import glob,os,datetime as d; T=sorted(os.path.getmtime(p) for p in glob.glob('runs/su2/*/*/result.json')); print(d.datetime.fromtimestamp(T[0]), d.datetime.fromtimestamp(T[-1]))"`.)*

Commands (`runs/run_s1.sh`, train pass then fine pass, both exit 0; logs `results/s1/{train,fine}.log`):
`SU2_BIN_DIR=$HOME/theLab/bwb-tanker-scratch/bin python scripts/s1_campaign.py train` and the same with `fine`.
Summaries: `results/s1/campaign_{train,fine}.json`. Dataset: `python scripts/build_dataset.py`
(`results/s1/build.log`, `failures_{256x97,512x193}.json`, `showcase.json`; HDF5 under `data/hdf5/`).

| Pass | Level | Requested | Converged | Failed | Summed CPU | Summed wall |
|---|---|---|---|---|---|---|
| train | 256x97 | 1,104 | 1,084 | 20 (19 residual, 1 CL) | 69.0 h | 79.1 h (sum over workers) |
| fine | 512x193 | 80 | 75 | 5 (4 residual, 1 CL) | 42.3 h | 44.1 h (sum over workers) |

Per case, medians over all finished cases (`runs/su2/*/*/result.json`): 256x97 took 2,057 iterations,
227.5 s wall and 198.6 s CPU; 512x193 took 3,976.5 iterations, 1,554 s wall and 1,501.7 s CPU. Over
the converged cases only they are 2,050 iterations, 225.9 s and 197.9 s (1,084 cases) and 3,720
iterations, 1,493.8 s and 1,432.3 s (75 cases): the failed cases, which run longer, raise the
fine-grid medians by 7 %, 4 % and 5 % and the 256x97 ones by under 1 %. The 256x97 pass covers all four splits (train 896, val 128,
test_in 64, test_ext 16); the 512x193 pass covers the two test splits only (64 and 16). Command for
the case counts, the summed hours of the table and both sets of medians:
`python -c "import json,glob,statistics as st; [print(lv, len(R:=[json.load(open(p)) for p in glob.glob(f'runs/su2/{lv}/*/result.json')]), 'cpu_h %.1f wall_h %.1f'%(sum(r['cpu_s'] for r in R)/3600, sum(r['wall_s'] for r in R)/3600), 'all: it %s wall %.1f cpu %.1f'%(st.median(r['iterations'] for r in R), st.median(r['wall_s'] for r in R), st.median(r['cpu_s'] for r in R)), 'converged(%d): it %s wall %.1f cpu %.1f'%(len(C:=[r for r in R if r['converged']]), st.median(r['iterations'] for r in C), st.median(r['wall_s'] for r in C), st.median(r['cpu_s'] for r in C))) for lv in ('256x97','512x193')]"`
(the converged and failed counts per pass are in `results/s1/campaign_{train,fine}.json`).

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
shock-dominated cases on which the residual stalls while CL and CD are still moving, i.e. unsteady and
not converged. No force or residual history was examined for periodicity, so whether the solver
limit-cycles there is not measured. *(Corrected 2026-10-06, second final review: this said "where the
Euler solver limit-cycles", a mechanism the data does not show.)* The 256x97 failures of the test splits
recur at 512x193 (test_in-0024, test_in-0032, test_ext-0014); the fine grid adds test_in-0048 and
test_ext-0008. Failed cases are excluded from the HDF5 files, not patched.

**Dataset (HDF5).** `256x97.h5`: train 881, val 126, test_in 62, test_ext 15 (1,084 cases).
`512x193.h5`: test_in 61, test_ext 14 (75 cases); command:
`python -c "import h5py; [print(f, {k: len(h[k]['case_id']) for k in h}) for f in ('data/hdf5/256x97.h5','data/hdf5/512x193.h5') for h in [h5py.File(f)]]"`. The showcase takes the 61 test_in cases that
survive at both levels, sorted by fine-grid maximum surface Mach, index round(p(n-1)):
test_in-0022 (0.512, subcritical), test_in-0021 (0.684, subcritical), test_in-0043 (0.847,
subcritical), test_in-0004 (1.206, supercritical), test_in-0031 (1.377, supercritical),
test_in-0040 (1.610, supercritical). Three subcritical, three supercritical (`results/s1/showcase.json`).

**Machine sleeps.** The machine entered sleep repeatedly during the campaign; `pmset` shows "Entering
Sleep state due to 'Dark Wake Thermal Emergency'" (a thermal sleep; pmset does not show whether the lid
was closed, and an earlier "(lid closed / dark wake)" here was the ledger's guess), at least on 2026-10-02 and
again twice before 05:36 on the same night. Throughput fell from about 120 to about 19 cases per hour
overnight while per-case wall time was unchanged (median 197 s at that point), so the sleeps paused the
campaign rather than slowing the solver. The mitigation was `caffeinate -i -s -m -w <run_s1 pid>`,
which keeps the machine from idle or system sleep; awake throughput was then about 90 cases per hour.
It cannot override a closed lid or a thermal emergency (controller's ledger): sleeps recurred despite it
after S1 ended, on 2026-10-05 and 2026-10-06 (controller's ledger).
The campaign is resumable, so the pauses cost time only.
*(Commands added 2026-10-06, second final review.)* The sleeps:
`pmset -g log | grep "Entering Sleep" | awk '$1"T"$2 >= "2026-10-01T22:35" && $1"T"$2 <= "2026-10-02T20:53"'`
(read 2026-10-06; `pmset`'s log rolls over, so it may not reproduce later) lists 24 sleeps in the
campaign's window, from 2026-10-01 23:33 to 2026-10-02 05:21: 17 'Dark Wake Thermal Emergency', 4
'Maintenance Sleep', 2 'Sleep Service Back to Sleep' and 1 'Idle Sleep', and none after 05:21. The
throughput, as 256x97 cases finished per clock hour:
`python -c "import glob,os,collections as c,datetime as d; h=c.Counter(d.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H') for p in glob.glob('runs/su2/256x97/*/result.json')); [print(k,v) for k,v in sorted(h.items())]"`
gives 52 in the first 22 minutes (about 140 per hour), 78 in the 23:00 hour, 6-8 per hour from 00:00 to
04:59 (19 per hour averaged over 23:00-04:59, the figure above), then 87-112 per hour from 06:00 to
13:59 (the "about 90" above was a ledger reading taken earlier in that recovery). The median wall time
of the 166 cases finished before 05:00 is 195.9 s ("197 s" above) against 232.3 s for the rest:
`python -c "import glob,os,json,statistics as st,datetime as d; R=[(d.datetime.fromtimestamp(os.path.getmtime(p)), json.load(open(p))['wall_s']) for p in glob.glob('runs/su2/256x97/*/result.json')]; c=d.datetime(2026,10,2,5,0); print(len([w for t,w in R if t<c]), st.median(w for t,w in R if t<c), st.median(w for t,w in R if t>=c))"`.

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
the two shards ran concurrently). Command for every row:
`python -c "import json,glob; [print(p[-14:-5], k, round(r['wall_s']/3600,2), round(r['cpu_s']/3600,2), *('%.3g'%r['final'][x] for x in ('residual','wall','kutta','total'))) for p in sorted(glob.glob('results/s3/pinn_runs_shard*of2.json')) for k,r in json.load(open(p)).items()]"`. The learned Gamma is NOT recorded in the run records or logs; it
is recovered from the saved predictions in S4, not stated here. The circulation term's loss is 0.0 in
every run (as recorded). `pure` and `nokutta` use fixed weights 10 / 10; `nokutta` sets the Kutta
loss to exactly 0.

**Failure-mode result: gradnorm's wall weight left the equation unenforced.** *(Rewritten 2026-10-06,
second final review: the first version attributed the failure to the Kutta weight; the
decomposition in the 2026-10-06 corrections entry below shows the wall term carried the total.)*
On test_in-0004 the gradient-norm balancing drove `w_kutta` from 1.06e4 at Adam step 0 (fixed runs:
10) to 1.9e6 by step 1,500, a logged peak of 7.0e10 at step 16,500 (the history is sampled every 500
steps, so the true peak is not known) and 5.8e9 at step 29,999, while the Kutta loss fell to 7e-10
or below from step 7,000 (1.4e-14 at step 29,999, exactly 0 at steps 8,000 and 9,500);
the weighted Kutta term was therefore small, at most 3.9 % of the total from step 500 on (616 of
15,880 at step 6,500) and 8e-5 of 719.5 at step 29,999. The total was the wall term: `w_wall`, at
692 to 1.3e5 from step 500 on, times a wall loss that stayed at 0.03-0.25 (0.093 at the end of Adam,
against 0.00285 for fixed-weight `pure` on the same case, about 33 times larger), was at least 96 %
of the weighted total at every logged step from 500 on, and 719.4 of 719.5 at step 29,999 (`w_wall`
7.75e3). The weighted total ranged from 67 (step 25,000) to 15,880 (step 6,500). The Adam phase ended
with residual 0.024 (`pure`: 0.020); L-BFGS, minimising an objective that was 99.99 % wall term,
cut the wall loss tenfold (0.093 to 0.0094) and the wall term from 719 to 73, and in exchange made
the PDE residual about 380 times worse (0.024 to 9.11; total 82.2). The result is a run whose
equation is effectively unenforced: compare `pure` on the same case at 2.98e-3. Reported as a result
of the method; nothing was tuned. Commands: `grep -n "w_kutta" results/s3/shard0.log` (the log's
`final:` line is the L-BFGS end state) and the decomposition command in the corrections entry.

**nokutta** (test_in-0022, `kutta=False`): final residual 1.82e-3, wall 7.8e-5, total 2.61e-3 — the
same case under `pure` ended at 2.25e-3 / 1.3e-4 / 3.58e-3. The PDE and wall losses alone converge
equally well, so the loss does not reveal whether the circulation is pinned; that is for the S4 lift
comparison.

**Attempts that did not run to completion (kept under explicit names in `results/s3`).**
1. `*_stopped_te_singularity.log` (3 shards, §7.8 code): shards 0 and 2 trained on the
   trailing-edge singularity with the residual at 67-995 (to Adam step 12,000) and 26-472 (to step
   8,500) and were stopped; shard 1 refused at its preflight (1,202.8 h against a 1,000 h budget,
   exit 3) and never trained. Fixed as in the 2026-10-05 entry (commit 8023caf). *(Corrected
   2026-10-06, second final review: this said "residual 150-1000 after 11,000 Adam steps" for all
   three.)*
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
of 2026-10-05; they are not part of S3 and are selectable with `--viscous`. *(Corrected 2026-10-06,
second final review: this cited "projected 119-271 h for one viscous run", which are contended
measurements; the uncontended projection is about a day per run. See the corrections entry below.)*

## 2026-10-06 — stage S4: head-to-head results, and the three predictions checked

Command: `python scripts/s4_score.py` (commit e06ff06), which wrote `results/s4/scores.md`,
`scores.json`, `costs.json`, `break_even.png`, `data_efficiency.png` and six `showcase_*.png`.
*(2026-10-06, second final review: the same command at commit f9878ff re-wrote `costs.json`, now
with both cost bases per method and the PINN costed per phase, and `break_even.png`, with the FNO's
curve visible under POD's; every other output came out byte-identical. The cost figures below are
read from the re-written file through the keys named in each command.)* Every
figure below is read from those files; the per-case figures come from `scores.json`'s `rows`, e.g.
`python -c "import json; [print(r['tag'], r['case_id'], r['score']['cl_err'], r['score']['cl_true']) for r in json.load(open('results/s4/scores.json'))['rows'] if r['method']=='pinn']"`.

*(Amended 2026-10-26: this entry's figures for the FNO at 256x97 were made with the padding rule of
the time, 17 rows. On 2026-10-26 the rule became 16 rows (spec §6.3). S2's predictions were remade
from the same checkpoints and S4 was rerun, so `results/s4/` now holds the 16-row scores, and the
commands below reproduce those, not the figures quoted. To reproduce the 17-row figures, read
`git show 7b39c58:results/s4/scores.json`. Each affected figure is marked, and the new ones are in
the 2026-10-26 entry on the amended padding rule. The figures at 128x49, and those for SU2, POD and the PINN, the costs and the break-even, are
unchanged.)*

**Four things the brief for this stage predates, stated before any result.**

1. **The PINN is the full-potential formulation** (author's decision 2026-10-04), not the Euler PINN
   of spec §7.1-7.7, which converges to fluid at rest (the five refuted hypotheses and the decisive
   probe are in the 2026-10-04 entry). It is exact only for shock-free inviscid flow and is an
   isentropic approximation at shocks, so on supercritical cases it is solving a different equation
   from SU2.
2. **Its airfoil runs use the trailing-edge fixes of 2026-10-05** (quarter-chord vortex angle,
   0.02-chord residual exclusion, equal-speed Kutta condition), **and its lift is under-predicted**
   and the equal-speed Kutta term is met (~1e-6) at a circulation far below the Kutta value
   (2026-10-05 entry). *(Corrected in place 2026-10-06, controller, from "because none of the Kutta
   formulations tried constrains the circulation", which is true of the first, (∂φ/∂ξ)² form and false
   of the equal-speed one: on an exact incompressible flat-plate (Joukowski) flow at α 2.19°, the
   equal-speed quantity |u|²(upper) − |u|²(lower) at 0.03 chord upstream of the TE is −0.421 at Γ = 0,
   −0.277 at Γ = 0.0387 (the `pure`/0022 value) and +0.027 at the Kutta value Γ = π sin α = 0.120, so a
   wrong-Γ potential flow pays ~100x the penalty of the right one. A network meeting the term at the
   wrong Γ is therefore not a potential-flow solution near the TE; whether the network's
   representation, the 0.02-chord exclusion or both is the cause is not isolated here (the 0.005-chord
   probes, `runs/probe_eqk_excl005_d*.log`, gave Γ ≈ 0.011 at 4,000 steps, no better, but they lost the
   suction peak altogether, Cp_min ≈ −0.07 against −0.942, so they are a weak comparator). Found by the second final
   review's verifier; command:
   `python -c "import numpy as np; al=np.radians(2.19); a=0.25; th=np.arccos(0.94); f=lambda G,t: abs((np.exp(-1j*al)-a*a*np.exp(1j*al)/(a*np.exp(1j*t))**2+1j*G/(2*np.pi*a*np.exp(1j*t)))/(1-np.exp(-2j*t)))**2; [print(G, f(G,th)-f(G,-th)) for G in (0, 0.0387, np.pi*np.sin(al))]"`.)* That
   entry's "about 8x" on test_in-0022 was Γ ≈ 0.02 against CL/2 ≈ 0.16 at 3,000 Adam steps; at the end
   of the run S4 measures CL 0.0732 against SU2's 0.3192, **4.4x too small**. Both are measurements,
   at different points of the run.
3. **The four viscous runs were dropped** (author's decision 2026-10-05). Prediction 3 (no Kutta) is
   still testable; the spec's comparison of the pure and artificially-viscous PINN at shocks (§7.4) is
   not made, so nothing here says whether viscosity fixes the PINN's shocks.
4. **The costs mix hardware.** SU2 is CPU time on this Mac (one process per case, measured with 8
   concurrent workers). The FNO's training is GPU wall time on this Mac's Apple GPU (MPS). The PINN is
   wall time of Adam on MPS plus L-BFGS on the CPU, two runs at once. POD is CPU time. Every
   break-even or ratio below inherits that mix. *(2026-10-06: the PINN's L-BFGS phase is CPU work
   and is charged here at wall time; costed per phase, as the scorer's own rule implies, Prediction 1
   comes out differently — see the corrections entry at the end.)*

**Test sets.** test_in (62 cases at 256x97, 61 scored: 37 subcritical, 24 supercritical) and
test_ext (14 scored, **all supercritical**: the extrapolation set has no subcritical case, so no
subcritical-extrapolation figure exists). Truth is SU2 at 512x193 sampled every second node; **the
floor** is SU2's own 256x97 solution scored against that truth. No contestant has a missing or
non-finite prediction on any case (`n_missing` and `n_non-finite` are 0 in every row of scores.md).

### The three predictions

**Prediction 1 — "For a single new case, SU2 beats the PINN on cost by one to two orders of
magnitude." HELD ON WALL TIME ONLY; NOT HELD on CPU time or per phase** (verdict restated in place
2026-10-06, controller, after the second final review: the bold "HELD." that stood here rested on
one cost basis; see the note below and the corrections entry at the end). `costs.json`, key `per_wall`
(written as `per` by the first S4 run): PINN per case 20,201.3 s (mean wall time of the six `pure` runs,
5.61 h) against SU2 per case 209.87 s (median CPU time of the 256x97 test cases): **96.3x**, 1.98
orders. On CPU time instead (mean 36,464 s per `pure` run, from `results/s3/pinn_runs_shard*of2.json`)
it is 174x, 2.24 orders, just outside the band; the first S4 run held the prediction on its own basis
(superseded: that basis charges the CPU L-BFGS phase at wall time), and
the mixed hardware of point 4 above is what separates the two figures. The ratio is for an answer of
very different quality: the PINN's near-field error is 0.52-0.63 against the floor's 0.008-0.024
(table below). Command (on the re-written `costs.json`, where the PINN's wall time is `per_wall`):
`python -c "import json; c=json.load(open('results/s4/costs.json'))['costs']; print(c['pinn']['per_wall']/c['su2']['per_cpu'])"`.
*(Restated 2026-10-06, second final review: 96.3x divides the PINN's wall time by SU2's CPU time.
On one basis for both it is 88.5x on wall time (HELD) and 174x on CPU time; per phase, the scorer's
own rule, it is 174-220x, 2.24-2.34 orders, outside the predicted band: **NOT HELD** on that basis,
SU2 winning by more than predicted. See the corrections entry at the end.)*

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
these figures, not a measured curve — but at 7.6x the floor rather than 3.4x. *(Amended
2026-10-26, 16 padding rows: the FNO column is 0.0918, 0.0634, 0.0460, 0.0310 and 0.0256, 6.4x to
1.8x the floor; the clause still does not fire, and the break-even is unchanged.)* Command for the table (prints the mean per tag and the case count):
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

*(Amended 2026-10-26, 16 padding rows: the FNO 256x97 rows are 0.0169 / 0.0181 / 41.6 / 0.0432
(subcritical), 0.0391 / 0.0403 / 64.6 / 0.0491 (supercritical) and 0.2238 / 0.1835 / 203.5 / 0.1933
(extrapolation); see the 2026-10-26 entry on the amended padding rule.)*

What this says:
- **In range and subcritical the FNO at its own 128x49 resolution is near the floor** (near field
  1.3x, CL 1.3x, CD 1.1x), **but evaluated at 256x97 it is 5.3x the floor** and POD beats it there on the near field, CL and CD
  (not on wall-Cp RMS, 0.0371 against 0.0358). The resolution transfer costs the FNO more than any other single factor in range.
  *(Amended 2026-10-26: at 16 padding rows it is 2.1x the floor at 256x97, level with POD on the
  near field; POD's lift and drag errors are smaller.)*
- **Supercritical, in range, the FNO beats POD by 2.5x** on the near field and 3x on CD at 256x97,
  and is 2.4x the floor. *(Amended 2026-10-26: at 16 padding rows, 3.7x better than POD on the
  near field and 1.9x on CD, and 1.6x the floor.)*
- **Extrapolation is where every surrogate fails**: the FNO is 3.6x the floor on the near field and
  its CD error reaches 181 counts (5.6x the floor's 32); POD is 7.9x and 530 counts. The FNO's CD
  errors on test_ext are biased, not scattered: median −78 counts at 256x97. *(Amended
  2026-10-26: at 16 padding rows, 3.4x the floor, CD error up to 204 counts, median −102.)*
- **The PINN is not competitive on any column**: 0.52-0.63 near field, 65x and 26x the floor
  (0.5155 / 0.007953 = 64.8 and 0.6273 / 0.02413 = 26.0; corrected 2026-10-06 from "64x"; command:
  `python -c "import json; G=json.load(open('results/s4/scores.json'))['groups']; f=lambda m,v: {(x['split'],x['regime']): x['near_field_rel_l2_mean'] for x in G if x['method']==m and x['variant']==v and x['grid']=='256x97' and x['split']=='test_in'}; fl,pi=f('su2-256x97','floor'),f('pinn','pure'); [print(k, pi[k], fl[k], pi[k]/fl[k]) for k in pi]"`).

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
*(Amended 2026-10-26: the 256x97 rows are at 17 padding rows. At 16, in-range shocks missed are 0
for every seed, the subcritical near-field spread is 12 % and the extrapolation CL maximum still
varies by 40 %; see the 2026-10-26 entry on the amended padding rule.)*

**Loss A against loss B (n = 896, seed 0).** Loss B (A plus the wall row weighted 10) is **worse than
loss A on the near field and CD in every set and regime at both grids, and on CL in every one except
in-range subcritical at 256x97**, where its CL maximum is 0.0239 against loss A's 0.0312, 23 % better
*(corrected 2026-10-06, second final review: this
said worse on CL "in every set and regime"; command:
`python -c "import json; G=json.load(open('results/s4/scores.json'))['groups']; f=lambda v,g: {(x['split'],x['regime']): x['cl_err_absmax'] for x in G if x['method']=='fno' and x['n_train']==896 and x['seed']==0 and x['variant']==v and x['grid']==g}; [print(g, k, a, f('lossB',g)[k]) for g in ('256x97','128x49') for k,a in f('',g).items()]"`)*.
It is worse most at the supercritical lift: test_in
supercritical CL max 0.2489 against 0.0431 (256x97) and 0.2633 against 0.0314 (128x49), CD max 256
against 40 counts; near field 0.1660 against 0.0575 (supercritical), 0.0694 against 0.0423
(subcritical) and 0.4506 against 0.2318 (test_ext) at 256x97. Its wall-Cp RMS is marginally better
in range subcritical (0.0337 against 0.0358 at 256x97, 0.0333 against 0.0334 at 128x49) and worse
in every other set and regime. Loss B stopped early by
patience at epoch 81 (S2 entry); the comparison is of the runs as made, not of converged runs.
*(Amended 2026-10-26: at 16 padding rows, loss B is worse at 256x97 on the near field and CL in
every set and regime, and on CD in every one except in-range subcritical, CD max 37.2 against 41.6
counts.)*

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
*(Amended 2026-10-26: at 16 padding rows the FNO at 256x97 misses no in-range shock at n = 896
(any seed), overshoot 0.36 / 0.85; loss B misses 1; n = 64 misses 1.)*

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
  worst 5,770; FNO n = 64 worst 543 at 256x97). No method has one in range. *(Amended 2026-10-26:
  at 16 padding rows the FNO has 5 and 4 such cases at n = 64 and 128, worst deviation 702.)*
- **Entropy decrease:** the subcritical near-field mean entropy is **below free stream** on 9-24 of 37
  cases for the FNO (9 for the n = 896 seed-0 run; 16 and 19 for seeds 1 and 2) and 10-23 of 37 for
  POD, against 0 of 37 for SU2 at either grid. *(Amended 2026-10-26: at 16 padding rows the FNO's
  counts are 8-21, and 8, 14 and 20 for the three seeds.)* Command (counts per tag; tags with no case are absent):
  `python -c "import json,collections as c; R=json.load(open('results/s4/scores.json'))['rows']; d=c.Counter(r['tag'] for r in R if r['regime']=='subcritical' and r['grid']=='256x97' and (r['score'].get('physics') or {}).get('entropy_mean',0)<0); print(sorted(d.items()))"`. That is an unphysical entropy
  decrease somewhere in the field, but it is **not an expansion-shock test**: the scorer has no
  check for an entropy decrease across a compression, so whether any surrogate predicts an expansion
  shock is **not measured**. For the potential PINN entropy is identically constant, so an expansion
  shock (which isentropic potential flow admits) could not be detected by any entropy check at all.
- **Missed shocks:** FNO up to 3 of 27 in range (n = 64), 1-2 at n = 896 *(amended 2026-10-26: at
  16 padding rows, 1 at n = 64 and none at n = 896)*; PINN 1 of 4 (`pure`) and 1
  of 1 (`gradnorm`); POD and the floor none.
- **Wrong lift from the wrong circulation:** the PINN's CL is 64-101 % low on five of six `pure` runs
  (table above).
- **PINN wall tangency and mass**, which the potential formulation enforces only softly: wall-normal
  RMS 0.011 median (subcritical) and 0.020 (supercritical) against SU2's 1e-8; mass-flux medians at
  j = 48 and 72 of 0.019-0.069 (`pure`), one to two orders above the FNO at n = 896 (4e-4 to 7e-3
  at 256x97; 3e-4 to 5e-3 at 16 padding rows, amended 2026-10-26) and three to five above SU2's truth.
- **A PINN run that did not converge:** `gradnorm` on test_in-0004 ended with residual 9.11 after
  L-BFGS made it about 380x worse than Adam left it (S3 entry), and scores CL −0.33 against 1.01,
  near field 1.27. The six `pure` runs and `nokutta` ended with residual losses 1.1e-3 to 2.4e-2; no
  convergence threshold was fixed for the PINN in advance, so they are reported, not judged.
- **FNO ringing** (spec §8.4) has no metric in the scorer and is not measured; the showcase plots
  (`results/s4/showcase_*.png`) are the only view of it.
- **Not measured at all:** the viscous PINN (dropped), and the Vassberg & Jameson comparison (still
  open from S0). *(2026-10-07: the Vassberg & Jameson comparison is now made; see the 2026-10-07
  entry.)*

## 2026-10-06 — corrections from the second final review, and the S3a warm-up pass

Each correction below is also marked where the corrected text sits, in the entry it belongs to.

**S3a warm-up: the potential formulation PASSED** (the record the 2026-10-04 entry's paths now
hold). `results/s3/warmup.json`: `passed` true, max|ΔCp| 0.0272 against the 0.1 bound (the Euler
PINN: 3.94), CL −3.9e-4 and CD −1.0e-3 against 1e-2 (CM −9.7e-5), final residual 3.4e-6, wall loss
6.0e-9, total 3.5e-6; 28,628.7 s wall (7.95 h) and 130,246.6 s CPU (36.2 CPU-h); `WARMUP_EXIT=0`
at the end of `results/s3/warmup.log`. Commands:
`python -c "import json; print(json.load(open('results/s3/warmup.json')))"` and
`tail -1 results/s3/warmup.log`. It ran **Adam on the CPU**, not on the Apple GPU (which S2's FNO
training held), and L-BFGS on the CPU in float64, with `OMP_NUM_THREADS=8`, through this launcher
(`runs/run_s3_warmup_potential.sh`, not committed; `runs/` is ignored), whose process-local override
makes `device="auto"` resolve to the CPU:

```bash
#!/bin/bash
# S3a warm-up, potential formulation (Task 25), on the CPU: S2's FNO training holds the GPU and its wall time is a
# recorded cost. Process-local override: torch.backends.mps.is_available -> False (device 'auto' resolves to cpu).
cd /Users/esousa/theLab/airfoil-pinn-no
export PATH=/Users/esousa/miniforge3/envs/airfoil-ml/bin:$PATH
export OMP_NUM_THREADS=8
python -u -c "
import runpy, sys, torch
torch.backends.mps.is_available = lambda: False
sys.argv = ['s3_warmup.py', '--formulation', 'potential']
runpy.run_path('scripts/s3_warmup.py', run_name='__main__')
" > results/s3/warmup.log 2>&1
echo "WARMUP_EXIT=$?" >> results/s3/warmup.log
```

The code was commit **0893de4** (the Task 25 PINN code of 2954e1c; the record does not say so, the
dates do): the launcher was written 2026-10-04 23:07, after 0893de4 (23:00:03), and the record
finished 2026-10-05 11:57:31, while the next commit touching the PINN, 8023caf (the trailing-edge
fixes), is from 16:49:50 that day and the two between (781e16d, c834166) touched only S2's files and
this document (`git log --format='%h %ad %s' --date=iso --until='2026-10-05 11:57' -1`, then
`git diff --stat 0893de4 c834166`). 8023caf does not change the cylinder configuration (Γ fixed at 0,
Kutta off, no trailing edge to exclude: spec §7.9), but the warm-up was not re-run on it. The CPU was
shared with S2's FNO training for part of the run (controller's ledger), so its times are not an
uncontended cost. `scripts/s3_warmup.py` now records `adam_device`, `lbfgs_device`,
`omp_num_threads`, `argv`, `git_commit` and `git_dirty`, takes `--device`, and refuses to overwrite
an existing `warmup*.json` or `warmup_q*.npy`; the same hardware without the override is
`OMP_NUM_THREADS=8 python scripts/s3_warmup.py --device cpu --tag <name>`.

**The gradnorm run, decomposed** (correcting the S3 entry, which blamed the Kutta weight). The
weighted total is residual + `w_wall`·wall + `w_kutta`·kutta. Per logged Adam step and at the end of
L-BFGS (`results/s3/pinn_runs_shard0of2.json`, `gradnorm/test_in-0004`):

| step | residual | `w_wall` | wall | `w_wall`·wall | `w_kutta` | `w_kutta`·kutta | total |
|---|---|---|---|---|---|---|---|
| 0 | 9.0 | 1.44e3 | 1.70 | 2,458 | 1.06e4 | 2,400 | 4,867 |
| 1,500 | 2.09 | 4.73e4 | 0.0509 | 2,408 | 1.92e6 | 15.9 | 2,426 |
| 6,500 | 0.43 | 9.02e4 | 0.169 | 15,260 | 4.14e8 | 616 | 15,880 |
| 16,500 | 0.065 | 1.25e4 | 0.114 | 1,425 | 7.04e10 | 0.0040 | 1,425 |
| 25,000 | 0.0087 | 692 | 0.0968 | 67.0 | 7.84e8 | 4.5e-5 | 67.0 |
| 29,999 | 0.024 | 7.75e3 | 0.0928 | 719.4 | 5.76e9 | 8.2e-5 | 719.5 |
| L-BFGS end | 9.11 | 7.75e3 | 0.00943 | 73.1 | 5.76e9 | 1.1e-3 | 82.2 |

From step 500 on the wall term is at least 96.1 % of the total at every one of the 60 logged steps,
the Kutta term at most 3.9 % (step 6,500) and the residual at most 0.5 %. So the balanced `w_wall`
(692 to 1.3e5) held a wall loss of 0.03-0.25 that fixed-weight `pure` on the same case had brought
to 0.00285 by the end of Adam (gradnorm: 0.0928, 33x), and L-BFGS started on an objective that was
99.99 % wall term: it cut the wall loss tenfold (0.0928 to 0.00943) and the wall term from 719 to
73, and traded the residual for it, 0.024 to 9.11 (380x). The runaway `w_kutta` (logged peak 7.04e10
at step 16,500; the history is sampled every 500 steps) multiplied a Kutta loss of 7e-10 or less
from step 7,000 and moved the total little; the controller's in-run note of "2.6e10 at step 8000"
is the logged value at that step (2.58e10), taken while the run was in progress, not a peak. The
weighted total ran from 67 to 15,880 over the Adam phase from step 500, so "tens to tens of
thousands", not the "hundreds to thousands" first written. Command (prints every logged step):
`python -c "import json; r=json.load(open('results/s3/pinn_runs_shard0of2.json'))['gradnorm/test_in-0004']; [print(h['step'], '%.3g'%h['residual'], '%.3g'%h['w_wall'], '%.3g'%h['wall'], '%.4g'%(h['w_wall']*h['wall']), '%.3g'%h['w_kutta'], '%.3g'%(h['w_kutta']*h['kutta']), '%.4g'%h['total']) for h in r['history'][:-1]]; f,w=r['final'],r['weights']; print('lbfgs', '%.3g'%f['residual'], '%.3g'%w['wall'], '%.3g'%f['wall'], '%.4g'%(w['wall']*f['wall']), '%.3g'%w['kutta'], '%.3g'%(w['kutta']*f['kutta']), '%.4g'%f['total'])"`;
the shares: `python -c "import json; r=json.load(open('results/s3/pinn_runs_shard0of2.json'))['gradnorm/test_in-0004']; A=[h for h in r['history'] if h['step']!='lbfgs' and h['step']>=500]; print(len(A), min(h['w_wall']*h['wall']/h['total'] for h in A), max(h['w_kutta']*h['kutta']/h['total'] for h in A), max(h['residual']/h['total'] for h in A))"`;
`pure`'s wall loss at the end of Adam:
`python -c "import json; print([h for h in json.load(open('results/s3/pinn_runs_shard1of2.json'))['pure/test_in-0004']['history'] if h['step']!='lbfgs'][-1]['wall'])"`.

**What the viscous runs would have cost** (correcting the S3 entry's "projected 119-271 h for one
viscous run"). Those two figures are line 7 of `results/s3/shard0_stopped_te_singularity.log` and of
`results/s3/shard0_refused_budget60.log`, preflights taken with three shards timing at once, the
contention the 2026-10-05 entry and spec §7.9 call noise: across the six logs of those two attempts
one viscous run was projected at anywhere from 50.5 to 840.7 h
(`grep -h "visc.* h$" results/s3/shard*_refused_budget60.log results/s3/shard*_stopped_te_singularity.log`). The
uncontended preflight of the 12-run plan on the Task 25 code, `results/s3/preflight_12run_task25code.json`,
projects **23.3 h per `visc1e-3` run and 22.1 h for `visc3e-4`, 91.8 h for the four**, and 125.3 h
for the whole 12-run plan against the 48 h budget, which it refused (`within_budget` false). The
projection is a lower bound (one closure per L-BFGS iteration): the six `pure` runs took 1.04-1.35
times their uncontended 4.78 h (`results/s3/preflight.json`), so about a day to a day and a half per
viscous run, not five to eleven. Command:
`python -c "import json; d=json.load(open('results/s3/preflight_12run_task25code.json')); v=[r for r in d['runs'] if r['variant'].startswith('visc')]; [print(r) for r in v]; print(sum(r['hours'] for r in v), d['projected_total_hours'], d['budget_hours'], d['within_budget'])"`
and `awk 'FNR==7{print FILENAME": "$0}' results/s3/shard0_stopped_te_singularity.log results/s3/shard0_refused_budget60.log`.
The recorded reasons for the cut are the author's scope decision of 2026-10-05 ("Fix + non-viscous
runs", spec §7.9) and the machine: three concurrent shards thrash its 16 GB of memory, so S3 ran as
two shards, and on two shards the four viscous runs add about 46 h of wall time to a plan already
projected at 35.2 h serial.

**Prediction 1 on each cost basis** (correcting the S4 entry's "HELD", which rests on one basis).
Spec §8.2 asks for wall time and CPU time. `costs.json` now holds both for every method, and the
PINN's per-phase cost — its Adam phase on the GPU at wall time, its L-BFGS phase on the CPU at CPU
time, S4's own rule — is a bracket, because the S3 runs did not record the Adam phase's CPU time:
the upper bound counts all of a run's CPU time as L-BFGS time, and `per_lower` assumes the Adam
phase's host CPU time equalled its wall time. *(Qualified 2026-10-06, controller: `per_lower` is a
lower bound only if that host CPU time never exceeds the wall time, which a multithreaded host
(`OMP_NUM_THREADS=6`) does not guarantee; the measured fraction 0.346 sits inside the bracket, so no
verdict depends on it. `scripts/s4_score.py` and `costs.json` still call it a lower bound.)*

| basis | PINN per case, s | SU2 per case, s | ratio | orders | prediction (one to two) |
|---|---|---|---|---|---|
| wall / wall | 20,201.3 | 228.1 (median wall) | 88.5x | 1.95 | held |
| PINN wall / SU2 CPU (the S4 entry's) | 20,201.3 | 209.87 (median CPU) | 96.3x | 1.98 | held |
| CPU / CPU | 36,464.4 | 209.87 | 173.7x | 2.24 | not held |
| per phase, upper bound | 46,122.1 | 209.87 | 219.8x | 2.34 | not held |
| per phase, host fraction 1 (`per_lower`) | 36,464.4 | 209.87 | 173.7x | 2.24 | not held |
| per phase, measured host fraction 0.346 | 42,783.4 | 209.87 | 203.9x | 2.31 | not held |

Command: `python -c "import json,glob,math; c=json.load(open('results/s4/costs.json'))['costs']; p,s=c['pinn'],c['su2']; R={}; [R.update(json.load(open(f))) for f in glob.glob('results/s3/pinn_runs_shard*of2.json')]; aw=sum(r['adam_wall_s'] for k,r in R.items() if k.startswith('pure/'))/6; ph=p['per']-0.3457*aw; [print(n, round(a,1), round(b,2), round(a/b,1), round(math.log10(a/b),2)) for n,a,b in [('wall/wall',p['per_wall'],s['per_wall']),('mixed',p['per_wall'],s['per_cpu']),('cpu/cpu',p['per_cpu'],s['per_cpu']),('phase upper',p['per'],s['per_cpu']),('phase lower',p['per_lower'],s['per_cpu']),('phase f=0.346',ph,s['per_cpu'])]]"`.
The host fraction is a measurement taken now, not during S3: 1,000 Adam steps of the production
configuration on test_in-0022 on the Apple GPU, uncontended, `OMP_NUM_THREADS=6`, at commit f9878ff
took 190.6 s wall and 65.9 s CPU (0.346), command
`OMP_NUM_THREADS=6 python -c "from apno.data.sampling import case_specs; from apno.geometry.ogrid import TRAIN, Section; from apno.models.pinn import PinnCase; from apno.models.train_pinn import PinnConfig, train; s={c.case_id: c for c in case_specs('test_in',TRAIN)}['test_in-0022']; _,r=train(PinnCase(s.mach,s.alpha_deg,Section(s.t,s.m)), PinnConfig(adam_steps=1000, lbfgs_iters=1), log=lambda *a: None); print(r['adam_device'], r['adam_wall_s'], r['adam_cpu_s'], r['adam_cpu_s']/r['adam_wall_s'])"`;
any fraction from 0 to 1 gives 174-220x. The mean `pure` run spent 9,658 s of wall time in Adam and
10,544 s in L-BFGS, so charging the CPU L-BFGS phase at wall time, as the S4 entry's 96.3x does,
contradicts the rule the scorer states for itself. The wall figures are also inflated by the two
shards sharing the GPU and the CPU, by at most about 1.2x on the mean (above), which moves no row
across the band's edge. **Restated: Prediction 1 holds on wall time alone (88.5x, 1.95 orders) and
fails on CPU time and on the per-phase basis (174-220x, 2.24-2.34 orders): SU2 beats the PINN by
more than predicted on every basis that charges CPU work as CPU time.** The prediction text above is
unchanged; this is its outcome.

**The break-even plot** (`results/s4/break_even.png`) is redrawn: the FNO's curve, which POD's hid
exactly, is now wide and translucent under a dotted POD, and the PINN's line is its per-phase upper
bound. The FNO and POD break-even points are unchanged, 1,126 and 1,094 new cases (the S4 entry's
command, on the re-written `costs.json`).

**Corrected in place** (each marked there): the S2 tuning paragraph's mode-count-against-seed claim
(false at width 48); the S1 heading's dates (the campaign ran 2026-10-01 to 2026-10-02), its
"limit-cycles" (not measured), and commands for its medians (with converged-only medians), summed
hours, HDF5 counts, sleeps and throughput; the 2026-10-05 entry's and the S3 entry's
"residual 150-1000 after 11,000 Adam steps" (two shards trained to steps 12,000 and 8,500, and the
third refused at its preflight), and the 2026-10-05 probe's script and log, now committed as
`scripts/probes/probe_task26.py` and `results/s3/probe_task26.log`; the S3 entry's gradnorm
paragraph, its viscous-cost sentence and a command for its run table; the S4 entry's Prediction 1
command and verdict, "64x" (65x) and the loss-B sentence (loss B is better on CL in one cell: test_in
subcritical at 256x97, 0.0239 against 0.0312); and a pointer from the 2026-10-04 entry to the Euler
warm-up record's current paths.

## 2026-10-07 — the two open references read: Vassberg & Jameson (2010), and Wassing, Langer & Bekemeyer

Both papers were supplied by the author on 2026-10-07 and read in full: `references/in-pursuit-of-grid-convergence-for-two-dimensional-euler-solutions.pdf`
and `references/2408.17364.pdf` (kept outside git: `references/` is untracked). This closes the two
literature items of spec §10.

**Vassberg, J. C., and Jameson, A., "In Pursuit of Grid Convergence for Two-Dimensional Euler
Solutions," Journal of Aircraft 47(4):1152-1166, 2010, DOI 10.2514/1.46737.** What the paper computes:
- **The geometry is not ours.** It is the NACA 0012 equation with its standard x⁴ coefficient,
  −0.1015, extended in chord to the equation's root at x = 1.0089304115, so the trailing edge is
  sharp (included angle 16.15°), with a reference chord of 1. Our section closes the trailing edge at
  x = 1 with −0.1036.
- **The grids and solvers are different.** Karman-Trefftz O-meshes of aspect-ratio-one cells,
  32x32 to 4096x4096, far field about 150 chords away. Three solvers (FLO82 H-CUSP, with a point-vortex
  far-field correction; OVERFLOW 2.1t central, without one; CFL3D v6 flux splitting, with and
  without), converged to machine zero in 64-bit arithmetic.
- **The reference values** are Richardson continuum estimates from the three finest meshes.

At M 0.8, α 1.25° (Table 11, p. 1165) the four continuum estimates are:
- CL 0.356208937 (FLO82), 0.351662793 (OVERFLOW), 0.348226045 (CFL3D), 0.351596613 (CFL3D +
  vortex), so **0.3482-0.3562**, a spread of 0.0080;
- CD 0.022684938, 0.022453440, 0.022501430, 0.022674853, so 224.5-226.8 counts;
- Cm −0.038774022, −0.037946129, −0.037353559, −0.037838046.

The memory value in spec §2, "≈0.35", is inside the CL range. FLO82's own sequence (Table 3,
p. 1159) converges **from above**: CL 0.368980 at 256x256, 0.363748 at 512x512, 0.357142 at
4096x4096, observed order 1.151. The paper calls all three solvers first-order or worse on this
lifting transonic case.

Our anchor (S0, `results/s0/summary.json` "mesh"/"anchor", MGLEVEL 0, 100 chords) against that
range. Cm is compared in magnitude: our CM is SU2's CMz, positive nose-down, and the paper's Cm is
negative nose-down, so both describe the same nose-down moment.

| quantity | ours 256x97 | ours 512x193 | V&J continuum | 512x193 against it |
|---|---|---|---|---|
| CL | 0.317363 | 0.326032 | 0.348226-0.356209 | −0.0222 to −0.0302 (−6.4 to −8.5 %) |
| CD (counts) | 208.76 | 212.28 | 224.53-226.85 | −12.3 to −14.6 counts (−5.5 to −6.4 %) |
| \|CM\| | 0.028791 | 0.031140 | 0.037354-0.038774 | −16.6 to −19.7 % |

Command:
`python -c "import json; m=json.load(open('results/s0/summary.json'))['mesh']['anchor']; vj={'cl':(0.348226045,0.356208937),'cd':(0.022453440,0.022684938),'cm':(0.037353559,0.038774022)}; [print(k,g,abs(m[k][g]),round(100*(abs(m[k][g])/lo-1),2),round(100*(abs(m[k][g])/hi-1),2)) for k,(lo,hi) in vj.items() for g in ('256x97','512x193')]"`.

What this says:
- **The truth is low.** SU2 at 512x193 is low on every coefficient of this case, by 6-8 % in lift,
  and it is still rising from below on our grids: CL 0.311764 → 0.317363 → 0.326032 at 128x49,
  256x97 and 512x193 (S0's non-asymptotic sequence).
- **It is low by more than the floor suggests.** The 512x193 truth's own CL error here, at least
  0.022, is about 2.6 times the floor's train-minus-fine CL difference on the same case, −0.0087.
  The 256x97 floor solution is 8.9-10.9 % below the continuum, against the 2.7 % the floor reports.
- **Every comparison in S4 is relative to SU2 at 512x193, not to the exact Euler solution.** The
  rankings and the ratios to the floor are unaffected, because every contestant meets the same
  truth. But the floor understates the error any 256x97 answer carries on transonic lifting flow,
  and "near the floor" is a statement about agreement with SU2.
- **The causes are not separated.** Four differences could contribute, none isolated here:
  - the section (closed trailing edge against the paper's sharp extended one);
  - the far field: 100 chords with no vortex correction, against about 150 chords. S0 measured CL
    +2.1e-3 from 100 to 200 chords at 256x97, and the paper's CFL3D moves 0.0034 with the vortex
    correction;
  - the grid: 512 by 192 stretched cells, against aspect-ratio-one meshes up to 4096²;
  - the scheme: JST, against H-CUSP, central and flux splitting.
  The size and direction of our refinement trend point at resolution first; that is an inference,
  not a measurement.

**Wassing, S., Langer, S., and Bekemeyer, P., "Physics-Informed Neural Networks for Transonic Flows
around an Airfoil," arXiv:2408.17364v3 [physics.flu-dyn], 15 April 2025 (DLR; submitted to Physics
of Fluids).** The spec's memory, "Wassing, Langer & Bekemeyer (DLR, around 2023-24) trained PINNs on
transonic NACA 0012 flow with artificial viscosity", is correct: v1 is from August 2024.

What they do:
- **The equations:** the steady 2D Euler equations in primitive variables (ρ, u, v, p) with the
  energy equation, on a 400x200 elliptic O-grid. The network takes the computational coordinates
  (ξ, η) of the grid points, and physical derivatives come from inverse metrics by finite
  differences. Predictions exist only at grid points.
- **The network:** a random Fourier-feature embedding (128 frequencies, σ = 1), then 5 layers of
  128 adaptive-tanh units with weight normalisation.
- **The loss:** soft far field, wall and periodicity terms. The residual is volume-weighted and
  scaled by λ_res = 2·10⁴. There is **no explicit Kutta condition**.
- **The optimiser:** L-BFGS only, in mini-batches of 5,000 points, reset after each outer
  iteration, 40 epochs.
- **Artificial viscosity:** added to the residual as a scalar μ = ν·s·(c + |q|) or in matrix form;
  the best variants apply it to the density and energy equations only. Here s is an analytic sensor: a shock sensor, the pressure gradient
  along the flow, times a stagnation-point sensor that switches viscosity off at stagnation points.
  The viscosity factor is ν ≈ 1.9e-3.

Results on NACA 0012 against a CODA finite-volume reference (Roe, 400x96, 90 chords):
- **Cp error:** for the best variant (matrix viscosity, both sensors) the relative mean absolute
  error of Cp is 0.42, 0.3 and 0.33 % at (M, α) = (0.70, 4°), (0.72, 3°) and (0.75, 2°), and 3 % at
  (0.78, 3°), where the shock sits upstream of the reference's (their Table III, p. 26).
- **Without viscosity** the PINN captures no shock at all (their Fig. 11). **Global** scalar
  viscosity without sensors (their model I) also fails to capture the shock.
- **Cost:** 30-45 min of training per case on DLR's HPC resources (the hardware is not stated).
- **Parametric:** one network trained over α ∈ (0, 4)° at M 0.72 gives CL in agreement with the
  reference (their Fig. 10). It took 1.5-2.75 h to train and 35 ms per prediction.
- **Their conclusion:** PINNs do not currently outperform CFD in speed and accuracy for
  non-parametric forward problems, and their case rests on parametric use.

How this bears on what this study found:
1. **The Euler PINN's collapse to rest is not a property of Euler PINNs in general.** Their Euler
   PINN, also with a strong-form residual and a soft far field, converges to the right transonic
   flow. The 2026-10-04 root cause is qualified in place accordingly. The differences from the
   Euler PINN here are many:
   - residual weight 2·10⁴ against 1, with volume weighting;
   - Fourier features of grid indices, against (cos 2πξ, sin 2πξ, η);
   - primitive outputs with the energy equation, against a constant-enthalpy closure with exact
     wall tangency;
   - L-BFGS only, in mini-batches, against Adam then L-BFGS;
   - collocation at grid points, against random points;
   - their Mach 0.70-0.78, against our warm-up at Mach 0.1 and showcase case at 0.36.
   The 2026-10-04 probes tested J-weighting (J¹ is close to volume weighting: it did not collapse
   but stayed at its initial state) and a low-Mach reparameterisation, but not the others. Which
   difference rescues the Euler PINN is open.
2. **Lift without an explicit Kutta condition.** Their circulation comes out right with no Kutta
   term, while ours came out 64-101 % low on five of six cases with one. Their sensor is active near the trailing edge
   (their Fig. 7), which may supply the dissipation that selects the circulation, as SU2's
   numerical dissipation does. This is a possibility, not tested here.
3. **The dropped viscous variants.** The viscosity specified here (spec §7.4: constant ν times the
   Laplacian, everywhere) is a global form, like their failing model I, though on a different
   formulation, so this does not predict how those runs would have done. Their working viscosity
   is localised and scaled by the spectral radius.
4. **Cost.** Their 30-45 min per non-parametric case is 7-11 times shorter than the 5.6 h mean
   `pure` run here, on unstated, different hardware. Both studies find a per-case PINN slower than
   CFD.

## 2026-10-07 — S5 predictions, recorded before any S5 code or run

Stage S5 (`docs/superpowers/specs/2026-10-07-s5-euler-pinn-av-design.md`, approved by the author
on 2026-10-07): an Euler PINN built to Wassing, Langer & Bekemeyer's recipe, gated on their NACA 0012
case B against SU2, ablated one ingredient at a time, and compared with the FNO on three in-range and
three extrapolation cases. These predictions are recorded before any S5 code exists or any S5 run,
and are never edited afterwards; outcomes are appended below them.

- **P4.** Recipe R passes the gate (RMAE of Cp and of local Mach each ≤ 1 % against SU2 at 512x193)
  within the four tuning configurations T1-T4.
- **P5.** Reverting the residual weighting (A1) breaks R; reverting the input encoding (A4) or the
  collocation (A5) does not. R on the cylinder at M 0.1 (A6a) fails or collapses: the Mach range is
  part of why this study's Euler PINN failed.
- **P6.** On the three extrapolation cases the new PINN degrades less than the FNO (a smaller
  degradation factor: mean extrapolation near-field ratio to the floor over mean in-range ratio),
  but its mean in-range near-field ratio to the floor stays above the FNO's at 256x97.
- **P7.** Per case, the new PINN still costs at least 10x SU2's CPU time.

## 2026-10-08 — S5 implemented; the recipe as built is trapped in a uniform field, and the gate over-runs its budget

S5's code is on branch `s5` (commits a813c61 to 04bc30b; the plan's "Implementation notes" list
where it departs from the plan's text). Full suite 363 passed, 1 skipped; `ruff check .` clean. **No
S5 stage has run**: no SU2 on the validation case, no gate. The three figures below come from one
probe that writes no S5 record. Command:
`python scripts/probes/probe_s5_prelaunch.py > results/s5/probe_prelaunch.log 2>&1` (Apple GPU
available, `mps available: True` on the log's first line).

**1. The recipe falls into a uniform field in its first L-BFGS outer iteration.** One epoch of T1
(`AvConfig()` defaults, seed 0) on validation case B (NACA 0012, M 0.72, α 3°), float32 on the Apple
GPU, 39.1 s:
- The first outer iteration took the loss from 279.6 to 0.05075 in 97 closures.
- Layer 4's adaptive slope went to exactly zero: ω_k = −0.221, −0.215, −0.207, **−0.0**, 0.134. That
  layer's output is tanh(0) = 0, so the network is a constant.
- The field at the 256x97 nodes is uniform: every variable spreads by less than 2.9e-8, the wall Cp
  by 6.6e-8, and the wall speed is 0.930 U∞ everywhere.
- The other 19 outer iterations stalled at the same loss after 3-5 closures each.

A whole-branch reviewer found the same state for T2-T4, through a different layer's slope each
time. With ω frozen at 0.1 the reviewer found a drift towards a near-uniform field instead. Both of
these are the reviewer's measurements, not re-measured here.

T1-T4 vary only the sensor reading, the viscosity form and the trailing-edge exclusion. None of them
acts on a uniform field, whose residual and viscous term are exactly zero, so the gate as specified
would fail all four configurations for this one reason and use up the pre-registered tuning budget.

The slope parameters n = 10, ω₀ = 0.1 were a declared assumption (design §2.2 and §2.6, item 5).
Jagtap, Kawaguchi & Karniadakis's layer-wise adaptive activations ("Locally adaptive activation
functions with slope recovery…", Proc. R. Soc. A 476, 20200334, 2020), which Wassing et al. cite as
their ref. 20, add a slope-recovery term to the loss. This implementation has no such term, and
Wassing et al. do not say whether they used it.

**2. Where the residual's weight goes.** The volume weights on this 100-chord grid put 1.25e-4 of the
residual weight inside the RMAE box, which holds 43.0 % of the residual nodes; the outer 10 rows
carry 56.4 % of it. λ_res·w has a median of 1.67 in the box, 0.106 on the wall row and 1.62e5 on the
outer row. The residual loss is therefore almost entirely the far field's, which a uniform free
stream nearly satisfies.

**3. The gate's upper bound exceeds its budget.** time_closure on case B with `AvConfig()` (one
batch of 5,000 points, the viscous term forced on everywhere):

| device | closure | L-BFGS two-loop per iteration | upper bound per gate configuration |
|---|---|---|---|
| Apple GPU | 0.408 s | 0.425 s | 207.8 h |
| CPU | 2.233 s | 0.142 s | 651.8 h |

The four configurations project to about 830 h against the 72 h budget, so `s5_pinn.py gate` would
refuse (exit 3), as designed. The bound charges every outer iteration 1,250 closures and 1,000
L-BFGS iterations. The real first epoch above took 39 s, but only because it was trapped.

**Decisions needed before any gate run (the author's, to be recorded here):**
- (a) How to keep the network out of the uniform state, for example Jagtap et al.'s slope-recovery
  term, a fixed slope, or ω bounded away from zero; and whether the residual weighting on this
  100-chord grid should change.
- (b) Whether that is an amendment to the recipe made before the gate, so that T1-T4 stay the
  pre-registered tuning budget. Any probe that informs it runs on validation case B, which is not a
  test case.
- (c) The gate's budget: raise `--budget-hours` on record, cut epochs, max_iter or max_eval, or
  project from a measured first epoch rather than the upper bound.

**The author's decisions (2026-10-08), taken on the measurements above, before any gate run:**
- **The trap: probe, then amend.** A few short probes run on validation case B, each changing one
  thing from T1: Jagtap et al.'s slope-recovery term; a fixed slope; residual weights capped at
  their near-field value; and the slope-recovery term with the capped weights together. Each runs
  the recipe's first outer iterations and is judged structurally, without SU2's truth (which is not
  computed): does the field leave the uniform state, do the slopes stay off zero, does the loss
  fall, and does the wall Cp have a stagnation point and a suction peak? The gate metric is not
  used, so the gate's own measurement on case B stays unused until the gate. A dated recipe
  amendment (design §2.7) follows from what works. T1-T4 stay the pre-registered tuning budget on
  the amended recipe. The cost, accepted: the amendment is informed by probes on the gate's own case.
- **The budget: project from a measured epoch.** Before each gate configuration trains, one real
  epoch of it is timed. The configuration is projected at 40 times that, and the phase refuses
  (exit 3) when the time already spent in the phase plus that projection exceeds `--budget-hours`
  (72). The steps phase keeps its re-projection from R's measured run.

## 2026-10-08 — S5 trap probes: no candidate leaves the uniform field; the gate's budget is now measured

**No S5 stage has run.** The probes write no S5 record. Each arm changes one thing from T1
(`AvConfig()` defaults, seed 0) on validation case B and repeats the recipe's first three
mini-batch outer iterations exactly as `train` runs them: the same seeds, batches and L-BFGS
settings. To keep each arm to a minute or two, every outer iteration is cut at 120 L-BFGS iterations
and 150 closures. Float32 on the Apple GPU. Commands:
`python scripts/probes/probe_s5_trap.py > results/s5/probe_trap.log 2>&1`, then
`python scripts/probes/probe_s5_trap.py --round 2 > results/s5/probe_trap_round2.log 2>&1`.

**Round 1: the author's candidates.** The control repeats the pre-launch probe exactly: 279.6 to
0.05075 in 97 closures, with layer 4's slope at 0.

| arm | loss after outer iterations 1, 2, 3 | n ω_k at the end | end state |
|---|---|---|---|
| control | 0.05075, 0.05075, 0.05075 | −2.21, −2.15, −2.07, **0**, 1.34 | uniform: spread < 3e-8, wall Cp 0 |
| slope recovery, S = 1 / mean_k exp(ω_k) | 0.05085, same, same | 12.5, 9.75, 9.31, **0**, 108 | uniform: spread < 2e-9 |
| fixed slope (ω_k = 0.1) | 2.856, 2.587, 1.957 | 1, 1, 1, 1, 1 | near-vacuum: wall Cp −2.73 everywhere (p ≈ 0.01), L_∞ 1.77 |
| weights capped at the box's largest (λ_res·cap 61.5) | 0.05075 (in 30 closures), same, same | **0**, −0.03, 0.12, −0.69, −1.56 | uniform: spread < 9e-8 |
| slope recovery and capped weights | 0.06008, 0.05914, 0.05914 | 5.78, −1.41, 10.5, 64.2, **0** | uniform: spread < 7e-10 |

None meets the criteria the author set: no arm has a stagnation point or a suction peak on the wall.
- Slope recovery fails as a mean over layers can. The term is satisfied by inflating the other
  slopes (n ω up to 108) while one layer stays dead. The term's form, exp(ω_k) averaged over the
  layers, is recalled from Jagtap et al.'s layer-wise variant and was not checked against the paper,
  which is not in `references/`.
- A fixed slope keeps every layer alive. The network then finds the other trivial state, a near
  vacuum (p, and with it every flux, close to zero), which the residual also nearly satisfies. The
  far-field term (1.77) is still falling there, slowly.
- Capping the weights changes the initial loss only from 279.6 to 252.2, and reaches the uniform
  field three times faster.

**Correction to point 2 of the entry above.** The far field holds the weight, not the loss. At
initialisation (T1, every residual node, in the training batches):

| region | share of the residual nodes | share of λ_res·L_res |
|---|---|---|
| wall rows, j < 10 | 5.2 % | 42.9 % |
| rest of the RMAE box | 37.8 % | 42.8 % |
| outer rows, j ≥ 183 | 5.2 % | 1.3 % |
| elsewhere | 51.8 % | 13.0 % |

The weight w ∝ cell area meets r² ∝ 1/(cell size)², so w·r² measures the field's variation in index
space. The Fourier features act on the integer indices with σ = 1, so the network at initialisation
varies from cell to cell everywhere. λ_res·L_res = 235.7 against L_∞ + L_af + L_lr = 2.516, a ratio
of about 94.

**Round 2: balancing the terms (an addition to the author's list).** Their reason for λ_res = 2·10⁴
is to bring λ_res L_res level with the boundary loss (their §II H). Applied to this grid at
initialisation, that reason gives λ_res = 2.516 / 0.01179 ≈ **210**.

| arm | loss after outer iterations 1, 2, 3 | n ω_k at the end | end state |
|---|---|---|---|
| balanced (λ_res = 210) | 0.05075 (from 5.425, in 60 closures), same, same | −0.99, −0.95, −0.61, **0**, 1.74 | uniform: spread < 6e-6 |
| balanced, fixed slope | 0.05356, 0.0538, 0.05075 | 1, 1, 1, 1, 1 | uniform: spread < 5e-4, reached in the third outer iteration |

**What this establishes.** The uniform field is reached from initialisation however the terms are
balanced and however the slopes are treated. Its residual and viscous term are exactly zero and its
far-field term is near zero, so its loss is the wall-tangency term alone: 0.0469 = mean (u∞·n̂)².
It is a local minimum. With one layer dead, the only directions out are perturbations of that
layer's random, cell-scale pre-activations, which no Euler solution resembles, so each costs far
more residual than it saves at the wall. This failure belongs to the recipe as specified, on this
grid. The four candidates do not touch it. The changes that might (Adam before L-BFGS, a hard wall
condition, a smoother embedding, another initialisation) are departures from their recipe, and the
first two are ablation arms A3 and A2.

**No recipe amendment is written** (design §2.7 stays unwritten): nothing worked. The gate's budget
is amended as the author decided (design §7.1, dated 2026-10-08). `time_epoch` times one real epoch,
and `s5_pinn.py gate` checks each configuration just before it trains: the time spent in the gate
plus 40 measured epochs, against 72 h. Tests first; full suite 369 passed, 1 skipped; `ruff check`
clean. A trapped epoch takes about 40 s, so a trapped gate configuration takes about half an hour.

**Decision needed (the author's):** either run the gate as pre-registered on the recipe as built,
where the expected outcome is "no configuration passes; S5 stops after step 1", or probe
departures from their recipe. Probes of departures should run on a case that is not the gate's
(validation case A or C), so that case B's gate stays unused.

**The author's decision (2026-10-08): probe departures from their recipe on validation case A**
(NACA 0012, M 0.70, α 4°). Case A is the last fallback, so case B's gate stays unused. The probes
are judged by the same structural criteria, without SU2's truth. Each changes one thing from T1:
Adam before L-BFGS, a hard wall condition, a smoother Fourier embedding, or Glorot initialisation.
A dated amendment (design §2.7) follows from what works, before the gate. Ablation arms that the
amendment would make identical to R are re-specified on record.

## 2026-10-08 — S5 departure probes on case A: every arm ends in a trivial or non-physical state

**No S5 stage has run**, and the probes write no S5 record. This is the author's round of
departures from their recipe, on validation case A (NACA 0012, M 0.70, α 4°), so case B's gate
stays unused. The method is that of the case-B probes: the recipe's first three outer iterations,
each cut at 120 L-BFGS iterations and 150 closures, float32 on the Apple GPU. A non-finite loss
stops an arm, as it stops `train`. Commands:
`python scripts/probes/probe_s5_departures.py --arms control,adam,hard_wall,sigma_0.1,glorot > results/s5/probe_departures.log 2>&1`;
`python scripts/probes/probe_s5_departures.py --arms hard_both,hard_both+adam > results/s5/probe_departures_hard.log 2>&1`.
The first log was written before the `--arms` option existed, by the script run with no arguments
(the same five arms).

| arm (one change from T1 unless stated) | loss after outer iterations 1, 2, 3 | n ω_k at the end | end state |
|---|---|---|---|
| control (T1 on case A) | diverged in the first, at closure 37 (last finite loss 1.4e25) | — | — |
| adam: 400 Adam steps at 1e-3 first | Adam reaches 1.677 (L_∞ 1.62); then 0.04857, same, same | −0.37, **0**, 0.38, 1.05, 1.73 | uniform: spread < 5e-8 |
| hard_wall: u −= (1 − η)⁴ (u·n̂) n̂ | 0.6378, 0.6448, 0.6384 (wall term 1e-18) | −3.64, −5.43, −5.04, **0**, −3.82 | near rest: wall speed ≤ 0.073 U∞, L_∞ 0.591 |
| sigma_0.1: frequencies from N(0, 0.1²) | 0.04857, same, same | −1.71, −2.41, −2.41, **0**, 1.66 | uniform: spread < 7e-6 |
| glorot: Glorot-normal weights, zero biases | 1.965e6, 1.72e6, 4.125e6 (initial 4.45e6) | 0.95, 0.95, 0.93, 0.92, 0.90 | non-physical: mean ρ 0.0006, mean p −0.0039 |
| hard_both: q += η⁴ (w∞ − q), then hard_wall | 3.914, 3.858, 3.825 (initial 897) | −2.13, −3.54, −3.46, ≈0 (−0.001), 0.10 | low-density interior: box spread of ρ 0.033 and p 0.041, mean ρ 0.45, wall Cp −2.46 to −2.46 |
| hard_both and Adam first | Adam reaches 9.712; then 3.674, 4.117, 3.852 | 12.4, −2.97, −0.94, **0**, 2.63 | the same: mean ρ 0.43, wall Cp −2.52 to −2.52 |

No arm meets the author's criteria. None has a stagnation point (case A's stagnation Cp is 1.129)
or a suction peak on the wall.
- **Each trap is a state the residual nearly satisfies.** Each one violates only what is left soft:
  - with a soft wall, the uniform field;
  - with a soft far field, the fluid at rest or a vacuum;
  - with both conditions exact, a low-density, low-pressure interior whose small fluxes make the
    residual small (λ_res·L_res 3.8, still falling slowly).
- **One layer's slope goes to zero in six of the seven trained arms.** Adam alone keeps every slope
  near 0.85 for 400 steps, and the first L-BFGS outer iteration after it still kills one.
- **T1 on case A diverges** where on case B it collapsed: the recipe is also fragile to the free
  stream.
- **Glorot initialisation** multiplies the initial loss by about 1.6e4 and makes no progress in
  three outer iterations.

**The limits of this evidence.** Each arm ran 3 of the recipe's 800 outer iterations, each cut at
150 of its 1,250 closures. The uniform states are exact (a dead layer, the outer iterations stalling
in 5 to 7 closures), so they are traps in practice. The both-hard state was still falling, by 1 %
per outer iteration, and the probes do not show where it ends. All arms used seed 0.

**No recipe amendment is written.** Design §2.7 stays unwritten: nothing the author listed, nor the
two additions (the balanced λ_res on case B, both conditions exact here), worked.

**The author's decision (2026-10-08): one full epoch of both-hard on case A** before deciding about
the gate. That is hard_both through the recipe's own L-BFGS limits: all 20 mini-batch outer
iterations of an epoch, max_iter 1,000, max_eval 1,250, history 1,000, with no cut. The state is
reported every 5 outer iterations. The question is whether the slowly falling low-density state
ever turns physical.

**The full epoch (2026-10-08): the low-density state is a stall, not a slow descent.** Command:
`python scripts/probes/probe_s5_departures.py --arms hard_both --full-epoch > results/s5/probe_hard_both_epoch.log 2>&1`.
hard_both on case A ran through all 20 outer iterations at the recipe's own L-BFGS limits (max_iter
1,000, max_eval 1,250, history 1,000, no cut). It took 699 closures in 782 s on the Apple GPU. No
outer iteration came near max_eval: the first used 210 closures, and the last twelve 5 to 36 each,
L-BFGS stopping on its own tolerance.
- **The loss did not fall across the epoch.** The outer iterations ended at 3.914 (from 897),
  4.091, 3.798, 3.851, 3.57, 3.944, 3.622, 3.669, 3.825, 3.868, 3.895, 3.903, 3.705, 3.829, 3.883,
  3.682, 3.692, 3.731, 4.024 and 3.784. The spread, 3.57 to 4.09, is batch to batch. The loss is
  λ_res·L_res (L_res 1.8e-4 to 2.0e-4) plus L_lr ≈ 0.018. Both hard conditions hold to round-off.
- **A dead layer throughout.** Layer 4's slope sat near zero for the first two outer iterations
  (n ω 0.063, 0.070). From the third on, layer 5's is exactly zero: n ω ends −0.898, −3.09, −2.35,
  −0.586, 0.
- **The same state at every report.** After 5, 10, 15 and 20 outer iterations the wall Cp is
  −2.390, −2.431, −2.423 and −2.423, the same all round the wall. At the end the mean ρ is 0.458
  and the mean p 0.338; in the RMAE box ρ spreads by 0.033 and p by 0.040, and the largest wall
  speed is 1.707 U∞. There is no stagnation point or suction peak.
- The network is saved to `runs/probes/s5_hard_both_epoch1.pt` (git-ignored).

So the most constrained variant probed (both conditions exact) stalls in a non-physical state as
firmly as T1 does in the uniform field. The probes give no amendment to write.

**The author's decision (2026-10-08): run the gate as pre-registered**, on the recipe as built (no
amendment), with the gate's budget measured as amended in design §7.1. First SU2 on the validation
case (`scripts/s5_su2.py`), then `scripts/s5_pinn.py gate`, T1 to T4 in order. Expected from the
probes: no configuration passes, and S5 stops after step 1 (design §3.4).

## 2026-10-08 — S5 gate: no configuration passes; S5 stops after step 1

The pre-registered gate ran on the recipe as built, with no amendment, under the measured budget of
design §7.1 (amended 2026-10-08). All runs were at commit d3da6d1, with a clean tree.

**The reference.** Command:
`SU2_BIN_DIR=/Users/esousa/theLab/bwb-tanker-scratch/bin python scripts/s5_su2.py --workers 2 > results/s5/s5_su2.log 2>&1`.
Case B (NACA 0012, M 0.72, α 3°) converged at both levels under the frozen settings and the
convergence rule, so `results/s5/su2.json` selects `s5-val-B`.

| level | iterations | wall | CL | CD (counts) | CM | largest surface Mach |
|---|---|---|---|---|---|---|
| 256x97 | 2,405 | 149 s | 0.5687 | 148.5 | −0.00445 | 1.381 |
| 512x193 | 4,537 | 1,304 s | 0.5744 | 149.4 | −0.00340 | 1.387 |

The floor (SU2 at 256x97 against the truth) has RMAE 0.054 % on Cp and 0.051 % on Mach.

**The gate.** Command: `python scripts/s5_pinn.py gate > results/s5/s5_pinn_gate.log 2>&1`, on the
Apple GPU, T1 to T4 in order. Scores: `python scripts/s5_score.py > results/s5/s5_score.log 2>&1`
(`results/s5/scores.md`, `gate.json`).

| config | RMAE Cp | RMAE Mach | passed | final loss (L_∞, L_af) | outer iterations stalled | closures | timed epoch | projected | wall |
|---|---|---|---|---|---|---|---|---|---|
| T1 | 5.177 % | 5.632 % | no | 0.05075 (0.0039, 0.0469) | 798 / 800 | 3,416 | 37 s | 0.41 h | 675 s |
| T2 (sensor reading B) | 5.177 % | 5.631 % | no | 0.05075 (0.0039, 0.0469) | 799 / 800 | 881 | 8 s | 0.09 h | 76 s |
| T3 (scalar viscosity) | 5.179 % | 5.634 % | no | 0.05076 (0.0038, 0.0470) | 799 / 800 | 3,660 | 142 s | 1.58 h | 1,696 s |
| T4 (0.02-chord TE exclusion) | 5.177 % | 5.631 % | no | 0.05075 (0.0039, 0.0469) | 797 / 800 | 3,462 | 341 s | 3.79 h | 6,142 s |

- **All four configurations end in the probes' uniform field.** The residual is about 1e-15, and the
  loss is the wall-tangency term plus a small far-field term. Every one predicts CL ≈ 0 (error
  −0.574) and misses the upper-surface shock (SU2: at x = 0.42, Mach 1.34 ahead of it). The
  near-field relative L2 error is 1.32 to 1.33.
- **A uniform stream scores 5.2 % on the gate's metric.** RMAE divides by the field's range, and
  most of the RMAE box is near free stream. For scale, on step-3 case `test_in-0004` the metric
  gives 0.30 % for the FNO at 256x97 and 1.23 % for S3's potential PINN. *(Amended 2026-10-26:
  0.13 % for the FNO at the amended 16 padding rows.)*
- **The scorer's collapse flag is False for all four.** Its test, the largest wall speed below half
  the truth's, catches fluid at rest. A uniform stream at 0.93 U∞ passes it. The gate's verdict does
  not depend on the flag.
- **The gate took 9,117 s (2.53 h) against 72 h**, timing epochs included. Each timed first epoch
  overestimated its run, by 2.2 to 4.2 times, because a trapped run's only real descent is in its
  first epoch.
- **Closure cost varied from 0.086 s (T2) to 1.77 s (T4)** for runs in the same state. The cause is
  not identified.

`results/s5/recipe.json` records config null, so **S5 stops after step 1** (design §3.4).
`s5_pinn.py steps` exits 0 on it with "the gate failed", and no ablation or step-3 run is made.

**Predictions (2026-10-07 entry; the text there is unchanged, these are its outcomes):**
- **P4 fails.** No configuration among T1–T4 passes the gate: every one misses the 1 % threshold by
  about five times, in the same uniform state.
- **P5, P6 and P7 are not tested.** Without a recipe R there is no ablation (P5), no step-3
  comparison (P6) and no per-case cost (P7).

**What this does and does not show.** The result concerns their recipe as specified and as built
here, on this study's 100-chord O-grid. Within S5, it cannot separate a flaw in the recipe from a
difference in what the paper leaves unstated: the far-field distance of their PINN grid, the
weights' normalisation, the initialisation, the precision. What the probes add is that the trap does
not depend on any single ingredient tried: the slope treatment, the residual weighting and balance,
the optimiser, the embedding scale, the initialisation, or hard boundary conditions. It is the
residual's preference for trivial states (uniform, at rest, vacuum, low density), reached from
initialisation, under every ingredient tried. This study's earlier Euler PINN converged to fluid at
rest (S3). S5 finds that artificial viscosity, localised by their sensors, does not change that on
this grid: the viscous term is zero in each trivial state, so it never acts.

## 2026-10-22 — the FNO on a Colab A100: S2 reproduced, and the FNOs of two libraries scored

Not a stage of the design: a probe of two questions the author asked on 2026-10-21. Does S2's FNO
come out the same when trained on an NVIDIA GPU, and how do the FNOs of NeuralOperator 2.0.0 and
PhysicsNeMo 2.2.2 compare with this study's `FNO2d` on the same data? The probe and what each
library's FNO is are in `docs/notes/2026-10-21-fno-libraries.md`; the steps are in
`docs/notes/2026-10-22-colab-fno.md`.

**The runs.** By the author, on a Google Colab A100 runtime, with
`NVIDIA_TF32_OVERRIDE=0` (author's statement, 2026-10-22: TF32 off, so float32 throughout as on the
Mac), commands of the form
`python scripts/probes/probe_fno_libraries.py --library L --n-train N --seed S --epochs 300 --device cuda`:
`apno` (this study's `FNO2d` through the same loop as S2) at n = 64, 128, 256, 512 and 896 (seed 0)
and at 896 with seeds 1 and 2; `neuralop` and `physicsnemo` at 896, seed 0. Every run used the S2
configuration (width 48, modes 24 × 8, 4 layers, loss A, batch 8, AdamW 1e-3 with cosine decay,
patience 30). The records (`runs/fno_lib/*/train.json`) and predictions are git-ignored; their
fields are copied into `results/fno_lib/scores.json` (`runs`). The records do not carry the torch
version or the GPU model: "cuda" and the author's A100 are all that is known of the hardware.

**Scoring.** `python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2`
(commit fa9ce84, which holds its output, `results/fno_lib/scores.md` and `scores.json`), with S4's
truth, floor and per-case code. The S2 files scored beside them (variant `s2`) reproduce S4's own
committed scores exactly: 18 groups, the largest difference over four metrics 0.0. Command:
`python -c "import json; L=json.load(open('results/fno_lib/scores.json'))['groups']; S=json.load(open('results/s4/scores.json'))['groups']; k=lambda g:(g['grid'],g['n_train'],g['seed'],g['split'],g['regime']); s={k(g):g for g in S if g['method']=='fno' and g['variant']==''}; print(sum(g['variant']=='s2' for g in L), max(abs(g[m]-s[k(g)][m]) for g in L if g['variant']=='s2' for m in ('near_field_rel_l2_mean','cl_err_absmax','cd_err_counts_absmax','cp_max_mean')))"`.
All 1,875 case scores are `ok`: no missing or non-finite prediction.

### S2 on CUDA

| run | best val, S2 (MPS) | at epoch | best val, CUDA | at epoch | epochs run | s per epoch, MPS / CUDA |
|---|---|---|---|---|---|---|
| n64 | 0.08493 | 297 | 0.08493 | 297 | 300 | 5.12 |
| n128 | 0.05065 | 288 | 0.05065 | 288 | 300 | 5.70 |
| n256 | 0.03320 | 284 | 0.03322 | 284 | 300 | 5.77 |
| n512 | 0.01784 | 274 | 0.01781 | 287 | 300 | 5.81 |
| n896 seed 0 | 0.01260 | 278 | 0.01256 | 284 | 300 | 5.27 |
| n896 seed 1 | 0.01274 | 293 | **0.02035** | 160 | **191** | 5.47 |
| n896 seed 2 | 0.01178 | 294 | 0.01169 | 294 | 300 | 5.47 |

Command:
`python -c "import json; r={x['tag']:x for x in json.load(open('results/fno_lib/scores.json'))['runs']}; s={x['tag']:x for x in json.load(open('results/s2/runs.json'))}; [print(t, s[t]['best_val'], s[t]['best_epoch'], r[a]['best_val'], r[a]['best_epoch'], r[a]['epochs_run'], round(s[t]['wall_s']/s[t]['epochs_run']/r[a]['s_per_epoch'],2)) for n,sd in ((64,0),(128,0),(256,0),(512,0),(896,0),(896,1),(896,2)) for t,a in [(f'fno-n{n}-s{sd}',f'apno-n{n}-s{sd}')]]"`.

- **Six of the seven runs reproduce S2** to within 0.8 % in validation loss, and the n64 and n128
  runs to four digits at the same best epoch. The starting weights and the batch order are drawn
  on the CPU from the same seed on both machines, so only the floating-point arithmetic differs.
- **The data-efficiency curve reproduces to 1 %.** Mean near-field relative L2 over the 61 test_in
  cases at 256x97, CUDA against S4: 0.1088 / 0.1088 (n64), 0.0857 / 0.0857, 0.0699 / 0.0698,
  0.0558 / 0.0561, 0.0487 / 0.0483 (n896). S4's data-efficiency result and the break-even it feeds
  do not depend on the device. Command:
  `python -c "import json,numpy as np; R=json.load(open('results/fno_lib/scores.json'))['rows']; S=json.load(open('results/s4/scores.json'))['rows']; f=lambda rows,v,n: np.mean([r['score']['near_field_rel_l2'] for r in rows if r['method']=='fno' and r['variant']==v and r['seed']==0 and r['n_train']==n and r['grid']=='256x97' and r['split']=='test_in']); [print(n, round(f(R,'apno',n),4), round(f(S,'',n),4)) for n in (64,128,256,512,896)]"`.
- **Seed 1 did not reproduce.** Its validation loss stopped improving at epoch 160, and early
  stopping (patience 30) ended it at epoch 191 with 0.02035, 1.6 times S2's 0.01274. On the test
  cases it is the worst of the six 896-case `FNO2d` runs: 6.37 times the floor on in-range
  subcritical cases at 256x97, against 5.28 to 5.50 for the other five (table below). Arithmetic
  differences alone moved the run into early stopping. S2's three seeds (validation loss 0.0118 to
  0.0127) therefore understate how much one run of this protocol can vary. *(Outcome 2026-10-22,
  reruns entry below: trained to 300 epochs, seed 1 reaches 0.01264 at epoch 293, S2's 0.01274 at
  epoch 293; the early stop, not the arithmetic, made it the outlier.)*
- **Cost.** An epoch took 5.1 to 5.8 times less wall time on the A100 than on the Mac's GPU; the
  nine Colab runs took 2,237 s in all. This is GPU wall time on different hardware, its own cost
  basis, and is not substituted into S4's costs. It is below the 7–14 times estimated beforehand
  (to the author, 2026-10-22, not recorded here) from the step's operation count, which assumed
  about 6 ms of launch overhead per training step; the measured step is about 12 ms (1.37 s over
  111 steps).

### The library FNOs

Near-field relative L2 against the floor (SU2 at 256x97), n = 896; S4 reported the S2 FNO, seed 0,
at 5.32 / 2.38 / 3.56 times at 256x97. Command:
`python -c "import json; [print(r['variant'], r['seed'], r['grid'], r['split'], r['regime'], round(r['x_floor'],2)) for r in json.load(open('results/fno_lib/scores.json'))['floor_ratios'] if r['n_train']==896]"`.

| contestant | grid | test_in subcritical | test_in supercritical | test_ext supercritical |
|---|---|---|---|---|
| `FNO2d`, CUDA, seeds 0 / 2 | 256x97 | 5.38 / 5.50 | 2.40 / 2.54 | 3.57 / 3.68 |
| `FNO2d`, CUDA, seeds 0 / 2 | 128x49 | 1.32 / 1.27 | 1.56 / 1.59 | 3.58 / 3.70 |
| NeuralOperator | 256x97 | **3.59** | 2.73 | 4.10 |
| NeuralOperator | 128x49 | 2.37 | 2.31 | 4.20 |
| PhysicsNeMo | 256x97 | 5.53 | 3.45 | 4.62 |
| PhysicsNeMo | 128x49 | 1.59 | 2.65 | 4.64 |

- **NeuralOperator trades accuracy on its training grid for transfer to the finer one.** On in-range
  subcritical cases at 128x49 it is 2.37 times the floor, worse than every 896-case `FNO2d` run that
  trained to the end (1.27 to 1.32; the early-stopped seed 1 is 2.40), and at 256x97 it is the best
  of the eight 896-case runs (3.59 against 5.28 to 6.37): its error grows 1.5 times between the
  grids, `FNO2d`'s about 4 times. Its validation transfer is likewise the smallest, 0.0259 against
  0.0346 to 0.0381 for `FNO2d` and 0.0468 for PhysicsNeMo. Which of its differences does this (the
  channel MLP in each layer, the symmetric padding, the forward-normalised FFT, the initialisation)
  is not isolated. *(Corrected 2026-10-24: the forward-normalised FFT is not a difference, since
  both normalisations scale the transform pair by the same 1/N; the fourth difference is the
  two-layer lifting and projection. See the corrections entry of 2026-10-24.)* POD, at 2.12 times
  the floor there (S4, `pod-n896`), is still the best contestant on in-range subcritical cases at
  256x97. *(Outcome 2026-10-22, reruns entry below: these are the early-stopped run's figures.
  Trained to 300 epochs, NeuralOperator is 1.47 times the floor at 128x49 and 2.26 at 256x97.)*
  Command for POD:
  `python -c "import json; G=json.load(open('results/s4/scores.json'))['groups']; fl={(g['split'],g['regime']):g['near_field_rel_l2_mean'] for g in G if g['method']=='su2-256x97'}; [print(g['split'], g['regime'], round(g['near_field_rel_l2_mean']/fl[(g['split'],g['regime'])],2)) for g in G if g['method']=='pod' and g['n_train']==896 and g['grid']=='256x97']"`.
- **On supercritical and extrapolation cases neither library beats `FNO2d`.** *(Superseded for
  NeuralOperator 2026-10-22, reruns entry below: trained to 300 epochs it is 1.84 times the floor
  on in-range supercritical cases and 3.59 in extrapolation. What follows is the early-stopped
  run.)* NeuralOperator is
  2.73 and 4.10 times the floor, PhysicsNeMo 3.45 and 4.62, against 2.40 and 3.57 for `FNO2d`
  seed 0. In extrapolation NeuralOperator's drag error is the largest of the four compared: mean
  |CD error| 148.9 counts (largest 268.3), against 79.3 (183.1) for `FNO2d` seed 0, 79.3 (180.6)
  for the S2 file, and 54.6 (126.4) for PhysicsNeMo. In range, NeuralOperator's mean lift and drag
  errors are the smallest of the four (|CL error| 0.0102, |CD error| 10.6 counts, against 0.0170
  and 16.4 for `FNO2d` seed 0).
  Command:
  `python -c "import json,numpy as np; R=json.load(open('results/fno_lib/scores.json'))['rows']; [print(v, s, sp, round(np.mean([abs(r['score']['cd_err_counts']) for r in x]),1), round(max(abs(r['score']['cd_err_counts']) for r in x),1), round(np.mean([abs(r['score']['cl_err']) for r in x]),4)) for v,s in (('s2',0),('apno',0),('neuralop',0),('physicsnemo',0)) for sp in ('test_in','test_ext') for x in [[r for r in R if r['variant']==v and r['seed']==s and r['n_train']==896 and r['grid']=='256x97' and r['split']==sp]]]"`.
- **Shocks.** Of the 27 true in-range shocks at 256x97, NeuralOperator misses none, PhysicsNeMo 2,
  the `FNO2d` runs 1 or 2; none of the eight runs misses any of the 21 in extrapolation. Command:
  `python -c "import json; [print(g['variant'], g['seed'], g['split'], g['n_true_shocks'], g['n_missed']) for g in json.load(open('results/fno_lib/scores.json'))['shock_groups'] if g['grid']=='256x97' and g['n_train']==896]"`.

**What this does and does not show.** S2's results, its data-efficiency curve included, do not
depend on training on the Apple GPU. The library comparison rests on one seed per library, and
NeuralOperator's run, like `FNO2d`'s seed 1, was ended by early stopping at epoch 183 with its best
at 152, so its figures are not those of a converged run; seed 1 shows that one run can move by
more than the gaps between the libraries. Both libraries were run with their positional embeddings
off and their modes matched to `FNO2d`'s, as the probe configures them, not with their defaults.
Neither library's FNO, so configured, is better than this study's on supercritical or extrapolated
flow. *(Superseded for NeuralOperator 2026-10-22, reruns entry below.)* Reruns of seed 1 and of
NeuralOperator with early stopping off (`--patience 300`) would settle whether the early stop, not
the model, is behind their numbers; they have not been made.
*(Made 2026-10-22: the next entry. The early stop was behind both.)*

## 2026-10-22 — the two early-stopped FNO runs, rerun with early stopping off

The previous entry's `FNO2d` seed 1 and NeuralOperator run were both ended by early stopping
(patience 30). By the author, on a Colab A100, with `NVIDIA_TF32_OVERRIDE=0`, both were rerun for
the full 300 epochs:
`python scripts/probes/probe_fno_libraries.py --library apno --n-train 896 --seed 1 --epochs 300 --patience 300 --suffix nopatience --device cuda`
and the same with `--library neuralop` (seed 0). `--suffix` keeps the early-stopped runs: the
reruns are `apno-n896-s1-nopatience` and `neuralop-n896-s0-nopatience`, variants
`apno-nopatience` and `neuralop-nopatience`.

**A failure first.** The first attempt at the seed-1 rerun stopped in its first epoch with "CUDA
error: an illegal memory access was encountered", in a training step identical to the nine
earlier runs'. After a fresh Colab runtime (`docs/notes/2026-10-22-colab-fno.md`, "If a run stops
with a CUDA error") both reruns completed. The cause is not identified; nothing was recorded of the
runtime that failed.

**The environment.** These are the first records that carry it: an NVIDIA A100-SXM4-80GB, torch
2.11.0+cu130, `NVIDIA_TF32_OVERRIDE` 0. Torch's own `cudnn.allow_tf32` reads True; the override
switches TF32 off inside cuDNN and cuBLAS whatever the framework's flags say. The nine earlier runs
did not record their environment, so whether they ran on the same GPU model and torch is not known.

**Scoring.** `python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2`
(commit 3ed15ea, `results/fno_lib/scores.md` and `scores.json`): 2,175 case scores, all `ok`. The
75 groups of the first scoring (commit fa9ce84) are unchanged, the largest difference 0.0.
Command:
`git show fa9ce84:results/fno_lib/scores.json > /tmp/fno_lib_fa9ce84.json && python -c "import json; k=lambda g:(g['method'],g['variant'],g['grid'],g['n_train'],g['seed'],g['split'],g['regime']); o={k(g):g for g in json.load(open('/tmp/fno_lib_fa9ce84.json'))['groups']}; n={k(g):g for g in json.load(open('results/fno_lib/scores.json'))['groups']}; print(len(o), max(abs(o[x][m]-n[x][m]) for x in o for m in o[x] if isinstance(o[x][m],float)))"`.

| run | epochs run | best epoch | best val | transfer | wall (s) |
|---|---|---|---|---|---|
| S2 `fno-n896-s1` (MPS) | 300 | 293 | 0.01274 | 0.0355 | 2,242 |
| `FNO2d` seed 1, early-stopped | 191 | 160 | 0.02035 | 0.0355 | 261 |
| `FNO2d` seed 1, rerun | 300 | 293 | 0.01264 | 0.0355 | 408 |
| NeuralOperator, early-stopped | 183 | 152 | 0.02450 | 0.0259 | 243 |
| NeuralOperator, rerun | 300 | 294 | 0.01225 | 0.0271 | 398 |

Command:
`python -c "import json; r={x['tag']:x for x in json.load(open('results/fno_lib/scores.json'))['runs']}; [print(t, r[t]['epochs_run'], r[t]['best_epoch'], round(r[t]['best_val'],5), round(r[t]['transfer_val_rel_l2'],4), round(r[t]['wall_s']), r[t].get('env',{}).get('gpu'), r[t].get('env',{}).get('torch'), r[t].get('env',{}).get('NVIDIA_TF32_OVERRIDE')) for t in ('fno-n896-s1','apno-n896-s1','apno-n896-s1-nopatience','neuralop-n896-s0','neuralop-n896-s0-nopatience')]"`.

Near-field relative L2 against the floor, n = 896, beside the previous entry's runs and S4's POD
(`pod-n896`, 256x97 only). Command for the reruns:
`python -c "import json; [print(r['variant'], r['seed'], r['grid'], r['split'], r['regime'], round(r['x_floor'],2)) for r in json.load(open('results/fno_lib/scores.json'))['floor_ratios'] if r['n_train']==896 and r['variant'] in ('apno-nopatience','neuralop-nopatience')]"`;
the others are the previous entry's commands.

| contestant | grid | test_in subcritical | test_in supercritical | test_ext supercritical |
|---|---|---|---|---|
| `FNO2d` seed 1, rerun | 256x97 | 5.30 | 2.52 | 3.94 |
| S2 `fno-n896-s1` | 256x97 | 5.28 | 2.52 | 3.92 |
| `FNO2d` seeds 0 and 2 (CUDA and S2) | 256x97 | 5.32 to 5.50 | 2.38 to 2.54 | 3.56 to 3.73 |
| NeuralOperator, rerun | 256x97 | **2.26** | **1.84** | 3.59 |
| NeuralOperator, rerun | 128x49 | 1.47 | 1.63 | 3.68 |
| POD (S4) | 256x97 | 2.12 | 6.06 | 7.91 |

- **The early stop, not the model or the arithmetic, was behind both outliers.** Seed 1, trained to
  the end, lands on S2's seed 1: the same best epoch, 0.8 % lower validation loss, and the same
  test scores to 0.02 times the floor. With it, all seven `FNO2d` runs reproduce S2. NeuralOperator,
  trained to the end, halves its validation loss, from 0.0245 to 0.01225, inside the range of the
  six fully trained `FNO2d` runs (0.01169 to 0.01274, CUDA and S2).
- **Trained to the end, NeuralOperator is the best FNO here on in-range cases at 256x97.** It is
  2.26 times the floor on subcritical cases and 1.84 on supercritical ones. The six fully trained
  `FNO2d` runs score 5.28 to 5.50 and 2.38 to 2.54, PhysicsNeMo 5.53 and 3.45. In range it beats POD
  on supercritical cases (6.06) and comes within 7 % of it on subcritical ones (2.12). *(Outcome
  2026-10-24, five-seed entry below: over five seeds NeuralOperator averages 2.37 there, 1.94 to
  3.09; POD is ahead of four of the five.)* It keeps the smaller resolution transfer of the
  early-stopped run (0.0271 against 0.0346 to 0.0381). On its own 128x49 grid it is still a little
  behind `FNO2d` on subcritical cases (1.47 against 1.27 to 1.32).
- **In extrapolation it matches `FNO2d` on the near field but not on drag.** It is 3.59 times the
  floor (`FNO2d` 3.56 to 3.94). Its mean |CD error| is 139.0 counts (largest 243.6), against 79.3
  (183.1) for `FNO2d` seed 0 and 82.0 (171.7) for the seed-1 rerun. In range its mean |CL error|
  is the smallest measured here, 0.0066, against 0.0170 for `FNO2d` seed 0 and 0.0145 for the
  seed-1 rerun; its mean |CD error| is 13.9 counts, against 16.4 and 16.7. Command:
  `python -c "import json,numpy as np; R=json.load(open('results/fno_lib/scores.json'))['rows']; [print(v, s, sp, round(np.mean([abs(r['score']['cd_err_counts']) for r in x]),1), round(max(abs(r['score']['cd_err_counts']) for r in x),1), round(np.mean([abs(r['score']['cl_err']) for r in x]),4)) for v,s in (('apno',0),('apno-nopatience',1),('neuralop-nopatience',0)) for sp in ('test_in','test_ext') for x in [[r for r in R if r['variant']==v and r['seed']==s and r['n_train']==896 and r['grid']=='256x97' and r['split']==sp]]]"`.
- **Shocks.** The NeuralOperator rerun misses none of the 27 in-range shocks and none of the 21 in
  extrapolation; the seed-1 rerun misses 2 in range and none in extrapolation. Command:
  `python -c "import json; [print(g['variant'], g['seed'], g['split'], g['n_true_shocks'], g['n_missed']) for g in json.load(open('results/fno_lib/scores.json'))['shock_groups'] if g['grid']=='256x97' and g['n_train']==896 and 'nopatience' in g['variant']]"`.

**What this does and does not show.** Patience 30 ended two of the five 896-case runs of the first
round, at epochs 191 and 183, when the cosine schedule still had the learning rate at 0.29 and 0.33
of its starting value; one of them was a run S2 had trained to the end. Of S2's FNO runs only the
loss-B run was stopped early (81 epochs, best at 50), as the 2026-10-06 S4 entry reports; the loss-A
runs that S4's comparisons rest on all ran 300 epochs, so S4's results stand. The NeuralOperator
result is one seed, and the seed-1 runs show that a single run can move by 1.6 times its loss.
*(Outcome 2026-10-24, five-seed entry below: the in-range lead at 256x97 holds over five seeds of
each model.)* On in-range supercritical cases at 256x97 it is the best near field of any learned
model in this study, and on subcritical ones second to POD; it pays for it with larger drag errors
in extrapolation. Which of its differences from `FNO2d` gives it the smaller resolution transfer
(the channel MLP in each layer, the symmetric padding, the forward-normalised FFT, the
initialisation) is still not isolated. *(Corrected 2026-10-24: the forward-normalised FFT is not a
difference, since both normalisations scale the transform pair by the same 1/N; the fourth
difference is the two-layer lifting and projection. See the corrections entry of 2026-10-24.)*
Whether the seed-1 rerun retraced the early-stopped run's first 191 epochs is not recorded here: the
per-epoch logs are in the git-ignored `train.json` files.

## 2026-10-24 — NeuralOperator and `FNO2d` over five seeds each

The previous entry's NeuralOperator result rested on one seed. By the author, on the same Colab
A100 (A100-SXM4-80GB, torch 2.11.0+cu130, `NVIDIA_TF32_OVERRIDE` 0, as each record's `env` shows),
NeuralOperator seeds 1 to 4 and `FNO2d` seeds 3 and 4 were trained for the full 300 epochs:
`python scripts/probes/probe_fno_libraries.py --library L --n-train 896 --seed S --epochs 300 --patience 300 --suffix nopatience --device cuda`.
Scored with
`python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2`
(commit 0705cd1): 3,075 case scores, all `ok`; the 87 groups of the previous scoring (commit
3ed15ea) are unchanged, the largest difference 0.0. Command:
`git show 3ed15ea:results/fno_lib/scores.json > /tmp/fno_lib_3ed15ea.json && python -c "import json; k=lambda g:(g['method'],g['variant'],g['grid'],g['n_train'],g['seed'],g['split'],g['regime']); o={k(g):g for g in json.load(open('/tmp/fno_lib_3ed15ea.json'))['groups']}; n={k(g):g for g in json.load(open('results/fno_lib/scores.json'))['groups']}; print(len(o), max(abs(o[x][m]-n[x][m]) for x in o for m in o[x] if isinstance(o[x][m],float)))"`.

**The two sets.** Five runs of each model, all on the A100, all 300 epochs. `FNO2d`: seeds 0 and
2 (patience 30, which never fired, so their training is the same as with it off), and seeds 1, 3
and 4 with patience off. NeuralOperator: seeds 0 to 4 with patience off. Each comparison below is
a two-sided permutation test of the difference in means over all 252 ways of splitting the ten
runs into two sets of five. p = 0.008 is the smallest such a test can give; in every case here
where it is reached, the two sets of five do not overlap.

| quantity | `FNO2d`, seeds 0 to 4 | mean | NeuralOperator, seeds 0 to 4 | mean | p |
|---|---|---|---|---|---|
| best validation loss (128x49) | 0.01256, 0.01264, 0.01169, 0.01265, 0.01324 | 0.01255 | 0.01225, 0.01219, 0.01161, 0.01248, 0.01162 | 0.01203 | 0.119 |
| validation transfer | 0.0346, 0.0355, 0.0381, 0.0339, 0.0451 | 0.0374 | 0.0271, 0.0271, 0.0256, 0.0255, 0.0222 | 0.0255 | 0.008 |

Command:
`python -c "import json,itertools,numpy as np; r={x['tag']:x for x in json.load(open('results/fno_lib/scores.json'))['runs']}; F=['apno-n896-s0','apno-n896-s1-nopatience','apno-n896-s2','apno-n896-s3-nopatience','apno-n896-s4-nopatience']; N=[f'neuralop-n896-s{s}-nopatience' for s in range(5)]; P=lambda a,b: np.mean([abs(np.mean([(a+b)[i] for i in c])-np.mean([(a+b)[i] for i in range(10) if i not in c]))>=abs(np.mean(a)-np.mean(b))-1e-12 for c in itertools.combinations(range(10),5)]); [print(k, [round(r[t][k],5) for t in F], round(np.mean([r[t][k] for t in F]),5), [round(r[t][k],5) for t in N], round(np.mean([r[t][k] for t in N]),5), round(P([r[t][k] for t in F],[r[t][k] for t in N]),3)) for k in ('best_val','transfer_val_rel_l2')]"`.

Near-field relative L2 against the floor (SU2 at 256x97), per seed:

| grid | split, regime | `FNO2d` | mean | NeuralOperator | mean | p |
|---|---|---|---|---|---|---|
| 256x97 | test_in, subcritical | 5.38, 5.30, 5.50, 5.20, 5.33 | 5.34 | 2.26, 2.23, 2.32, 3.09, 1.94 | **2.37** | 0.008 |
| 256x97 | test_in, supercritical | 2.40, 2.52, 2.54, 2.33, 3.02 | 2.56 | 1.84, 1.82, 1.78, 1.84, 1.50 | **1.76** | 0.008 |
| 256x97 | test_ext, supercritical | 3.57, 3.94, 3.68, 4.45, 4.09 | 3.94 | 3.59, 3.77, 3.44, 4.26, 3.62 | 3.74 | 0.349 |
| 128x49 | test_in, subcritical | 1.32, 1.30, 1.27, 1.29, 1.29 | **1.29** | 1.47, 1.45, 1.41, 1.53, 1.39 | 1.45 | 0.008 |
| 128x49 | test_in, supercritical | 1.56, 1.65, 1.59, 1.54, 1.61 | 1.59 | 1.63, 1.73, 1.62, 1.65, 1.53 | 1.63 | 0.294 |
| 128x49 | test_ext, supercritical | 3.58, 3.83, 3.70, 4.36, 4.07 | 3.91 | 3.68, 3.92, 3.54, 4.43, 3.83 | 3.88 | 0.905 |

Command:
`python -c "import json,itertools,numpy as np; X={(r['variant'],r['seed'],r['grid'],r['split'],r['regime']):r['x_floor'] for r in json.load(open('results/fno_lib/scores.json'))['floor_ratios'] if r['n_train']==896}; F=[('apno',0),('apno-nopatience',1),('apno',2),('apno-nopatience',3),('apno-nopatience',4)]; N=[('neuralop-nopatience',s) for s in range(5)]; P=lambda a,b: np.mean([abs(np.mean([(a+b)[i] for i in c])-np.mean([(a+b)[i] for i in range(10) if i not in c]))>=abs(np.mean(a)-np.mean(b))-1e-12 for c in itertools.combinations(range(10),5)]); [print(g, sp, rg, [round(X[f+(g,sp,rg)],2) for f in F], round(np.mean([X[f+(g,sp,rg)] for f in F]),2), [round(X[n+(g,sp,rg)],2) for n in N], round(np.mean([X[n+(g,sp,rg)] for n in N]),2), round(P([X[f+(g,sp,rg)] for f in F],[X[n+(g,sp,rg)] for n in N]),3)) for g in ('256x97','128x49') for sp,rg in (('test_in','subcritical'),('test_in','supercritical'),('test_ext','supercritical'))]"`.

Forces at 256x97, per-seed means over the cases of each split:

| quantity | `FNO2d` mean (range) | NeuralOperator mean (range) | p |
|---|---|---|---|
| test_in mean \|CL error\| | 0.0166 (0.0145 to 0.0178) | **0.0075** (0.0061 to 0.0088) | 0.008 |
| test_in mean \|CD error\|, counts | 16.5 (13.7 to 18.5) | 15.3 (13.9 to 16.5) | 0.23 |
| test_ext mean \|CL error\| | 0.0505 (0.0443 to 0.0609) | 0.0516 (0.0431 to 0.0673) | 0.857 |
| test_ext mean \|CD error\|, counts | **83.8** (76.4 to 94.6) | 136.2 (104.6 to 177.9) | 0.008 |
| test_ext largest \|CD error\|, counts | **180.1** (162.6 to 197.6) | 258.4 (199.6 to 341.1) | 0.008 |

Command:
`python -c "import json,itertools,numpy as np; R=json.load(open('results/fno_lib/scores.json'))['rows']; F=[('apno',0),('apno-nopatience',1),('apno',2),('apno-nopatience',3),('apno-nopatience',4)]; N=[('neuralop-nopatience',s) for s in range(5)]; P=lambda a,b: np.mean([abs(np.mean([(a+b)[i] for i in c])-np.mean([(a+b)[i] for i in range(10) if i not in c]))>=abs(np.mean(a)-np.mean(b))-1e-12 for c in itertools.combinations(range(10),5)]); m=lambda v,s,sp,k,f: float(f([abs(r['score'][k]) for r in R if r['variant']==v and r['seed']==s and r['n_train']==896 and r['grid']=='256x97' and r['split']==sp])); [print(sp, k, f.__name__, [round(m(*x,sp,k,f),4) for x in F], round(np.mean([m(*x,sp,k,f) for x in F]),4), [round(m(*x,sp,k,f),4) for x in N], round(np.mean([m(*x,sp,k,f) for x in N]),4), round(P([m(*x,sp,k,f) for x in F],[m(*x,sp,k,f) for x in N]),3)) for sp,k,f in (('test_in','cl_err',np.mean),('test_in','cd_err_counts',np.mean),('test_ext','cl_err',np.mean),('test_ext','cd_err_counts',np.mean),('test_ext','cd_err_counts',np.max))]"`.

**Shocks** at 256x97: over the five seeds, NeuralOperator misses 1 of the 135 in-range shock
instances (5 x 27) and `FNO2d` 7; neither misses any of the 105 in extrapolation. Command:
`python -c "import json; G=json.load(open('results/fno_lib/scores.json'))['shock_groups']; [print(v, s, sp, g['n_true_shocks'], g['n_missed']) for v,s in [('apno',0),('apno-nopatience',1),('apno',2),('apno-nopatience',3),('apno-nopatience',4)]+[('neuralop-nopatience',s) for s in range(5)] for sp in ('test_in','test_ext') for g in G if g['variant']==v and g['seed']==s and g['grid']=='256x97' and g['n_train']==896 and g['split']==sp]"`.

- **The in-range lead at 256x97 holds over five seeds.** NeuralOperator is 2.37 times the floor
  on in-range subcritical cases and 1.76 on supercritical ones, against 5.34 and 2.56 for
  `FNO2d`, with no overlap between the two sets of five. Its in-range lift error is less than
  half of `FNO2d`'s, also without overlap.
- **It comes from the transfer, not from the fit.** On the 128x49 grid both models train on, the
  validation loss cannot be told apart (p 0.119), NeuralOperator is behind on in-range
  subcritical cases (1.45 against 1.29, no overlap) and level elsewhere. Its transfer from 128x49
  to 256x97 is smaller in all five seeds (0.0255 against 0.0374). The previous entry's single
  seed showed the same pattern.
- **In extrapolation it is level on the near field and worse on drag.** The near field and the
  lift cannot be told apart (p 0.349 and 0.857). Its mean drag error, 136.2 counts against 83.8,
  and its largest, 258.4 against 180.1, are larger in all five seeds.
- **POD remains ahead on in-range subcritical cases.** POD's 2.12 times the floor (S4,
  `pod-n896`) is below four of NeuralOperator's five seeds, whose mean is 2.37. On in-range
  supercritical cases NeuralOperator's 1.76 is far ahead of POD's 6.06.
- **Seed spread.** `FNO2d`'s five fully trained seeds give validation losses of 0.01169 to
  0.01324 and NeuralOperator's 0.01161 to 0.01248. On test cells the spread within one model
  reaches 1.94 to 3.09 times the floor (NeuralOperator, in-range subcritical). Single runs of
  either model should be read with that spread in mind.

**What this does and does not show.** Five seeds of each model, as the probe configures them
(positional embeddings off, modes matched to `FNO2d`'s, the same loop, loss and schedule), on one
GPU model. NeuralOperator's FNO gives a better near field than this study's `FNO2d` at the 256x97
evaluation grid on in-range cases, because less is lost between the 128x49 training grid and 256x97;
it does not fit the training grid better, and it gives larger drag errors in extrapolation. Which of
its differences causes the smaller transfer (the channel MLP in each layer, the symmetric padding,
the forward-normalised FFT, the initialisation) is still not isolated: that needs runs that change
one of them at a time. *(Outcome 2026-10-25, ablation entry below: it is the padding, and only
through its row count at 256x97.)* *(Corrected 2026-10-24: the forward-normalised FFT is not a
difference, since both normalisations scale the transform pair by the same 1/N; the fourth
difference is the two-layer lifting and projection. See the corrections entry of 2026-10-24.)*
PhysicsNeMo was run once, with early stopping on (it did not fire), and is not part of this
comparison. S4's results, which use S2's `FNO2d`, are unchanged.

## 2026-10-24 — corrections: what NeuralOperator's FNO differs in

Found while setting up the ablation of the transfer (`scripts/probes/fno_ablation.py`): the list
of NeuralOperator's differences from `FNO2d` given as candidates for its smaller transfer, in the
entries of 2026-10-22 (twice) and 2026-10-24 (five seeds), named one that is not a difference and
missed one that is. Each place is marked.

- **The FFT normalisation is not a difference.** NeuralOperator transforms with
  `norm="forward"` (1/N on the forward transform) and `FNO2d` with PyTorch's default
  `"backward"` (1/N on the inverse). A spectral layer multiplies between the two transforms, so
  the 1/N lands once either way and the output is the same. One NeuralOperator spectral layer
  under both, with the same weights: largest difference 1.3e-7 of the output's largest value,
  float32 rounding. Command:
  `python -c "import torch; from neuralop.layers.spectral_convolution import SpectralConv as S; torch.manual_seed(0); a=S(8,8,(8,6)); b=S(8,8,(8,6),fft_norm='backward'); b.load_state_dict(a.state_dict()); x=torch.randn(2,8,32,17); print(float(((a(x)-b(x)).abs().max()/a(x).abs().max()).detach()))"`.
  Pinned by `tests/test_fno_ablation.py::test_the_fft_normalisation_is_not_a_difference`.
- **The two-layer lifting and projection are a difference.** NeuralOperator lifts with a
  two-layer channel MLP (6 to 96 to 48, GELU) where `FNO2d` lifts linearly (6 to 48), and projects
  through 96 hidden channels where `FNO2d` uses 128.
- **The initialisation also places the bias differently.** NeuralOperator's spectral weights
  have a standard deviation of 0.144 at width 48 against `FNO2d`'s 4.3e-4, and its bias sits on
  the spectral output (drawn with the same deviation) where `FNO2d`'s sits on the skip
  convolution (PyTorch's default). The two placements give the same functions; only the starting
  values differ.

The differences are therefore four: the channel MLP in each layer (with its gated skip), the
symmetric padding, the spectral initialisation with its bias, and the two-layer lifting and
projection. `AblationFno` switches each on alone; tests show that with all four off it gives
`FNO2d`'s outputs and with all four on NeuralOperator's, from copied weights, at 128x49 and
256x97.

## 2026-10-25 — the FNO transfer ablation, phase 1: one row of padding at 256x97

The design, the runs and the planned analysis are in
`docs/notes/2026-10-24-fno-transfer-ablation.md`, written before the runs; no prediction was
recorded. By the author, on the Colab A100
(A100-SXM4-80GB, torch 2.11.0+cu130, `NVIDIA_TF32_OVERRIDE` 0 in every record): the two controls
`abl-none` (`FNO2d`) and `abl-MPIL` (NeuralOperator's FNO), and each of NeuralOperator's four
differences added alone to `FNO2d` (`abl-M` channel MLP, `abl-P` symmetric padding, `abl-I`
spectral initialisation, `abl-L` two-layer lifting and projection), seeds 0 to 4, 300 epochs,
early stopping off:
`python scripts/probes/probe_fno_libraries.py --library ablate --factors F --n-train 896 --seed S --epochs 300 --patience 300 --device cuda`.
Scored with
`python scripts/probes/score_fno_libraries.py --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2`
and analysed with `python scripts/probes/fno_ablation.py summary` (commit 9780063,
`results/fno_lib/ablation.md`): 7,575 case scores, all `ok`.

**A failure first.** The first scoring (commit c5314bf) was made with the 30 runs' prediction
files but without their training records, so the validation columns, the primary measure, were
blank, and nothing said so. The scorer and the summary now name such runs (commit 7cfd328); the
records were copied and everything rescored.

**The stop rule fired for one control.** The plan's rule: stop if `abl-none`'s mean validation
transfer falls outside `FNO2d`'s five-seed range (0.0339 to 0.0451) or `abl-MPIL`'s outside
NeuralOperator's (0.0222 to 0.0271). `abl-none` passes at 0.0375. `abl-MPIL` fails at 0.0281.
The two controls are not the same kind of check. `abl-none` builds its lifting, skips and
projection in `FNO2d`'s order and so draws exactly `FNO2d`'s initial values for them (only its
spectral weights, of deviation 4.3e-4, differ); it tracks `FNO2d` to 0.00029 in transfer on every
seed. NeuralOperator builds its blocks before its lifting, so `abl-MPIL` shares only the
projection's initial values with it and is a new sample of five; against NeuralOperator's five,
p = 0.111. A mean inside another sample's five-seed range was a stricter test than a resample can
be expected to pass. The rule is recorded as fired. The add-one effects below are measured
against `abl-none`, which passed; only remove-one comparisons would rest on `abl-MPIL`.
Commands:
`python -c "import sys,torch; sys.path.insert(0,'scripts/probes'); import probe_fno_libraries as p, fno_ablation as ab; from apno.models.train_fno import FnoConfig; c=FnoConfig(width=48,modes_i=24,modes_j=8,layers=4); d=lambda b: (torch.manual_seed(0), b())[1]; f,a,n,m=d(lambda: p.build_model('apno',c)),d(lambda: ab.AblationFno(c,'')),d(lambda: p.NeuralopFno(c)),d(lambda: ab.AblationFno(c,'MPIL')); print('FNO2d/abl-none lift, skips, projection, spectral:', torch.equal(f.lift.weight[...,0],a.lift.fcs[0].weight), all(torch.equal(f.pointwise[k].weight[...,0],a.skips[k].conv.weight) for k in range(4)), torch.equal(f.proj[0].weight[...,0],a.proj.fcs[0].weight), torch.equal(torch.cat([f.spectral[0].w_neg,f.spectral[0].w_pos],2),a.convs[0].weight.tensor)); B=n.net.fno_blocks; print('NeuralOperator/abl-MPIL lift, skips, projection, spectral:', torch.equal(n.net.lifting.fcs[0].weight,m.lift.fcs[0].weight), torch.equal(B.fno_skips[0].conv.weight,m.skips[0].conv.weight), torch.equal(n.net.projection.fcs[0].weight,m.proj.fcs[0].weight), torch.equal(B.convs[0].weight.tensor,m.convs[0].weight.tensor))"`
and
`python -c "import sys,json; sys.path.insert(0,'scripts/probes'); from fno_ablation import permutation_p as P; r={x['tag']:x for x in json.load(open('results/fno_lib/scores.json'))['runs']}; t=lambda tags: [r[x]['transfer_val_rel_l2'] for x in tags]; F=t(['apno-n896-s0','apno-n896-s1-nopatience','apno-n896-s2','apno-n896-s3-nopatience','apno-n896-s4-nopatience']); N=t([f'neuralop-n896-s{s}-nopatience' for s in range(5)]); A=lambda c: t([f'abl-{c}-n896-s{s}' for s in range(5)]); print('abl-none vs FNO2d, largest per-seed difference', round(max(abs(a-b) for a,b in zip(A('none'),F)),5)); print('abl-MPIL vs NeuralOperator p', round(P(A('MPIL'),N),3)); print('abl-P vs NeuralOperator p', round(P(A('P'),N),3), '| abl-P vs abl-MPIL p', round(P(A('P'),A('MPIL')),3)); print('best val abl-P minus abl-none per seed', [round(r[f'abl-P-n896-s{s}']['best_val']-r[f'abl-none-n896-s{s}']['best_val'],6) for s in range(5)])"`.

| configuration | validation transfer, seeds 0 to 4 | mean | best val loss | x floor 256x97 in range sub / sup | x floor 128x49 in range sub |
|---|---|---|---|---|---|
| `abl-none` (`FNO2d`) | 0.0348, 0.0355, 0.0380, 0.0342, 0.0451 | 0.0375 | 0.01262 | 5.36 / 2.56 | 1.29 |
| `abl-P` | 0.0206, 0.0205, 0.0216, 0.0203, 0.0245 | **0.0215** | 0.01263 | **2.18 / 1.68** | 1.30 |
| `abl-M` | 0.0463, 0.0491, 0.0450, 0.0526, 0.0589 | 0.0504 | 0.01270 | 5.76 / 2.92 | 1.34 |
| `abl-I` | 0.0467, 0.0348, 0.0338, 0.0371, 0.0493 | 0.0404 | 0.01313 | 4.95 / 2.84 | 1.39 |
| `abl-L` | 0.0431, 0.0435, 0.0424, 0.0484, 0.0474 | 0.0449 | 0.01451 | 4.80 / 2.77 | 1.40 |
| `abl-MPIL` (NeuralOperator) | 0.0299, 0.0298, 0.0301, 0.0256, 0.0249 | 0.0281 | 0.01171 | 2.52 / 1.89 | 1.47 |

Commands: transfer and validation loss,
`python -c "import json,numpy as np; r={x['tag']:x for x in json.load(open('results/fno_lib/scores.json'))['runs']}; [print(c, [round(r[f'abl-{c}-n896-s{s}']['transfer_val_rel_l2'],4) for s in range(5)], round(np.mean([r[f'abl-{c}-n896-s{s}']['transfer_val_rel_l2'] for s in range(5)]),4), round(np.mean([r[f'abl-{c}-n896-s{s}']['best_val'] for s in range(5)]),5)) for c in ('none','P','M','I','L','MPIL')]"`;
near field,
`python -c "import sys,json,numpy as np; sys.path.insert(0,'scripts/probes'); from fno_ablation import permutation_p as P; X={(r['variant'],r['seed'],r['grid'],r['split'],r['regime']):r['x_floor'] for r in json.load(open('results/fno_lib/scores.json'))['floor_ratios'] if r['n_train']==896}; S={c:[(f'abl-{c}',s) for s in range(5)] for c in ('none','P','M','I','L','MPIL')}; S['neuralop']=[('neuralop-nopatience',s) for s in range(5)]; v=lambda k,g,sp,rg: [X[a+(g,sp,rg)] for a in S[k]]; [print(g, sp, rg, {k: round(float(np.mean(v(k,g,sp,rg))),2) for k in S}, 'p P-none', round(P(v('P',g,sp,rg),v('none',g,sp,rg)),3), 'p P-neuralop', round(P(v('P',g,sp,rg),v('neuralop',g,sp,rg)),3)) for g,sp,rg in (('256x97','test_in','subcritical'),('256x97','test_in','supercritical'),('256x97','test_ext','supercritical'),('128x49','test_in','subcritical'))]"`.
The add-one effects and their p values are in `results/fno_lib/ablation.md`.

- **The padding alone accounts for the transfer.** Adding P to `FNO2d` lowers the mean transfer
  from 0.0375 to 0.0215 (p = 0.008, no overlap). That is more than the 0.0119 gap it was meant to
  explain: `abl-P` transfers better than NeuralOperator's FNO (0.0255, p = 0.016) and than
  `abl-MPIL` (p = 0.008). On the test cases it takes in-range subcritical cases at 256x97 from
  5.36 to 2.18 times the floor and supercritical ones from 2.56 to 1.68 (both p = 0.008), level
  with NeuralOperator (p = 0.492 and 0.31).
- **The other three, alone, leave the transfer as it is or make it worse:** the channel MLP by
  0.0128 (p = 0.016), the two-layer lifting and projection by 0.0074 (p = 0.032), the
  initialisation by 0.0028 (p = 0.444). They are where NeuralOperator's small deficit on the
  128x49 training grid comes from (1.45 times the floor on in-range subcritical cases, against
  1.30 for `abl-P`, p = 0.008; 1.34, 1.39 and 1.40 for M, I and L alone).
- **P does not change the training at all, only the evaluation.** Every layer commutes with a
  circular shift of the padded radial column, and the radial FFT treats that column as periodic.
  So 4 + 4 rows at 128x49 are the same operator as `FNO2d`'s 8 at the far end, and 8 + 8 at
  256x97 the same as 16 at the far end. With the same weights, the outputs agree to 5.4e-7 in
  both cases, against 1.3e-3 for 8 + 8 against `FNO2d`'s 17 at 256x97. Pinned by
  `tests/test_fno_ablation.py::test_p_is_a_circular_shift_of_one_sided_padding_with_the_same_row_count`.
  Command:
  `python -c "import sys,torch,torch.nn.functional as F; sys.path.insert(0,'scripts/probes'); import fno_ablation as ab; from apno.models.train_fno import FnoConfig; c=FnoConfig(width=48,modes_i=24,modes_j=8,layers=4); torch.manual_seed(0); a=ab.AblationFno(c,'').eval(); torch.manual_seed(0); b=ab.AblationFno(c,'P').eval(); [m.convs[k].weight.tensor.data.mul_(100) for m in (a,b) for k in range(4)]; e=lambda u,v: float((u-v).abs().max()/v.abs().max()); x1,x2=torch.randn(2,6,128,49),torch.randn(2,6,256,97); y1,y2=b(x1).detach(),b(x2).detach(); print('4+4 vs 8 at 128x49', e(y1,a(x1).detach()), '| 8+8 vs 17 at 256x97', e(y2,a(x2).detach())); a.padding.pad=lambda x: (setattr(a.padding,'_nj',x.shape[-1]), F.pad(x,(0,16)))[1]; print('8+8 vs 16 at 256x97', e(y2,a(x2).detach()))"`.
  P adds no parameters, so `abl-P` and `abl-none` draw the same initial weights and train the same
  way: their best validation losses differ by at most 1.2e-4 per seed (GPU arithmetic is not
  bit-reproducible), and on the 128x49 test grid they score 1.30 and 1.29 (p = 0.841). `abl-P` is
  `abl-none`'s models evaluated at 256x97 with 16 padding rows instead of 17.
- **The forces move both ways.** In range at 256x97, `abl-P`'s mean |CL error| falls from 0.0168
  to 0.0102 and its mean |CD error| rises from 16.7 to 20.1 counts (both p = 0.008). In
  extrapolation its mean |CD error| is 101.2 counts against 84.5 (p = 0.063), and its near field
  is level (3.81 against 3.97 times the floor, p = 0.444). Command:
  `python -c "import sys,json,numpy as np; sys.path.insert(0,'scripts/probes'); from fno_ablation import permutation_p as P; R=json.load(open('results/fno_lib/scores.json'))['rows']; S={c:[(f'abl-{c}',s) for s in range(5)] for c in ('none','P','MPIL')}; S['neuralop']=[('neuralop-nopatience',s) for s in range(5)]; m=lambda v,s,sp,k,f: float(f([abs(r['score'][k]) for r in R if r['variant']==v and r['seed']==s and r['n_train']==896 and r['grid']=='256x97' and r['split']==sp])); q=lambda c,sp,k,f: [m(v,s,sp,k,f) for v,s in S[c]]; [print(sp, k, f.__name__, {c: round(float(np.mean(q(c,sp,k,f))),4) for c in S}, 'p P-none', round(P(q('P',sp,k,f),q('none',sp,k,f)),3)) for sp,k,f in (('test_in','cl_err',np.mean),('test_in','cd_err_counts',np.mean),('test_ext','cd_err_counts',np.mean),('test_ext','cd_err_counts',np.max))]"`.

**What this does and does not show.** The smaller transfer of NeuralOperator's FNO comes from its
radial padding rule, and only through the number of padding rows at the evaluation grid: 16 at
256x97 where `FNO2d` uses 17. `FNO2d`'s rule was chosen (design §6.3, amended 2026-10-01) to keep
the radial DFT period the same at both grids, 19/16 in η, on the argument that a learned radial mode
then means the same physical mode, supported by a one-layer test on a wrap-periodic field. On the
trained networks the matched period transfers worse than the 0.88 % mismatch of 16 rows (period
113/96). Why is not shown here. Nor is 16 shown to be the best count: only 16 and 17 were evaluated.
*(Outcome 2026-10-26, padding-sweep entry below: of 0 to 34 rows, 16 gives the smallest transfer for
all 18 checkpoints swept.)* Because P changes only the evaluation, other counts can be measured on
the existing checkpoints without training, and NeuralOperator's model without P (phase 2's
`abl-MIL`) needs no runs either. The design's padding rule and S4's results are unchanged: S4
measured `FNO2d` as designed. These findings say that its error at 256x97, and the resolution
transfer S4 charged to it, depend on a choice of one padding row.

## 2026-10-26 — the FNO padding sweep at 256x97: 16 rows, and two measures that disagree

The ablation found that the transfer depends on the number of radial padding rows at 256x97 and on
nothing in training. The sweep measures that dependence on existing checkpoints, without training
(`scripts/probes/probe_fno_pad_sweep.py`; design and tests in
`docs/notes/2026-10-24-fno-transfer-ablation.md`). By the author, on the Mac's Apple GPU:
`python scripts/probes/probe_fno_pad_sweep.py` (commit 8e796a6, `results/fno_lib/pad_sweep.md`,
`.json` and `.png`), 1,168 s in all. Eighteen checkpoints: S2's `fno-n896-s0` to `s2` (`FNO2d`,
trained on the Mac), and five seeds each of `abl-none` (`FNO2d`), `abl-MPIL` and
`neuralop-n896-*-nopatience` (NeuralOperator's FNO), trained on the A100. Each is evaluated on the
126 validation cases at 256x97 with 0 to 34 padding rows. Its 128x49 prediction keeps its own 8
rows. Two measures: the transfer, as S2 records it; and the error against SU2 at 256x97 (relative
L2 over the whole field in target space, per case and channel, then averaged), as the probe
records it. The test set is not read.

| group | checkpoints | 128x49 error | error at 256x97, rows 0 / 15 / 16 / 17 / 18 / 34 | transfer, rows 0 / 15 / 16 / 17 / 18 / 34 |
|---|---|---|---|---|
| S2 `fno-n896` (own rows 17) | 3 | 0.0124 | 0.720 / 0.0846 / **0.0411** / 0.0411 / 0.0859 / 0.532 | 0.714 / 0.0731 / **0.0209** / 0.0360 / 0.0896 / 0.527 |
| `abl-none` (`FNO2d`, own rows 17) | 5 | 0.0126 | 0.720 / 0.0845 / **0.0410** / 0.0421 / 0.0874 / 0.535 | 0.715 / 0.0726 / **0.0215** / 0.0375 / 0.0916 / 0.531 |
| `abl-MPIL` (own rows 16) | 5 | 0.0117 | 0.920 / 0.0998 / **0.0420** / 0.0476 / 0.1027 / 0.654 | 0.908 / 0.0912 / **0.0281** / 0.0450 / 0.1063 / 0.642 |
| NeuralOperator (own rows 16) | 5 | 0.0120 | 0.946 / 0.0976 / **0.0412** / 0.0475 / 0.1024 / 0.641 | 0.940 / 0.0887 / **0.0255** / 0.0446 / 0.1065 / 0.631 |

Command (group means):
`python -c "import json; d=json.load(open('results/fno_lib/pad_sweep.json')); r=d['rows']; [print(g, c['n'], round(c['val_rel_l2_128x49'],4), [round(c['val_rel_l2_256x97'][r.index(k)],4) for k in (0,15,16,17,18,34)], [round(c['transfer'][r.index(k)],4) for k in (0,15,16,17,18,34)]) for g,c in sorted(d['groups'].items())]"`.

- **16 rows gives the smallest transfer for every checkpoint.** All 18 are lowest at 16 among the
  35 counts. One row more raises the transfer by 1.33 to 1.94 times, one row fewer by 2.99 to
  3.67 times, two more by 3.31 to 4.55 times. The curve is a sharp, narrow minimum, and with no
  padding the transfer is 0.71 to 0.94. Commands:
  `python -c "import json,collections; d=json.load(open('results/fno_lib/pad_sweep.json')); b=lambda x,m: int(min(x['rows'], key=lambda k: x['rows'][k][m])); [print(m, sorted(collections.Counter(b(x,m) for x in d['records']).items()), [x['tag'] for x in d['records'] if b(x,m)!=16]) for m in ('transfer','val_rel_l2_256x97')]"`
  and
  `python -c "import json; d=json.load(open('results/fno_lib/pad_sweep.json')); t=lambda x,k: x['rows'][str(k)]['transfer']; v=[(t(x,15)/t(x,16), t(x,17)/t(x,16), t(x,18)/t(x,16)) for x in d['records']]; print([round(min(a),2) for a in zip(*v)], [round(max(a),2) for a in zip(*v)])"`.
- **16 is the count the design's first rule gave.** (NJ − 1)/6 rows is 8 at 128x49 and 16 at
  256x97. The 2026-10-01 amendment (design §6.3) replaced it with `radial_pad`, 8 and 17, to keep
  the radial DFT period the same at both grids. The amendment changed the 256x97 count from 16 to
  17, and with it `FNO2d`'s transfer from about 0.021 to about 0.036. Command:
  `python -c "from apno.models.fno import radial_pad; print([(nj, (nj-1)//6, radial_pad(nj)) for nj in (49, 97, 193)])"`.
- **The error against SU2 at 256x97 agrees for NeuralOperator's models but hardly separates 16
  from 17 for `FNO2d`.** It is lowest at 16 for 14 of the 18 checkpoints and at 17 for the other
  four (S2's seed 0, `abl-none`'s seeds 0 and 3, `abl-MPIL`'s seed 0). For NeuralOperator's
  models, 16 rows give 0.0412 to 0.0420 and 17 rows 0.0475 to 0.0476. For `FNO2d`, 16 and 17
  rows give 0.0411 and 0.0411 (S2) and 0.0410 and 0.0421 (`abl-none`). Away from 16 and 17 the
  error rises as steeply as the transfer.
- **That disagrees with the test cases.** The ablation's `abl-P` is `abl-none`'s models evaluated
  with 16 rows instead of 17. On the test cases at 256x97, S4's near-field measure took them from
  5.36 to 2.18 times the floor on in-range subcritical cases (2026-10-25). The validation error
  above barely moves for the same change. The two measures differ in several ways. The near field is
  weighted by node area within 2.5 chords of mid-chord, with the free stream subtracted from the
  state, against the 512x193 truth. The validation error weights every node of the whole field
  equally, in target space, against SU2 at 256x97, and divides by each case's and channel's norm.
  Which of these differences decides it is not established. *(Outcome 2026-10-26, validation
  near-field entry below: S4's near-field measure on the validation cases agrees with the test
  cases, so it is the whole-field measure that does not separate 16 from 17 rows.)* The S2
  checkpoints behave like `abl-none`'s throughout.

**What this does and does not show.** Among 0 to 34 rows at 256x97, 16 gives the smallest transfer
for all 18 checkpoints of both models. 16 is also the count of the rule the design used before its
2026-10-01 amendment. Whether 16 also gives a smaller error at 256x97 than `FNO2d`'s 17 depends on
the measure. On the test cases' near field it does, by 2.4 times. On the validation cases'
whole-field error it does not, within 0.001. Measuring the validation near field the way S4 measures
the test near field would show which. *(Made 2026-10-26, next entry: on the validation near field,
16 rows is better for `FNO2d` too, by 1.65 to 2.26 times.)* Only the 256x97 grid was evaluated, at
one training grid (128x49). The design's padding rule, S2's predictions and S4's results are
unchanged. Whether to amend the rule is the author's decision.

## 2026-10-26 — the padding sweep's validation near field: 16 rows for `FNO2d` too

The previous entry left a disagreement open. For `FNO2d`, the whole-field validation error tied 16
and 17 padding rows at 256x97. Yet S4's near-field measure on the test cases separated them 2.4
times. The sweep now also records S4's near-field measure on the validation cases
(`score_case`'s `near_field_rel_l2`, against SU2 at 256x97, the only solution the validation
cases have), overall and by regime. 76 of the 126 cases are subcritical by SU2's own largest
surface Mach number. A test checks the measure against `score_case` case by case. By the author,
on the Mac's Apple GPU: `python scripts/probes/probe_fno_pad_sweep.py` (commit 2b18c0a), the same
18 checkpoints and 0 to 34 rows. The first sweep's two measures came out the same for every
checkpoint and row count (largest difference 0.0). Command:
`git show 8e796a6:results/fno_lib/pad_sweep.json > /tmp/pad_sweep_8e796a6.json && python -c "import json; a={x['tag']:x for x in json.load(open('/tmp/pad_sweep_8e796a6.json'))['records']}; b={x['tag']:x for x in json.load(open('results/fno_lib/pad_sweep.json'))['records']}; print(len(a), max(abs(a[t]['rows'][k][m]-b[t]['rows'][k][m]) for t in a for k in a[t]['rows'] for m in ('val_rel_l2_256x97','transfer')))"`.

| group | near field at 128x49: all / sub / sup | near field at 256x97, 16 rows: all / sub / sup | 17 rows: all / sub / sup |
|---|---|---|---|
| S2 `fno-n896` (own rows 17) | 0.0136 / 0.0057 / 0.0257 | **0.0195 / 0.0126 / 0.0299** | 0.0423 / 0.0380 / 0.0489 |
| `abl-none` (`FNO2d`, own rows 17) | 0.0139 / 0.0056 / 0.0263 | **0.0201 / 0.0129 / 0.0311** | 0.0437 / 0.0379 / 0.0525 |
| `abl-MPIL` (own rows 16) | 0.0143 / 0.0078 / 0.0242 | **0.0263 / 0.0188 / 0.0377** | 0.0501 / 0.0467 / 0.0553 |
| NeuralOperator (own rows 16) | 0.0143 / 0.0076 / 0.0245 | **0.0244 / 0.0174 / 0.0352** | 0.0513 / 0.0481 / 0.0563 |

Command (group means, rows 15 to 18):
`python -c "import json; d=json.load(open('results/fno_lib/pad_sweep.json')); r=d['rows']; [print(g, c['n'], [round(c[m+'_128x49'],4) for m in ('near_field','near_field_subcritical','near_field_supercritical')], [[round(c[m][r.index(k)],4) for k in (15,16,17,18)] for m in ('near_field','near_field_subcritical','near_field_supercritical')]) for g,c in sorted(d['groups'].items())]"`.

- **On the validation near field, 16 rows is best for every checkpoint, `FNO2d` included.** All
  18 are lowest at 16, overall and in each regime. Going from 16 to 17 rows raises the near-field
  error 1.65 to 2.26 times over all cases, 2.22 to 3.19 times on subcritical cases and 1.18 to
  1.93 times on supercritical ones. That matches the test cases, where the same change took
  `FNO2d` from 2.18 to 5.36 times the floor on in-range subcritical cases (2026-10-25). The
  disagreement is the whole-field measure's alone. Command:
  `python -c "import json,collections; d=json.load(open('results/fno_lib/pad_sweep.json')); b=lambda x,m: int(min(x['rows'], key=lambda k: x['rows'][k][m])); [print(m, sorted(collections.Counter(b(x,m) for x in d['records']).items()), round(min(x['rows']['17'][m]/x['rows']['16'][m] for x in d['records']),2), round(max(x['rows']['17'][m]/x['rows']['16'][m] for x in d['records']),2)) for m in ('near_field','near_field_subcritical','near_field_supercritical')]"`.
- **At 16 rows, `FNO2d` has the smaller near-field error.** At its own 17 rows, `FNO2d` was
  behind NeuralOperator's model; evaluated with the same 16 it is ahead. Its mean is 0.0201
  against 0.0244 over all cases (p = 0.016), 0.0129 against 0.0174 on subcritical cases (p =
  0.008, no overlap), and 0.0311 against 0.0352 on supercritical ones (p = 0.071). This fits
  the ablation: NeuralOperator's channel MLP, initialisation and lifting cost it a little on the
  training grid, and its advantage at 256x97 came from the padding alone. Command:
  `python -c "import sys,json; sys.path.insert(0,'scripts/probes'); from fno_ablation import permutation_p as P; R=json.load(open('results/fno_lib/pad_sweep.json'))['records']; v=lambda g,m: [x['rows']['16'][m] for x in R if x['group']==g]; [print(m, [round(a,4) for a in v('abl-none-n896',m)], [round(a,4) for a in v('neuralop-n896-nopatience',m)], round(P(v('abl-none-n896',m), v('neuralop-n896-nopatience',m)),3)) for m in ('near_field','near_field_subcritical','near_field_supercritical','transfer')]"`.
- **16 rows reduces the loss between the grids; it does not remove it.** At 16 rows, `FNO2d`'s
  near-field error at 256x97 is still 2.3 times its error at 128x49 on subcritical cases (0.0129
  against 0.0056), and 1.2 times on supercritical ones.

**What this does and does not show.** All three measures that look at the near field agree: the
transfer, the validation near field and the test near field. On each, 16 padding rows at 256x97
beat the design's 17 for both models. The whole-field validation error does not separate the
two for `FNO2d`. Why it does not is not shown here: it weights every node of the 100-chord grid
equally, in target space, per case and channel. The test cases were not reread for this entry.
The design's rule, S2's predictions and S4's results are unchanged. Whether to amend the rule,
and to rescore `FNO2d` at 16 rows, is the author's decision.

## 2026-10-26 — S2's `FNO2d` rescored on the test cases with 16 padding rows at 256x97

The sweep found 16 padding rows at 256x97 better than the design's 17 for every checkpoint, on
the validation cases. This entry scores S2's own models on the test cases with 16 rows. Nothing
is retrained, since the padding does not change training at 128x49. By the author, on the Mac's
Apple GPU: `python scripts/probes/probe_fno_pad_predict.py` re-predicted the 77 test cases of S2's
seven loss-A runs (three seeds at 896 cases; 64, 128, 256 and 512 cases at seed 0) with 16 rows,
into `runs/predictions_pad16/`. Then
`python scripts/probes/score_fno_libraries.py --pred-dir runs/predictions_pad16 --out results/fno_pad16 --reference fno-n896-s0 --reference fno-n896-s1 --reference fno-n896-s2 --reference fno-n64-s0 --reference fno-n128-s0 --reference fno-n256-s0 --reference fno-n512-s0`
scored them beside S2's own 17-row files (variant `s2`) with S4's code (commit fc84fcf,
`results/fno_pad16/`). 1,650 case scores, all `ok`.

**The re-evaluation reproduces S2's pipeline.** Each checkpoint also predicted at its own 17 rows.
That matched S2's committed prediction file exactly (largest relative difference 0.0) for all
seven, on all 77 cases. So the only change below is the row count. Command:
`python -c "import json; p=json.load(open('results/fno_pad16/predict.json')); [print(r['tag'], r['native_rows'], r['n_cases'], r['reproduces_s2']['max_rel_diff'], r['reproduced']) for r in p['records']]"`.

Near field against the floor at 256x97, 896 cases, seeds 0, 1 and 2:

| split, regime | 17 rows (S2, as in S4) | 16 rows | 128x49, for reference | POD at 256x97 (S4) |
|---|---|---|---|---|
| test_in, subcritical | 5.32, 5.28, 5.50 | **2.12, 2.03, 2.27** | 1.28, 1.30, 1.27 | 2.12 |
| test_in, supercritical | 2.38, 2.52, 2.54 | **1.62, 1.70, 1.69** | 1.55, 1.65, 1.58 | 6.06 |
| test_ext, supercritical | 3.56, 3.92, 3.73 | **3.44, 3.74, 3.58** | 3.56, 3.81, 3.74 | 7.91 |

Command:
`python -c "import json,numpy as np; X={(r['variant'],r['n_train'],r['seed'],r['grid'],r['split'],r['regime']):r['x_floor'] for r in json.load(open('results/fno_pad16/scores.json'))['floor_ratios']}; [print(sp, rg, [round(X[('s2',896,s,'256x97',sp,rg)],2) for s in range(3)], [round(X[('pad16',896,s,'256x97',sp,rg)],2) for s in range(3)], [round(X[('s2',896,s,'128x49',sp,rg)],2) for s in range(3)]) for sp,rg in (('test_in','subcritical'),('test_in','supercritical'),('test_ext','supercritical'))]"`.

- **Each of the three models improves in every cell.** On in-range subcritical cases the mean
  goes from 5.37 to 2.14 times the floor. POD's 2.12 had beaten `FNO2d` there in S4; at 16 rows
  the two are level. On in-range supercritical cases the mean goes from 2.48 to 1.67, close to the
  model's own 1.59 at 128x49. In extrapolation it moves least, from 3.74 to 3.59.
- **Shocks.** At 17 rows the three models miss 5 of their 81 in-range shock instances (3 x 27).
  At 16 rows they miss none. Neither misses any of the 63 in extrapolation. Command:
  `python -c "import json; [print(g['variant'], g['seed'], g['split'], g['n_true_shocks'], g['n_missed']) for g in json.load(open('results/fno_pad16/scores.json'))['shock_groups'] if g['grid']=='256x97' and g['n_train']==896]"`.
- **The forces move both ways.** In range, the mean |CL error| falls from 0.0145–0.0173 to
  0.0085–0.0108 and the mean |CM error| from 0.0027–0.0035 to 0.0017–0.0020. The mean |CD error|
  rises from 16.3–17.3 to 19.2–20.6 counts, and its largest from 35.9–40.4 to 46.6–64.6. The
  mean largest wall |ΔCp| rises from 0.31–0.35 to 0.37–0.41. In extrapolation, lift and moment
  improve slightly, while the mean |CD error| rises from 79.3–85.4 to 98.5–99.3 counts. Ranges
  are over the three seeds. Command:
  `python -c "import json,numpy as np; R=json.load(open('results/fno_pad16/scores.json'))['rows']; m=lambda v,s,sp,k,f: float(f([abs(r['score'][k]) for r in R if r['variant']==v and r['n_train']==896 and r['seed']==s and r['grid']=='256x97' and r['split']==sp])); [print(sp, k, f.__name__, [round(m('s2',s,sp,k,f),4) for s in range(3)], [round(m('pad16',s,sp,k,f),4) for s in range(3)]) for sp in ('test_in','test_ext') for k,f in (('cl_err',np.mean),('cm_err',np.mean),('cd_err_counts',np.mean),('cd_err_counts',np.max),('cp_max',np.mean))]"`.

The data-efficiency curve (S4's: mean near-field relative L2 over the 61 test_in cases at
256x97, seed 0, floor 0.01432):

| n | 17 rows (S4) | 16 rows | 16 rows / floor | 128x49 (17 rows' model) | POD (S4) |
|---|---|---|---|---|---|
| 64 | 0.1088 | **0.0918** | 6.41 | 0.0900 | 0.1379 |
| 128 | 0.0857 | **0.0634** | 4.43 | 0.0600 | 0.1044 |
| 256 | 0.0698 | **0.0460** | 3.21 | 0.0418 | 0.0883 |
| 512 | 0.0561 | **0.0310** | 2.16 | 0.0264 | 0.0753 |
| 896 | 0.0483 | **0.0256** | 1.79 | 0.0209 | 0.0677 |

Command:
`python -c "import json,numpy as np; R=json.load(open('results/fno_pad16/scores.json'))['rows']; S=json.load(open('results/s4/scores.json'))['rows']; f=lambda rows,meth,v,n,g: np.mean([r['score']['near_field_rel_l2'] for r in rows if r['method']==meth and r['variant']==v and r['n_train']==n and r['seed']==0 and r['grid']==g and r['split']=='test_in']); fl=np.mean([r['score']['near_field_rel_l2'] for r in R if r['variant']=='floor' and r['split']=='test_in']); print('floor', round(fl,5)); [print(n, round(f(R,'fno','s2',n,'256x97'),4), round(f(R,'fno','pad16',n,'256x97'),4), round(f(R,'fno','s2',n,'128x49'),4), round(f(S,'pod','',n,'256x97'),4), round(f(R,'fno','pad16',n,'256x97')/fl,2)) for n in (64,128,256,512,896)]"`.

- **The gain grows with the training set.** 16 rows lowers the error by 16 % at 64 cases and by
  47 % at 896, and the curve follows the model's own 128x49 curve more closely as n grows. At
  896 cases `FNO2d` is 1.79 times the floor at 256x97, where S4 recorded 3.4. It stays below POD
  at every n, as at 17 rows.

**What this does and does not show.** S2's trained `FNO2d`, evaluated at 256x97 with one padding row
fewer than the design's rule, has about half the near-field error on in-range test cases. It is
level with POD on subcritical ones and misses no in-range shock. In extrapolation it barely changes,
and its drag error rises in and out of range. Nothing was retrained or retuned. The row count was
identified by the ablation's validation transfer and confirmed on the validation near field, but the
test cases were not unseen when it was chosen: the ablation had already scored its own `FNO2d`
models on them at 16 and 17 rows (2026-10-25). The costs, and so the break-even of Prediction 2, do
not change: one padding row fewer is a slightly cheaper evaluation, not a different model. S4's
records and the README's summary describe `FNO2d` as designed, with 17 rows. They are unchanged, and
so is the design's rule. Amending the rule, and with it S4's figures for `FNO2d` at 256x97, is the
author's decision.

## 2026-10-26 — the padding rule amended to 16 rows: S2 re-predicted, S4 and S5 rescored

**The decision.** The author amended the design's radial padding rule to (NJ − 1)/6 rows, 8 at
128x49 and 16 at 256x97 (spec §6.3, amended 2026-10-26; commit 7b39c58). The evidence is the three
entries above. The test cases had been scored at 16 rows before the choice was made, so the test
figures below are not a blind test of it. The old rule is kept as `radial_pad_period_matched` for the
probes whose records used it.

**S2 re-predicted, not retrained.** Both rules pad 8 rows at 128x49, where `FNO2d` trains, so S2's
checkpoints, training costs and 128x49 predictions stand. By the author, on the Mac's Apple GPU:
`python scripts/s2_operator.py repredict` remade the 256x97 test predictions of S2's eight final
runs (`results/s2/repredict.log`, `repredict_2026-10-26.json`). The 17-row files are kept in
`runs/predictions_rows17/`.

- Before each file was replaced, its own checkpoint reproduced it under the old rule exactly
  (largest relative difference 0.0, at 256x97 and 128x49, all eight).
- The largest relative change at 16 rows is 0.136 to 0.168 for the loss-A runs and 0.289 for loss B.
  Command: `python -c "import json; [print(e['tag'], e['reproduces_file_at_old_rows'], round(e['max_rel_change'],3)) for e in json.load(open('results/s2/repredict_2026-10-26.json'))['files']]"`.
- **The validation transfer** (S2 entry, 2026-10-05) falls from 0.0348 / 0.0355 / 0.0379 to
  0.0207 / 0.0206 / 0.0215 for seeds 0 / 1 / 2 at 896 cases. Loss B's goes from 0.0595 to 0.0243.
  At 64, 128, 256 and 512 cases it goes from 0.0443 / 0.0476 / 0.0474 / 0.0431 to
  0.0255 / 0.0271 / 0.0268 / 0.0254. `runs.json` keeps the old values as
  `transfer_val_rel_l2_rows17`. The tuning runs' transfers in `tuning.json` stay at 17 rows. The
  tuning selected on validation loss at 128x49, which the padding does not change. Command:
  `python -c "import json; [print(r['tag'], round(r['transfer_val_rel_l2_rows17'],4), round(r['transfer_val_rel_l2'],4)) for r in json.load(open('results/s2/runs.json'))]"`.

Then `python scripts/s4_score.py`, `python scripts/s5_score.py` and
`python docs/report/make_figures.py` (commit 19199e5). S4's FNO groups at 256x97 equal the
2026-10-26 rescore's `pad16` groups exactly, for the seven loss-A runs. Loss B is new here.
`costs.json` and `break_even.png` did not change, and neither did anything at 128x49 or for SU2,
POD or the PINN.

**S4 at 256x97, 896 cases, seed 0, beside the floor and POD** (17-row values from the S4 entry in
brackets):

| set / regime | near field | ×floor | CL max | CD max (counts) | CD median (counts) | Cp RMS |
|---|---|---|---|---|---|---|
| test_in subcritical | 0.0169 (0.0423) | 2.1 (5.3) | 0.0181 (0.0312) | 41.6 (31.3) | −17.1 (+16.5) | 0.0432 (0.0358) |
| test_in supercritical | 0.0391 (0.0575) | 1.6 (2.4) | 0.0403 (0.0431) | 64.6 (40.4) | −19.2 (+12.6) | 0.0491 (0.0521) |
| test_ext supercritical | 0.2238 (0.2318) | 3.4 (3.6) | 0.1835 (0.1847) | 203.5 (180.6) | −101.8 (−78.2) | 0.1933 (0.2038) |

Floor: near field 0.0080, 0.0241 and 0.0651. POD: 0.0168, 0.1462 and 0.5151. Command:
`python -c "import json; G=json.load(open('results/s4/scores.json'))['groups']; [print(x['method'], x['variant'], x['grid'], x['seed'], x['split'], x['regime'], round(x['near_field_rel_l2_mean'],4), round(x['cl_err_absmax'],4), round(x['cd_err_counts_absmax'],1), round(x['cd_err_counts_median'],1), round(x['cp_l2_mean'],4)) for x in G if x['n_train'] in (0, 896) and x['method'] in ('fno', 'pod', 'su2-256x97')]"`.
The bracketed values are the same command on the file at commit 7b39c58
(`git show 7b39c58:results/s4/scores.json`).

- **In range, subcritical, the FNO at 256x97 is level with POD on the near field** (2.1x the floor
  each; 0.0169 against 0.0168). POD's lift and drag errors are smaller (CL max 0.0097, CD max 14.2
  counts). At 17 rows POD beat it on all three.
- **In range, supercritical, the FNO beats POD by 3.7x on the near field** (2.5x at 17 rows) and
  1.9x on the CD maximum (3x at 17 rows).
- **In extrapolation it barely moves**: 3.4x the floor against POD's 7.9x. Its largest drag error
  rises from 181 to 204 counts.
- **The in-range drag error changes sign with the padding.** At 256x97 the median in-range CD error
  is +15.0 / +16.9 / +17.4 counts at 17 rows (5 of 61 cases negative, every seed), and
  −17.4 / −17.9 / −16.9 at 16 rows (60 of 61 negative). At 128x49 the same models' median is −0.2 /
  +0.1 / +0.4 counts, and the floor's +0.2. So at 256x97 an offset of about 17 counts sits on the
  FNO's drag, and one padding row flips its sign. This entry does not establish the cause. The
  largest in-range drag errors rise for that reason, while the near field, lift and moment improve.
  Command:
  `python -c "import json,numpy as np; R=json.load(open('results/s4/scores.json'))['rows']; [print(g, s, round(float(np.median(v)),1), sum(c<0 for c in v), len(v)) for g in ('256x97','128x49') for s in range(3) for v in [[r['score']['cd_err_counts'] for r in R if r['tag']==f'fno-n896-s{s}' and r['grid']==g and r['split']=='test_in']]]"`,
  and the same on the file at commit 7b39c58.
- **Seeds** (896 cases, 256x97, seeds 0 / 1 / 2). The in-range near field is 0.0161 to 0.0181
  subcritical (12 % spread, against 4 % at 17 rows) and 0.0391 to 0.0409 supercritical. The
  extrapolation CL maximum still varies by 40 % (0.171 to 0.238). Same command as the table.
- **Loss B against loss A at 256x97.** Loss B is worse on the near field and CL in every set and
  regime. It is worse on CD in every one except in-range subcritical, where its CD maximum is
  37.2 counts against loss A's 41.6. At 17 rows the exception was CL, not CD. The 128x49
  comparison is unchanged. Same command as the table.

**Shocks at 256x97** (scores.md, "Shocks"). Command:
`python -c "import json; [print(g['variant'], g['grid'], g['n_train'], g['seed'], g['split'], g['n_missed'], round(g['abs_x_err_median'],4), round(g['abs_x_err_max'],4), round(g['overshoot_median'],2), round(g['overshoot_max'],2)) for g in json.load(open('results/s4/scores.json'))['shock_groups'] if g['method']=='fno' and g['grid']=='256x97']"`.

- **In range**, the three loss-A seeds at 896 cases miss none of their 27 shocks each (2, 2 and 1 at
  17 rows). Seed 0's position error is 0 / 0.058 chord (median / max), unchanged. Its overshoot is
  0.36 / 0.85 (0.42 / 1.05), against the floor's 0.19 / 0.54. Loss B misses 1 (2). Across the
  training-set sizes, only n = 64 misses one (3 at 17 rows).
- **In extrapolation** none is missed at any n, as before. Seed 0's position error is 0.040 / 0.213
  chord, and its overshoot is 1.07 / 1.56 (1.13 / 1.58).

**Data efficiency** (S4 entry's command, unchanged; mean near field over the 61 test_in cases at
256x97, seed 0, floor 0.01432). For n = 64, 128, 256, 512 and 896 it is 0.0918, 0.0634, 0.0460,
0.0310 and 0.0256: 6.4x, 4.4x, 3.2x, 2.2x and 1.8x the floor, where S4 recorded 7.6x to 3.4x. It is
below POD at every n. **Prediction 2 is unchanged: held.** The break-even is still 1,126 new cases,
since the costs are unchanged. The "unless" clause still does not fire: the curve is still falling
at 896 cases and is 1.8x the floor there.

**Physics checks on the FNO at 256x97** (the S4 entry's commands, on the new file):

- Non-positive density or pressure on test_ext: 5 of 14 cases at n = 64 (4 at 17 rows), 4 at
  n = 128 (4), none at n ≥ 256 (none). The worst enthalpy deviation at n = 64 is 702 (543).
- Subcritical mean entropy below free stream: 8 to 21 of 37 cases across the FNO's runs (9 to 24),
  with 8 / 14 / 20 for the three 896-case seeds (9 / 16 / 19).
- Mass-flux medians at j = 48 and 72, 896 cases, in range: 2.8e-4 to 5.4e-3 (4.3e-4 to 7.4e-3).
  Command: `python -c "import json; [print(g['seed'], g['split'], g['regime'], [f\"{g[f'mass_flux_line{k}_median']:.1e}\" for k in (2, 3)]) for g in json.load(open('results/s4/scores.json'))['physics_groups'] if g['method']=='fno' and g['grid']=='256x97' and g['n_train']==896 and g['variant']=='']"`.

**S5's comparison columns** (`results/s5/scores.md`). S5 stopped after its gate, so P6 and P7
remain untested, and only the FNO's reference columns move. At 256x97, the FNO's mean near-field
ratio to the floor on the three step-3 cases goes from 2.71 to 1.75 in range. In extrapolation it
goes from 4.84 to 4.69. Its degradation factor therefore rises from 1.79 to 2.69. On
`test_in-0004`, its gate-metric RMAE of Cp goes from 0.30 % to 0.13 %. Command:
`grep -E "^\| (fno-256|test_in-0004 \| fno-256) " results/s5/scores.md`.

**Figures marked in place** with a pointer here: the S2 entry's transfer table, and the S4
entry's figures for the FNO at 256x97. That covers its data-efficiency table and the "unless"
sentence, the results table and the bullets under it, the seeds table, loss B, the shocks table
and paragraph, and the failures list. Also the S5 gate entry's 0.30 %. The README's summary, the
report's text and figures, and `CLAUDE.md` are updated to the new rule and figures.

## 2026-10-26 — the FNO's drag offset at 256x97: the right row count lies between 16 and 17

The previous entry found an in-range drag offset at 256x97 of about 17 counts, whose sign flips
between 16 and 17 padding rows, while the same models at 128x49 have none. This entry measures what
the committed records and the network without training can show. Commands, at commit 3b7b2b5:
`python scripts/probes/probe_fno_drag.py --from-scores --synthetic`, which wrote
`results/fno_drag/from_scores.*` and `synthetic.*`. `--from-scores` reads the test scores at
17 rows from `git show 7b39c58:results/s4/scores.json`, and at 16 rows and 128x49 from the current
file.

**In range, every S2 run's drag crosses its own 128x49 value at 16.5 rows.** Per case, take the
line through the 16- and 17-row drag errors and find where it meets the same model's 128x49 error.
Over the 61 in-range cases the median crossing is 16.48 to 16.54 rows, for all eight final runs:
three seeds, 64 to 512 cases, and loss B. The interquartile ranges lie within 16.41 to 16.62. The
change from 16 to 17 rows is +31 to +36 counts at the median. The mean of the 16- and 17-row errors
is within 1.8 counts of the 128x49 error at the median (−1.8 to +0.6). So at 256x97 the drag sits
about half a row's change, 15 to 18 counts, either side of the model's own 128x49 drag: high at 17
rows, low at 16. Two points per case cannot show that the dependence is a straight line. That is
measured next.

**In extrapolation the pattern is weaker.** The change from 16 to 17 rows is smaller: +5 to +20
counts at the median, and +60 for loss B. The crossings spread from 16.2 to 18.0, with wide
interquartile ranges. There the drag error is mostly the model's own: −69 to −167 counts at the
median at 128x49 already.

**Every measure is V-shaped in the row count, with no odd/even effect.** In the committed padding
sweep (`results/fno_lib/pad_sweep.json`), the validation transfer and near field fall steadily to a
minimum at 16 rows and rise from 17. For S2's three models the near field is 0.1157, 0.0784,
0.0434, 0.0195, 0.0423, 0.0756, 0.1094, 0.1427 at 13 to 20 rows. Straight branches through the two
points either side of the transfer's minimum meet at 16.34 to 16.36 rows, for all four groups of
the sweep. Command:
`python -c "import json,numpy as np; R=json.load(open('results/fno_lib/pad_sweep.json'))['records']; G=sorted({r['group'] for r in R}); [print(g, [round(float(np.mean([r['rows'][str(k)]['near_field'] for r in R if r['group']==g])),4) for k in range(13,21)]) for g in G]"`.
The vertex is `_vertex` in `probe_fno_drag.py`, applied to the group means of `transfer` over 12 to
22 rows.

**The same optimum appears without training** (`results/fno_drag/synthetic.md`):

- **One spectral layer** at 128x49 (8 rows) against 256x97, at its shared nodes. On a radial field
  that is smooth across the wrap of the padded column, the transfer is exact at 17 rows: 0.0000,
  vertex 16.99. That is the period-matched rule's design case (spec §6.3, 2026-10-01). On fields
  that are still non-zero at the far-field row, where the zero padding starts with a jump, the
  minimum is at 16 rows and the vertex at 16.16 to 16.25. The fields were a constant, 1 + η and
  exp(−3η).
- **An untrained `FNO2d`** (S2's sizes, spectral weights scaled by 20, 50 and 150, three seeds),
  on the stand-in dataset's input features: vertex 16.28 at every scale.

Every input and output channel of the real problem is non-zero at the far-field row (free stream,
position). So the padding starts with a jump, and the one-layer case suggests why no whole row
count is right. 17 rows matches the DFT period, 114 spacings of the 256x97 grid, as at 128x49. 16
rows nearly matches the share of the period that holds data: 97/113 = 0.858 against 49/57 = 0.860.
This is offered as a reading of these measurements; it is not established.

**What this establishes.** The drag offset is the same row-count mismatch the near field and the
transfer show, seen in a quantity with a sign. Their minimum lies between 16 and 17 rows. A norm
shows the mismatch as a residual error at either neighbour, and the drag shows it as an offset of
opposite sign at each. It is a property of the operator on a field cut off by the padding, not of
the training.

**Not yet established:**

- whether the drag is a straight line in the row count;
- where on the wall the change from 16 to 17 rows acts;
- whether averaging the 16- and 17-row predictions helps beyond the drag. That average's drag
  error is the mean of the two by arithmetic, since forces are linear in the wall pressure. For
  one spectral layer, though, the averaged output transfers worse than 16 rows alone (0.0240
  against 0.0144 for the constant field; inline check, not in a script).

`python scripts/probes/probe_fno_drag.py` measures these on the validation cases from S2's
checkpoints (`docs/notes/2026-10-24-fno-transfer-ablation.md`).
