# ruff: noqa
# Task 26 evidence probe, kept verbatim below this header (copied 2026-10-06 from /tmp/t26, where
# it was run on 2026-10-05 16:37-16:47 as `python /tmp/t26/probe_task26.py`; its output is
# results/s3/probe_task26.log). Re-run from the repository root with
# `python scripts/probes/probe_task26.py` (needs data/hdf5/512x193.h5 and an Apple GPU for the
# recorded device). It reads only library code and the truth file, and writes nothing.
"""Task 26 evidence probe: the first 3,000 Adam steps of the production S3 'pure' run on
test_in-0022 with PinnConfig defaults (same lr schedule, resampling, TE exclusion, equal-speed
Kutta), on MPS; wall Cp against SU2's fine-grid truth every 1,000 steps."""
import sys, time
import h5py, numpy as np, torch
from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN, Section, stretch_ratio
from apno.models import train_pinn as tp
from apno.models.pinn import PinnCase
from apno.physics.flow import pressure_coefficient

cid, steps = "test_in-0022", 3000
s = {c.case_id: c for c in case_specs("test_in", TRAIN)}[cid]
case = PinnCase(s.mach, s.alpha_deg, Section(s.t, s.m))
cfg = tp.PinnConfig()
with h5py.File("data/hdf5/512x193.h5") as f:
    ids = list(f["test_in"]["case_id"].asstr()[...])
    cp_true = pressure_coefficient(f["test_in"]["q"][ids.index(cid)][::2, ::2][:, 0, 3], case.mach)
dev = tp.resolve_device(cfg.device)
print(f"{cid}: M {s.mach:.4f} alpha {s.alpha_deg:.3f} t {s.t:.4f} m {s.m:.4f}; device {dev}; "
      f"te_exclusion {cfg.te_exclusion} kutta_delta {cfg.kutta_delta}", flush=True)
torch.manual_seed(cfg.seed)
gen = torch.Generator().manual_seed(cfg.seed)
ratio = stretch_ratio(case.radius)
pinn = tp.make_pinn(case, cfg, dev)
opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg.lr1 / cfg.lr0) ** (1 / cfg.adam_steps))
w = tp.initial_weights(pinn, cfg)
t0 = time.time()
for step in range(steps + 1):
    if step % cfg.resample_every == 0:
        pts, n = tp._sample(pinn, cfg, gen, torch.float32, dev, ratio)
        print(f"resample at {step}: dropped {n} of {cfg.n_interior}", flush=True)
    opt.zero_grad()
    L = tp.losses(pinn, pts, cfg, ratio, w)
    if step % 1000 == 0:
        cpu = tp.make_pinn(case, cfg, "cpu", torch.float64)
        cpu.net.load_state_dict({k: v.detach().cpu().double() for k, v in pinn.net.state_dict().items()})
        q = tp.evaluate_on_grid(cpu, TRAIN)
        cp = pressure_coefficient(q[:, 0, 3], case.mach); d = cp - cp_true
        print(f"step {step:5d} t={time.time()-t0:5.0f}s res {float(L['residual']):.3e} "
              f"wall {float(L['wall']):.3e} kutta {float(L['kutta']):.2e} "
              f"Gamma {float(pinn.net.gamma):+.4f} | wall Cp L2 {np.sqrt(np.mean(d*d)):.3f} "
              f"max {np.abs(d).max():.3f} Cp_min {cp.min():.3f} (SU2 {cp_true.min():.3f})",
              flush=True)
    if step == steps:
        break
    L["total"].backward(); opt.step(); sched.step()
print("SENTINEL", flush=True)
