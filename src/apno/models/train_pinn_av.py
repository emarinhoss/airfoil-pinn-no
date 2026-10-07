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
float64 on the CPU. A non-finite loss stops the run (rec["diverged"]; rec["reason"] names where:
the epoch and batch, the Adam step or the final L-BFGS, design §7.3); an outer iteration whose
loss did not fall is counted in rec["stalled"].
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
    where = ["setup"]  # the outer iteration in progress, for a divergence's reason

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

        before = float(opt.step(closure).detach())
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
                    where[0] = f"epoch {epoch}, batch {b}"
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
                    where[0] = f"adam step {step + 1}"
                    opt.zero_grad()
                    L = objective(batch)
                    L["total"].backward()
                    opt.step()
                    sched.step()
                    step += 1
                    if step % 500 == 0 or step == cfg.adam_steps:
                        rec["history"].append({"epoch": "adam", "step": step, **_snapshot(L)})
            where[0] = "final L-BFGS"
            record("final", 0, *lbfgs_step(fixed, cfg.final_lbfgs_iters,
                                           math.ceil(1.25 * cfg.final_lbfgs_iters),
                                           cfg.final_lbfgs_history))
        else:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    except NonFiniteLoss as exc:
        rec["diverged"], rec["reason"] = True, f"{where[0]}: {exc}"
        log(f"diverged: {rec['reason']}")
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
