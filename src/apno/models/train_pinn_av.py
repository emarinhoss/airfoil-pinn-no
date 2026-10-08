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
loss did not fall is counted in rec["stalled"]. rec["func_evals"] counts the L-BFGS closures
completed (loss and backward), those of an outer iteration that diverged included. The default log
prints flushed at every line, so a run's progress shows under nohup or a redirection.
"""

import dataclasses
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


def _log(msg):
    """print, flushed at every line: a run's progress stays visible under nohup or > file."""
    print(msg, flush=True)


def train(case, cfg, log=_log):
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
            rec["func_evals"] += 1  # here, not in record(): a diverging iteration never gets there
            return total

        before = float(opt.step(closure).detach())
        return before, objective(batch), evals[0]

    def record(epoch, b, before, after, evals):
        snap = _snapshot(after)
        rec["history"].append({"epoch": epoch, "batch": b, "loss_before": before,
                               "func_evals": evals, **snap})
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


def time_epoch(case, cfg, log=_log):
    """{"device", "epoch_s", "func_evals", "stalled", "diverged"}: one real epoch of the run, timed
    (design §7.1 as amended 2026-10-08): `train` with epochs = 1 and the run's own seeds, so it is
    the run's first epoch exactly, its work then discarded. The gate projects a configuration at
    cfg.epochs times epoch_s. L-BFGS only: A3's Adam steps make no epochs to scale."""
    if cfg.optimizer != "lbfgs":
        raise ValueError(f"time_epoch: optimizer {cfg.optimizer!r} has no L-BFGS epochs")
    _, rec = train(case, dataclasses.replace(cfg, epochs=1), log)
    return {"device": rec["device"], "epoch_s": rec["wall_s"], "func_evals": rec["func_evals"],
            "stalled": rec["stalled"], "diverged": rec["diverged"]}


def time_closure(case, cfg, n=3):
    """{"device", "closure_s", "lbfgs_iter_s"}: median wall seconds, on cfg.device, of one closure
    (loss and backward) on one batch of cfg.batch residual nodes, from a fresh network, with the
    viscous term forced on at every point (the cost's upper bound); and of one L-BFGS iteration's
    own work, the two-loop recursion at the full history (cfg.lbfgs_history; final_lbfgs_history
    for "adam_lbfgs") over vectors of the network's parameter count (`time_two_loop`). For each,
    the first call is a warm-up and is not counted."""
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
    history = cfg.final_lbfgs_history if cfg.optimizer == "adam_lbfgs" else cfg.lbfgs_history
    n_params = sum(p.numel() for p in pinn.parameters())
    return {"device": device, "closure_s": float(np.median(times[1:])),
            "lbfgs_iter_s": time_two_loop(n_params, history, device, n=n)}


def _two_loop(flat_grad, old_dirs, old_stps, ro, H_diag):
    """The L-BFGS direction -H g, by the two-loop recursion exactly as torch.optim.LBFGS.step runs
    it (one buffer, dot and add_ over the history lists, the 0-dim tensors ro and al as alpha)."""
    num_old = len(old_dirs)
    al = [None] * num_old
    q = flat_grad.neg()
    for i in range(num_old - 1, -1, -1):
        al[i] = old_stps[i].dot(q) * ro[i]
        q.add_(old_dirs[i], alpha=-al[i])
    d = r = torch.mul(q, H_diag)
    for i in range(num_old):
        be_i = old_dirs[i].dot(r) * ro[i]
        r.add_(old_stps[i], alpha=al[i] - be_i)
    return d


def time_two_loop(n_params, history, device, n=3, dtype=torch.float32):
    """Median wall seconds of one two-loop recursion (`_two_loop`) over a full history of `history`
    pairs of n_params-vectors, on `device`, synchronised before and after; the first call is a
    warm-up and is not counted. On the Apple GPU this is ~4 x history small kernels per iteration,
    each alpha read back to the host, as in PyTorch's L-BFGS. The pairs are y = 2 s (y.s > 0, as
    L-BFGS keeps), so the recursion is a sequence of projections and stays finite."""
    gen = torch.Generator().manual_seed(0)
    old_stps = list(torch.randn(history, n_params, generator=gen).to(device, dtype).unbind(0))
    old_dirs = [2.0 * s for s in old_stps]
    ro = [1.0 / y.dot(s) for y, s in zip(old_dirs, old_stps)]
    H_diag = old_dirs[-1].dot(old_stps[-1]) / old_dirs[-1].dot(old_dirs[-1])
    flat_grad = torch.randn(n_params, generator=gen).to(device, dtype)
    times = []
    for _ in range(n + 1):
        _sync(device)
        t = time.perf_counter()
        _two_loop(flat_grad, old_dirs, old_stps, ro, H_diag)
        _sync(device)
        times.append(time.perf_counter() - t)
    del old_stps, old_dirs
    if device == "mps":
        torch.mps.empty_cache()  # the history is ~2 x history x n_params floats: give it back
    return float(np.median(times[1:]))


def projected_upper_s(cfg, n_nodes, closure_s, lbfgs_iter_s=0.0):
    """A run's upper-bound wall seconds (design §7.1), from `time_closure`'s closure_s and
    lbfgs_iter_s: every L-BFGS outer iteration at max_eval closures plus max_iter iterations of the
    optimiser's own two-loop recursion at the full history, over epochs x ceil(n_nodes / batch)
    batches; for A3, every Adam step plus, per final L-BFGS iteration, 1.25 closures (its max_eval
    over max_iter) and one recursion. lbfgs_iter_s = 0 gives the closures-only count."""
    if cfg.optimizer == "adam_lbfgs":
        return (cfg.adam_steps * closure_s
                + cfg.final_lbfgs_iters * (1.25 * closure_s + lbfgs_iter_s))
    return (cfg.epochs * math.ceil(n_nodes / cfg.batch)
            * (cfg.lbfgs_max_eval * closure_s + cfg.lbfgs_max_iter * lbfgs_iter_s))


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
