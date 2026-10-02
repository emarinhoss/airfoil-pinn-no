"""PINN training (spec 7.5): Adam with exponential decay, then L-BFGS in float64.

Points are uniform in (xi, eta), which the map turns into clustering near the wall. The Kutta
condition is a penalty on the velocity at the trailing-edge wall point (xi, eta) = (0, 0); the
cylinder warm-up replaces it with zero circulation, since a cylinder has no sharp edge to fix it.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from apno.geometry.ogrid import stretch_ratio
from apno.models.pinn import Pinn, PinnNet, to_dataset_q
from apno.physics.euler import residual


@dataclass
class PinnConfig:
    hidden: int = 96
    depth: int = 7
    n_interior: int = 20000
    resample_every: int = 1000
    n_outer: int = 1000
    n_wall: int = 512
    adam_steps: int = 30000
    lr0: float = 1e-3
    lr1: float = 1e-4
    lbfgs_iters: int = 5000
    w_res: float = 1.0
    w_outer: float = 10.0
    w_kutta: float = 10.0
    viscosity: float = 0.0
    kutta: bool = True
    zero_circulation: bool = False
    seed: int = 0


def _points(cfg, gen, dtype):
    xi = torch.rand(cfg.n_interior, generator=gen, dtype=dtype)
    eta = 1.0 - torch.rand(cfg.n_interior, generator=gen, dtype=dtype)  # (0, 1]
    xo = torch.rand(cfg.n_outer, generator=gen, dtype=dtype)
    xw = torch.arange(cfg.n_wall, dtype=dtype) / cfg.n_wall
    return {"xi": xi, "eta": eta, "xi_outer": xo, "xi_wall": xw}


def losses(pinn, pts, cfg, ratio):
    case = pinn.case
    r = residual(pts["xi"], pts["eta"], pinn.state, case.geometry, case.radius, ratio,
                 cfg.viscosity)
    out = {"residual": (r * r).sum(-1).mean()}
    xo = pts["xi_outer"]
    rho, u, v, _ = pinn.state(xo, torch.ones_like(xo))
    out["outer"] = ((rho - 1) ** 2 + (u - pinn.ca) ** 2 + (v - pinn.sa) ** 2).mean()
    zero = torch.zeros(1, dtype=xo.dtype)
    if cfg.kutta:
        _, uk, vk, _ = pinn.state(zero, zero)
        out["kutta"] = (uk * uk + vk * vk).sum()
    else:
        out["kutta"] = zero.sum()
    if cfg.zero_circulation:
        xw = pts["xi_wall"].clone().requires_grad_(True)
        _, uw, vw, _ = pinn.state(xw, torch.zeros_like(xw))
        x, y = case.geometry.wall(xw, xp=torch)
        dx = torch.autograd.grad(x.sum(), xw, create_graph=True)[0]
        dy = torch.autograd.grad(y.sum(), xw, create_graph=True)[0]
        gamma = (uw * dx + vw * dy).mean()
        out["circulation"] = gamma * gamma
    else:
        out["circulation"] = zero.sum()
    out["total"] = (cfg.w_res * out["residual"] + cfg.w_outer * out["outer"]
                    + cfg.w_kutta * (out["kutta"] + out["circulation"]))
    return out


def _snapshot(L):
    return {k: float(v.detach()) for k, v in L.items()}


def train(case, cfg, log=print):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ratio = stretch_ratio(case.radius)
    pinn = Pinn(case, PinnNet(cfg.hidden, cfg.depth).float())
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg.lr1 / cfg.lr0) ** (1 / cfg.adam_steps))
    history = []
    t0, c0 = time.perf_counter(), time.process_time()
    pts = None
    for step in range(cfg.adam_steps):
        if step % cfg.resample_every == 0:
            pts = _points(cfg, gen, torch.float32)
        opt.zero_grad()
        L = losses(pinn, pts, cfg, ratio)
        L["total"].backward()
        opt.step()
        sched.step()
        if step % max(1, cfg.adam_steps // 60) == 0 or step == cfg.adam_steps - 1:
            history.append({"step": step, **_snapshot(L)})
            log(f"adam {step}: {history[-1]}")
    pinn.double()
    pts = {k: v.double() for k, v in pts.items()}
    lbfgs = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        lbfgs.zero_grad()
        total = losses(pinn, pts, cfg, ratio)["total"]
        total.backward()
        return total

    lbfgs.step(closure)
    final = _snapshot(losses(pinn, pts, cfg, ratio))
    history.append({"step": "lbfgs", **final})
    log(f"final: {final}")
    rec = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "history": history, "final": final}
    return pinn, rec


def evaluate_on_grid(pinn, shape):
    ni, nj = shape
    xi = (torch.arange(ni, dtype=torch.float64) / ni)[:, None].expand(ni, nj).reshape(-1)
    eta = (torch.arange(nj, dtype=torch.float64) / (nj - 1))[None, :].expand(ni, nj).reshape(-1)
    pinn.double()
    with torch.no_grad():
        rho, u, v, p = pinn.state(xi, eta)
        q = to_dataset_q(rho, u, v, p, pinn.case.mach)
    return q.reshape(ni, nj, 4).detach().numpy()


def circle_cp_error(q, mach):
    """For the warm-up: max |Cp - (1 - 4 sin^2 theta)| on the cylinder's wall row."""
    ni = q.shape[0]
    th = -2 * math.pi * np.arange(ni) / ni
    cp = (q[:, 0, 3] - 1.0) / (0.5 * 1.4 * mach * mach)
    return float(np.abs(cp - (1 - 4 * np.sin(th) ** 2)).max()), cp
