"""PINN training (spec 7.5): Adam with exponential decay, then L-BFGS in float64.

Points are uniform in (xi, eta), which the map turns into clustering near the wall. The Kutta
condition is a penalty on the velocity at the trailing-edge wall point (xi, eta) = (0, 0); the
cylinder warm-up replaces it with zero circulation, since a cylinder has no sharp edge to fix it.

Loss weights are fixed (residual 1, outer 10, Kutta 10) unless `grad_balance` is set, the spec's
variant: gradient-norm balancing after Wang, Teng & Perdikaris (2021, "learning-rate annealing"),
every `balance_every` Adam steps lambda_k <- (1 - a) lambda_k + a max|grad L_res| / mean|grad L_k|
for the outer and Kutta (or circulation) terms, a = `balance_alpha`. L-BFGS keeps the last weights.
`time_steps` measures what one Adam step and one L-BFGS closure of a configuration cost, so a stage
can project its total before committing days to it.
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
    grad_balance: bool = False
    balance_every: int = 100
    balance_alpha: float = 0.9
    seed: int = 0


def _points(cfg, gen, dtype):
    xi = torch.rand(cfg.n_interior, generator=gen, dtype=dtype)
    eta = 1.0 - torch.rand(cfg.n_interior, generator=gen, dtype=dtype)  # (0, 1]
    xo = torch.rand(cfg.n_outer, generator=gen, dtype=dtype)
    xw = torch.arange(cfg.n_wall, dtype=dtype) / cfg.n_wall
    return {"xi": xi, "eta": eta, "xi_outer": xo, "xi_wall": xw}


def losses(pinn, pts, cfg, ratio, weights=None):
    """Each loss term and their weighted total; `weights` overrides cfg's outer and Kutta ones."""
    w = weights or {"outer": cfg.w_outer, "kutta": cfg.w_kutta}
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
    out["total"] = (cfg.w_res * out["residual"] + w["outer"] * out["outer"]
                    + w["kutta"] * (out["kutta"] + out["circulation"]))
    return out


def balance_weights(L, params, weights, alpha):
    """One gradient-norm balancing update (Wang, Teng & Perdikaris 2021). A term whose gradient is
    zero (e.g. the Kutta term of a no-Kutta run) keeps its weight."""
    params = [p for p in params if p.requires_grad]

    def grads(term):
        g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1).abs()
                          for x, p in zip(g, params)])

    g_res = grads(L["residual"]).max()
    out = dict(weights)
    for name, term in (("outer", L["outer"]), ("kutta", L["kutta"] + L["circulation"])):
        if not term.requires_grad:
            continue
        mean = grads(term).mean()
        if mean > 0:
            out[name] = float((1 - alpha) * weights[name] + alpha * g_res / mean)
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
    weights = {"outer": cfg.w_outer, "kutta": cfg.w_kutta}
    t0, c0 = time.perf_counter(), time.process_time()
    pts = None
    for step in range(cfg.adam_steps):
        if step % cfg.resample_every == 0:
            pts = _points(cfg, gen, torch.float32)
        opt.zero_grad()
        L = losses(pinn, pts, cfg, ratio, weights)
        if cfg.grad_balance and step % cfg.balance_every == 0:
            weights = balance_weights(L, pinn.parameters(), weights, cfg.balance_alpha)
            L = losses(pinn, pts, cfg, ratio, weights)
        L["total"].backward()
        opt.step()
        sched.step()
        if step % max(1, cfg.adam_steps // 60) == 0 or step == cfg.adam_steps - 1:
            history.append({"step": step, **_snapshot(L), "w_outer": weights["outer"],
                            "w_kutta": weights["kutta"]})
            log(f"adam {step}: {history[-1]}")
    pinn.double()
    pts = {k: v.double() for k, v in pts.items()}
    lbfgs = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        lbfgs.zero_grad()
        total = losses(pinn, pts, cfg, ratio, weights)["total"]
        total.backward()
        return total

    lbfgs.step(closure)
    final = _snapshot(losses(pinn, pts, cfg, ratio, weights))
    history.append({"step": "lbfgs", **final})
    log(f"final: {final}")
    rec = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0,
           "history": history, "final": final, "weights": weights}
    return pinn, rec


def time_steps(case, cfg, n_adam=5, n_closure=2):
    """Wall seconds per Adam step (float32) and per L-BFGS closure (float64) for this config on this
    case, from a fresh network at the configured point counts; the first of each is a warm-up and
    not counted. With grad_balance, also seconds per balancing update.
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ratio = stretch_ratio(case.radius)
    pinn = Pinn(case, PinnNet(cfg.hidden, cfg.depth).float())
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
    weights = {"outer": cfg.w_outer, "kutta": cfg.w_kutta}
    pts = _points(cfg, gen, torch.float32)

    def timed(fn, n):
        out = []
        for _ in range(n + 1):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return float(np.median(out[1:]))

    def adam():
        opt.zero_grad()
        losses(pinn, pts, cfg, ratio, weights)["total"].backward()
        opt.step()

    rec = {"adam_s": timed(adam, n_adam)}
    if cfg.grad_balance:
        rec["balance_s"] = timed(lambda: balance_weights(losses(pinn, pts, cfg, ratio, weights),
                                                         pinn.parameters(), weights,
                                                         cfg.balance_alpha), 1)
    pinn.double()
    pts = {k: v.double() for k, v in pts.items()}

    def closure():
        for p in pinn.parameters():
            p.grad = None
        losses(pinn, pts, cfg, ratio, weights)["total"].backward()

    rec["closure_s"] = timed(closure, n_closure)
    return rec


def projected_s(cfg, rec):
    """A run's projected wall seconds from time_steps: every Adam step, at least lbfgs_iters
    closures (L-BFGS evaluates one or more per iteration, up to 1.25 per iteration by default),
    and the balancing updates."""
    total = cfg.adam_steps * rec["adam_s"] + cfg.lbfgs_iters * rec["closure_s"]
    if cfg.grad_balance:
        total += math.ceil(cfg.adam_steps / cfg.balance_every) * rec["balance_s"]
    return total


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
