"""PINN training (spec 7.5): Adam with exponential decay, then L-BFGS in float64.

Adam runs in float32 on `PinnConfig.device` ("auto": the Apple GPU when MPS is available); L-BFGS
and grid evaluation run on the CPU in float64, which MPS does not have. Points are drawn from a CPU
generator and moved, so a seed gives the same points on every device.

Points are uniform in (xi, eta), which the map turns into clustering near the wall. For the
potential PINN on a section, the interior points within `te_exclusion` chords of the trailing edge
are dropped from the residual after every (re)sampling (`exclude_trailing_edge`; the corner flow is
singular there), and the count dropped at each resample is recorded in the run record's
`te_dropped`. The Kutta condition is the formulation's own (potential: equal wall speeds at
xi = kutta_delta and 1 - kutta_delta; Euler: zero velocity at the trailing-edge wall point); the
cylinder warm-up replaces it with zero circulation, since a cylinder has no sharp edge to fix it.

`PinnConfig.formulation` picks the PINN: "potential" (the default since the author's decision of
2026-10-04, `apno.models.pinn_potential`: a velocity potential, continuity only, soft wall, exact
far field) or "euler" (spec 7 as first written, `apno.models.pinn`: kept runnable as the documented
negative result, since it converges to fluid at rest; docs/findings.md, 2026-10-04).

Loss weights are fixed (residual 1, soft boundary 10, Kutta 10) unless `grad_balance` is set, the
spec's variant: gradient-norm balancing after Wang, Teng & Perdikaris (2021, "learning-rate
annealing"), every `balance_every` Adam steps lambda_k <- (1 - a) lambda_k + a max|grad L_res| /
mean|grad L_k| for the soft boundary term (Euler: the outer circle; potential: the wall) and the
Kutta (or circulation) term, a = `balance_alpha`. L-BFGS keeps the last weights.
`time_steps` measures what one Adam step and one L-BFGS closure of a configuration cost, so a stage
can project its total before committing days to it.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from apno.geometry.ogrid import Section, map_xy, stretch_ratio
from apno.models import pinn_potential
from apno.models.pinn import Pinn, PinnNet, to_dataset_q
from apno.models.pinn_potential import PotentialNet, PotentialPinn
from apno.physics.euler import residual

FORMULATIONS = ("potential", "euler")


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
    w_outer: float = 10.0  # Euler: the soft free-stream penalty on the outer circle
    w_wall: float = 10.0  # potential: the soft no-penetration penalty on the wall
    w_kutta: float = 10.0
    kutta_delta: float = 0.055  # potential: the equal-speed Kutta points, xi = delta and 1 - delta
    te_exclusion: float = 0.02  # potential, sections: residual-free radius about the TE, chords
    viscosity: float = 0.0
    kutta: bool = True
    zero_circulation: bool = False
    grad_balance: bool = False
    balance_every: int = 100
    balance_alpha: float = 0.9
    seed: int = 0
    device: str = "auto"  # the Adam phase's; L-BFGS and evaluation are always CPU float64
    formulation: str = "potential"  # or "euler", the recorded negative result (see the docstring)


def resolve_device(name):
    """'auto' is the Apple GPU when PyTorch's MPS backend is available, else the CPU."""
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def make_pinn(case, cfg, device="cpu", dtype=torch.float32):
    """The PINN of cfg.formulation, its network in `dtype` on `device`. The potential PINN learns
    its circulation unless cfg.zero_circulation (the cylinder warm-up), which fixes it at 0."""
    if cfg.formulation == "euler":
        return Pinn(case, PinnNet(cfg.hidden, cfg.depth).to(dtype).to(device))
    if cfg.formulation == "potential":
        net = PotentialNet(cfg.hidden, cfg.depth, circulation=not cfg.zero_circulation)
        return PotentialPinn(case, net.to(dtype).to(device))
    raise ValueError(f"unknown formulation {cfg.formulation!r}; expected one of {FORMULATIONS}")


def initial_weights(pinn, cfg):
    """The soft-term weights the PINN's losses take, before any balancing."""
    if getattr(pinn, "formulation", "euler") == "potential":
        return {"wall": cfg.w_wall, "kutta": cfg.w_kutta}
    return {"outer": cfg.w_outer, "kutta": cfg.w_kutta}


def _points(cfg, gen, dtype, device="cpu"):
    """Drawn from the CPU generator `gen` and then moved, so a seed gives the same points on
    every device."""
    xi = torch.rand(cfg.n_interior, generator=gen, dtype=dtype)
    eta = 1.0 - torch.rand(cfg.n_interior, generator=gen, dtype=dtype)  # (0, 1]
    xo = torch.rand(cfg.n_outer, generator=gen, dtype=dtype)
    xw = torch.arange(cfg.n_wall, dtype=dtype) / cfg.n_wall
    return {k: v.to(device) for k, v in
            {"xi": xi, "eta": eta, "xi_outer": xo, "xi_wall": xw}.items()}


def exclude_trailing_edge(pts, geom, radius, ratio, te_exclusion):
    """(pts, n_dropped): the interior points (xi, eta) whose physical distance from the trailing
    edge (1, 0) is below te_exclusion removed, for a Section; the wall and outer points, and every
    point of any other geometry (the cylinder has no corner), unchanged. The distance is taken in
    float64 on the CPU, so the same points are dropped on every device."""
    if not isinstance(geom, Section) or te_exclusion <= 0:
        return pts, 0
    xi, eta = pts["xi"].detach().cpu().double(), pts["eta"].detach().cpu().double()
    x, y = map_xy(xi, eta, geom, radius, ratio, xp=torch)
    keep = torch.hypot(x - 1.0, y) >= te_exclusion
    out = dict(pts)
    keep = keep.to(pts["xi"].device)
    out["xi"], out["eta"] = pts["xi"][keep], pts["eta"][keep]
    return out, int((~keep).sum())


def _sample(pinn, cfg, gen, dtype, device, ratio):
    """Fresh points, with the potential PINN's trailing-edge exclusion applied."""
    pts = _points(cfg, gen, dtype, device)
    if getattr(pinn, "formulation", "euler") != "potential":
        return pts, 0
    case = pinn.case
    return exclude_trailing_edge(pts, case.geometry, case.radius, ratio, cfg.te_exclusion)


def losses(pinn, pts, cfg, ratio, weights=None):
    """Each loss term and their weighted total; `weights` overrides cfg's soft-term ones. The
    formulation is the PINN object's, so an Euler `Pinn` gets the Euler losses whatever cfg says."""
    w = weights or initial_weights(pinn, cfg)
    if getattr(pinn, "formulation", "euler") == "potential":
        return pinn_potential.losses(pinn, pts, cfg, ratio, w)
    case = pinn.case
    r = residual(pts["xi"], pts["eta"], pinn.state, case.geometry, case.radius, ratio,
                 cfg.viscosity)
    out = {"residual": (r * r).sum(-1).mean()}
    xo = pts["xi_outer"]
    rho, u, v, _ = pinn.state(xo, torch.ones_like(xo))
    out["outer"] = ((rho - 1) ** 2 + (u - pinn.ca) ** 2 + (v - pinn.sa) ** 2).mean()
    zero = torch.zeros(1, dtype=xo.dtype, device=xo.device)  # the Kutta point, on xo's device
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
    for name in weights:
        term = L["kutta"] + L["circulation"] if name == "kutta" else L[name]
        if not term.requires_grad:
            continue
        mean = grads(term).mean()
        if mean > 0:
            out[name] = float((1 - alpha) * weights[name] + alpha * g_res / mean)
    return out


def _snapshot(L):
    return {k: float(v.detach()) for k, v in L.items()}


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()


def _to_cpu64(pinn, pts):
    """The L-BFGS phase: network and points on the CPU in float64 (MPS has no float64)."""
    pinn.net.to("cpu").double()  # device first: MPS cannot cast to float64 in place
    return {k: v.to("cpu").double() for k, v in pts.items()}


def train(case, cfg, log=print):
    """Adam in float32 on cfg.device ('auto': the Apple GPU if available), then L-BFGS in float64
    on the CPU. Points come from one CPU generator, so a seed draws the same points on any device.
    """
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ratio = stretch_ratio(case.radius)
    pinn = make_pinn(case, cfg, device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg.lr1 / cfg.lr0) ** (1 / cfg.adam_steps))
    history = []
    weights = initial_weights(pinn, cfg)
    t0, c0 = time.perf_counter(), time.process_time()
    pts, te_dropped = None, []
    for step in range(cfg.adam_steps):
        if step % cfg.resample_every == 0:
            pts, n = _sample(pinn, cfg, gen, torch.float32, device, ratio)
            te_dropped.append({"step": step, "dropped": n})
        opt.zero_grad()
        L = losses(pinn, pts, cfg, ratio, weights)
        if cfg.grad_balance and step % cfg.balance_every == 0:
            weights = balance_weights(L, pinn.parameters(), weights, cfg.balance_alpha)
            L = losses(pinn, pts, cfg, ratio, weights)
        L["total"].backward()
        opt.step()
        sched.step()
        if step % max(1, cfg.adam_steps // 60) == 0 or step == cfg.adam_steps - 1:
            history.append({"step": step, **_snapshot(L),
                            **{f"w_{k}": v for k, v in weights.items()}})
            log(f"adam {step}: {history[-1]}")
    _sync(device)
    adam_wall_s = time.perf_counter() - t0
    pts = _to_cpu64(pinn, pts)
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
           "adam_wall_s": adam_wall_s, "adam_device": device, "lbfgs_device": "cpu",
           "formulation": cfg.formulation, "history": history, "te_dropped": te_dropped,
           "final": final, "weights": weights}
    return pinn, rec


def time_steps(case, cfg, n_adam=5, n_closure=2):
    """Wall seconds per Adam step (float32, on cfg.device) and per L-BFGS closure (float64, on the
    CPU) for this config on this case, from a fresh network at the configured point counts; the
    first of each is a warm-up and not counted. With grad_balance, also seconds per balancing
    update (on the Adam device). The devices are recorded beside the times.
    """
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    ratio = stretch_ratio(case.radius)
    pinn = make_pinn(case, cfg, device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr0)
    weights = initial_weights(pinn, cfg)
    pts, _ = _sample(pinn, cfg, gen, torch.float32, device, ratio)

    def timed(fn, n, on=device):
        out = []
        for _ in range(n + 1):
            _sync(on)
            t0 = time.perf_counter()
            fn()
            _sync(on)
            out.append(time.perf_counter() - t0)
        return float(np.median(out[1:]))

    def adam():
        opt.zero_grad()
        losses(pinn, pts, cfg, ratio, weights)["total"].backward()
        opt.step()

    rec = {"adam_device": device, "lbfgs_device": "cpu", "formulation": cfg.formulation,
           "adam_s": timed(adam, n_adam)}
    if cfg.grad_balance:
        rec["balance_s"] = timed(lambda: balance_weights(losses(pinn, pts, cfg, ratio, weights),
                                                         pinn.parameters(), weights,
                                                         cfg.balance_alpha), 1)
    pts = _to_cpu64(pinn, pts)

    def closure():
        for p in pinn.parameters():
            p.grad = None
        losses(pinn, pts, cfg, ratio, weights)["total"].backward()

    rec["closure_s"] = timed(closure, n_closure, on="cpu")
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
    """The network on the dataset grid, on the CPU in float64 (whatever device trained it)."""
    ni, nj = shape
    xi = (torch.arange(ni, dtype=torch.float64) / ni)[:, None].expand(ni, nj).reshape(-1)
    eta = (torch.arange(nj, dtype=torch.float64) / (nj - 1))[None, :].expand(ni, nj).reshape(-1)
    pinn.net.to("cpu").double()
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
