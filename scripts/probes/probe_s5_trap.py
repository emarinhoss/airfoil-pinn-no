"""Probes of the S5 recipe's uniform-field trap (docs/findings.md, 2026-10-08, the author's
decision (a)), writing no S5 record.

Each arm changes one thing from T1 (AvConfig() defaults, seed 0) on validation case B (NACA 0012,
M 0.72, alpha 3) and runs the recipe's first OUTER mini-batch outer iterations exactly as
`train_pinn_av.train` does (the same seeds, batches and L-BFGS settings), except that each outer
iteration is cut at MAX_ITER iterations and MAX_EVAL closures so a probe takes minutes:

- control: T1 as built (its first outer iteration fell into the uniform state in 97 closures);
- slope_recovery: Jagtap et al.'s slope-recovery term for layer-wise slopes added to the loss,
  S = 1 / mean_k exp(omega_k), weight 1;
- fixed_slope: the slopes omega_k frozen at omega0 (n omega = 1, plain tanh);
- capped_weights: the residual weights capped at their largest value in the RMAE box, not
  renormalised, so the near field keeps the recipe's weights and the far field loses its excess;
- slope_recovery+capped: both.

Round 2, after none of those left the trap (`--round 2`, results/s5/probe_trap_round2.log): the
loss at initialisation split by term and by region, then
- balanced: lambda_res set so that lambda_res L_res = L_far + L_wall + L_lr at initialisation over
  all the residual nodes, their stated reason for 2e4 (their section II H), applied on this grid;
- balanced+fixed_slope: both.

Judged without SU2's truth: does the field leave the uniform state, do the slopes stay off zero,
does the loss fall, and does the wall Cp have a stagnation point and a suction peak?

    python scripts/probes/probe_s5_trap.py > results/s5/probe_trap.log 2>&1
    python scripts/probes/probe_s5_trap.py --round 2 > results/s5/probe_trap_round2.log 2>&1

MPS is often unavailable inside a sandbox; the log's first line says which device ran.
"""

import dataclasses
import sys
import time

import numpy as np
import torch

from apno.geometry.ogrid import FINE, TRAIN, Section, grid
from apno.models.pinn import PinnCase
from apno.models.pinn_euler_av import AdaptiveTanh, AvConfig, losses
from apno.models.train_pinn import resolve_device
from apno.models.train_pinn_av import (
    _to, boundary_points, evaluate_on_grid, make_pinn, residual_nodes,
)
from apno.physics.flow import pressure_coefficient
from apno.score.rmae import BOX

CASE = PinnCase(0.72, 3.0, Section(0.12, 0.0))
OUTER, MAX_ITER, MAX_EVAL = 3, 120, 150
ARMS = {
    "control": {},
    "slope_recovery": {"slope_recovery": True},
    "fixed_slope": {"fixed_slope": True},
    "capped_weights": {"capped": True},
    "slope_recovery+capped": {"slope_recovery": True, "capped": True},
}


def stagnation_cp(mach, g=1.4):
    return 2 / (g * mach ** 2) * ((1 + (g - 1) / 2 * mach ** 2) ** (g / (g - 1)) - 1)


def in_box(cfg):
    """The residual nodes' membership of the RMAE box, in residual_nodes' order."""
    nodes = residual_nodes(CASE, cfg)
    X, Y = grid(CASE.geometry, FINE)
    i = np.rint(nodes["xi"].numpy() * FINE[0]).astype(int)
    j = np.rint(nodes["eta"].numpy() * (FINE[1] - 1)).astype(int)
    x, y = X[i, j], Y[i, j]
    (x0, x1), (y0, y1) = BOX
    return torch.tensor((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))


def slopes(pinn):
    return [m.omega for m in pinn.net.mlp if isinstance(m, AdaptiveTanh)]


def initial_split(cfg):
    """The loss terms at initialisation over every residual node (in batches, as trained), and
    lambda_res L_res by region: wall rows j < 10, the rest of the RMAE box, outer rows j >= NJ - 10,
    everything else. Returns lambda_res for which lambda_res L_res equals the boundary terms."""
    device, dtype = resolve_device(cfg.device), torch.float32
    torch.manual_seed(cfg.seed)
    pinn = make_pinn(CASE, cfg, device, dtype)
    nodes = residual_nodes(CASE, cfg)
    bnd = _to(boundary_points(cfg), device, dtype)
    n = len(nodes["xi"])
    wr2 = torch.empty(n, dtype=torch.float64)
    for k in range(0, n, cfg.batch):
        b = _to({key: v[k:k + cfg.batch] for key, v in nodes.items()}, device, dtype)
        r = pinn.residual(b["xi"], b["eta"])
        wr2[k:k + cfg.batch] = (b["w"] * (r * r).sum(-1)).detach().cpu().double()
    L = {key: float(v.detach()) for key, v in losses(pinn, b, bnd, cfg).items()}
    res = float(wr2.mean())
    bdr = L["far"] + L["wall"] + L["periodic"]
    j = np.rint(nodes["eta"].numpy() * (FINE[1] - 1)).astype(int)
    box = in_box(cfg).numpy()
    parts = {"wall rows j < 10": j < 10, "rest of the RMAE box": box & (j >= 10),
             "outer rows j >= 183": j >= FINE[1] - 10}
    parts["elsewhere"] = ~np.logical_or.reduce(list(parts.values()))
    print(f"   lambda_res L_res {cfg.lambda_res * res:.4g} over all nodes; far {L['far']:.4g}, "
          f"wall {L['wall']:.4g}, periodic {L['periodic']:.4g}", flush=True)
    for label, m in parts.items():
        share = float(wr2[torch.tensor(m)].sum() / wr2.sum())
        print(f"   {label}: {m.mean():.1%} of the nodes, {share:.1%} of lambda_res L_res",
              flush=True)
    lam = float(f"{bdr / res:.2g}")
    print(f"   balanced lambda_res = {bdr:.4g} / {res:.4g} = {lam:.2g}", flush=True)
    return lam


def run(name, slope_recovery=False, fixed_slope=False, capped=False, lambda_res=None):
    cfg = AvConfig()
    if lambda_res is not None:
        cfg = dataclasses.replace(cfg, lambda_res=lambda_res)
    device, dtype = resolve_device(cfg.device), torch.float32
    torch.manual_seed(cfg.seed)  # as train() does, so the control repeats its first batches
    gen = torch.Generator().manual_seed(cfg.seed)
    pinn = make_pinn(CASE, cfg, device, dtype)
    if fixed_slope:
        for om in slopes(pinn):
            om.requires_grad_(False)
    nodes = residual_nodes(CASE, cfg)
    if capped:
        box = in_box(cfg)
        cap = nodes["w"][box].max()
        nodes["w"] = torch.minimum(nodes["w"], cap)
        w = nodes["w"]
        print(f"   cap {float(cap):.3g} (lambda_res * cap {cfg.lambda_res * float(cap):.3g}); "
              f"the box now holds {float(w[box].sum() / w.sum()):.1%} of the residual weight",
              flush=True)
    bnd = _to(boundary_points(cfg), device, dtype)
    perm = torch.randperm(len(nodes["xi"]), generator=gen)
    params = [p for p in pinn.parameters() if p.requires_grad]

    def objective(batch):
        L = losses(pinn, batch, bnd, cfg)
        if slope_recovery:
            L["slope"] = 1.0 / torch.stack([torch.exp(om) for om in slopes(pinn)]).mean()
            L["total"] = L["total"] + L["slope"]
        return L

    t0, evals_all = time.perf_counter(), 0
    for b in range(OUTER):
        batch = _to({k: v[perm[b * cfg.batch:(b + 1) * cfg.batch]] for k, v in nodes.items()},
                    device, dtype)
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=MAX_ITER, max_eval=MAX_EVAL,
                                tolerance_grad=cfg.lbfgs_tol_grad,
                                tolerance_change=cfg.lbfgs_tol_change,
                                history_size=cfg.lbfgs_history, line_search_fn="strong_wolfe")
        evals = [0]

        def closure():
            opt.zero_grad()
            total = objective(batch)["total"]
            total.backward()
            evals[0] += 1
            return total

        before = float(opt.step(closure).detach())
        L = {k: float(v.detach()) for k, v in objective(batch).items()}
        evals_all += evals[0]
        terms = ", ".join(f"{k} {v:.3g}" for k, v in L.items() if k != "total")
        print(f"   {b}: {before:.4g} -> {L['total']:.4g} ({evals[0]} evals); {terms}; "
              f"n omega {[round(cfg.act_scale * float(om.detach()), 3) for om in slopes(pinn)]}",
              flush=True)
    wall_s = time.perf_counter() - t0

    q = evaluate_on_grid(pinn, TRAIN)
    X, Y = grid(CASE.geometry, TRAIN)
    xw, yw = X[:, 0], Y[:, 0]
    cp = pressure_coefficient(q[:, 0, 3], CASE.mach)
    near = (X >= BOX[0][0]) & (X <= BOX[0][1]) & (np.abs(Y) <= BOX[1][1])
    upper = yw > 0
    k_max, k_min = int(np.argmax(cp)), int(np.argmin(np.where(upper, cp, np.inf)))
    print(f"   {evals_all} closures in {wall_s:.0f} s on {device}", flush=True)
    print("   spread (max - min) of rho, u, v, p: all nodes",
          [f"{np.ptp(q[..., k]):.2e}" for k in range(4)], "; RMAE box",
          [f"{np.ptp(q[near][:, k]):.2e}" for k in range(4)], flush=True)
    print(f"   mean rho {q[..., 0].mean():.4f}, mean p {q[..., 3].mean():.4f}", flush=True)
    print(f"   wall Cp: max {cp[k_max]:.3f} at x {xw[k_max]:.3f}; upper-surface min "
          f"{cp[k_min]:.3f} at x {xw[k_min]:.3f}; largest wall speed "
          f"{np.hypot(q[:, 0, 1], q[:, 0, 2]).max():.3f} U_inf", flush=True)


def main(argv):
    print("mps available:", torch.backends.mps.is_available(), flush=True)
    print(f"case B: stagnation Cp {stagnation_cp(CASE.mach):.3f}; each arm runs {OUTER} outer "
          f"iterations of at most {MAX_ITER} L-BFGS iterations and {MAX_EVAL} closures", flush=True)
    if argv[:2] != ["--round", "2"]:
        for name, opts in ARMS.items():
            print(f"{name}:", flush=True)
            run(name, **opts)
        return
    print("initial split (T1):", flush=True)
    lam = initial_split(AvConfig())
    for name, opts in {"balanced": {}, "balanced+fixed_slope": {"fixed_slope": True}}.items():
        print(f"{name} (lambda_res {lam:.2g}):", flush=True)
        run(name, lambda_res=lam, **opts)


if __name__ == "__main__":
    main(sys.argv[1:])
