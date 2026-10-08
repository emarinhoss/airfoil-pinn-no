"""Probes of departures from the S5 recipe that might keep it out of the uniform field
(docs/findings.md, 2026-10-08, the author's decision to probe departures), writing no S5 record.

On validation case A (NACA 0012, M 0.70, alpha 4), the last fallback, so that case B's gate stays
unused. Each arm changes one thing from T1 (AvConfig() defaults, seed 0) and repeats the recipe's
first OUTER mini-batch outer iterations as `train_pinn_av.train` runs them (the same seeds, batches
and L-BFGS settings), each cut at MAX_ITER iterations and MAX_EVAL closures:

- control: T1 as built;
- adam: ADAM Adam steps at lr 1e-3 over the recipe's mini-batches in order before the L-BFGS outer
  iterations (this study's schedule, A3, shortened);
- hard_wall: the velocity's normal component removed near the wall, u <- u - (1 - eta)^4 (u.n) n,
  on the primitive outputs (the projection A2's closure uses), so the wall term is zero exactly;
- sigma_0.1: the Fourier frequencies drawn from N(0, 0.1^2), a period of about 60 cells, not 6;
- glorot: Glorot-normal weights and zero biases under the weight normalisation, not PyTorch's
  default initialisation.

Then, after none of those left a trivial state (each trap a PDE solution violating one soft
boundary term: uniform, the wall's; at rest or a vacuum, the far field's), `--arms` the two below,
results/s5/probe_departures_hard.log:
- hard_both: both conditions exact, q <- q + eta^4 (w_inf - q) (the far field), then hard_wall's
  projection, which vanishes at eta = 1; no trivial state satisfies the residual and both;
- hard_both+adam: and the Adam steps first.

A non-finite loss stops an arm, as it stops `train` (rec["diverged"]), and is reported with where
it happened. Judged without SU2's truth, as before: does the field leave the uniform state, do the
slopes stay off zero, does the loss fall, and does the wall Cp have a stagnation point and a
suction peak?

    python scripts/probes/probe_s5_departures.py \
        --arms control,adam,hard_wall,sigma_0.1,glorot > results/s5/probe_departures.log 2>&1
    python scripts/probes/probe_s5_departures.py \
        --arms hard_both,hard_both+adam > results/s5/probe_departures_hard.log 2>&1

MPS is often unavailable inside a sandbox; the log's first line says which device ran.
"""

import dataclasses
import sys
import time

import numpy as np
import torch
from torch import nn

from apno.geometry.ogrid import TRAIN, Section, grid
from apno.models.pinn import PinnCase, wall_normal
from apno.models.pinn_euler_av import AdaptiveTanh, AvConfig, losses
from apno.models.train_pinn import resolve_device
from apno.models.train_pinn_av import (
    _to, boundary_points, evaluate_on_grid, make_pinn, residual_nodes,
)
from apno.physics.flow import pressure_coefficient
from apno.score.rmae import BOX

CASE = PinnCase(0.70, 4.0, Section(0.12, 0.0))
OUTER, MAX_ITER, MAX_EVAL, ADAM = 3, 120, 150, 400
ARMS = {
    "control": {},
    "adam": {"adam": True},
    "hard_wall": {"hard_wall": True},
    "sigma_0.1": {"sigma": 0.1},
    "glorot": {"glorot": True},
    "hard_both": {"hard_far": True, "hard_wall": True},
    "hard_both+adam": {"hard_far": True, "hard_wall": True, "adam": True},
}


def stagnation_cp(mach, g=1.4):
    return 2 / (g * mach ** 2) * ((1 + (g - 1) / 2 * mach ** 2) ** (g / (g - 1)) - 1)


def slopes(pinn):
    return [m.omega for m in pinn.net.mlp if isinstance(m, AdaptiveTanh)]


def n_omega(pinn):
    return [round(pinn.cfg.act_scale * float(om.detach()), 3) for om in slopes(pinn)]


def glorot(pinn):
    """Glorot-normal weights and zero biases, set through the weight normalisation (assigning the
    weight sets g and v from it)."""
    with torch.no_grad():
        for m in pinn.net.mlp:
            if isinstance(m, nn.Linear):
                w = torch.empty_like(m.weight)
                nn.init.xavier_normal_(w)
                m.weight = w
                m.bias.zero_()


def hard_wall(pinn):
    """The velocity's normal component removed near the wall, as A2's closure does."""
    primitive = pinn.primitive

    def projected(xi, eta):
        rho, u, v, p = primitive(xi, eta)
        nx, ny = wall_normal(xi, pinn.case.geometry)
        un = (1.0 - eta) ** 4 * (u * nx + v * ny)
        return rho, u - un * nx, v - un * ny, p

    pinn.primitive = projected  # residual, viscous and losses all call pinn.primitive


def hard_far(pinn):
    """The far field exact: q <- q + eta^4 (w_inf - q), so q = w_inf on the outer row."""
    primitive = pinn.primitive

    def blended(xi, eta):
        b = eta ** 4
        return tuple(c + b * (w - c) for c, w in zip(primitive(xi, eta), pinn.w_inf))

    pinn.primitive = blended


def end_state(pinn):
    q = evaluate_on_grid(pinn, TRAIN)
    X, Y = grid(CASE.geometry, TRAIN)
    xw, yw = X[:, 0], Y[:, 0]
    cp = pressure_coefficient(q[:, 0, 3], CASE.mach)
    near = (X >= BOX[0][0]) & (X <= BOX[0][1]) & (np.abs(Y) <= BOX[1][1])
    k_max = int(np.argmax(cp))
    k_min = int(np.argmin(np.where(yw > 0, cp, np.inf)))
    print("   spread (max - min) of rho, u, v, p: all nodes",
          [f"{np.ptp(q[..., k]):.2e}" for k in range(4)], "; RMAE box",
          [f"{np.ptp(q[near][:, k]):.2e}" for k in range(4)], flush=True)
    print(f"   mean rho {q[..., 0].mean():.4f}, mean p {q[..., 3].mean():.4f}", flush=True)
    print(f"   wall Cp: max {cp[k_max]:.3f} at x {xw[k_max]:.3f}; upper-surface min "
          f"{cp[k_min]:.3f} at x {xw[k_min]:.3f}; largest wall speed "
          f"{np.hypot(q[:, 0, 1], q[:, 0, 2]).max():.3f} U_inf", flush=True)


class Diverged(RuntimeError):
    pass


def finite(L, where):
    if not torch.isfinite(L["total"]):
        raise Diverged(where)
    return L


def run(adam=False, hard_wall_on=False, sigma=None, glorot_on=False, hard_far_on=False):
    cfg = AvConfig() if sigma is None else dataclasses.replace(AvConfig(), sigma=sigma)
    device, dtype = resolve_device(cfg.device), torch.float32
    torch.manual_seed(cfg.seed)  # as train() does
    gen = torch.Generator().manual_seed(cfg.seed)
    pinn = make_pinn(CASE, cfg, device, dtype)
    if glorot_on:
        glorot(pinn)
    if hard_far_on:
        hard_far(pinn)
    if hard_wall_on:  # after the far field's blend: the projection vanishes at eta = 1
        hard_wall(pinn)
    nodes = residual_nodes(CASE, cfg)
    bnd = _to(boundary_points(cfg), device, dtype)
    n = len(nodes["xi"])
    perm = torch.randperm(n, generator=gen)
    nb = -(-n // cfg.batch)

    def batch(k):
        idx = perm[(k % nb) * cfg.batch:(k % nb + 1) * cfg.batch]
        return _to({key: v[idx] for key, v in nodes.items()}, device, dtype)

    def snapshot(L):
        return ", ".join(f"{k} {float(v.detach()):.3g}" for k, v in L.items() if k != "total")

    t0, evals_all = time.perf_counter(), 0
    if adam:
        opt = torch.optim.Adam(pinn.parameters(), lr=cfg.adam_lr0)
        for step in range(ADAM):  # a divergence here stops the arm in main()
            opt.zero_grad()
            L = finite(losses(pinn, batch(step), bnd, cfg), f"adam step {step + 1}")
            L["total"].backward()
            opt.step()
            evals_all += 1
            if (step + 1) % 100 == 0:
                print(f"   adam {step + 1}: {float(L['total'].detach()):.4g}; {snapshot(L)}; "
                      f"n omega {n_omega(pinn)}", flush=True)
    for b in range(OUTER):
        bt = batch(b)
        opt = torch.optim.LBFGS(pinn.parameters(), lr=1.0, max_iter=MAX_ITER, max_eval=MAX_EVAL,
                                tolerance_grad=cfg.lbfgs_tol_grad,
                                tolerance_change=cfg.lbfgs_tol_change,
                                history_size=cfg.lbfgs_history, line_search_fn="strong_wolfe")
        evals, last = [0], [None]

        def closure():
            opt.zero_grad()
            total = finite(losses(pinn, bt, bnd, cfg),
                           f"outer iteration {b}, closure {evals[0] + 1}")["total"]
            total.backward()
            evals[0] += 1
            last[0] = float(total.detach())
            return total

        try:
            before = float(opt.step(closure).detach())
        except Diverged as exc:
            evals_all += evals[0]
            print(f"   diverged: non-finite loss at {exc}, the last finite {last[0]:.4g}; "
                  f"{evals_all} closures in {time.perf_counter() - t0:.0f} s on {device}",
                  flush=True)
            return
        L = losses(pinn, bt, bnd, cfg)
        evals_all += evals[0]
        print(f"   {b}: {before:.4g} -> {float(L['total'].detach()):.4g} ({evals[0]} evals); "
              f"{snapshot(L)}; n omega {n_omega(pinn)}", flush=True)
    print(f"   {evals_all} closures in {time.perf_counter() - t0:.0f} s on {device}", flush=True)
    end_state(pinn)


def main(argv):
    names = argv[1].split(",") if argv[:1] == ["--arms"] else list(ARMS)
    print("mps available:", torch.backends.mps.is_available(), flush=True)
    print(f"case A: stagnation Cp {stagnation_cp(CASE.mach):.3f}; each arm runs {OUTER} outer "
          f"iterations of at most {MAX_ITER} L-BFGS iterations and {MAX_EVAL} closures", flush=True)
    for name in names:
        opts = ARMS[name]
        print(f"{name}:", flush=True)
        try:
            run(adam=opts.get("adam", False), hard_wall_on=opts.get("hard_wall", False),
                sigma=opts.get("sigma"), glorot_on=opts.get("glorot", False),
                hard_far_on=opts.get("hard_far", False))
        except Diverged as exc:
            print(f"   diverged: non-finite loss at {exc}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
