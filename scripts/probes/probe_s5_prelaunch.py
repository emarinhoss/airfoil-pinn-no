"""Pre-launch checks of the S5 recipe (docs/findings.md, 2026-10-08), writing no S5 record:

1. one epoch of T1 (AvConfig() defaults, seed 0) on validation case B (NACA 0012, M 0.72, alpha 3)
   on the default device: the loss per outer iteration, the adaptive slopes, and whether the
   field at the 256x97 nodes is uniform;
2. time_closure and the gate's upper-bound projection, on the default device and on the CPU;
3. where the volume weights put the residual weight on the 512x193 collocation grid.

    python scripts/probes/probe_s5_prelaunch.py > results/s5/probe_prelaunch.log 2>&1

MPS is often unavailable inside a sandbox; the log's first line says which device ran.
"""

import dataclasses
import time

import numpy as np
import torch

from apno.geometry.ogrid import FINE, TRAIN, Section, grid
from apno.models.pinn import PinnCase
from apno.models.pinn_euler_av import AvConfig
from apno.models.train_pinn_av import (
    evaluate_on_grid, projected_upper_s, residual_nodes, time_closure, train,
)
from apno.physics.flow import pressure_coefficient

CASE = PinnCase(0.72, 3.0, Section(0.12, 0.0))


def main():
    print("mps available:", torch.backends.mps.is_available(), flush=True)
    cfg = AvConfig(epochs=1)
    t0 = time.perf_counter()
    pinn, rec = train(CASE, cfg)
    print(f"1. one epoch of T1: {time.perf_counter() - t0:.1f} s on {rec['device']}, "
          f"{rec['func_evals']} closures, {rec['stalled']} of 20 outer iterations stalled, "
          f"diverged {rec['diverged']}", flush=True)
    slopes = [float(m.omega.detach()) for m in pinn.net.mlp if hasattr(m, "omega")]
    print("   adaptive slopes omega_k:", [round(s, 6) for s in slopes], flush=True)
    q = evaluate_on_grid(pinn, TRAIN)
    cp = pressure_coefficient(q[:, 0, 3], CASE.mach)
    print("   spread (max - min) of rho, u, v, p over the 256x97 nodes:",
          [f"{np.ptp(q[..., k]):.2e}" for k in range(4)], flush=True)
    print(f"   wall Cp spread {np.ptp(cp):.2e}, largest wall speed "
          f"{np.hypot(q[:, 0, 1], q[:, 0, 2]).max():.4f} U_inf", flush=True)

    n_nodes = len(residual_nodes(CASE, AvConfig())["xi"])
    for device in ("auto", "cpu"):
        c = dataclasses.replace(AvConfig(), device=device)
        t = time_closure(CASE, c)
        hours = projected_upper_s(c, n_nodes, t["closure_s"], t["lbfgs_iter_s"]) / 3600
        print(f"2. {t['device']}: closure {t['closure_s']:.3f} s, L-BFGS two-loop "
              f"{t['lbfgs_iter_s']:.3f} s per iteration, upper bound {hours:.1f} h per gate "
              "configuration", flush=True)

    nodes = residual_nodes(CASE, AvConfig())
    X, Y = grid(CASE.geometry, FINE)
    i = np.rint(nodes["xi"].numpy() * FINE[0]).astype(int)
    j = np.rint(nodes["eta"].numpy() * (FINE[1] - 1)).astype(int)
    x, y = X[i, j], Y[i, j]
    w = nodes["w"].numpy() * AvConfig().lambda_res
    box = (x >= -0.5) & (x <= 1.5) & (np.abs(y) <= 1.0)
    print(f"3. RMAE box: {box.mean():.1%} of the residual nodes, {w[box].sum() / w.sum():.2e} of "
          f"the residual weight; outer 10 rows: {w[j >= FINE[1] - 10].sum() / w.sum():.1%}",
          flush=True)
    print(f"   lambda_res * w, median: box {np.median(w[box]):.3g}, wall row "
          f"{np.median(w[j == 0]):.3g}, outer row {np.median(w[j == FINE[1] - 1]):.3g}", flush=True)


if __name__ == "__main__":
    main()
