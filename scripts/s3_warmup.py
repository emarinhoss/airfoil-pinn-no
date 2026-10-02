"""Stage S3a: the cylinder warm-up (spec 7.6). Exits 1 if it fails its pass criterion.

Pass: max |Cp - (1 - 4 sin^2 theta)| <= 0.1 around the cylinder at Mach 0.1, and lift and drag zero
to the scorer's resolution. Writes results/s3/warmup.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from apno.geometry.ogrid import TRAIN, Cylinder, grid
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, circle_cp_error, evaluate_on_grid, train
from apno.score.forces import wall_forces


def main():
    case = PinnCase(0.1, 0.0, Cylinder())
    cfg = PinnConfig(kutta=False, zero_circulation=True)
    pinn, rec = train(case, cfg)
    q = evaluate_on_grid(pinn, TRAIN)
    err, cp = circle_cp_error(q, case.mach)
    X, Y = grid(Cylinder(), TRAIN)
    cl, cd, cm = wall_forces(X[:, 0], Y[:, 0], cp, 0.0)
    passed = err <= 0.1 and abs(cl) < 1e-2 and abs(cd) < 1e-2
    out = {"max_cp_error": err, "cl": cl, "cd": cd, "cm": cm, "passed": passed,
           "wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"], "final_losses": rec["final"]}
    Path("results/s3").mkdir(parents=True, exist_ok=True)
    Path("results/s3/warmup.json").write_text(json.dumps(out, indent=2))
    np.save("runs/warmup_q.npy", q)
    print(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
