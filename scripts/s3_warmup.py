"""Stage S3a: the cylinder warm-up (spec 7.6). Exits 1 if it fails its pass criterion.

    python scripts/s3_warmup.py [--formulation potential|euler] [--device auto|cpu|mps] [--tag T]

Pass: max |Cp - (1 - 4 sin^2 theta)| <= 0.1 around the cylinder at Mach 0.1, and lift and drag zero
to the scorer's resolution. The potential formulation (the default, spec 7 as amended 2026-10-04)
fixes the circulation at zero; the Euler one (the recorded negative result) penalises it. Writes
results/s3/warmup.json and runs/warmup_q.npy, or warmup_euler.json and warmup_q_euler.npy for the
Euler formulation, so the two never overwrite each other's record; `--tag T` appends _T to both
names. `--device` is the Adam phase's (PinnConfig.device; L-BFGS is always on the CPU), and the
record names both phases' devices, OMP_NUM_THREADS, the command line and the code's git commit.

It REFUSES to start if either output already exists: results/s3/warmup.json is the committed
record of the potential formulation's pass (2026-10-05, Adam on the CPU), and a plain re-run would
otherwise replace it, and its field, with a run on other hardware. Use --tag for a new run.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from apno.geometry.ogrid import TRAIN, Cylinder, grid
from apno.models.pinn import PinnCase
from apno.models.train_pinn import (
    FORMULATIONS, PinnConfig, circle_cp_error, evaluate_on_grid, train,
)
from apno.score.forces import wall_forces

REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version():
    """{"git_commit": HEAD or None, "git_dirty": tracked files modified, or None}."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": head, "git_dirty": None if status is None else bool(status)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--formulation", choices=FORMULATIONS, default="potential")
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto",
                    help="the Adam phase's device (default auto: MPS when available)")
    ap.add_argument("--tag", default="", help="append _TAG to both output names")
    args = ap.parse_args(argv)
    suffix = "" if args.formulation == "potential" else f"_{args.formulation}"
    suffix += f"_{args.tag}" if args.tag else ""
    record, field = Path(f"results/s3/warmup{suffix}.json"), Path(f"runs/warmup_q{suffix}.npy")
    for p in (record, field):
        if p.exists():
            raise SystemExit(f"{p} exists and is a warm-up record; refusing to overwrite it. "
                             "Move it aside deliberately, or pass --tag for a distinct name.")
    case = PinnCase(0.1, 0.0, Cylinder())
    cfg = PinnConfig(kutta=False, zero_circulation=True, formulation=args.formulation,
                     device=args.device)
    pinn, rec = train(case, cfg)
    q = evaluate_on_grid(pinn, TRAIN)
    err, cp = circle_cp_error(q, case.mach)
    X, Y = grid(Cylinder(), TRAIN)
    cl, cd, cm = wall_forces(X[:, 0], Y[:, 0], cp, 0.0)
    passed = err <= 0.1 and abs(cl) < 1e-2 and abs(cd) < 1e-2
    out = {"formulation": cfg.formulation, "max_cp_error": err, "cl": cl, "cd": cd, "cm": cm,
           "passed": passed,
           "wall_s": rec["wall_s"], "cpu_s": rec["cpu_s"], "final_losses": rec["final"],
           "adam_device": rec["adam_device"], "lbfgs_device": rec["lbfgs_device"],
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "argv": sys.argv[1:] if argv is None else list(argv), **code_version()}
    record.parent.mkdir(parents=True, exist_ok=True)
    field.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(out, indent=2))
    np.save(field, q)
    print(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
