"""Stage S1: the data campaign. Resumable: rerun the same command after any interruption.

    SU2_BIN_DIR=... python scripts/s1_campaign.py train   # 1,104 cases at 256x97
    SU2_BIN_DIR=... python scripts/s1_campaign.py fine    # 80 cases at 512x193
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from apno.data.sampling import campaign_specs
from apno import settings
from apno.settings import CL_BOUND, CONVERGENCE, WORKERS, settings_for
from apno.su2.run import run_campaign

ROOT = Path("runs/su2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("train", "fine"))
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    settings.check_frozen()  # S1 never starts on unfrozen or self-inconsistent settings
    specs = campaign_specs(args.which)
    res = run_campaign(specs, ROOT, Path(os.environ["SU2_BIN_DIR"]), settings_for, args.workers,
                       CONVERGENCE)
    failures = [{"case_id": r["case_id"], "level": r["level"], "reason": r["reason"],
                 "inputs": r["inputs"]} for r in res if not r["converged"]]
    summary = {
        "which": args.which, "requested": len(specs), "finished": len(res),
        "converged": sum(r["converged"] for r in res), "failures": failures,
        "failure_reasons": Counter(f["reason"].split(" ")[0] for f in failures),
        "wall_s_sum": sum(r["wall_s"] for r in res), "cpu_s_sum": sum(r["cpu_s"] for r in res),
        "max_abs_cl": max((abs(r["cl"]) for r in res if r["converged"]), default=None),
        "cl_bound": CL_BOUND,
    }
    if summary["max_abs_cl"] is not None and summary["max_abs_cl"] > CL_BOUND:
        print(f"WARNING: max |CL| {summary['max_abs_cl']:.3f} exceeds CL_BOUND {CL_BOUND}; the "
              "Cauchy-stop consistency check in apno.settings assumed less", flush=True)
    out = Path("results/s1")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"campaign_{args.which}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    return 0 if summary["finished"] == summary["requested"] else 2


if __name__ == "__main__":
    sys.exit(main())
