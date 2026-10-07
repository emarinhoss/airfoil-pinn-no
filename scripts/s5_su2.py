"""Stage S5, the SU2 reference (S5 design §3.1-3.2): the validation case at 256x97 and 512x193.

    SU2_BIN_DIR=... python scripts/s5_su2.py [--workers 2]

Runs Wassing, Langer & Bekemeyer's case B (NACA 0012, M 0.72, alpha 3 deg) at both levels with the
frozen S0 settings and the convergence rule, into runs/su2/<level>/s5-val-B/; if SU2 fails the rule
at either level (a recorded result with converged false), case C, then case A: the fallback order
fixed in the design. A level with no result (SU2 killed by a signal, or a driver error) is not a
failure: it is not a result and simply runs again (spec §4.5), so the script stops at that case
without trying the next one and says to re-run. The selection therefore depends only on the stored
result.json files, and a re-run resumes the same case.

Writes results/s5/su2.json with every case decided, the selected one and the unfinished one
(regenerated from the resumable runs on every call). A selection on record is never changed: if the
runs now give another one, the script refuses and leaves su2.json as it is. Exit 0 when a case is
selected, 1 when a case is unfinished (re-run to resume), 2 when none converged at both levels,
3 when the runs disagree with the selection on record.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from apno import settings
from apno.data.sampling import s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN
from apno.settings import CONVERGENCE, settings_for
from apno.su2.run import run_campaign

ROOT, RES = Path("runs/su2"), Path("results/s5")
KEEP = ("converged", "reason", "iterations", "cl", "cd", "cm", "wall_s", "cpu_s",
        "max_surface_mach")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    settings.check_frozen()
    su2_bin = Path(os.environ["SU2_BIN_DIR"])
    tried, selected, unfinished = [], None, None
    for coarse, fine in zip(s5_validation_specs(TRAIN), s5_validation_specs(FINE)):
        res = run_campaign([coarse, fine], ROOT, su2_bin, settings_for, args.workers, CONVERGENCE)
        levels = {r["level"]: {k: r[k] for k in KEEP} for r in res}
        entry = {"case_id": coarse.case_id, "inputs": coarse.inputs(), "levels": levels}
        missing = [s.level for s in (coarse, fine) if s.level not in levels]
        if missing:  # not a result, so not a failure: resume this case, never fall back on it
            unfinished = {**entry, "missing": missing}
            break
        ok = all(v["converged"] for v in levels.values())
        tried.append({**entry, "converged_both": ok})
        if ok:
            selected = coarse.case_id
            break
    path = RES / "su2.json"
    before = json.loads(path.read_text()).get("selected") if path.exists() else None
    if before and before != selected:
        print(f"REFUSING TO OVERWRITE {path}: it selects {before}, the stored runs now give "
              f"{selected}. A recorded selection is never changed; resolve it by hand.", flush=True)
        return 3
    RES.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tried": tried, "selected": selected, "unfinished": unfinished},
                               indent=2))
    print(json.dumps({"tried": [t["case_id"] for t in tried], "selected": selected,
                      "unfinished": unfinished and unfinished["case_id"]}), flush=True)
    if unfinished:
        print(f"re-run to resume {unfinished['case_id']}: no result at "
              f"{', '.join(unfinished['missing'])}", flush=True)
        return 1
    return 0 if selected else 2


if __name__ == "__main__":
    sys.exit(main())
