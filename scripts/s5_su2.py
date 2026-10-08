"""Stage S5, the SU2 reference (S5 design §3.1-3.2): the validation case at 256x97 and 512x193.

    SU2_BIN_DIR=... python scripts/s5_su2.py [--workers 2]

Runs Wassing, Langer & Bekemeyer's case B (NACA 0012, M 0.72, alpha 3 deg) at both levels with the
frozen S0 settings and the convergence rule, into runs/su2/<level>/s5-val-B/; if SU2 fails the rule
at either level (a recorded result with converged false), case C, then case A: the fallback order
fixed in the design. A recorded failure at one level decides the case even when the other level
has no result: the script falls back at once, and a re-run never runs that other level again. A
level with no result (SU2 killed by a signal, or a driver error) is not a failure: it is not a
result and simply runs again (spec §4.5), so unless the other level failed, the script stops at
that case without trying the next one and says to re-run. The selection therefore depends only on
the stored result.json files, and a re-run resumes the same case. SU2_BIN_DIR is read only when
SU2 must launch: a re-run that resumes every level needs none, and a wrong one (unset, or with no
executable SU2_CFD in it) stops the call with exit 4 instead of leaving a level with no result.

Writes results/s5/su2.json (atomically) with every case decided, the selected one and the
unfinished one (regenerated from the resumable runs on every call). A selection on record is never
changed: if the runs now give another one, or record a failure of the selected case, the script
refuses and leaves su2.json as it is; if they are missing (runs/ is not tracked, results/ is) or
unfinished, it keeps the record and says to re-run.

Exit codes:
  0  a case is selected;
  1  a case is unfinished, or the runs behind the selection on record are missing or unfinished:
     re-run to resume;
  2  no case converged at both levels;
  3  the runs disagree with the selection on record (resolve it by hand);
  4  cannot run: SU2 must launch and SU2_BIN_DIR is unset or wrong (it holds no executable
     SU2_CFD), another driver holds runs/su2/.campaign.lock, or a finished run was made with
     other settings; re-running as is cannot help, and the one-line "cannot run:" message says
     what to do.
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
from apno.su2.run import LOCK_NAME, ResumeMismatch, case_dir, is_done, run_campaign

ROOT, RES = Path("runs/su2"), Path("results/s5")
KEEP = ("converged", "reason", "iterations", "cl", "cd", "cm", "wall_s", "cpu_s",
        "max_surface_mach")


class CannotRun(Exception):
    """Exit 4: the environment or another driver stops this call; the message says what to do."""


class _Su2Bin(os.PathLike):
    """SU2_BIN_DIR, read only when SU2 is about to launch: a call that resumes every run needs none.

    run_case turns su2_bin into a path only to start SU2_CFD. If SU2_BIN_DIR is unset or holds no
    executable SU2_CFD, that start fails (a driver error, so the case is left without a result,
    which no re-run would fix) and `problem` records why for _levels to raise on.
    """

    def __init__(self):
        self.problem = None

    def __fspath__(self):
        value = os.environ.get("SU2_BIN_DIR")
        exe = os.path.join(value, "SU2_CFD") if value else None
        if not exe:
            self.problem = "SU2_BIN_DIR is not set"
        elif not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            self.problem = f"SU2_BIN_DIR holds no executable SU2_CFD ({exe})"
        else:
            return value
        raise LookupError(self.problem)


def _levels(specs, su2_bin, workers):
    """{level: the KEEP fields} of the specs' results; a level with no result is absent."""
    if not specs:
        return {}
    try:
        res = run_campaign(specs, ROOT, su2_bin, settings_for, workers, CONVERGENCE)
    except ResumeMismatch as exc:
        raise CannotRun(str(exc)) from None
    except RuntimeError as exc:
        if LOCK_NAME not in str(exc):
            raise
        raise CannotRun(f"{exc}; wait for it to finish, then re-run") from None
    if su2_bin.problem:
        raise CannotRun(f"{su2_bin.problem}, and SU2 must launch: set SU2_BIN_DIR to the "
                        "directory holding SU2_CFD, then re-run")
    return {r["level"]: {k: r[k] for k in KEEP} for r in res}


def _decide(su2_bin, workers):
    """(tried, selected, unfinished) from the runs, in the fallback order of design §3.1."""
    tried, selected, unfinished = [], None, None
    for pair in zip(s5_validation_specs(TRAIN), s5_validation_specs(FINE)):
        stored = [s for s in pair if is_done(case_dir(ROOT, s))]
        levels = _levels(stored, su2_bin, workers)  # a recorded failure here decides the case
        if all(v["converged"] for v in levels.values()):
            levels.update(_levels([s for s in pair if s not in stored], su2_bin, workers))
        entry = {"case_id": pair[0].case_id, "inputs": pair[0].inputs(), "levels": levels}
        missing = [s.level for s in pair if s.level not in levels]
        failed = not all(v["converged"] for v in levels.values())
        if missing and not failed:  # not a result, so not a failure: resume this case
            unfinished = {**entry, "missing": missing}
            break
        case = {**entry, "converged_both": not failed}
        if missing:  # failed at the other level: this one is not waited for
            case["missing"] = missing
        tried.append(case)
        if not failed:
            selected = pair[0].case_id
            break
    return tried, selected, unfinished


def _write_atomic(path, text):
    """A kill mid-write leaves the previous file (or none), never a truncated one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    settings.check_frozen()
    try:
        tried, selected, unfinished = _decide(_Su2Bin(), args.workers)
    except CannotRun as exc:
        print(f"cannot run: {exc}", flush=True)
        return 4
    path = RES / "su2.json"
    before = json.loads(path.read_text()).get("selected") if path.exists() else None
    failed = {t["case_id"] for t in tried if not t["converged_both"]}
    if before and before != selected and (unfinished is None or before in failed):
        now = f"record a failure of {before}" if before in failed else f"now give {selected}"
        print(f"REFUSING TO OVERWRITE {path}: it selects {before}, the stored runs {now}. A "
              "recorded selection is never changed; resolve it by hand.", flush=True)
        return 3
    if before and unfinished:
        print(f"{path} selects {before}, but the runs behind it are missing or unfinished (no "
              f"result for {unfinished['case_id']} at {', '.join(unfinished['missing'])}): "
              "re-run to resume; the record is kept.", flush=True)
        return 1
    RES.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps({"tried": tried, "selected": selected,
                                    "unfinished": unfinished}, indent=2))
    print(json.dumps({"tried": [t["case_id"] for t in tried], "selected": selected,
                      "unfinished": unfinished and unfinished["case_id"]}), flush=True)
    if unfinished:
        print(f"re-run to resume {unfinished['case_id']}: no result at "
              f"{', '.join(unfinished['missing'])}", flush=True)
        return 1
    return 0 if selected else 2


if __name__ == "__main__":
    sys.exit(main())
