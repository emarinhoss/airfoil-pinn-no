"""Run SU2 cases, one directory each, resumably.

A case is done iff its result.json exists. result.json is written last and atomically (temporary
file, then rename), so a killed run leaves no result and simply runs again. Stale outputs from a
killed run are deleted before SU2 starts, so they can never be read as the new run's result.
Per-case CPU time comes from os.wait4 on that one child, because RUSAGE_CHILDREN mixes concurrent
cases.

An SU2 killed by SIGINT, SIGTERM, SIGKILL or SIGHUP is an interruption, not a result: run_case
raises CaseInterrupted and writes nothing (spec 4.5). SIGSEGV, SIGABRT, SIGFPE and any other
signal are SU2 failures, recorded with the signal in the reason. A case counts as converged only if
the history passes our rule AND SU2 exited 0 AND a restart on this case's grid was read back.

A resumed case is returned only if it was run with the configuration, mesh and inputs this call
would use. A converged case run with anything else raises ResumeMismatch naming both hashes; an
unconverged one is rerun. A stored result whose exit code is an interruption signal (written by an
earlier version of this driver) is not a result and is rerun.

run_campaign holds an exclusive lock on its root, and on Ctrl-C, SIGTERM or SIGHUP it cancels the
queued cases, terminates the live SU2 processes, waits for their threads and re-raises.
"""

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from apno.geometry.ogrid import FAR_RADIUS, Section, grid, level_name, su2_text
from apno.physics.flow import local_mach, nondimensionalise
from apno.su2.config import render_config
from apno.su2.convergence import judge, read_history
from apno.su2.restart import on_grid, read_restart

OUTPUTS = ("restart.dat", "history.csv", "surface.csv", "su2.log", "result.json")
INTERRUPTS = frozenset({signal.SIGINT, signal.SIGTERM, signal.SIGKILL, signal.SIGHUP})
LOCK_NAME = ".campaign.lock"
TERMINATE_GRACE_S = 10.0


class CaseInterrupted(RuntimeError):
    """SU2 was interrupted (or the campaign is stopping): no result.json, the case reruns."""


class ResumeMismatch(RuntimeError):
    """A finished case was run with other settings than the ones now requested."""


class CampaignStopped(BaseException):
    """SIGTERM or SIGHUP reached the campaign driver."""


_LIVE = set()
_LIVE_LOCK = threading.Lock()
_STOP = threading.Event()


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    mach: float
    alpha_deg: float
    t: float
    m: float
    shape: tuple
    radius: float = FAR_RADIUS

    @property
    def level(self):
        return level_name(self.shape)

    def inputs(self):
        return {"mach": self.mach, "alpha_deg": self.alpha_deg, "t": self.t, "m": self.m}


def case_dir(root, spec):
    return Path(root) / spec.level / spec.case_id


def is_done(d):
    return (Path(d) / "result.json").exists()


def _write_json_atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _sha(path_or_text):
    data = path_or_text.read_bytes() if isinstance(path_or_text, Path) else path_or_text.encode()
    return hashlib.sha256(data).hexdigest()


def _max_surface_mach(d, X, Y, mach):
    try:
        f = on_grid(read_restart(d / "restart.dat"), X, Y)
    except (OSError, ValueError):
        return None
    q = nondimensionalise(f["Density"][:, 0], f["Momentum_x"][:, 0], f["Momentum_y"][:, 0],
                          f["Energy"][:, 0], mach)
    return float(local_mach(*q, mach).max())


def _signal_name(code):
    try:
        return signal.Signals(-code).name
    except ValueError:
        return f"signal {-code}"


def _resume(d, root, spec, cfg, mesh):
    """The stored result if it is this call's case, None if the case must be rerun; else raise."""
    res = json.loads((d / "result.json").read_text())
    code = res.get("exit_code")
    if isinstance(code, int) and code < 0 and -code in INTERRUPTS:
        return None  # a kill is not a result (spec 4.5); an earlier driver wrote this one
    stored = {"config": res.get("config_sha256"), "mesh": res.get("mesh_sha256"),
              "inputs": res.get("inputs"), "level": res.get("level"), "radius": res.get("radius")}
    current = {"config": _sha(cfg), "mesh": _sha(mesh), "inputs": spec.inputs(),
               "level": spec.level, "radius": spec.radius}
    diff = [f"{k}: stored {stored[k]} != requested {current[k]}" for k in stored
            if stored[k] != current[k]]
    if not diff:
        return res
    if not res.get("converged"):
        return None
    raise ResumeMismatch(
        f"{d}: this converged case was run with other settings ({'; '.join(diff)}). Move "
        f"{root} aside to start a campaign with the new settings, or restore the old ones.")


def run_case(spec, root, su2_bin, settings, judge_kwargs=None):
    d = case_dir(root, spec)
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    cfg = render_config(spec.mach, spec.alpha_deg, settings)
    mesh = su2_text(X, Y)
    if is_done(d):
        res = _resume(d, root, spec, cfg, mesh)
        if res is not None:
            return res
    if _STOP.is_set():
        raise CaseInterrupted(f"{spec.case_id}: campaign stopping, not started")
    d.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS + ("result.json.tmp",):
        (d / name).unlink(missing_ok=True)
    (d / "mesh.su2").write_text(mesh)
    (d / "case.cfg").write_text(cfg)

    t0 = time.perf_counter()
    with open(d / "su2.log", "w") as log:
        with _LIVE_LOCK:
            if _STOP.is_set():
                raise CaseInterrupted(f"{spec.case_id}: campaign stopping, not started")
            proc = subprocess.Popen([str(Path(su2_bin) / "SU2_CFD"), "case.cfg"], cwd=d,
                                    stdout=log, stderr=subprocess.STDOUT)
            _LIVE.add(proc)
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        finally:
            with _LIVE_LOCK:
                _LIVE.discard(proc)
    code = proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    if code < 0 and -code in INTERRUPTS:
        raise CaseInterrupted(f"{spec.case_id}: SU2 killed by {_signal_name(code)}; no result "
                              "written, the case reruns")

    hist = read_history(d / "history.csv")
    verdict = judge(hist, **(judge_kwargs or {}))
    hist = hist or {}
    last = {k: float(hist[k][-1]) if k in hist else None for k in ("CL", "CD", "CMz")}
    version = re.search(r"Release (\S+)", (d / "su2.log").read_text())
    mach_max = _max_surface_mach(d, X, Y, spec.mach)
    failures = []
    if code < 0:
        failures.append(f"SU2 killed by {_signal_name(code)}")
    elif code != 0:
        failures.append(f"SU2 exited with code {code}")
    if not (d / "restart.dat").exists():
        failures.append("no restart.dat")
    elif mach_max is None:
        failures.append("restart.dat unreadable or not on this case's grid")
    result = {
        "case_id": spec.case_id, "inputs": spec.inputs(), "level": spec.level,
        "radius": spec.radius, "converged": verdict.converged and not failures,
        "reason": "; ".join(failures + [verdict.reason]) if failures else verdict.reason,
        "verdict": verdict.as_dict(), "iterations": verdict.iterations,
        "cl": last["CL"], "cd": last["CD"], "cm": last["CMz"],
        "max_surface_mach": mach_max,
        "wall_s": wall, "cpu_s": usage.ru_utime + usage.ru_stime,
        "exit_code": code, "su2_version": version.group(1) if version else None,
        "config_sha256": _sha(cfg), "mesh_sha256": _sha(d / "mesh.su2"),
        "mglevel": settings.mglevel, "settings": asdict(settings),
    }
    _write_json_atomic(d / "result.json", result)
    return result


def _terminate_live():
    """SIGTERM every live SU2, then SIGKILL whatever is still alive after the grace period."""
    with _LIVE_LOCK:
        for p in list(_LIVE):
            p.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + TERMINATE_GRACE_S
    while time.monotonic() < deadline:
        with _LIVE_LOCK:
            if not _LIVE:
                return
        time.sleep(0.05)
    with _LIVE_LOCK:
        for p in list(_LIVE):
            p.send_signal(signal.SIGKILL)


@contextmanager
def _campaign_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    fh = open(root / LOCK_NAME, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RuntimeError(f"another campaign driver holds {root / LOCK_NAME}; refusing to run "
                           "two drivers on one root") from None
    try:
        yield
    finally:
        fh.close()


@contextmanager
def _stop_on_signals():
    """In the main thread, SIGTERM and SIGHUP raise CampaignStopped where the driver waits."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, _frame):
        raise CampaignStopped(signal.Signals(signum).name)

    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _check_resumable(specs, root, settings_for):
    """Before any SU2 starts: every finished case must match its requested settings, else raise.

    Checked up front so a mismatch cannot leave a root holding cases run with two settings.
    A settings_for error is left for the worker to report.
    """
    for spec in specs:
        d = case_dir(root, spec)
        if not is_done(d):
            continue
        try:
            settings = settings_for(spec)
        except Exception:  # noqa: BLE001 -- reported per case by the worker
            continue
        X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
        _resume(d, root, spec, render_config(spec.mach, spec.alpha_deg, settings), su2_text(X, Y))


def _run_one(spec, root, su2_bin, settings_for, judge_kwargs):
    return run_case(spec, root, su2_bin, settings_for(spec), judge_kwargs)


def run_campaign(specs, root, su2_bin, settings_for, workers, judge_kwargs=None):
    """Run every spec; a driver exception is reported, not recorded as a result, so it reruns.

    Ctrl-C, SIGTERM and SIGHUP stop the campaign: queued cases are cancelled, live SU2 processes
    are terminated (their cases leave no result.json), and the exception is re-raised. A finished
    case run with other settings raises ResumeMismatch before any SU2 starts.
    """
    root = Path(root)
    results = []
    with _campaign_lock(root), _stop_on_signals():
        _STOP.clear()
        _check_resumable(specs, root, settings_for)
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(_run_one, s, root, su2_bin, settings_for, judge_kwargs): s
                       for s in specs}
            for n, fut in enumerate(as_completed(futures), 1):
                spec = futures[fut]
                try:
                    res = fut.result()
                except ResumeMismatch:
                    raise
                except Exception as exc:  # noqa: BLE001 -- reported, case left not-done
                    print(f"[{n}/{len(specs)}] {spec.case_id}: DRIVER ERROR {exc!r}", flush=True)
                    continue
                print(f"[{n}/{len(specs)}] {spec.case_id}: {res['reason']} "
                      f"({res['iterations']} it, {res['wall_s']:.0f} s)", flush=True)
                results.append(res)
        except BaseException as exc:
            _STOP.set()
            pool.shutdown(wait=False, cancel_futures=True)
            _terminate_live()
            pool.shutdown(wait=True)
            print(f"campaign stopped ({exc!r}): {len(results)} cases finished; the rest rerun "
                  "on the next launch", flush=True)
            raise
        finally:
            pool.shutdown(wait=True)
            _STOP.clear()
    return results
