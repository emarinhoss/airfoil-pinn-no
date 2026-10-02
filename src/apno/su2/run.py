"""Run SU2 cases, one directory each, resumably.

A case is done iff its result.json exists. result.json is written last and atomically (temporary
file, then rename), so a killed run leaves no result and simply runs again. Stale outputs from a
killed run are deleted before SU2 starts, so they can never be read as the new run's result.
Per-case CPU time comes from os.wait4 on that one child, because RUSAGE_CHILDREN mixes concurrent
cases.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from apno.geometry.ogrid import FAR_RADIUS, Section, grid, level_name, write_su2
from apno.physics.flow import local_mach, nondimensionalise
from apno.su2.config import render_config
from apno.su2.convergence import judge, read_history
from apno.su2.restart import on_grid, read_restart

OUTPUTS = ("restart.dat", "history.csv", "surface.csv", "su2.log", "result.json")


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


def run_case(spec, root, su2_bin, settings, judge_kwargs=None):
    d = case_dir(root, spec)
    if is_done(d):
        return json.loads((d / "result.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS + ("result.json.tmp",):
        (d / name).unlink(missing_ok=True)
    X, Y = grid(Section(spec.t, spec.m), spec.shape, spec.radius)
    write_su2(d / "mesh.su2", X, Y)
    cfg = render_config(spec.mach, spec.alpha_deg, settings)
    (d / "case.cfg").write_text(cfg)

    t0 = time.perf_counter()
    with open(d / "su2.log", "w") as log:
        proc = subprocess.Popen([str(Path(su2_bin) / "SU2_CFD"), "case.cfg"], cwd=d,
                                stdout=log, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0

    hist = read_history(d / "history.csv")
    verdict = judge(hist, **(judge_kwargs or {}))
    hist = hist or {}
    last = {k: float(hist[k][-1]) if k in hist else None for k in ("CL", "CD", "CMz")}
    version = re.search(r"Release (\S+)", (d / "su2.log").read_text())
    result = {
        "case_id": spec.case_id, "inputs": spec.inputs(), "level": spec.level,
        "radius": spec.radius, "converged": verdict.converged, "reason": verdict.reason,
        "verdict": verdict.as_dict(), "iterations": verdict.iterations,
        "cl": last["CL"], "cd": last["CD"], "cm": last["CMz"],
        "max_surface_mach": _max_surface_mach(d, X, Y, spec.mach),
        "wall_s": wall, "cpu_s": usage.ru_utime + usage.ru_stime,
        "exit_code": proc.returncode, "su2_version": version.group(1) if version else None,
        "config_sha256": _sha(cfg), "mesh_sha256": _sha(d / "mesh.su2"),
        "mglevel": settings.mglevel,
    }
    _write_json_atomic(d / "result.json", result)
    return result


def run_campaign(specs, root, su2_bin, settings_for, workers, judge_kwargs=None):
    """Run every spec; a driver exception is reported, not recorded as a result, so it reruns."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, s, root, su2_bin, settings_for(s), judge_kwargs): s
                   for s in specs}
        for n, fut in enumerate(as_completed(futures), 1):
            spec = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- reported, case left not-done
                print(f"[{n}/{len(specs)}] {spec.case_id}: DRIVER ERROR {exc!r}", flush=True)
                continue
            print(f"[{n}/{len(specs)}] {spec.case_id}: {res['reason']} "
                  f"({res['iterations']} it, {res['wall_s']:.0f} s)", flush=True)
            results.append(res)
    return results
