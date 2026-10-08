"""Stage S5: the Euler PINN with localised artificial viscosity (S5 design §3-§5, §7).

    python scripts/s5_pinn.py gate  [--budget-hours 72] [--preflight-only] [--preflight-file PATH]
    python scripts/s5_pinn.py steps [--budget-hours 72] [--preflight-only] [--preflight-file PATH]
                                    [--shard K/N]

gate: the tuning configurations T1-T4 (design §3.4) on the validation case selected by
scripts/s5_su2.py (results/s5/su2.json), in order; the first that passes the gate (RMAE of Cp and of
local Mach each <= 1 % against SU2 at 512x193) is recipe R, written to results/s5/recipe.json. A
configuration already in results/s5/gate.json is not re-run; once recipe.json exists the gate is
decided and does not run again. A gate.json already decided (the first configuration in order that
passed, every one before it a recorded failure) whose recipe.json was never written (a stop between
the two) writes it from the recorded run straight away, before any timing, budget or truth check.
If none passes, recipe.json records config null: S5 stops after step 1 and the author decides.

steps: the ablation arms A1-A6b (design §4) and the step-3 cases (§5), each a variant of R, sharded
by plan index K mod N, recorded in results/s5/runs{suffix}.json.

Budget (design §7.1, amended 2026-10-08), checked per phase against --budget-hours. The gate times
each configuration just before it trains, by one real epoch of it (`time_epoch`: the run's first
epoch exactly, its work discarded), and REFUSES TO START it (exit 3) when the time the phase has
spent (its recorded runs, the epochs that timed them, and this timing epoch) plus cfg.epochs such
epochs exceeds the budget; --preflight-only times the next configuration to train and stops. The
steps time each unfinished run for one closure on its own case (`time_closure`, viscous term forced
on everywhere) and one iteration of L-BFGS's own recursion, re-project it from R's measured gate
run, scaled by the ratio of timed closures (A3 keeps its upper bound), and REFUSE TO START (exit 3)
when the projection of their unfinished runs exceeds the budget. --preflight-file reuses an earlier
preflight's timings by name; for the gate only a measured epoch, never an upper-bound timing.
Prediction files go to runs/predictions_s5/pinn-euler_av-<name>.h5, never runs/predictions/.

Before either phase trains or scores, the truth of every airfoil case it still has to run is loaded
(`truth_and_floor`): results/ is tracked by git but runs/ and data/ are not, so a fresh clone can
hold su2.json or recipe.json without the truth, and a run trained without it would be scored a
failure, deciding the gate or the ablation wrongly. A missing truth refuses the phase (exit 2).

Resume (design §7.3): a run recorded in gate.json, or for the steps in any runs*.json whatever the
shard layout that ran it, is finished and never re-run; a prediction file is never overwritten.
A prediction file without a record (the run stopped between the two), or a recorded run whose
scoring failed, is scored from that file, never re-trained; for the steps under any shard layout,
the re-scored entry goes back into the runs*.json that holds it. A scoring failure is recorded with
the run (status "scoring failed: ...") and exits 4: the gate stops undecided on it, the steps finish
their other runs first, and exit 4 while any run of their plan holds one, in whichever runs*.json.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

from apno.data.predictions import read_predictions, write_predictions
from apno.geometry.ogrid import TRAIN
from apno.models.pinn_euler_av import AvConfig, config_from_dict
from apno.models.train_pinn_av import (
    evaluate_on_grid, projected_upper_s, time_closure, time_epoch, train,
)
from apno.score.s5 import (
    ABLATION_AIRFOIL, CYLINDER, STEP3_EXT, STEP3_IN, all_specs, pinn_case, score_run,
    truth_and_floor,
)

REPO = Path(__file__).resolve().parents[1]
RES, PRED5 = Path("results/s5"), Path("runs/predictions_s5")
BUDGET_HOURS = 72.0
SCORING_FAILED = "scoring failed"


def gate_configs():
    base, rep = AvConfig(), dataclasses.replace
    return [("T1", base), ("T2", rep(base, sensor="B")), ("T3", rep(base, viscosity="scalar")),
            ("T4", rep(base, te_exclusion=0.02))]


def steps_plan(r, validation_id):
    """[(name, case_id, cfg)] in priority order (design §4, §5)."""
    rep = dataclasses.replace
    runs = [("A1", validation_id, rep(r, weighting="uniform", lambda_res=1.0, w_boundary=10.0)),
            ("A2", validation_id, rep(r, closure="enthalpy")),
            ("A3", validation_id, rep(r, optimizer="adam_lbfgs")),
            ("A4", validation_id, rep(r, encoding="periodic")),
            ("A5", validation_id, rep(r, collocation="random")),
            ("A6a", CYLINDER, r), ("A6b", ABLATION_AIRFOIL, r)]
    return runs + [(cid, cid, r) for cid in STEP3_IN + STEP3_EXT]


def shard(runs, k, n):
    return [x for i, x in enumerate(runs) if i % n == k]


def parse_shard(text):
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard wants K/N, got {text!r}") from None
    if not 0 <= k < n:
        raise argparse.ArgumentTypeError(f"--shard needs 0 <= K < N, got {text!r}")
    return k, n


def code_version():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def prediction_path(name):
    return PRED5 / f"pinn-euler_av-{name}.h5"


def _write(path, obj):
    """Write through a temporary file, so a stop mid-write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


def _score(q, case_id, specs):
    """score_run, a failure kept as a status so the trained run is recorded, never lost."""
    try:
        return score_run(q, case_id, specs)
    except Exception as exc:
        print(f"{SCORING_FAILED} on {case_id}: {exc!r}", flush=True)
        return {"status": f"{SCORING_FAILED}: {type(exc).__name__}: {exc}", "passed": False,
                "collapsed": None}


def unscored(entry):
    return entry["score"].get("status", "").startswith(SCORING_FAILED)


def next_action(name, entry):
    """What a run still needs: "train", "score" (from its prediction file) or None (finished)."""
    has_file = prediction_path(name).exists()
    if entry is None:
        return "score" if has_file else "train"
    return "score" if unscored(entry) and has_file else None


def truth_ok(case_ids):
    """True when every airfoil case among case_ids has its truth; the missing ones are printed.
    The cylinder is scored against potential flow and needs none."""
    missing = {}
    for cid in dict.fromkeys(case_ids):
        if cid == CYLINDER:
            continue
        try:
            if truth_and_floor(cid)[0] is not None:
                continue
            missing[cid] = "not run, not converged, or not in the dataset"
        except Exception as exc:
            missing[cid] = f"{type(exc).__name__}: {exc}"
    for cid, why in missing.items():
        print(f"no truth for {cid} ({why}): SU2 under runs/su2, or data/hdf5", flush=True)
    if missing:
        print("REFUSING TO START: a run without its truth would be trained, then scored a "
              "failure", flush=True)
    return not missing


def run_one(name, case_id, cfg, specs, argv=None, code=None):
    """Train, evaluate, write the prediction file (unless diverged), score: the record entry. It
    carries the command line (argv), the code (`code_version()` read when the process started,
    passed by main: the process runs the code it imported then, so a commit or an edit made while
    an earlier run trained is not the code that ran; read here when not passed) and both devices:
    training's, and evaluation's, always the CPU (design §2.5, §3.5)."""
    path = prediction_path(name)
    if path.exists():
        raise FileExistsError(f"{path} is finished: it is scored, never overwritten")
    code = code_version() if code is None else code
    pinn, rec = train(pinn_case(case_id, specs), cfg)
    keep = ("wall_s", "cpu_s", "device", "func_evals", "stalled", "diverged", "reason", "final",
            "history")
    entry = {"case_id": case_id, "cfg": dataclasses.asdict(cfg), "argv": argv, **code,
             **{k: rec[k] for k in keep}, "eval_device": "cpu"}
    if rec["diverged"]:
        entry["score"] = {"status": "diverged", "passed": False, "collapsed": None}
        return entry
    q = evaluate_on_grid(pinn, TRAIN)
    tmp = path.with_name(path.name + ".tmp")  # a file at `path` is always whole: resume reads it
    write_predictions(tmp, [case_id], q[None], {
        "method": "pinn", "tag": path.stem, "variant": f"euler_av-{name}",
        "formulation": "euler_av", "seed": cfg.seed, "n_train": 0,
        "train_wall_s": rec["wall_s"], "train_cpu_s": rec["cpu_s"], "query_s": rec["wall_s"],
        "train_device": rec["device"]})
    tmp.replace(path)
    entry["score"] = _score(q, case_id, specs)
    return entry


def recover(name, case_id, cfg, specs, entry=None):
    """Score a finished run from its prediction file, never training it again (design §7.3): a
    recorded run whose scoring failed keeps its entry; a file with no record gets an entry from
    the file's attributes (times, device), marked `recovered_from`."""
    path = prediction_path(name)
    by_case, attrs = read_predictions(path)
    if entry is None:
        entry = {"case_id": case_id, "cfg": dataclasses.asdict(cfg), "argv": None,
                 "git_commit": None, "git_dirty": None, "wall_s": attrs["train_wall_s"],
                 "cpu_s": attrs["train_cpu_s"], "device": attrs["train_device"],
                 "func_evals": None, "stalled": None, "diverged": False, "reason": None,
                 "final": None, "history": None, "eval_device": "cpu",
                 "recovered_from": str(path)}
    print(f"{name}: scoring {path}, finished, not re-trained", flush=True)
    return {**entry, "score": _score(by_case[case_id], case_id, specs)}


def held_timings(preflight_file):
    return json.loads(Path(preflight_file).read_text())["timing_s"] if preflight_file else {}


def timings(runs, specs, preflight_file=None):
    """{name: {"device", "closure_s", "lbfgs_iter_s"}} per run, timed on its own case or reused
    by name; an older preflight file's timings have no lbfgs_iter_s, projected as 0."""
    held = held_timings(preflight_file)
    out = {}
    for name, cid, cfg in runs:
        out[name] = held.get(name) or time_closure(pinn_case(cid, specs), cfg)
        print(f"preflight {name} on {cid}: {out[name]}", flush=True)
    return out


def reprojected_hours(cfg, closure_s, recipe, lbfgs_iter_s=0.0):
    if cfg.optimizer == "adam_lbfgs":
        return projected_upper_s(cfg, 0, closure_s, lbfgs_iter_s) / 3600
    g = recipe["gate_run"]
    return g["wall_s"] * closure_s / g["closure_s"] / 3600


def _budget(path, timing, hours, budget, note):
    total = sum(hours.values())
    _write(path, {"timing_s": timing, "hours": hours, "projected_total_hours": total,
                  "budget_hours": budget, "within_budget": total <= budget, "note": note})
    print(f"projected {total:.1f} h for {len(hours)} runs against {budget:.1f} h", flush=True)
    if total > budget:
        print("REFUSING TO START: the projection exceeds the budget. Raise --budget-hours "
              "deliberately, on record, or cut runs as a recorded decision.", flush=True)
        return False
    return True


GATE_NOTE = ("measured: cfg.epochs times one real epoch, after the time the gate has spent "
             "(design §7.1, amended 2026-10-08)")


def gate_spent_s(gate):
    """Seconds the gate has spent on its recorded runs: each run's training and the epoch that
    timed it."""
    return sum((e.get("wall_s") or 0.0) + (e.get("epoch_timing") or {}).get("epoch_s", 0.0)
               for e in gate["configs"].values())


def gate_epoch(name, case, cfg, held):
    """One measured epoch of a gate configuration: reused by name when an earlier preflight holds
    one, else timed now."""
    t = held.get(name) or {}
    return t if "epoch_s" in t else time_epoch(case, cfg)


def gate_check(pre, name, cfg, t, spent_s, budget):
    """One gate configuration against the budget: spent_s (the phase's, this configuration's timing
    epoch included) plus cfg.epochs measured epochs. Recorded in results/s5/preflight_gate.json;
    False refuses it."""
    hours = cfg.epochs * t["epoch_s"] / 3600
    total = spent_s / 3600 + hours
    ok = total <= budget
    pre["timing_s"][name], pre["hours"][name] = t, hours
    pre["checks"][name] = {"spent_hours": spent_s / 3600, "projected_hours": hours,
                           "total_hours": total, "within_budget": ok}
    pre.update(projected_total_hours=total, budget_hours=budget, within_budget=ok)
    _write(RES / "preflight_gate.json", pre)
    print(f"{name}: {spent_s / 3600:.1f} h spent in the gate, {hours:.1f} h projected "
          f"({cfg.epochs} epochs of {t['epoch_s']:.0f} s), {total:.1f} h against {budget:.1f} h",
          flush=True)
    if not ok:
        print("REFUSING TO START: the projection exceeds the budget. Raise --budget-hours "
              "deliberately, on record, or cut runs as a recorded decision.", flush=True)
    return ok


def decided(gate, configs):
    """The configuration a recorded gate is decided on: in order, the first that passed, every one
    before it a recorded failure; None while one of them still needs training or scoring."""
    for name, _ in configs:
        entry = gate["configs"].get(name)
        if next_action(name, entry) or unscored(entry):
            return None
        if entry["score"].get("passed"):
            return name
    return None


def write_recipe(name, entry, vid):
    _write(RES / "recipe.json", {"config": name, "cfg": entry["cfg"], "validation_case": vid,
                                 "gate_run": {"wall_s": entry["wall_s"],
                                              "closure_s": entry["closure_s"]}})
    print(f"{name} passed the gate: it is recipe R", flush=True)
    return 0


def run_gate(args, specs):
    su2_path = RES / "su2.json"
    vid = json.loads(su2_path.read_text()).get("selected") if su2_path.exists() else None
    if not vid:
        print("no validation case: run scripts/s5_su2.py first (results/s5/su2.json)", flush=True)
        return 2
    if (RES / "recipe.json").exists():
        print("results/s5/recipe.json exists: the gate is decided", flush=True)
        return 0
    gate_path = RES / "gate.json"
    gate = (json.loads(gate_path.read_text()) if gate_path.exists()
            else {"validation_case": vid, "configs": {}})
    configs = gate_configs()
    r = decided(gate, configs)
    if r:  # a stop between the gate.json and recipe.json writes: nothing to time, budget or check
        return write_recipe(r, gate["configs"][r], vid)
    act = {n: next_action(n, gate["configs"].get(n)) for n, _ in configs}
    if any(act.values()) and not truth_ok([vid]):
        return 2
    case, held = pinn_case(vid, specs), held_timings(args.preflight_file)
    pre = {"timing_s": {}, "hours": {}, "checks": {}, "note": GATE_NOTE}
    if args.preflight_only:  # the next configuration to train, timed and checked; nothing runs
        nxt = [(n, c) for n, c in configs if act[n] == "train"][:1]
        for name, cfg in nxt:
            t = gate_epoch(name, case, cfg, held)
            if not gate_check(pre, name, cfg, t, gate_spent_s(gate) + t["epoch_s"],
                              args.budget_hours):
                return 3
        return 0
    for name, cfg in configs:
        entry = gate["configs"].get(name)
        action = next_action(name, entry)
        if action == "train":
            t = gate_epoch(name, case, cfg, held)
            if not gate_check(pre, name, cfg, t, gate_spent_s(gate) + t["epoch_s"],
                              args.budget_hours):
                return 3
            entry = {**run_one(name, vid, cfg, specs, args.argv, args.code), "epoch_timing": t}
        elif action == "score":
            entry = recover(name, vid, cfg, specs, entry)
        if action:
            if entry["score"].get("passed"):  # kept with the run: recipe.json is written from it
                entry["closure_s"] = time_closure(case, cfg)["closure_s"]
            gate["configs"][name] = entry
            _write(gate_path, gate)
        entry = gate["configs"][name]
        if unscored(entry):
            p = prediction_path(name)
            print(f"{name}: {entry['score']['status']}; the gate stops undecided. "
                  + (f"Re-run once fixed: it is scored from {p}, not re-trained." if p.exists()
                     else f"{p} is gone: remove {name} from {gate_path} deliberately to re-train."),
                  flush=True)
            return 4
        if entry["score"].get("passed"):
            return write_recipe(name, entry, vid)
    _write(RES / "recipe.json", {"config": None, "cfg": None, "validation_case": vid,
                                 "gate_run": None})
    print("no configuration passed the gate: S5 stops after step 1", flush=True)
    return 0


def merged_runs():
    """Every steps record, whatever its shard layout, as ({name: entry}, {name: the file holding
    it}). A file that sorts later wins, as in scripts/s5_score.py's load_runs."""
    done, owner = {}, {}
    for p in sorted(RES.glob("runs*.json")):
        for name, entry in json.loads(p.read_text()).items():
            done[name], owner[name] = entry, p
    return done, owner


def record_entry(path, name, entry):
    """Put one entry into a steps record, read again first so an entry written since is kept."""
    rec = json.loads(path.read_text()) if path.exists() else {}
    rec[name] = entry
    _write(path, rec)


def run_steps(args, specs):
    rp = RES / "recipe.json"
    if not rp.exists():
        print("no results/s5/recipe.json: run the gate phase first", flush=True)
        return 2
    recipe = json.loads(rp.read_text())
    if recipe["config"] is None:
        print("the gate failed: S5 stopped after step 1", flush=True)
        return 0
    runs = shard(steps_plan(config_from_dict(recipe["cfg"]), recipe["validation_case"]),
                 *args.shard)
    k, n = args.shard
    suffix = "" if n == 1 else f"_shard{k}of{n}"
    rec_path = RES / f"runs{suffix}.json"
    done, owner = merged_runs()
    act = {name: next_action(name, done.get(name)) for name, _, _ in runs}
    work = [x for x in runs if act[x[0]]]
    if not truth_ok(cid for _, cid, _ in work):
        return 2
    todo = [x for x in work if act[x[0]] == "train"]
    timing = timings(todo, specs, args.preflight_file)
    hours = {name: reprojected_hours(cfg, timing[name]["closure_s"], recipe,
                                     timing[name].get("lbfgs_iter_s", 0.0))
             for name, _, cfg in todo}
    if not _budget(RES / f"preflight_steps{suffix}.json", timing, hours, args.budget_hours,
                   "re-projected from R's measured gate run; A3 at its upper bound"):
        return 3
    if args.preflight_only:
        return 0
    for name, cid, cfg in work:
        if act[name] == "train":
            entry, path = run_one(name, cid, cfg, specs, args.argv, args.code), rec_path
        else:  # back into the file that holds it: a copy elsewhere could hide behind the stale one
            entry, path = recover(name, cid, cfg, specs, done.get(name)), owner.get(name, rec_path)
        record_entry(path, name, entry)
        done[name], owner[name] = entry, path
    failed = [name for name, _, _ in runs if name in done and unscored(done[name])]
    if failed:
        print(f"{SCORING_FAILED}: {', '.join(failed)}", flush=True)
        for name in failed:
            p = prediction_path(name)
            print(f"{name} ({owner[name]}): " + (
                f"re-run once fixed: it is scored from {p}, not re-trained." if p.exists()
                else f"{p} is gone: remove {name} from {owner[name]} deliberately to re-train."),
                flush=True)
        return 4
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("gate", "steps"))
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--preflight-file", type=Path, default=None, metavar="PATH")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="K/N")
    args = ap.parse_args(argv)
    args.argv = sys.argv[1:] if argv is None else list(argv)
    args.code = code_version()  # once: every run of this process trains the code imported now
    if args.phase == "gate" and args.shard != (0, 1):
        ap.error("the gate is sequential: --shard applies to the steps phase only")
    RES.mkdir(parents=True, exist_ok=True)
    specs = all_specs()
    return run_gate(args, specs) if args.phase == "gate" else run_steps(args, specs)


if __name__ == "__main__":
    sys.exit(main())
