import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from apno.geometry.ogrid import TRAIN
from apno.models.pinn_euler_av import AvConfig

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_pinn.py"


def _diff(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    return {k: v for k, v in db.items() if da[k] != v}


@pytest.fixture
def s5(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s5_pinn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    Path("results/s5").mkdir(parents=True)
    Path("results/s5/su2.json").write_text(json.dumps({"tried": [], "selected": "s5-val-B"}))
    mod.trained, mod.verdicts, mod.closure_s, mod.timed = [], [], 1e-6, []
    mod.no_truth = {mod.CYLINDER: None}  # case_id -> None (missing) or the exception it raises

    def fake_train(case, cfg, log=print):
        mod.trained.append(cfg)
        return object(), {"wall_s": 100.0, "cpu_s": 50.0, "device": "cpu", "func_evals": 10,
                          "stalled": 0, "diverged": False, "reason": None,
                          "final": {"total": 1.0}, "history": []}

    def fake_score(q, case_id, specs):
        ok = mod.verdicts.pop(0) if mod.verdicts else False
        if isinstance(ok, BaseException):
            raise ok
        return {"status": "ok", "passed": ok, "collapsed": False}

    def fake_truth(case_id):
        if case_id not in mod.no_truth:
            return np.ones(TRAIN + (4,)), np.ones(TRAIN + (4,)), 0.9
        if mod.no_truth[case_id] is not None:
            raise mod.no_truth[case_id]
        return None, None, None

    def fake_time(case, cfg):
        mod.timed.append(cfg)
        return {"device": "cpu", "closure_s": mod.closure_s}

    monkeypatch.setattr(mod, "train", fake_train)
    monkeypatch.setattr(mod, "evaluate_on_grid", lambda pinn, shape: np.ones(shape + (4,)))
    monkeypatch.setattr(mod, "score_run", fake_score)
    monkeypatch.setattr(mod, "truth_and_floor", fake_truth)
    monkeypatch.setattr(mod, "time_closure", fake_time)
    return mod


class Killed(BaseException):
    """A hard stop (a kill, a power cut) mid-run: not an Exception, so never caught as a failure."""


def _recipe(wall_s=3600.0, closure_s=0.1):
    Path("results/s5/recipe.json").write_text(json.dumps({
        "config": "T1", "cfg": dataclasses.asdict(AvConfig()), "validation_case": "s5-val-B",
        "gate_run": {"wall_s": wall_s, "closure_s": closure_s}}))


def test_gate_configs_change_one_thing_each(s5):
    base = AvConfig()
    got = {n: _diff(base, c) for n, c in s5.gate_configs()}
    assert list(got) == ["T1", "T2", "T3", "T4"]
    assert got == {"T1": {}, "T2": {"sensor": "B"}, "T3": {"viscosity": "scalar"},
                   "T4": {"te_exclusion": 0.02}}


def test_steps_plan_reverts_one_ingredient_per_arm(s5):
    r = AvConfig(sensor="B")
    plan = s5.steps_plan(r, "s5-val-B")
    names = ["A1", "A2", "A3", "A4", "A5", "A6a", "A6b", *s5.STEP3_IN, *s5.STEP3_EXT]
    assert [n for n, _, _ in plan] == names
    d = {n: _diff(r, c) for n, _, c in plan}
    assert d["A1"] == {"weighting": "uniform", "lambda_res": 1.0, "w_boundary": 10.0}
    assert d["A2"] == {"closure": "enthalpy"} and d["A3"] == {"optimizer": "adam_lbfgs"}
    assert d["A4"] == {"encoding": "periodic"} and d["A5"] == {"collocation": "random"}
    assert all(d[n] == {} for n in names[5:])
    cases = {n: c for n, c, _ in plan}
    assert cases["A1"] == "s5-val-B" and cases["A6a"] == "cylinder"
    assert cases["A6b"] == "test_in-0022" and cases["test_ext-0006"] == "test_ext-0006"


def test_the_gate_stops_at_the_first_pass(s5):
    s5.verdicts = [False, True]
    assert s5.main(["gate"]) == 0
    assert len(s5.trained) == 2
    recipe = json.loads(Path("results/s5/recipe.json").read_text())
    assert recipe["config"] == "T2" and recipe["cfg"]["sensor"] == "B"
    assert recipe["gate_run"] == {"wall_s": 100.0, "closure_s": 1e-6}
    gate = json.loads(Path("results/s5/gate.json").read_text())
    assert list(gate["configs"]) == ["T1", "T2"] and gate["validation_case"] == "s5-val-B"
    assert Path("runs/predictions_s5/pinn-euler_av-T1.h5").exists()
    assert not Path("runs/predictions").exists()


def test_the_gate_resumes_after_recorded_failures(s5):
    entry = {"score": {"status": "ok", "passed": False, "collapsed": False},
             "cfg": dataclasses.asdict(AvConfig()), "wall_s": 1.0}
    Path("results/s5/gate.json").write_text(json.dumps(
        {"validation_case": "s5-val-B", "configs": {"T1": entry}}))
    s5.verdicts = [True]
    assert s5.main(["gate"]) == 0
    assert len(s5.trained) == 1 and s5.trained[0].sensor == "B"
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] == "T2"


def test_a_decided_gate_does_not_run_again(s5):
    _recipe()
    assert s5.main(["gate"]) == 0 and s5.trained == []


def test_no_pass_records_a_null_recipe_and_the_steps_stop(s5):
    s5.verdicts = [False] * 4
    assert s5.main(["gate"]) == 0 and len(s5.trained) == 4
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] is None
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 4


def test_the_gate_refuses_past_its_budget(s5):
    s5.closure_s = 1.0
    assert s5.main(["gate"]) == 3 and s5.trained == []
    pre = json.loads(Path("results/s5/preflight_gate.json").read_text())
    assert pre["hours"]["T1"] == pytest.approx(40 * 20 * 1250 / 3600)
    assert pre["within_budget"] is False


def test_the_gate_needs_a_validation_case(s5):
    Path("results/s5/su2.json").write_text(json.dumps({"tried": [], "selected": None}))
    assert s5.main(["gate"]) == 2 and s5.trained == []


def test_steps_reproject_from_the_gate_run_and_write_only_under_predictions_s5(s5):
    _recipe(wall_s=3600.0, closure_s=0.1)
    s5.closure_s = 0.2
    assert s5.main(["steps", "--preflight-only"]) == 0 and s5.trained == []
    pre = json.loads(Path("results/s5/preflight_steps.json").read_text())
    assert pre["hours"]["A1"] == pytest.approx(2.0)
    assert pre["hours"]["A3"] == pytest.approx((30000 + 6250) * 0.2 / 3600)
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 13
    record = json.loads(Path("results/s5/runs.json").read_text())
    assert list(record) == [n for n, _, _ in s5.steps_plan(AvConfig(), "s5-val-B")]
    assert record["A6a"]["case_id"] == "cylinder" and record["A1"]["score"]["status"] == "ok"
    assert all(p.name.startswith("pinn-euler_av-") for p in Path("runs/predictions_s5").iterdir())
    assert not Path("runs/predictions").exists()


def test_steps_shard_by_plan_index(s5):
    _recipe()
    assert s5.main(["steps", "--shard", "1/2"]) == 0 and len(s5.trained) == 6
    record = json.loads(Path("results/s5/runs_shard1of2.json").read_text())
    assert list(record) == ["A2", "A4", "A6a", s5.STEP3_IN[0], s5.STEP3_IN[2], s5.STEP3_EXT[1]]


def test_steps_need_a_recipe(s5):
    assert s5.main(["steps"]) == 2 and s5.trained == []


def test_a_diverged_run_is_recorded_without_a_prediction_file(s5, monkeypatch):
    def diverging(case, cfg, log=print):
        s5.trained.append(cfg)
        return object(), {"wall_s": 5.0, "cpu_s": 5.0, "device": "cpu", "func_evals": 1,
                          "stalled": 0, "diverged": True, "reason": "non-finite loss",
                          "final": {}, "history": []}

    monkeypatch.setattr(s5, "train", diverging)
    assert s5.main(["gate"]) == 0
    gate = json.loads(Path("results/s5/gate.json").read_text())
    assert gate["configs"]["T1"]["score"] == {"status": "diverged", "passed": False,
                                              "collapsed": None}
    assert not Path("runs/predictions_s5/pinn-euler_av-T1.h5").exists()


def _files():
    return {p.name: p.read_bytes() for p in Path("runs/predictions_s5").iterdir()}


def test_steps_skip_runs_finished_under_another_shard_layout(s5):
    _recipe()
    assert s5.main(["steps", "--shard", "0/2"]) == 0 and s5.main(["steps", "--shard", "1/2"]) == 0
    assert len(s5.trained) == 13
    held = _files()
    assert s5.main(["steps"]) == 0 and s5.main(["steps", "--shard", "0/3"]) == 0
    assert len(s5.trained) == 13 and _files() == held
    assert not Path("results/s5/runs.json").exists()
    assert not Path("results/s5/runs_shard0of3.json").exists()


def test_a_run_recorded_by_a_shard_is_not_retrained_unsharded(s5):
    _recipe()
    entry = {"case_id": "s5-val-B", "cfg": dataclasses.asdict(AvConfig()), "wall_s": 1.0,
             "score": {"status": "ok", "passed": True, "collapsed": False}}
    Path("results/s5/runs_shard0of2.json").write_text(json.dumps({"A1": entry}))
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 12
    assert "A1" not in json.loads(Path("results/s5/runs.json").read_text())
    assert not Path("runs/predictions_s5/pinn-euler_av-A1.h5").exists()


def test_a_prediction_file_without_a_record_is_scored_not_retrained(s5):
    s5.verdicts = [Killed()]  # stopped between writing T1's file and recording it
    with pytest.raises(Killed):
        s5.main(["gate"])
    path = Path("runs/predictions_s5/pinn-euler_av-T1.h5")
    assert len(s5.trained) == 1 and path.exists() and not Path("results/s5/gate.json").exists()
    held = path.read_bytes()
    s5.verdicts = [True]
    assert s5.main(["gate"]) == 0 and len(s5.trained) == 1 and path.read_bytes() == held
    t1 = json.loads(Path("results/s5/gate.json").read_text())["configs"]["T1"]
    assert t1["recovered_from"] == str(path) and t1["wall_s"] == 100.0 and t1["score"]["passed"]
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] == "T1"


def test_a_steps_prediction_file_without_a_record_is_scored_not_retrained(s5):
    _recipe()
    s5.verdicts = [Killed()]
    with pytest.raises(Killed):
        s5.main(["steps"])
    held = _files()
    assert list(held) == ["pinn-euler_av-A1.h5"] and len(s5.trained) == 1
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 13
    assert _files()["pinn-euler_av-A1.h5"] == held["pinn-euler_av-A1.h5"]
    a1 = json.loads(Path("results/s5/runs.json").read_text())["A1"]
    assert a1["score"]["status"] == "ok" and a1["cpu_s"] == 50.0 and "recovered_from" in a1


def test_a_scoring_failure_keeps_the_run_and_leaves_the_gate_undecided(s5):
    s5.verdicts = [RuntimeError("no lattice")]
    assert s5.main(["gate"]) == 4 and len(s5.trained) == 1
    t1 = json.loads(Path("results/s5/gate.json").read_text())["configs"]["T1"]
    assert t1["score"]["status"] == "scoring failed: RuntimeError: no lattice"
    assert t1["score"]["passed"] is False and not Path("results/s5/recipe.json").exists()
    s5.verdicts = [True]
    assert s5.main(["gate"]) == 0 and len(s5.trained) == 1
    assert json.loads(Path("results/s5/recipe.json").read_text())["config"] == "T1"


def test_a_scoring_failure_in_the_steps_is_rescored_not_retrained(s5):
    _recipe()
    s5.verdicts = [RuntimeError("no lattice")]
    assert s5.main(["steps"]) == 4 and len(s5.trained) == 13
    rec = json.loads(Path("results/s5/runs.json").read_text())
    assert rec["A1"]["score"]["status"].startswith("scoring failed")
    s5.verdicts = [True]
    assert s5.main(["steps"]) == 0 and len(s5.trained) == 13
    a1 = json.loads(Path("results/s5/runs.json").read_text())["A1"]
    assert a1["score"]["passed"] is True and a1["func_evals"] == 10


@pytest.mark.parametrize("finish", [[], ["--shard", "0/3"]], ids=["unsharded", "shard0of3"])
def test_a_scoring_failure_under_another_shard_layout_is_rescored_in_its_file(s5, finish):
    _recipe()
    s5.verdicts = [RuntimeError("no lattice")]  # A1, the first run of shard 0/2
    assert s5.main(["steps", "--shard", "0/2"]) == 4 and s5.main(["steps", "--shard", "1/2"]) == 0
    assert len(s5.trained) == 13
    held = _files()
    s5.verdicts = [True]
    assert s5.main(["steps", *finish]) == 0
    assert len(s5.trained) == 13 and _files() == held
    a1 = json.loads(Path("results/s5/runs_shard0of2.json").read_text())["A1"]
    assert a1["score"]["passed"] is True and a1["func_evals"] == 10
    # written back where it was: a re-scored copy in another file would hide behind the stale one
    assert sorted(p.name for p in Path("results/s5").glob("runs*.json")) == [
        "runs_shard0of2.json", "runs_shard1of2.json"]


def test_a_scoring_failure_without_its_file_fails_the_steps_under_any_layout(s5):
    _recipe()
    s5.verdicts = [RuntimeError("no lattice")]
    assert s5.main(["steps", "--shard", "0/2"]) == 4 and s5.main(["steps", "--shard", "1/2"]) == 0
    Path("runs/predictions_s5/pinn-euler_av-A1.h5").unlink()
    assert s5.main(["steps"]) == 4 and len(s5.trained) == 13
    assert not Path("results/s5/runs.json").exists()


def test_the_gate_without_its_truth_trains_nothing(s5):
    s5.no_truth["s5-val-B"] = FileNotFoundError("runs/su2/.../restart.dat")
    assert s5.main(["gate"]) == 2 and s5.trained == [] and s5.timed == []
    assert not Path("results/s5/gate.json").exists()
    assert not Path("results/s5/recipe.json").exists()
    s5.no_truth["s5-val-B"] = None
    assert s5.main(["gate"]) == 2 and s5.trained == []
    assert not Path("results/s5/recipe.json").exists()


def test_steps_with_a_missing_truth_train_nothing(s5):
    _recipe()
    s5.no_truth[s5.STEP3_EXT[2]] = None
    assert s5.main(["steps"]) == 2 and s5.trained == [] and s5.timed == []
    assert not Path("results/s5/runs.json").exists()
    assert s5.main(["steps", "--shard", "1/2"]) == 0 and len(s5.trained) == 6  # not its shard
