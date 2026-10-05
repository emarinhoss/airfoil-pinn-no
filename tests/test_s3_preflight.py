import importlib.util
import json
from pathlib import Path

import pytest

from apno.data.sampling import case_specs
from apno.geometry.ogrid import TRAIN

SCRIPT = Path(__file__).parents[1] / "scripts" / "s3_pinn.py"


@pytest.fixture
def s3(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s3_pinn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    ids = [s.case_id for s in case_specs("test_in", TRAIN)[:6]]
    cases = [{"case_id": c, "max_surface_mach": 0.6 + 0.1 * k,
              "regime": "supercritical" if k >= 4 else "subcritical"} for k, c in enumerate(ids)]
    Path("results/s1").mkdir(parents=True)
    Path("results/s1/showcase.json").write_text(json.dumps({"cases": cases}))
    timed = []

    def fake_time_steps(case, cfg, n_adam, n_closure):
        timed.append(cfg)
        rec = {"adam_s": 10.0 if cfg.viscosity else 1.0, "closure_s": 2.0}
        return {**rec, "balance_s": 5.0} if cfg.grad_balance else rec

    def no_training(*a, **k):
        raise AssertionError("a run started")

    monkeypatch.setattr(mod, "time_steps", fake_time_steps)
    monkeypatch.setattr(mod, "train", no_training)
    mod.timed = timed
    return mod


def test_plan_has_the_gradient_balanced_variant_and_euler_runs_first(s3):
    variants = [v for v, _, _ in s3.plan()]
    assert variants[:7] == ["euler"] * 6 + ["gradnorm"]
    assert set(variants) == {"euler", "gradnorm", "visc1e-3", "visc3e-4", "nokutta"}


def test_s3_refuses_to_start_past_its_budget(s3):
    assert s3.main(["--budget-hours", "48"]) == 3
    pre = json.loads(Path("results/s3/preflight.json").read_text())
    euler = 30000 * 1.0 + 5000 * 2.0  # s per Euler run: 11.1 h
    visc = 30000 * 10.0 + 5000 * 2.0
    balance = 300 * 5.0  # gradnorm: one balancing update every 100 of 30,000 steps
    assert pre["projected_total_hours"] == pytest.approx((8 * euler + balance + 3 * visc) / 3600)
    assert not pre["within_budget"] and len(s3.timed) == 5  # one timing per variant
    assert not Path("runs/predictions").exists()


def test_s3_preflight_only_within_budget(s3):
    assert s3.main(["--budget-hours", "1000", "--preflight-only"]) == 0
    assert json.loads(Path("results/s3/preflight.json").read_text())["within_budget"]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_shards_partition_the_plan_in_its_order(s3, n):
    runs = [(v, c) for v, c, _ in s3.plan()]
    shards = [[(v, c) for v, c, _ in s3.shard(s3.plan(), k, n)] for k in range(n)]
    assert sorted(sum(shards, [])) == sorted(runs) and sum(map(len, shards)) == len(runs)
    for k, part in enumerate(shards):
        assert part == [r for i, r in enumerate(runs) if i % n == k]


def test_preflight_records_the_device_of_each_phase(s3):
    from apno.models.train_pinn import resolve_device

    assert s3.main(["--budget-hours", "1000", "--preflight-only"]) == 0
    timing = json.loads(Path("results/s3/preflight.json").read_text())["timing_s"]
    for t in timing.values():
        assert t["adam_device"] == resolve_device("auto") and t["lbfgs_device"] == "cpu"


def test_a_shard_writes_only_its_own_files_and_budgets_only_its_own_runs(s3):
    assert s3.main(["--budget-hours", "1000", "--preflight-only", "--shard", "1/2"]) == 0
    assert sorted(p.name for p in Path("results/s3").iterdir()) == ["preflight_shard1of2.json"]
    pre = json.loads(Path("results/s3/preflight_shard1of2.json").read_text())
    want = [(v, c) for i, (v, c, _) in enumerate(s3.plan()) if i % 2 == 1]
    assert [(r["variant"], r["case_id"]) for r in pre["runs"]] == want
    assert pre["shard"] == "1/2"


@pytest.mark.parametrize("bad", ["2/2", "-1/2", "0/0", "1", "a/b"])
def test_a_bad_shard_is_refused(s3, bad):
    with pytest.raises(SystemExit):
        s3.main(["--preflight-only", "--shard", bad])


def test_the_plan_is_the_potential_pinn_by_default_and_euler_on_request(s3):
    assert {cfg.formulation for _, _, cfg in s3.plan()} == {"potential"}
    euler = s3.plan("euler")
    assert {cfg.formulation for _, _, cfg in euler} == {"euler"}
    assert [(v, c) for v, c, _ in euler] == [(v, c) for v, c, _ in s3.plan()]  # same variants


@pytest.mark.parametrize("formulation, name", [("potential", "preflight.json"),
                                                ("euler", "preflight_euler.json")])
def test_preflight_records_the_formulation_and_keeps_euler_apart(s3, formulation, name):
    argv = ["--budget-hours", "1000", "--preflight-only"]
    extra = [] if formulation == "potential" else ["--formulation", formulation]
    assert s3.main(argv + extra) == 0
    assert sorted(p.name for p in Path("results/s3").iterdir()) == [name]
    pre = json.loads(Path("results/s3", name).read_text())
    assert pre["formulation"] == formulation
    assert {t["formulation"] for t in pre["timing_s"].values()} == {formulation}
    assert {cfg.formulation for cfg in s3.timed} == {formulation}


def test_a_run_records_its_formulation_in_the_prediction_and_the_run_record(s3, monkeypatch):
    import numpy as np

    from apno.data.predictions import read_predictions

    def fake_train(case, cfg):
        rec = {"wall_s": 1.0, "cpu_s": 1.0, "adam_wall_s": 0.5, "adam_device": "cpu",
               "lbfgs_device": "cpu", "formulation": cfg.formulation, "final": {}, "weights": {},
               "history": []}
        return None, rec

    monkeypatch.setattr(s3, "train", fake_train)
    monkeypatch.setattr(s3, "evaluate_on_grid", lambda pinn, shape: np.ones((*shape, 4)))
    monkeypatch.setattr(s3, "plan", lambda f="potential", _plan=s3.plan: _plan(f)[:1])
    assert s3.main(["--budget-hours", "1000", "--formulation", "euler"]) == 0
    (path,) = Path("runs/predictions").iterdir()
    assert path.name.endswith("_euler.h5")
    _, attrs = read_predictions(path)
    assert attrs["formulation"] == "euler" and attrs["variant"] == "euler"
    rec = json.loads(Path("results/s3/pinn_runs_euler.json").read_text())
    assert [r["formulation"] for r in rec.values()] == ["euler"]
