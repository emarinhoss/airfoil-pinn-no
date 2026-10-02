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
