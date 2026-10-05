import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "s3_warmup.py"


@pytest.fixture
def warmup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("s3_warmup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_train(case, cfg):
        seen.append(cfg)
        return None, {"wall_s": 1.0, "cpu_s": 1.0, "final": {}}

    monkeypatch.setattr(mod, "train", fake_train)
    monkeypatch.setattr(mod, "evaluate_on_grid", lambda pinn, shape: np.ones((*shape, 4)))
    mod.seen = seen
    return mod


@pytest.mark.parametrize("formulation, suffix", [("potential", ""), ("euler", "_euler")])
def test_warmup_runs_the_formulation_and_records_it_in_its_own_files(warmup, formulation, suffix):
    argv = [] if formulation == "potential" else ["--formulation", formulation]
    assert warmup.main(argv) == 1  # uniform free stream is not the cylinder flow: it fails
    (cfg,) = warmup.seen
    assert cfg.formulation == formulation and cfg.zero_circulation and not cfg.kutta
    assert sorted(p.name for p in Path("results/s3").iterdir()) == [f"warmup{suffix}.json"]
    assert json.loads(Path(f"results/s3/warmup{suffix}.json").read_text())["formulation"] == (
        formulation)
    assert Path(f"runs/warmup_q{suffix}.npy").exists()
