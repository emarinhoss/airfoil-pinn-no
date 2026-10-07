import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from apno.models.train_pinn import resolve_device

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
        return None, {"wall_s": 1.0, "cpu_s": 1.0, "final": {},
                      "adam_device": resolve_device(cfg.device), "lbfgs_device": "cpu"}

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


def test_warmup_records_the_device_of_each_phase_and_the_code(warmup, monkeypatch):
    # review 2, F3: the potential PASS ran Adam on the CPU through an uncommitted override, and its
    # record named neither device, the thread count nor the code version
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    warmup.main(["--device", "cpu"])
    (cfg,) = warmup.seen
    assert cfg.device == "cpu"
    out = json.loads(Path("results/s3/warmup.json").read_text())
    assert out["adam_device"] == "cpu" and out["lbfgs_device"] == "cpu"
    assert out["omp_num_threads"] == "8" and "git_commit" in out and "argv" in out


def test_warmup_refuses_to_overwrite_a_record_and_writes_a_tagged_one(warmup):
    # review 2, F3/M13: results/s3/warmup.json holds the committed potential PASS; a plain re-run
    # must not silently replace it (or its field), and --tag writes beside it instead
    Path("results/s3").mkdir(parents=True)
    Path("results/s3/warmup.json").write_text('{"passed": true}')
    with pytest.raises(SystemExit, match="warmup.json"):
        warmup.main([])
    assert warmup.seen == []  # refused before any training
    assert Path("results/s3/warmup.json").read_text() == '{"passed": true}'
    Path("results/s3/warmup.json").unlink()
    Path("runs").mkdir()
    Path("runs/warmup_q.npy").write_bytes(b"field")
    with pytest.raises(SystemExit, match="warmup_q.npy"):
        warmup.main([])
    assert warmup.main(["--tag", "rerun"]) == 1
    assert Path("results/s3/warmup_rerun.json").exists()
    assert Path("runs/warmup_q_rerun.npy").exists()
    assert Path("runs/warmup_q.npy").read_bytes() == b"field"
