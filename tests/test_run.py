import json
import os
import sys
from pathlib import Path

import pytest

from apno.geometry.ogrid import COARSE
from apno.su2.config import Su2Settings
from apno.su2.run import CaseSpec, case_dir, is_done, run_campaign, run_case

FAKE = Path(__file__).with_name("fake_su2.py")
SETTINGS = Su2Settings(iter_cap=1000)
JUDGE = dict(window=200)


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    exe = b / "SU2_CFD"
    exe.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_SU2_MODE", "ok")
    return b


def _spec(k=0, mach=0.5):
    return CaseSpec(f"test-{k:04d}", mach, 2.0, 0.12, 0.02, COARSE)


def test_ok_case(tmp_path, fake_bin):
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["converged"] and res["exit_code"] == 0 and res["su2_version"] == "8.5.0"
    assert res["max_surface_mach"] == pytest.approx(0.5, rel=1e-12)  # free-stream fake
    assert res["cl"] == 0.25 and res["level"] == "128x49"
    assert is_done(case_dir(tmp_path / "runs", _spec()))


def test_done_case_is_not_rerun(tmp_path, fake_bin, monkeypatch):
    first = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    monkeypatch.setenv("FAKE_SU2_MODE", "crash")
    assert run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE) == first


def test_crash_records_failure(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "crash")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not res["converged"] and res["exit_code"] == 1 and "no history" in res["reason"]
    assert res["max_surface_mach"] is None


def test_nan_history_records_failure(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "nan")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not res["converged"] and "non-finite" in res["reason"]


def test_stale_files_rerun(tmp_path, fake_bin):
    d = case_dir(tmp_path / "runs", _spec())
    d.mkdir(parents=True)
    (d / "restart.dat").write_bytes(b"garbage from a killed run")
    (d / "history.csv").write_text("garbage")
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["converged"]


def test_tmp_result_is_not_done(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    (d / "result.json.tmp").write_text("{}")
    assert not is_done(d)


def test_campaign(tmp_path, fake_bin):
    specs = [_spec(k, 0.3 + 0.1 * k) for k in range(3)]
    out = run_campaign(specs, tmp_path / "runs", fake_bin, lambda s: SETTINGS, 2, JUDGE)
    assert sorted(r["case_id"] for r in out) == [s.case_id for s in specs]
    assert all(r["converged"] for r in out)
    saved = json.loads((case_dir(tmp_path / "runs", specs[2]) / "result.json").read_text())
    assert saved["inputs"]["mach"] == pytest.approx(0.5)
    assert os.environ["FAKE_SU2_MODE"] == "ok"
