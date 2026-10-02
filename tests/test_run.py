import fcntl
import json
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from apno.geometry.ogrid import COARSE
from apno.su2 import run as run_mod
from apno.su2.config import Su2Settings
from apno.su2.run import (
    LOCK_NAME, CampaignStopped, CaseInterrupted, CaseSpec, ResumeMismatch, case_dir, is_done,
    run_campaign, run_case,
)

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
    assert res["settings"] == asdict(SETTINGS)
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


def test_signal_killed_case_leaves_no_result_and_reruns(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "sigkill")  # a steady 1000-row history, then SIGKILL
    with pytest.raises(CaseInterrupted, match="SIGKILL"):
        run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not is_done(case_dir(tmp_path / "runs", _spec()))
    monkeypatch.setenv("FAKE_SU2_MODE", "ok")
    assert run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)["converged"]


def test_su2_fault_signal_is_a_recorded_failure(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "segv")  # history and restart written, then SIGSEGV
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert not res["converged"] and res["exit_code"] == -signal.SIGSEGV
    assert "SIGSEGV" in res["reason"] and res["verdict"]["converged"]


@pytest.mark.parametrize("mode, why", [("norestart", "no restart.dat"),
                                       ("exit3", "exited with code 3")])
def test_steady_history_alone_is_not_converged(tmp_path, fake_bin, monkeypatch, mode, why):
    monkeypatch.setenv("FAKE_SU2_MODE", mode)
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["verdict"]["converged"] and not res["converged"] and why in res["reason"]


def test_resume_refuses_a_converged_case_run_with_other_settings(tmp_path, fake_bin):
    first = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    other = Su2Settings(iter_cap=1000, mglevel=3, cauchy_eps=1e-9)
    with pytest.raises(ResumeMismatch) as err:
        run_case(_spec(), tmp_path / "runs", fake_bin, other, JUDGE)
    assert first["config_sha256"] in str(err.value) and "move" in str(err.value).lower()
    assert run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE) == first


def test_resume_reruns_an_unconverged_case_under_new_settings(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "nan")
    assert not run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)["converged"]
    monkeypatch.setenv("FAKE_SU2_MODE", "ok")
    other = Su2Settings(iter_cap=1000, cauchy_eps=1e-9)
    res = run_case(_spec(), tmp_path / "runs", fake_bin, other, JUDGE)
    assert res["converged"] and res["settings"] == asdict(other)


def test_resume_reruns_a_stored_interruption(tmp_path, fake_bin):
    d = case_dir(tmp_path / "runs", _spec())
    run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    legacy = json.loads((d / "result.json").read_text())
    legacy.update(exit_code=-2, converged=False, reason="killed by an earlier driver")
    (d / "result.json").write_text(json.dumps(legacy))
    res = run_case(_spec(), tmp_path / "runs", fake_bin, SETTINGS, JUDGE)
    assert res["converged"] and res["exit_code"] == 0


def test_campaign_reports_a_driver_exception_and_runs_the_rest(tmp_path, fake_bin, capsys):
    specs = [_spec(k) for k in range(3)]

    def settings_for(s):
        if s.case_id == specs[1].case_id:
            raise RuntimeError("bad settings")
        return SETTINGS

    out = run_campaign(specs, tmp_path / "runs", fake_bin, settings_for, 2, JUDGE)
    assert sorted(r["case_id"] for r in out) == [specs[0].case_id, specs[2].case_id]
    assert not is_done(case_dir(tmp_path / "runs", specs[1]))
    assert "test-0001: DRIVER ERROR RuntimeError('bad settings')" in capsys.readouterr().out


def test_campaign_refuses_a_root_another_driver_holds(tmp_path, fake_bin):
    root = tmp_path / "runs"
    root.mkdir()
    with open(root / LOCK_NAME, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another campaign driver"):
            run_campaign([_spec()], root, fake_bin, lambda s: SETTINGS, 1, JUDGE)
    assert not is_done(case_dir(root, _spec()))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_sigterm_stops_the_campaign_and_its_su2_children(tmp_path, fake_bin, monkeypatch):
    monkeypatch.setenv("FAKE_SU2_MODE", "slow")
    specs = [_spec(k) for k in range(4)]
    timer = threading.Timer(1.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(CampaignStopped):
            run_campaign(specs, tmp_path / "runs", fake_bin, lambda s: SETTINGS, 2, JUDGE)
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 20  # the queued cases did not run, the live ones were stopped
    pids = [int(p.read_text()) for p in (tmp_path / "runs").rglob("pid")]
    assert 1 <= len(pids) <= 2 and not any(_alive(pid) for pid in pids)
    assert not any(is_done(case_dir(tmp_path / "runs", s)) for s in specs)
    assert not run_mod._LIVE and not run_mod._STOP.is_set()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
