import importlib.util
import json
import sys
from pathlib import Path

import pytest

from apno.data.sampling import s5_validation_specs
from apno.geometry.ogrid import FINE, TRAIN

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_su2.py"
FAKE_SU2 = Path(__file__).with_name("fake_su2.py")


def _load():
    spec = importlib.util.spec_from_file_location("s5_su2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def level_bin(fake_bin, tmp_path, monkeypatch):
    """A fake SU2_CFD whose mode is set per <level>/<case> directory.

    S5_FAKE_MODES="512x193/s5-val-B=sigkill 256x97/s5-val-C=nan" puts those directories in those
    tests/fake_su2.py modes (the rest stay in FAKE_SU2_MODE); every launch appends its
    <level>/<case> to the file returned as the fixture's second value.
    """
    b, log = tmp_path / "level_bin", tmp_path / "launches.txt"
    b.mkdir()
    exe = b / "SU2_CFD"
    exe.write_text(
        f"#!{sys.executable}\n"
        "import os, runpy\n"
        "from pathlib import Path\n"
        "here = Path.cwd().parent.name + '/' + Path.cwd().name\n"
        f"with open({str(log)!r}, 'a') as fh:\n"
        "    fh.write(here + '\\n')\n"
        "for item in os.environ.get('S5_FAKE_MODES', '').split():\n"
        "    key, _, mode = item.partition('=')\n"
        "    if key == here:\n"
        "        os.environ['FAKE_SU2_MODE'] = mode\n"
        f"runpy.run_path({str(FAKE_SU2)!r}, run_name='__main__')\n")
    exe.chmod(0o755)
    monkeypatch.setenv("SU2_BIN_DIR", str(b))
    monkeypatch.delenv("S5_FAKE_MODES", raising=False)
    return b, log


def _launches(log):
    return log.read_text().split() if log.exists() else []


def test_validation_specs_are_naca0012_in_fallback_order():
    for shape in (TRAIN, FINE):
        got = [(s.case_id, s.mach, s.alpha_deg, s.t, s.m, s.shape)
               for s in s5_validation_specs(shape)]
        assert got == [("s5-val-B", 0.72, 3.0, 0.12, 0.0, shape),
                       ("s5-val-C", 0.75, 2.0, 0.12, 0.0, shape),
                       ("s5-val-A", 0.70, 4.0, 0.12, 0.0, shape)]


def test_case_b_is_selected_when_it_converges(fake_bin, tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-B"
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B"]
    assert set(out["tried"][0]["levels"]) == {"256x97", "512x193"}
    assert Path("runs/su2/512x193/s5-val-B/result.json").exists()


def test_a_case_with_no_result_is_resumed_not_fallen_back_from(fake_bin, tmp_path, monkeypatch):
    # a kill is not a result (spec §4.5): B is unfinished, not failed, so C never runs
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    monkeypatch.setenv("FAKE_SU2_KILL", "s5-val-B")  # killed at both levels: no result
    assert mod.main([]) == 1
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] is None and out["tried"] == []
    assert out["unfinished"]["case_id"] == "s5-val-B"
    assert out["unfinished"]["missing"] == ["256x97", "512x193"]
    assert not list(Path("runs/su2").glob("*/s5-val-C"))
    monkeypatch.delenv("FAKE_SU2_KILL")  # nothing wrong any more: a plain re-run resumes B
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-B" and out["unfinished"] is None
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B"]
    assert not list(Path("runs/su2").glob("*/s5-val-C"))


def test_a_kill_at_512x193_only_resumes_b_and_never_runs_c(level_bin, tmp_path, monkeypatch):
    _, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=sigkill")
    assert mod.main([]) == 1
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] is None and out["tried"] == []
    assert out["unfinished"]["case_id"] == "s5-val-B"
    assert out["unfinished"]["missing"] == ["512x193"]
    assert set(out["unfinished"]["levels"]) == {"256x97"}
    monkeypatch.delenv("S5_FAKE_MODES")
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-B"
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B"]
    # the 256x97 result is resumed; only the killed level runs again; C is never launched
    assert sorted(_launches(log)) == ["256x97/s5-val-B", "512x193/s5-val-B", "512x193/s5-val-B"]


def test_falls_back_to_case_c_on_a_recorded_failure_of_b(level_bin, tmp_path, monkeypatch):
    _, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=nan")  # B fails the rule at 512x193
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-C" and out["unfinished"] is None
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B", "s5-val-C"]
    b = out["tried"][0]
    assert b["converged_both"] is False
    assert b["levels"]["256x97"]["converged"] is True
    assert b["levels"]["512x193"]["converged"] is False
    first = json.loads(Path("results/s5/su2.json").read_text())
    monkeypatch.delenv("S5_FAKE_MODES")  # the recorded failure stands: a re-run selects C again
    n = len(_launches(log))
    assert mod.main([]) == 0
    assert json.loads(Path("results/s5/su2.json").read_text()) == first
    assert len(_launches(log)) == n  # everything resumed from result.json, no SU2 launched


def test_a_selection_on_record_is_never_changed(fake_bin, tmp_path, monkeypatch):
    # e.g. C selected by an earlier run; the stored runs now give B: refuse, keep the record
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    Path("results/s5").mkdir(parents=True)
    record = json.dumps({"tried": [], "selected": "s5-val-C", "unfinished": None})
    Path("results/s5/su2.json").write_text(record)
    assert mod.main([]) == 3
    assert Path("results/s5/su2.json").read_text() == record


def test_exit_2_when_no_case_converges(fake_bin, tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    monkeypatch.setenv("FAKE_SU2_MODE", "nan")
    assert mod.main([]) == 2
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] is None and len(out["tried"]) == 3


# exit 4: the environment or another driver stops the call (not "unfinished, re-run")

def test_exit_4_when_su2_bin_dir_is_unset_and_a_run_must_launch(level_bin, tmp_path, monkeypatch,
                                                                  capsys):
    _, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SU2_BIN_DIR")
    assert mod.main([]) == 4
    assert capsys.readouterr().out.splitlines()[-1].startswith("cannot run: SU2_BIN_DIR")
    assert _launches(log) == [] and not list(Path("runs/su2").glob("*/*/result.json"))
    assert not Path("results/s5/su2.json").exists()


def test_su2_bin_dir_is_read_only_when_a_run_launches(level_bin, tmp_path, monkeypatch, capsys):
    b, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=sigkill")
    assert mod.main([]) == 1
    monkeypatch.delenv("S5_FAKE_MODES")
    monkeypatch.delenv("SU2_BIN_DIR")  # 512x193 has no result, so SU2 must launch: exit 4, not 1
    assert mod.main([]) == 4
    assert capsys.readouterr().out.splitlines()[-1].startswith("cannot run: SU2_BIN_DIR")
    monkeypatch.setenv("SU2_BIN_DIR", str(b))
    assert mod.main([]) == 0
    first, n = Path("results/s5/su2.json").read_text(), len(_launches(log))
    monkeypatch.delenv("SU2_BIN_DIR")  # a re-run after success launches nothing: no SU2 needed
    assert mod.main([]) == 0
    assert Path("results/s5/su2.json").read_text() == first and len(_launches(log)) == n


@pytest.mark.parametrize("bad", ["empty", "absent", "not_executable"])
def test_exit_4_when_su2_bin_dir_holds_no_executable_su2_cfd(level_bin, tmp_path, monkeypatch,
                                                             capsys, bad):
    # a wrong SU2_BIN_DIR is a driver error that no re-run can fix: exit 4, not 1
    b, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    wrong = tmp_path / "wrong_bin"
    if bad != "absent":
        wrong.mkdir()
    if bad == "not_executable":
        (wrong / "SU2_CFD").write_text((b / "SU2_CFD").read_text())  # no x bit
    monkeypatch.setenv("SU2_BIN_DIR", str(wrong))
    assert mod.main([]) == 4
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("cannot run: ") and str(wrong / "SU2_CFD") in last
    assert _launches(log) == [] and not list(Path("runs/su2").glob("*/*/result.json"))
    assert not Path("results/s5/su2.json").exists()


def test_exit_4_on_a_wrong_su2_bin_dir_with_a_record_whose_runs_are_missing(
        level_bin, tmp_path, monkeypatch, capsys):
    b, _ = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    assert mod.main([]) == 0
    record = Path("results/s5/su2.json").read_text()
    Path("runs").rename("runs-aside")  # a fresh clone: SU2 must launch again
    (tmp_path / "empty_bin").mkdir()
    monkeypatch.setenv("SU2_BIN_DIR", str(tmp_path / "empty_bin"))
    assert mod.main([]) == 4
    assert capsys.readouterr().out.splitlines()[-1].startswith("cannot run: ")
    assert Path("results/s5/su2.json").read_text() == record
    assert not list(Path("runs/su2").glob("*/*/result.json"))


def test_exit_4_on_a_run_made_with_other_settings(fake_bin, tmp_path, monkeypatch, capsys):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    assert mod.main([]) == 0
    record = Path("results/s5/su2.json").read_text()
    res = Path("runs/su2/512x193/s5-val-B/result.json")
    res.write_text(json.dumps({**json.loads(res.read_text()), "config_sha256": "0" * 64}))
    assert mod.main([]) == 4  # the library's ResumeMismatch, not a traceback
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("cannot run: ") and "other settings" in last
    assert Path("results/s5/su2.json").read_text() == record


def test_exit_4_when_another_driver_holds_the_lock(level_bin, tmp_path, monkeypatch, capsys):
    import fcntl

    _, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    Path("runs/su2").mkdir(parents=True)
    with open("runs/su2/.campaign.lock", "w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert mod.main([]) == 4
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("cannot run: another campaign driver holds") and "re-run" in last
    assert _launches(log) == [] and not Path("results/s5/su2.json").exists()


# a recorded failure at either level decides the case (design §3.1)

@pytest.mark.parametrize("failed,killed", [("256x97", "512x193"), ("512x193", "256x97")])
def test_falls_back_on_a_recorded_failure_even_if_the_other_level_has_no_result(
        level_bin, tmp_path, monkeypatch, failed, killed):
    _, log = level_bin
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("S5_FAKE_MODES", f"{failed}/s5-val-B=nan {killed}/s5-val-B=sigkill")
    assert mod.main([]) == 0
    out = json.loads(Path("results/s5/su2.json").read_text())
    assert out["selected"] == "s5-val-C" and out["unfinished"] is None
    assert [t["case_id"] for t in out["tried"]] == ["s5-val-B", "s5-val-C"]
    b = out["tried"][0]
    assert b["converged_both"] is False and b["missing"] == [killed]
    assert set(b["levels"]) == {failed} and b["levels"][failed]["converged"] is False
    monkeypatch.delenv("S5_FAKE_MODES")
    n = len(_launches(log))
    assert mod.main([]) == 0  # B has failed already: its unfinished level is not run again
    assert json.loads(Path("results/s5/su2.json").read_text()) == out
    assert len(_launches(log)) == n


# a record whose runs are missing or unfinished is a re-run (1), not a disagreement (3)

def test_a_record_with_missing_runs_asks_for_a_rerun_and_keeps_the_record(level_bin, tmp_path,
                                                                           monkeypatch, capsys):
    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=nan")
    assert mod.main([]) == 0
    record = Path("results/s5/su2.json").read_text()
    assert json.loads(record)["selected"] == "s5-val-C"
    Path("runs").rename("runs-aside")  # a fresh clone: results/ is tracked, runs/ is not
    monkeypatch.setenv("S5_FAKE_MODES", "256x97/s5-val-B=sigkill 512x193/s5-val-B=sigkill")
    assert mod.main([]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert "missing or unfinished" in last and "re-run" in last
    assert Path("results/s5/su2.json").read_text() == record
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=nan")  # the same SU2: the same choice
    assert mod.main([]) == 0
    assert json.loads(Path("results/s5/su2.json").read_text())["selected"] == "s5-val-C"


def test_unfinished_runs_that_already_contradict_the_record_refuse(level_bin, tmp_path,
                                                                   monkeypatch):
    # the record says B, the runs record a failure of B and C is unfinished: they disagree
    mod = _load()
    monkeypatch.chdir(tmp_path)
    Path("results/s5").mkdir(parents=True)
    record = json.dumps({"tried": [], "selected": "s5-val-B", "unfinished": None})
    Path("results/s5/su2.json").write_text(record)
    monkeypatch.setenv("S5_FAKE_MODES", "512x193/s5-val-B=nan 256x97/s5-val-C=sigkill "
                                        "512x193/s5-val-C=sigkill")
    assert mod.main([]) == 3
    assert Path("results/s5/su2.json").read_text() == record


# su2.json is replaced atomically

def test_a_kill_while_writing_su2_json_leaves_the_previous_one(fake_bin, tmp_path, monkeypatch):
    class Killed(BaseException):
        pass

    mod = _load()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    assert mod.main([]) == 0
    record = Path("results/s5/su2.json").read_text()
    real = Path.write_text

    def torn(self, data, *a, **kw):  # half the text reaches the disk, then the process dies
        if self.name.startswith("su2.json"):
            real(self, data[: len(data) // 2], *a, **kw)
            raise Killed
        return real(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", torn)
    with pytest.raises(Killed):
        mod.main([])
    monkeypatch.setattr(Path, "write_text", real)
    assert Path("results/s5/su2.json").read_text() == record
    assert mod.main([]) == 0
