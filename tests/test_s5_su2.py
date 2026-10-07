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
