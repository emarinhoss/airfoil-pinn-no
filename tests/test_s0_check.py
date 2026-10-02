import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "s0_check.py"


def test_an_interrupted_case_leaves_s0_incomplete(tmp_path, fake_bin, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SU2_BIN_DIR", str(fake_bin))
    monkeypatch.setenv("FAKE_SU2_KILL", "sub2-R50")  # killed by a signal: no result.json
    monkeypatch.setenv("FAKE_SU2_FORCES", "zero")  # what the scorer integrates from free stream
    monkeypatch.setattr(sys, "argv", ["s0_check.py", "--workers", "6"])
    with pytest.raises(SystemExit) as code:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    out = capsys.readouterr().out
    assert code.value.code == 2 and "sub2-R50@256x97: no result" in out
    assert "scorer check passed" in out
    assert not (tmp_path / "runs/s0/256x97/sub2-R50/result.json").exists()
