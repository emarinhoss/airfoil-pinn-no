import os
import sys
from pathlib import Path

import pytest

SPIKE_DIR = Path(
    os.environ.get(
        "APNO_SPIKE_DIR", str(Path.home() / "theLab/bwb-tanker-study/cache/su2_spike")
    )
)


@pytest.fixture
def su2_bin() -> Path:
    value = os.environ.get("SU2_BIN_DIR")
    if not value:
        pytest.skip("SU2_BIN_DIR not set")
    return Path(value)


@pytest.fixture
def spike_dir() -> Path:
    """The 2026-10-01 NACA 0012 spike: a real SU2 restart, history and surface file."""
    if not (SPIKE_DIR / "restart.dat").exists():
        pytest.skip(f"spike files not present at {SPIKE_DIR}")
    return SPIKE_DIR


FAKE_SU2 = Path(__file__).with_name("fake_su2.py")


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """A directory holding a fake SU2_CFD (tests/fake_su2.py), in mode ok."""
    b = tmp_path / "bin"
    b.mkdir()
    exe = b / "SU2_CFD"
    exe.write_text(f"#!{sys.executable}\n" + FAKE_SU2.read_text())
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_SU2_MODE", "ok")
    monkeypatch.delenv("FAKE_SU2_KILL", raising=False)
    monkeypatch.delenv("FAKE_SU2_FORCES", raising=False)
    return b
