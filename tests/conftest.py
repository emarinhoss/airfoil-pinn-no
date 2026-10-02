import os
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
