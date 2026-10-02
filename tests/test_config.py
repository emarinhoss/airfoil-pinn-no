import re
import subprocess

import pytest

from apno.su2.config import VERIFIED_KEYWORDS, Su2Settings, render_config


def _keys(cfg):
    return [ln.split("=")[0].strip() for ln in cfg.splitlines() if "=" in ln and ln[0] != "%"]


def test_every_keyword_is_verified_and_unique():
    keys = _keys(render_config(0.7, 2.5, Su2Settings(mglevel=3)))
    assert len(keys) == len(set(keys))
    assert set(keys) <= VERIFIED_KEYWORDS


def test_values_are_written():
    cfg = render_config(0.7, -1.5, Su2Settings(mglevel=0, iter_cap=1234))
    assert re.search(r"^MACH_NUMBER= 0\.7000000000$", cfg, re.M)
    assert re.search(r"^AOA= -1\.5000000000$", cfg, re.M)
    assert re.search(r"^ITER= 1234$", cfg, re.M)
    assert re.search(r"^MGLEVEL= 0$", cfg, re.M)
    assert "VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )" in cfg


@pytest.mark.su2
def test_keywords_exist_in_the_binary(su2_bin):
    out = subprocess.run(["strings", "-n", "3", str(su2_bin / "SU2_CFD")],
                         capture_output=True, text=True).stdout.splitlines()
    present = set(out)
    # REF_AREA is parsed by this build (the spike's log echoes it) but strings cannot see it whole
    missing = sorted(k for k in VERIFIED_KEYWORDS if k not in present and k != "REF_AREA")
    assert not missing
