from dataclasses import replace

import pytest

from apno import settings
from apno.geometry.ogrid import FINE, TRAIN
from apno.settings import BASE, CONVERGENCE, WORKERS, consistency_errors, settings_for
from apno.su2.config import Su2Settings
from apno.su2.run import CaseSpec


def test_caps_by_level_and_judge_defaults():
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, TRAIN)).iter_cap == 10000
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, FINE)).iter_cap == 20000
    assert CONVERGENCE == {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}


def test_settings_are_pinned():
    """The S0 freeze edits these lines, FROZEN and BASE together, with its findings entry. An edit
    anywhere else is a campaign change that resume and build_level would otherwise have to catch.
    """
    assert WORKERS == 6
    assert settings.FROZEN is False
    assert BASE == Su2Settings(mglevel=0, iter_cap=10000, cauchy_elems=500, cauchy_eps=1e-7)


def test_cauchy_stop_must_not_end_a_run_before_our_rule_can_pass():
    assert BASE.cauchy_elems == CONVERGENCE["window"]
    assert any("not below force_tol" in e for e in consistency_errors())  # 1e-7: 7.5e-5 > 1e-6
    assert consistency_errors(replace(BASE, cauchy_eps=1e-9)) == []  # 9.3e-7 < 1e-6
    assert any("window" in e for e in consistency_errors(replace(BASE, cauchy_eps=1e-9,
                                                                 cauchy_elems=250)))


def test_check_frozen(monkeypatch):
    with pytest.raises(RuntimeError, match="not FROZEN"):
        settings.check_frozen()
    monkeypatch.setattr(settings, "FROZEN", True)
    with pytest.raises(RuntimeError, match="inconsistent"):
        settings.check_frozen()
    monkeypatch.setattr(settings, "BASE", replace(BASE, cauchy_eps=1e-9))
    settings.check_frozen()


def test_s1_refuses_to_start_unfrozen(monkeypatch, tmp_path):
    import runpy
    import sys
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "s1_campaign.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["s1_campaign.py", "train"])
    monkeypatch.delenv("SU2_BIN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="not FROZEN"):
        runpy.run_path(str(script), run_name="__main__")
    assert not (tmp_path / "runs").exists()
