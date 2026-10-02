from apno.geometry.ogrid import FINE, TRAIN
from apno.settings import CONVERGENCE, WORKERS, settings_for
from apno.su2.run import CaseSpec


def test_caps_by_level_and_judge_defaults():
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, TRAIN)).iter_cap == 10000
    assert settings_for(CaseSpec("a", 0.5, 0, 0.1, 0, FINE)).iter_cap == 20000
    assert CONVERGENCE == {"residual_drop": 5.0, "window": 500, "force_tol": 1e-6}
    assert 1 <= WORKERS <= 8
