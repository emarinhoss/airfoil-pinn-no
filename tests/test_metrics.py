import numpy as np
import pytest

from apno.geometry.ogrid import TRAIN, Section, grid
from apno.physics.flow import freestream_q
from apno.score.metrics import node_areas, score_case, shocks

X, Y = grid(Section(0.12, 0.0), TRAIN)


def _free(alpha=2.0):
    return np.broadcast_to(freestream_q(alpha), X.shape + (4,)).copy()


def test_node_areas_sum_to_the_annulus():
    total = node_areas(X, Y).sum()
    airfoil = 0.685 * 0.12  # NACA 4-digit area is about 0.685 t c
    assert total == pytest.approx(np.pi * 100.0**2 - airfoil, rel=1e-3)


def test_identical_fields_score_zero():
    q = _free()
    s = score_case(q, q, X, Y, 0.6, 2.0, "subcritical")
    assert s["status"] == "ok"
    assert s["cl_err"] == 0 and s["cd_err_counts"] == 0 and s["cp_max"] == 0
    assert s["near_field_rel_l2"] == 0.0


def test_free_stream_physics_is_clean():
    s = score_case(_free(), _free(), X, Y, 0.6, 2.0, "subcritical")["physics"]
    assert abs(s["entropy_max"]) < 1e-14 and s["enthalpy_max_dev"] < 1e-14
    assert all(abs(v) < 1e-10 for v in s["mass_flux"].values())


def test_shock_found_on_a_synthetic_step():
    x = np.linspace(0, 1, 101)
    cp = np.where(x < 0.6, -1.0 + 0.2 * x, 0.1)  # suction plateau, then a jump at 0.6
    xw = np.concatenate([x[::-1], x[1:-1]])  # lower TE->LE then upper LE->TE
    cpw = np.concatenate([np.full(101, -0.2), cp[1:-1]])
    out = shocks(xw, cpw, 0.75)
    assert out["upper"]["x"] == pytest.approx(0.595, abs=0.006)
    assert out["lower"] is None


def test_shock_suction_peak_on_trailing_edge_node_returns_none():
    x = np.linspace(0, 1, 101)
    xw = np.concatenate([x[::-1], x[1:-1]])
    cpw = np.full(len(xw), -0.2)
    cpw[0] = -1.0  # node 0 is the TE node ending both surfaces; supersonic (M 1.19), p > 0
    out = shocks(xw, cpw, 0.75)
    assert out == {"lower": None, "upper": None}


# A real cosine-clustered wall: the last interval before the trailing edge is about 1.5e-4 chord, so
# the closing stagnation rise is far steeper (dCp/dx in the thousands) than any captured shock.
XW = X[:, 0]
NI = len(XW)
LE = NI // 2
UPPER = np.arange(LE + 1, NI)
MACH = 0.75


def _wall_cp(x_shock, supersonic=True):
    """Upper: a supersonic plateau, a jump at x_shock, recovery; lower subsonic; TE stagnation."""
    cp = np.full(NI, -0.2)
    xu = XW[UPPER]
    plateau = -0.9 if supersonic else -0.3  # Cp* is about -0.43 at M 0.75
    after = -0.1 + 0.3 * np.clip((xu - x_shock) / (1 - x_shock), 0, 1)
    cp[UPPER] = np.where(xu < x_shock, plateau, after)
    cp[LE] = 1.0  # leading-edge stagnation
    cp[0] = 0.9  # trailing-edge stagnation rise over the last ~1e-4 chord
    return cp


def _field_with_wall_cp(cp, alpha=2.0):
    q = _free(alpha)
    q[:, 0, 3] = 1.0 + 0.5 * 1.4 * MACH**2 * cp
    return q


def test_shock_on_a_cosine_clustered_wall_is_not_the_trailing_edge_recompression():
    cp = _wall_cp(0.65)
    out = shocks(XW, cp, MACH)
    assert out["lower"] is None
    assert out["upper"]["x"] == pytest.approx(0.65, abs=0.01)
    assert out["upper"]["mach_ahead"] > 1.0


def test_supersonic_leading_edge_node_is_not_a_lower_surface_shock():
    cp = _wall_cp(0.65)
    cp[LE] = -0.9  # the shared LE node is supersonic, at the upper plateau
    cp[LE - 1] = 1.0  # lower-surface stagnation just aft of the LE: a compression, not a shock
    out = shocks(XW, cp, MACH)
    assert out["lower"] is None
    assert out["upper"]["x"] == pytest.approx(0.65, abs=0.01)


def test_score_case_supercritical_branch():
    truth = _field_with_wall_cp(_wall_cp(0.65))
    pred = _field_with_wall_cp(_wall_cp(0.60))
    s = score_case(pred, truth, X, Y, MACH, 2.0, "supercritical")["shock"]
    assert set(s) == {"upper"}  # the truth's lower surface is subsonic
    u = s["upper"]
    assert u["x_true"] == pytest.approx(0.65, abs=0.01)
    assert u["x_err"] == pytest.approx(-0.05, abs=0.01) and u["missed"] is False
    assert u["mach_ahead_true"] > 1.0 and u["mach_ahead_err"] == pytest.approx(0.0, abs=1e-12)
    assert 0.8 <= u["overshoot"] <= 0.84  # recovery from 0.60 against the -0.9 plateau to 0.65

    smooth = _field_with_wall_cp(_wall_cp(0.65, supersonic=False))
    u = score_case(smooth, truth, X, Y, MACH, 2.0, "supercritical")["shock"]["upper"]
    assert u["missed"] is True and u["x_err"] is None and u["mach_ahead_err"] is None
    assert u["overshoot"] > 0


def test_missing_and_nonfinite_predictions():
    q = _free()
    assert score_case(None, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "missing"}
    bad = q.copy()
    bad[3, 4, 1] = np.nan
    assert score_case(bad, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "non-finite"}
