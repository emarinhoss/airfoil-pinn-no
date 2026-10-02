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
    cpw[0] = -5.0  # node 0 is the TE node ending both surfaces
    out = shocks(xw, cpw, 0.75)
    assert out == {"lower": None, "upper": None}


def test_missing_and_nonfinite_predictions():
    q = _free()
    assert score_case(None, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "missing"}
    bad = q.copy()
    bad[3, 4, 1] = np.nan
    assert score_case(bad, q, X, Y, 0.6, 2.0, "subcritical") == {"status": "non-finite"}
