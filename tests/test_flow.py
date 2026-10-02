import numpy as np
import pytest

from apno.constants import GAMMA, P_INF
from apno.physics.flow import (
    entropy, freestream, freestream_q, isentropic_mach_from_cp, local_mach, nondimensionalise,
    normal_shock, pressure_coefficient, total_enthalpy_ratio,
)


def test_free_stream_is_quiet():
    M, a = 0.7, 3.0
    q = freestream_q(a)
    assert pressure_coefficient(q[3], M) == 0.0
    assert entropy(q[0], q[3]) == 0.0
    assert total_enthalpy_ratio(*q, M) == pytest.approx(1.0, abs=1e-15)
    assert local_mach(*q, M) == pytest.approx(M, rel=1e-15)
    assert isentropic_mach_from_cp(0.0, M) == pytest.approx(M, rel=1e-12)


def test_nondimensionalise_free_stream():
    M = 0.5
    rho, U, p = freestream(M)
    a = np.radians(2.0)
    rhoE = p / (GAMMA - 1) + 0.5 * rho * U * U
    out = nondimensionalise(rho, rho * U * np.cos(a), rho * U * np.sin(a), rhoE, M)
    np.testing.assert_allclose(out, [1.0, np.cos(a), np.sin(a), 1.0], rtol=1e-13)
    assert p == P_INF


def test_normal_shock_jumps():
    rho2, u2, p2 = normal_shock(2.0)
    assert rho2 == pytest.approx(8 / 3, rel=1e-12)
    assert p2 == pytest.approx(4.5, rel=1e-12)
    assert u2 == pytest.approx(3 / 8, rel=1e-12)
    assert local_mach(rho2, u2, 0.0, p2, 2.0) == pytest.approx(0.5773502692, rel=1e-9)
    assert total_enthalpy_ratio(rho2, u2, 0.0, p2, 2.0) == pytest.approx(1.0, rel=1e-12)
    assert entropy(rho2, p2) > 0


def test_cp_and_isentropic_mach_round_trip():
    M = 0.75
    p_h = 0.6  # an expansion
    rho_h = p_h ** (1 / GAMMA)
    # isentropic, constant total enthalpy -> speed from H
    speed2 = 1.0 + 2.0 / ((GAMMA - 1) * M**2) * (1.0 - p_h / rho_h)
    m_loc = local_mach(rho_h, np.sqrt(speed2), 0.0, p_h, M)
    assert isentropic_mach_from_cp(pressure_coefficient(p_h, M), M) == pytest.approx(m_loc, 1e-12)
