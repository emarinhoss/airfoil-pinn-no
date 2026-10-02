"""Flow quantities, all from one tested place.

Non-dimensional state everywhere: rho/rho_inf, u/U_inf, v/U_inf, p/p_inf. Then
a^2/U_inf^2 = (p_h / rho_h) / M^2 and H/U_inf^2 = p_h/(rho_h (gamma - 1) M^2) + |u|^2/2.
"""

import numpy as np

from apno.constants import GAMMA, P_INF, R_GAS, T_INF

G = GAMMA


def freestream(mach):
    rho = P_INF / (R_GAS * T_INF)
    return rho, mach * np.sqrt(G * R_GAS * T_INF), P_INF


def nondimensionalise(rho, mx, my, rhoE, mach):
    rho_inf, u_inf, p_inf = freestream(mach)
    p = (G - 1.0) * (rhoE - 0.5 * (mx * mx + my * my) / rho)
    return rho / rho_inf, mx / rho / u_inf, my / rho / u_inf, p / p_inf


def freestream_q(alpha_deg):
    a = np.radians(alpha_deg)
    return np.array([1.0, np.cos(a), np.sin(a), 1.0])


def pressure_coefficient(p_h, mach):
    return (p_h - 1.0) / (0.5 * G * mach * mach)


def local_mach(rho_h, u_h, v_h, p_h, mach):
    return np.sqrt(u_h * u_h + v_h * v_h) * mach / np.sqrt(p_h / rho_h)


def entropy(rho_h, p_h):
    """(s - s_inf)/c_v."""
    return np.log(p_h / rho_h**G)


def total_enthalpy_ratio(rho_h, u_h, v_h, p_h, mach):
    h = p_h / (rho_h * (G - 1.0) * mach * mach) + 0.5 * (u_h * u_h + v_h * v_h)
    return h / (1.0 / ((G - 1.0) * mach * mach) + 0.5)


def normal_shock(mach):
    m2 = mach * mach
    rho2 = (G + 1.0) * m2 / ((G - 1.0) * m2 + 2.0)
    p2 = 1.0 + 2.0 * G / (G + 1.0) * (m2 - 1.0)
    return rho2, 1.0 / rho2, p2


def isentropic_mach_from_cp(cp, mach):
    p_h = 1.0 + 0.5 * G * mach * mach * cp
    t0 = 1.0 + 0.5 * (G - 1.0) * mach * mach
    return np.sqrt(2.0 / (G - 1.0) * (t0 * p_h ** (-(G - 1.0) / G) - 1.0))
