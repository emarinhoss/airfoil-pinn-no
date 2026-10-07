"""Localised artificial viscosity for the S5 Euler PINN (S5 design §2.4; Wassing, Langer &
Bekemeyer 2025, §II D-E and Appendix C).

Every function takes torch tensors of shape (n,) per variable, in the S5 PINN's scaling
(rho_inf = p_inf = 1, |u_inf| = sqrt(gamma) M_inf). The conserved state is W = (rho, rho u, rho v,
rho E). The viscous term D is dissipative: the S5 residual is div F - D (their Eq. 13 prints + D,
which with their Eq. 14 would be anti-dissipative; read as a sign typo, design §2.4).

|A| for the flux Jacobian A = dF_k/dW in the direction (kx, ky) is closed-form and exact:
A = q I + M with M = r_a l_b + (gamma - 1) r_b beta, of rank two with eigenvalues {-c, 0, 0, c}, so
M^3 = c^2 M and any function of A fixed by its values on the spectrum {q - c, q, q, q + c} is a
quadratic in M: |A| = |l3| I + (|l1| - |l2|)/(2c) M + ((|l1| + |l2|)/2 - |l3|)/c^2 M^2, with
l1 = q + c, l2 = q - c, l3 = q and each |l| floored at EIG_FLOOR (their limiter). This equals
T|Lambda|T^-1 to round-off. rho and p are floored at STATE_FLOOR inside c and H, so a non-physical
state (early in training) gives a finite viscosity rather than NaN.
"""

from dataclasses import dataclass

import torch

from apno.constants import GAMMA

EIG_FLOOR = 0.25  # their Appendix C limiter on |lambda|
STATE_FLOOR = 1e-6  # floor on rho and p inside c and H
SELECT = (1.0, 0.0, 0.0, 1.0)  # I_{rho, rho E}: viscosity on the density and energy equations


@dataclass(frozen=True)
class SensorConstants:
    k0_shock: float  # threshold
    k1_shock: float  # steepness
    k0_stag: float = 3.0
    k1_stag: float = 5.0


READING_A = SensorConstants(k0_shock=0.03, k1_shock=4.0)  # steepness 4, as in their Fig. 3
READING_B = SensorConstants(k0_shock=4.0, k1_shock=0.03)  # their Table VII, read literally
READINGS = {"A": READING_A, "B": READING_B}


def sound_speed(rho, p, gamma=GAMMA):
    return torch.sqrt(gamma * p.clamp_min(STATE_FLOOR) / rho.clamp_min(STATE_FLOOR))


def shock_sensor(u, v, p_x, p_y, k):
    """tanh(max(0, k1 (q/|q|).grad p - k0)): positive where pressure rises along the flow."""
    speed = torch.sqrt(u * u + v * v).clamp_min(STATE_FLOOR)
    along = (u * p_x + v * p_y) / speed
    return torch.tanh(torch.clamp(k.k1_shock * along - k.k0_shock, min=0.0))


def stagnation_sensor(grad_mx, grad_my, k):
    """sig(k1 (k0 - (|grad rho u| + |grad rho v|))) from the two gradient magnitudes: near 0 where
    the momentum gradients are large (stagnation points), near 1 elsewhere."""
    return torch.sigmoid(k.k1_stag * (k.k0_stag - (grad_mx + grad_my)))


def sensor(u, v, p_x, p_y, mx_x, mx_y, my_x, my_y, k):
    """s = s_shock s_stag, detached: the PINN must not move the shock to lower its loss."""
    with torch.no_grad():
        s = shock_sensor(u, v, p_x, p_y, k) * stagnation_sensor(
            torch.hypot(mx_x, mx_y), torch.hypot(my_x, my_y), k)
    return s.detach()


def flux_jacobian_parts(rho, u, v, p, kx, ky, gamma=GAMMA):
    """(M, q, c) with A = q I + M the Jacobian dF_k/dW of the Euler flux in direction (kx, ky)."""
    q = u * kx + v * ky
    c = sound_speed(rho, p, gamma)
    # H from the same floored c (c^2/(gamma-1) = gamma p/((gamma-1) rho)): M's spectrum is then
    # {-c, 0, 0, c} with this c, so |A| stays an absolute value (dissipative) when rho, p <= 0.
    h = c * c / (gamma - 1) + 0.5 * (u * u + v * v)
    one, zero = torch.ones_like(rho), torch.zeros_like(rho)
    r_a = torch.stack([one, u, v, h], -1)
    r_b = torch.stack([zero, kx * one, ky * one, q], -1)
    l_b = torch.stack([-q, kx * one, ky * one, zero], -1)
    beta = torch.stack([0.5 * (u * u + v * v), -u, -v, one], -1)
    m = r_a[..., :, None] * l_b[..., None, :] + (gamma - 1) * r_b[..., :, None] * beta[..., None, :]
    return m, q, c


def abs_jacobian(rho, u, v, p, kx, ky, eps=EIG_FLOOR, gamma=GAMMA):
    """|A| = T|Lambda|T^-1 of dF_k/dW, each |eigenvalue| floored at eps; shape (n, 4, 4)."""
    m, q, c = flux_jacobian_parts(rho, u, v, p, kx, ky, gamma)
    l1 = (q + c).abs().clamp_min(eps)
    l2 = (q - c).abs().clamp_min(eps)
    l3 = q.abs().clamp_min(eps)
    eye = torch.eye(4, dtype=rho.dtype, device=rho.device).expand(m.shape)
    a = (0.5 * (l1 - l2) / c)[:, None, None]
    b = ((0.5 * (l1 + l2) - l3) / (c * c))[:, None, None]
    return l3[:, None, None] * eye + a * m + b * (m @ m)


def viscous_term(w_xx, w_yy, rho, u, v, p, s, nu, form="matrix", eps=EIG_FLOOR, gamma=GAMMA):
    """D = mu_x W_xx + mu_y W_yy on the density and energy rows only; shape (n, 4).

    w_xx, w_yy: (n, 4) second derivatives of the conserved state in x and y; s: (n,) the sensor.
    form "matrix" (their model V): mu_x = nu s I |A|, mu_y = nu s I |B|; "scalar" (their model IV):
    mu = nu s I (c + |u|). The momentum rows of D are exactly zero.
    """
    select = torch.tensor(SELECT, dtype=rho.dtype, device=rho.device)
    if form == "matrix":
        ax = abs_jacobian(rho, u, v, p, 1.0, 0.0, eps, gamma)
        by = abs_jacobian(rho, u, v, p, 0.0, 1.0, eps, gamma)
        d = (ax @ w_xx[..., None])[..., 0] + (by @ w_yy[..., None])[..., 0]
    elif form == "scalar":
        lam = sound_speed(rho, p, gamma) + torch.sqrt(u * u + v * v)
        d = lam[:, None] * (w_xx + w_yy)
    else:
        raise ValueError(f"unknown viscosity form {form!r}; expected 'matrix' or 'scalar'")
    return nu * s[:, None] * select * d
