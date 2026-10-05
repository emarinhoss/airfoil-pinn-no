"""The full-potential PINN (spec 7, amended 2026-10-04 and 2026-10-05): one scalar network, one
equation.

Velocity potential, in units of U_inf x chord:

    phi = x cos(alpha) + y sin(alpha) + Gamma * Theta_v(x, y) + psi(xi, eta) * w(eta)

- psi is the network NN(cos 2 pi xi, sin 2 pi xi, eta): periodic in xi, so single-valued.
- w(eta) = 1 - eta vanishes on the 100-chord circle, so there phi is free stream plus the vortex
  term exactly: a hard Dirichlet condition (the neglected doublet is ~1e-3 there).
- Gamma is a learned scalar, the circulation, and Theta_v is the angle of a point vortex at the
  quarter chord (`vortex_angle`): Theta_v = 1 - (atan2(y, x - 0.25) mod 2 pi) / (2 pi). Its unit
  jump lies on the ray y = 0, x > 0.25, which in the domain is exactly the O-grid's xi = 0 line
  (the trailing edge out to the far field, the wake cut); its gradient, the vortex velocity
  1 / (2 pi r) clockwise, is continuous across that line and regular everywhere in the domain,
  the trailing edge included. Far from the body Theta_v ~ xi. (Until 2026-10-05 this term was
  Theta = xi, whose gradient is singular at the trailing edge, ~1/sqrt(r), because the section's
  cosine clustering makes x_xi = y_xi = 0 there: on test_in-0022 the Gamma-only residual was
  1.2e4 with Theta = xi and 8e-3 with Theta_v; docs/findings.md, 2026-10-05.) Convention: Gamma
  is the CLOCKWISE circulation (the counterclockwise line integral of the velocity is -Gamma), so
  Kutta-Joukowski gives L' = rho U Gamma and positive Gamma is positive lift. A cylinder warm-up
  fixes Gamma at 0.

The state follows algebraically: (u, v) = grad(phi) through the map's autograd metric terms (the
same ones as `apno.physics.euler`), rho from isentropic Bernoulli at the free-stream total enthalpy,
rho = (1 + (gamma-1)/2 M^2 (1 - |u|^2))^(1/(gamma-1)) with the base clamped at a small positive
floor, and p / p_inf = rho^gamma. `state` returns p in rho_inf U_inf^2 units like the Euler PINN, so
`to_dataset_q` reports (rho/rho_inf, u/U_inf, v/U_inf, p/p_inf) and the scorer is unchanged.

The trailing edge. The section's cosine clustering makes d(x, y)/dxi vanish at the trailing-edge
wall point, so the map is degenerate there (J = 0); at that one point, where 0/0 has no value,
`state` reports velocity zero. The network term's second derivatives are singular at the sharp
corner whatever its input features (as the true corner flow is), so the training excludes the
interior points within `PinnConfig.te_exclusion` chords of the trailing edge from the residual
(`apno.models.train_pinn.exclude_trailing_edge`). The Kutta condition is the equal-speed form,
|u|^2 at the wall point xi = delta equal to |u|^2 at xi = 1 - delta (`kutta_residual`), with
delta = `PinnConfig.kutta_delta` (0.055: about 0.03 chord upstream of the trailing edge on a NACA
4-digit section, just outside the exclusion). Equal speeds on the two surfaces a finite distance
upstream are an approximation to the trailing-edge limit, where they are equal exactly (no
loading at the edge). It replaces dphi/dxi = 0 at the trailing-edge point, which was degenerate:
there x_xi = y_xi = 0, so the free-stream and circulation terms contribute nothing to it and it
never constrains Gamma. Every point other than the trailing-edge wall point has J > 0.
"""

import math

import torch
from torch import nn

from apno.constants import GAMMA
from apno.geometry.ogrid import stretch_ratio
from apno.models.pinn import wall_normal
from apno.physics.euler import metrics
from apno.physics.potential import residual

G = GAMMA
QUARTER_CHORD = 0.25  # the circulation's vortex sits at (0.25, 0)
DENSITY_FLOOR = 1e-6  # floor on the Bernoulli base, so a supersonic overshoot cannot give NaN


class PotentialNet(nn.Module):
    """psi(xi, eta), plus the circulation Gamma: a parameter, or a zero buffer when fixed."""

    def __init__(self, hidden=96, depth=7, circulation=True):
        super().__init__()
        layers, width = [], 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)
        if circulation:
            self.gamma = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("gamma", torch.zeros(()))

    def forward(self, xi, eta):
        f = torch.stack([torch.cos(2 * math.pi * xi), torch.sin(2 * math.pi * xi), eta], -1)
        return self.mlp(f)[..., 0]


def vortex_angle(x, y, x0=QUARTER_CHORD):
    """Theta_v = 1 - (atan2(y, x - x0) mod 2 pi) / (2 pi), the clockwise point vortex's angle
    about (x0, 0) in units of one turn: a unit jump on the ray y = 0, x > x0 (the O-grid's wake
    cut xi = 0) and a gradient 1 / (2 pi r), clockwise, that is continuous across it."""
    return 1.0 - torch.remainder(torch.atan2(y, x - x0), 2 * math.pi) / (2 * math.pi)


def wall_weight(eta):
    """w(eta) = 1 - eta: 1 at the wall, exactly 0 on the outer circle (the Dirichlet condition)."""
    return 1.0 - eta


def bernoulli_density(speed2, mach):
    base = 1.0 + 0.5 * (G - 1.0) * mach * mach * (1.0 - speed2)
    return base.clamp_min(DENSITY_FLOOR) ** (1.0 / (G - 1.0))


def _grad(f, wrt):
    """d f / d wrt with the graph kept; zero where f does not depend on wrt (e.g. phi = xi)."""
    g = torch.autograd.grad(f.sum(), wrt, create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(wrt) if g is None else g


def _leaf(t):
    return t if t.requires_grad else t.detach().clone().requires_grad_(True)


def state_from_potential(phi_fn, xi, eta, geom, radius, ratio, mach):
    """(rho, u, v, p) of the potential phi_fn(xi, eta, x, y), p in rho_inf U_inf^2 units.

    Differentiable (create_graph), and usable under torch.no_grad. Where J = 0 (only the section's
    trailing-edge wall point) the velocity is reported as zero, the Kutta stagnation value.
    """
    with torch.enable_grad():
        xi, eta = _leaf(xi), _leaf(eta)
        x, y, x_xi, x_eta, y_xi, y_eta, J = metrics(xi, eta, geom, radius, ratio)
        phi = phi_fn(xi, eta, x, y)
        p_xi, p_eta = _grad(phi, xi), _grad(phi, eta)
        degenerate = J == 0
        Js = torch.where(degenerate, torch.ones_like(J), J)  # no 0/0, so no NaN gradient either
        u = torch.where(degenerate, torch.zeros_like(J), (p_xi * y_eta - p_eta * y_xi) / Js)
        v = torch.where(degenerate, torch.zeros_like(J), (-p_xi * x_eta + p_eta * x_xi) / Js)
        rho = bernoulli_density(u * u + v * v, mach)
        p = rho**G / (G * mach * mach)
    return rho, u, v, p


class PotentialPinn:
    formulation = "potential"

    def __init__(self, case, net):
        self.case, self.net = case, net
        self.ratio = stretch_ratio(case.radius)
        a = math.radians(case.alpha_deg)
        self.ca, self.sa = math.cos(a), math.sin(a)

    def parameters(self):
        return self.net.parameters()

    def double(self):
        self.net.double()
        return self

    @staticmethod
    def theta(x, y):
        """The circulation's angle term, `vortex_angle`."""
        return vortex_angle(x, y)

    def potential(self, xi, eta, x, y):
        return (x * self.ca + y * self.sa + self.net.gamma * self.theta(x, y)
                + self.net(xi, eta) * wall_weight(eta))

    def state(self, xi, eta):
        c = self.case
        return state_from_potential(self.potential, xi, eta, c.geometry, c.radius, self.ratio,
                                    c.mach)

    def kutta_residual(self, delta, like):
        """|u|^2 at the wall point xi = delta minus |u|^2 at xi = 1 - delta, on `like`'s device
        and dtype: zero is the equal-speed Kutta condition (see the module docstring)."""
        xk = torch.tensor([delta, 1.0 - delta], dtype=like.dtype, device=like.device)
        _, u, v, _ = self.state(xk, torch.zeros_like(xk))
        q2 = u * u + v * v
        return q2[0] - q2[1]

    def wall_normal_velocity(self, xi):
        """u . n_w at wall points xi (n_w from `wall_normal`, finite at the trailing edge)."""
        _, u, v, _ = self.state(xi, torch.zeros_like(xi))
        nx, ny = wall_normal(xi, self.case.geometry)
        return u * nx + v * ny


def losses(pinn, pts, cfg, ratio, weights):
    """The potential formulation's loss terms and weighted total.

    residual: mean square of the continuity residual over the interior points (for a section,
    those left after `train_pinn.exclude_trailing_edge`); wall: the soft Neumann condition, mean
    (u . n_w)^2 over the wall points shifted by half a spacing (so the degenerate trailing-edge
    point is not one of them); kutta: the equal-speed condition, `kutta_residual(cfg.kutta_delta)`
    squared, when cfg.kutta; circulation: zero, since a cylinder warm-up fixes Gamma = 0 instead of
    penalising it. There is no outer term: the far-field condition is exact.
    """
    case = pinn.case
    r = residual(pts["xi"], pts["eta"], pinn.state, case.geometry, case.radius, ratio,
                 cfg.viscosity)
    xw = pts["xi_wall"] + 0.5 / cfg.n_wall
    zero = torch.zeros((), dtype=xw.dtype, device=xw.device)
    out = {"residual": (r * r).mean(), "wall": (pinn.wall_normal_velocity(xw) ** 2).mean()}
    out["kutta"] = pinn.kutta_residual(cfg.kutta_delta, xw) ** 2 if cfg.kutta else zero
    out["circulation"] = zero
    out["total"] = (cfg.w_res * out["residual"] + weights["wall"] * out["wall"]
                    + weights["kutta"] * out["kutta"])
    return out
