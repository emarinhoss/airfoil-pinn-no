"""The full-potential PINN (spec 7, amended 2026-10-04): one scalar network, one equation.

Velocity potential, in units of U_inf x chord:

    phi = x cos(alpha) + y sin(alpha) + Gamma * Theta(xi) + psi(xi, eta) * w(eta)

- psi is the network NN(cos 2 pi xi, sin 2 pi xi, eta): periodic in xi, so single-valued.
- w(eta) = 1 - eta vanishes on the 100-chord circle, so there phi is free stream plus the vortex
  term exactly: a hard Dirichlet condition (the neglected doublet is ~1e-3 there).
- Gamma is a learned scalar, the circulation, and Theta = xi is the wake-cut angle term. xi runs
  once around the O-grid, so Theta jumps by one unit across the line xi = 0 (the trailing edge out
  to the far field) while its gradient, set by the map's metric terms, is periodic and therefore
  continuous across it. On the outer circle the far-field point sits at polar angle -2 pi xi about
  mid-chord, so grad(Gamma xi) -> Gamma / (2 pi r), clockwise. Convention: Gamma is the CLOCKWISE
  circulation (the counterclockwise line integral of the velocity is -Gamma), so Kutta-Joukowski
  gives L' = rho U Gamma and positive Gamma is positive lift. A cylinder warm-up fixes Gamma at 0.

The state follows algebraically: (u, v) = grad(phi) through the map's autograd metric terms (the
same ones as `apno.physics.euler`), rho from isentropic Bernoulli at the free-stream total enthalpy,
rho = (1 + (gamma-1)/2 M^2 (1 - |u|^2))^(1/(gamma-1)) with the base clamped at a small positive
floor, and p / p_inf = rho^gamma. `state` returns p in rho_inf U_inf^2 units like the Euler PINN, so
`to_dataset_q` reports (rho/rho_inf, u/U_inf, v/U_inf, p/p_inf) and the scorer is unchanged.

The trailing edge. The section's cosine clustering makes d(x, y)/dxi vanish at the trailing-edge
wall point, so the map is degenerate there (J = 0) and a smooth phi(xi, eta) gives a finite
velocity there iff dphi/dxi = 0 at (xi, eta) = (0, 0); otherwise the velocity ~ dphi/dxi / xi, the
corner singularity. So the Kutta condition is a penalty on (dphi/dxi)^2 at that point
(`trailing_edge_slope`), the finite form of the |u|^2 = 0 the spec states, and at that one point,
where 0/0 has no value, `state` reports the stagnation velocity zero that the Kutta condition
imposes. Every other point (and every cylinder point) has J > 0.
"""

import math

import torch
from torch import nn

from apno.constants import GAMMA
from apno.geometry.ogrid import map_xy, stretch_ratio
from apno.models.pinn import wall_normal
from apno.physics.euler import _d, metrics
from apno.physics.potential import residual

G = GAMMA
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
    def theta(xi):
        """The wake-cut angle term: one unit of jump across xi = 0, periodic gradient."""
        return xi

    def potential(self, xi, eta, x, y):
        return (x * self.ca + y * self.sa + self.net.gamma * self.theta(xi)
                + self.net(xi, eta) * wall_weight(eta))

    def state(self, xi, eta):
        c = self.case
        return state_from_potential(self.potential, xi, eta, c.geometry, c.radius, self.ratio,
                                    c.mach)

    def trailing_edge_slope(self, like):
        """dphi/dxi at the trailing-edge wall point (0, 0), on `like`'s device and dtype. Zero is
        the Kutta condition (the free-stream term contributes nothing, since dx/dxi = dy/dxi = 0
        there on a section)."""
        with torch.enable_grad():
            xi = torch.zeros(1, dtype=like.dtype, device=like.device, requires_grad=True)
            eta = torch.zeros_like(xi)
            x, y = map_xy(xi, eta, self.case.geometry, self.case.radius, self.ratio, xp=torch)
            return _d(self.potential(xi, eta, x, y), xi)

    def wall_normal_velocity(self, xi):
        """u . n_w at wall points xi (n_w from `wall_normal`, finite at the trailing edge)."""
        _, u, v, _ = self.state(xi, torch.zeros_like(xi))
        nx, ny = wall_normal(xi, self.case.geometry)
        return u * nx + v * ny


def losses(pinn, pts, cfg, ratio, weights):
    """The potential formulation's loss terms and weighted total.

    residual: mean square of the continuity residual over the interior points; wall: the soft
    Neumann condition, mean (u . n_w)^2 over the wall points shifted by half a spacing (the
    trailing-edge point itself has no finite velocity to penalise unless Kutta holds; see the
    module docstring); kutta: (dphi/dxi)^2 at the trailing edge when cfg.kutta; circulation: zero,
    since a cylinder warm-up fixes Gamma = 0 instead of penalising it. There is no outer term: the
    far-field condition is exact.
    """
    case = pinn.case
    r = residual(pts["xi"], pts["eta"], pinn.state, case.geometry, case.radius, ratio,
                 cfg.viscosity)
    xw = pts["xi_wall"] + 0.5 / cfg.n_wall
    zero = torch.zeros((), dtype=xw.dtype, device=xw.device)
    out = {"residual": (r * r).mean(), "wall": (pinn.wall_normal_velocity(xw) ** 2).mean()}
    out["kutta"] = (pinn.trailing_edge_slope(xw) ** 2).sum() if cfg.kutta else zero
    out["circulation"] = zero
    out["total"] = (cfg.w_res * out["residual"] + weights["wall"] * out["wall"]
                    + weights["kutta"] * out["kutta"])
    return out
