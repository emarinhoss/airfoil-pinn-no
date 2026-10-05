"""Steady 2D Euler residual in the O-grid's coordinates (xi, eta), by automatic differentiation.

Strong conservation form: div(F, G) = (1/J) [d/dxi (y_eta F - x_eta G) + d/deta (-y_xi F + x_xi G)],
with J = x_xi y_eta - x_eta y_xi. Because the metric terms come from autograd of one smooth map,
the mixed partials cancel exactly and a uniform flow gives zero residual (free-stream preservation).
The optional artificial viscosity subtracts nu * Laplacian of (rho, rho u, rho v). It is folded
into the fluxes, div(F - nu q_x, G - nu q_y), which is the same operator but shares the divergence's
derivative pass: two extra autograd passes per conserved variable instead of four (the nested form
made a viscous step 7-20x an Euler one at spec sizes).
Only continuity and momentum are solved: total enthalpy is constant and fixes the pressure.
"""

import torch

from apno.geometry.ogrid import FAR_RADIUS, map_xy, stretch_ratio


def _d(f, wrt):
    return torch.autograd.grad(f.sum(), wrt, create_graph=True)[0]


def metrics(xi, eta, geom, radius=FAR_RADIUS, ratio=None):
    if not xi.requires_grad:
        xi = xi.clone().requires_grad_(True)
    if not eta.requires_grad:
        eta = eta.clone().requires_grad_(True)
    ratio = stretch_ratio(radius) if ratio is None else ratio
    x, y = map_xy(xi, eta, geom, radius, ratio, xp=torch)
    x_xi, x_eta, y_xi, y_eta = _d(x, xi), _d(x, eta), _d(y, xi), _d(y, eta)
    return x, y, x_xi, x_eta, y_xi, y_eta, x_xi * y_eta - x_eta * y_xi


def residual(xi, eta, state_fn, geom, radius=FAR_RADIUS, ratio=None, viscosity=0.0):
    xi = xi.clone().requires_grad_(True)
    eta = eta.clone().requires_grad_(True)
    _, _, x_xi, x_eta, y_xi, y_eta, J = metrics(xi, eta, geom, radius, ratio)
    rho, u, v, p = state_fn(xi, eta)
    F = (rho * u, rho * u * u + p, rho * u * v)
    G = (rho * v, rho * u * v, rho * v * v + p)
    terms = (x_xi, x_eta, y_xi, y_eta, J)
    return torch.stack([divergence(f, g, q, xi, eta, terms, viscosity)
                        for f, g, q in zip(F, G, (rho, rho * u, rho * v))], -1)


def divergence(f, g, q, xi, eta, terms, viscosity=0.0):
    """div(f, g) - nu * Laplacian(q) in strong conservation form, with terms = (x_xi, x_eta, y_xi,
    y_eta, J) from `metrics` at the same (xi, eta). Shared by the Euler and full-potential
    residuals, so both use one operator, artificial viscosity included."""
    x_xi, x_eta, y_xi, y_eta, J = terms
    if viscosity:  # fold -nu grad(q) into the fluxes: one divergence pass, not two
        q_xi, q_eta = _d(q, xi), _d(q, eta)
        f = f - viscosity * (q_xi * y_eta - q_eta * y_xi) / J
        g = g - viscosity * (-q_xi * x_eta + q_eta * x_xi) / J
    return (_d(y_eta * f - x_eta * g, xi) + _d(-y_xi * f + x_xi * g, eta)) / J
