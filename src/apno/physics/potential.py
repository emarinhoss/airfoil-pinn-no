"""Steady full-potential residual in the O-grid's coordinates (xi, eta), by automatic
differentiation: the continuity equation div(rho grad phi) = 0 in strong conservation form.

Momentum is satisfied by construction when the state is irrotational, isentropic and at constant
total enthalpy (`apno.models.pinn_potential`), so continuity is the only equation. This is exact
for shock-free inviscid flow and an isentropic approximation at shocks (it has no entropy jump).
The artificial-viscosity variant subtracts nu * Laplacian(rho) from it, with the same operator
`apno.physics.euler` uses for its continuity row (`euler.divergence`), as a change to the equations.
"""

from apno.geometry.ogrid import FAR_RADIUS
from apno.physics.euler import divergence, metrics


def residual(xi, eta, state_fn, geom, radius=FAR_RADIUS, ratio=None, viscosity=0.0):
    """One value per point: (1/J) [d/dxi (y_eta F - x_eta G) + d/deta (-y_xi F + x_xi G)] with
    (F, G) = rho (u, v), minus nu * Laplacian(rho) when viscosity is set."""
    xi = xi.clone().requires_grad_(True)
    eta = eta.clone().requires_grad_(True)
    _, _, x_xi, x_eta, y_xi, y_eta, J = metrics(xi, eta, geom, radius, ratio)
    rho, u, v, _ = state_fn(xi, eta)
    return divergence(rho * u, rho * v, rho, xi, eta, (x_xi, x_eta, y_xi, y_eta, J), viscosity)
