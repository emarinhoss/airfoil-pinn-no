"""The PINN (spec 7): one case at a time, from the equations, boundary conditions and inputs only.

Units: rho/rho_inf, u/U_inf, p/(rho_inf U_inf^2). Exact by construction: positive density
(exponential output), constant total enthalpy (pressure from H_inf), and no flow through the wall
(normal component removed with weight (1 - eta)^4: 1 at the wall, 0 at the outer circle).
The network sees (cos 2 pi xi, sin 2 pi xi, eta), so periodicity around the airfoil is built in.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from apno.constants import GAMMA
from apno.geometry.ogrid import FAR_RADIUS

G = GAMMA


@dataclass(frozen=True)
class PinnCase:
    mach: float
    alpha_deg: float
    geometry: object
    radius: float = FAR_RADIUS


class PinnNet(nn.Module):
    def __init__(self, hidden=96, depth=7):
        super().__init__()
        layers, width = [], 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.Tanh()]
            width = hidden
        layers.append(nn.Linear(width, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xi, eta):
        f = torch.stack([torch.cos(2 * math.pi * xi), torch.sin(2 * math.pi * xi), eta], -1)
        return self.mlp(f)


def wall_normal(xi, geom):
    """Outward unit normal at wall point xi; differentiable in xi for the residual.

    A section is differentiated in s = cos(pi xi), because d/dxi of the cosine-clustered curve is
    exactly zero at the trailing edge (0/0 there). Since ds/dxi < 0 on (0, 1), the xi-tangent is
    -(x_s, y_s) and its left (outward) normal is (y_s, -x_s). At xi = 0 this is the lower surface's
    side of the trailing-edge corner, which is the named hazard of spec 7.7.
    """
    with torch.enable_grad():
        x = xi if xi.requires_grad else xi.detach().clone().requires_grad_(True)
        if hasattr(geom, "wall_s"):
            s = torch.cos(math.pi * x)
            xw, yw = geom.wall_s(s, xp=torch)
            xs = torch.autograd.grad(xw.sum(), s, create_graph=True)[0]
            ys = torch.autograd.grad(yw.sum(), s, create_graph=True)[0]
            nx, ny = ys, -xs
        else:
            xw, yw = geom.wall(x, xp=torch)
            dx = torch.autograd.grad(xw.sum(), x, create_graph=True)[0]
            dy = torch.autograd.grad(yw.sum(), x, create_graph=True)[0]
            nx, ny = -dy, dx  # left normal of the clockwise wall = outward
    norm = torch.sqrt(nx * nx + ny * ny)
    return nx / norm, ny / norm


def to_dataset_q(rho, u, v, p, mach):
    return torch.stack([rho, u, v, p * G * mach * mach], -1)


class Pinn:
    def __init__(self, case, net):
        self.case, self.net = case, net
        self.h_inf = 1.0 / ((G - 1.0) * case.mach**2) + 0.5
        a = math.radians(case.alpha_deg)
        self.ca, self.sa = math.cos(a), math.sin(a)

    def parameters(self):
        return self.net.parameters()

    def double(self):
        self.net.double()
        return self

    def state(self, xi, eta):
        out = self.net(xi, eta)
        rho = torch.exp(out[..., 0])
        u, v = self.ca + out[..., 1], self.sa + out[..., 2]
        nx, ny = wall_normal(xi, self.case.geometry)
        phi = (1.0 - eta) ** 4
        un = u * nx + v * ny
        u, v = u - phi * un * nx, v - phi * un * ny
        p = (G - 1.0) / G * rho * (self.h_inf - 0.5 * (u * u + v * v))
        return rho, u, v, p
