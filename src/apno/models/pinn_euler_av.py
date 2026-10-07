"""The S5 Euler PINN with localised artificial viscosity (S5 design §2; Wassing, Langer &
Bekemeyer 2025, arXiv:2408.17364).

The network outputs the primitive variables (rho, u, v, p) in their scaling: rho_inf = p_inf = 1,
(u_inf, v_inf) = sqrt(gamma) M_inf (cos alpha, sin alpha), coordinates in chords, so c_inf =
sqrt(gamma) and their viscosity factor and sensor constants keep their meaning. `to_dataset_q`
converts to the dataset's units exactly: rho and p unchanged, u / U_inf = u / (sqrt(gamma) M_inf).

Inputs are the collocation grid's index coordinates (NI xi, (NJ - 1) eta) through a random
Fourier-feature embedding (n_freq frequencies from N(0, sigma^2), sin and cos), then `depth`
weight-normalised layers of `hidden` adaptive-tanh units, tanh(n omega_k x) with n = act_scale and
omega_k from omega0 (their scale factor is not stated; n = 10, omega0 = 0.1 follows Jagtap et al.).
Ablation variants (design §4): encoding "periodic" feeds (cos 2 pi xi, sin 2 pi xi, eta) (A4);
closure "enthalpy" is this study's Euler closure (A2): outputs ln rho and the velocity's departure
from free stream, pressure from constant total enthalpy, wall tangency exact by the (1 - eta)^4
projection, and the energy equation, then an identity, dropped from the residual.

The residual is the steady Euler system in strong conservation form through the O-grid's metric
terms (`apno.physics.euler.divergence`), minus the dissipative viscous term D of
`apno.physics.viscosity` (design §2.4). D is evaluated only where the sensor exceeds
cfg.sensor_floor, on a re-evaluation of the state at those points; `viscous_everywhere` (set by the
preflight's timing) evaluates it at every point, the cost's upper bound.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from apno.constants import GAMMA
from apno.geometry.ogrid import FINE, stretch_ratio
from apno.models.pinn import wall_normal
from apno.physics.euler import divergence, metrics
from apno.physics.viscosity import READINGS, sensor, viscous_term

G = GAMMA
ENCODINGS = ("fourier", "periodic")
CLOSURES = ("primitive", "enthalpy")
VISCOSITIES = ("matrix", "scalar", "none")


@dataclass(frozen=True)
class AvConfig:
    # network (design §2.2)
    hidden: int = 128
    depth: int = 5
    n_freq: int = 128
    sigma: float = 1.0
    act_scale: float = 10.0
    omega0: float = 0.1
    encoding: str = "fourier"  # A4: "periodic"
    closure: str = "primitive"  # A2: "enthalpy"
    # loss and collocation (design §2.3)
    weighting: str = "volume"  # A1: "uniform"
    lambda_res: float = 2e4  # A1: 1.0
    w_boundary: float = 1.0  # A1: 10.0
    te_exclusion: float = 0.0  # T4: 0.02 chord
    collocation: str = "nodes"  # A5: "random"
    grid_shape: tuple = FINE
    # artificial viscosity (design §2.4)
    viscosity: str = "matrix"  # T3: "scalar"; "none" switches D off
    nu: float = 1.9e-3
    sensor: str = "A"  # T2: "B"
    sensor_floor: float = 1e-4
    # optimiser (design §2.5; A3)
    optimizer: str = "lbfgs"  # A3: "adam_lbfgs"
    batch: int = 5000
    epochs: int = 40
    lbfgs_max_iter: int = 1000
    lbfgs_max_eval: int = 1250
    lbfgs_tol_grad: float = 1e-6
    lbfgs_tol_change: float = 1e-10
    lbfgs_history: int = 1000
    adam_steps: int = 30000
    adam_lr0: float = 1e-3
    adam_lr1: float = 1e-4
    final_lbfgs_iters: int = 5000
    final_lbfgs_history: int = 50
    seed: int = 0
    device: str = "auto"  # training; evaluation is always CPU float64


def config_from_dict(d):
    """The inverse of dataclasses.asdict through JSON (which turns the grid shape into a list)."""
    return AvConfig(**{**d, "grid_shape": tuple(d["grid_shape"])})


def freestream(mach, alpha_deg):
    a, s = math.radians(alpha_deg), math.sqrt(G) * mach
    return 1.0, s * math.cos(a), s * math.sin(a), 1.0


def to_dataset_q(rho, u, v, p, mach):
    s = math.sqrt(G) * mach
    return torch.stack([rho, u / s, v / s, p], -1)


def from_dataset_q(q, mach):
    s = math.sqrt(G) * mach
    return q[..., 0], q[..., 1] * s, q[..., 2] * s, q[..., 3]


def conserved(rho, u, v, p):
    return rho, rho * u, rho * v, p / (G - 1) + 0.5 * rho * (u * u + v * v)


def fluxes(rho, u, v, p):
    e = conserved(rho, u, v, p)[3]
    return ((rho * u, rho * u * u + p, rho * u * v, u * (e + p)),
            (rho * v, rho * u * v, rho * v * v + p, v * (e + p)))


def _grad(f, wrt):
    g = torch.autograd.grad(f.sum(), wrt, create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(wrt) if g is None else g


def physical_grad(f, xi, eta, terms):
    """(df/dx, df/dy) by the chain rule through the map's metric terms (x_xi, x_eta, y_xi, y_eta,
    J), differentiable again."""
    x_xi, x_eta, y_xi, y_eta, J = terms
    f_xi, f_eta = _grad(f, xi), _grad(f, eta)
    return (f_xi * y_eta - f_eta * y_xi) / J, (-f_xi * x_eta + f_eta * x_xi) / J


class FourierEmbedding(nn.Module):
    def __init__(self, n_freq, sigma, generator):
        super().__init__()
        self.register_buffer("phi", sigma * torch.randn(n_freq, 2, generator=generator))

    def forward(self, r):
        z = r @ self.phi.T
        return torch.stack([torch.sin(z), torch.cos(z)], -1).flatten(-2)


class AdaptiveTanh(nn.Module):
    def __init__(self, scale, omega0):
        super().__init__()
        self.scale = scale
        self.omega = nn.Parameter(torch.tensor(float(omega0)))

    def forward(self, x):
        return torch.tanh(self.scale * self.omega * x)


class EulerAvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        gen = torch.Generator().manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.ni, self.nj1 = cfg.grid_shape[0], cfg.grid_shape[1] - 1
        if cfg.encoding == "fourier":
            self.embed, width = FourierEmbedding(cfg.n_freq, cfg.sigma, gen), 2 * cfg.n_freq
        else:
            self.embed, width = None, 3
        layers = []
        for _ in range(cfg.depth):
            layers += [weight_norm(nn.Linear(width, cfg.hidden)),
                       AdaptiveTanh(cfg.act_scale, cfg.omega0)]
            width = cfg.hidden
        layers.append(weight_norm(nn.Linear(width, 3 if cfg.closure == "enthalpy" else 4)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xi, eta):
        if self.embed is None:
            f = torch.stack([torch.cos(2 * math.pi * xi), torch.sin(2 * math.pi * xi), eta], -1)
        else:
            f = self.embed(torch.stack([self.ni * xi, self.nj1 * eta], -1))
        return self.mlp(f)


class EulerAvPinn:
    formulation = "euler_av"

    def __init__(self, case, cfg):
        if (cfg.encoding not in ENCODINGS or cfg.closure not in CLOSURES
                or cfg.viscosity not in VISCOSITIES):
            raise ValueError(f"unknown option in {cfg}")
        self.case, self.cfg = case, cfg
        self.net = EulerAvNet(cfg)
        self.ratio = stretch_ratio(case.radius)
        self.w_inf = freestream(case.mach, case.alpha_deg)
        self.k = READINGS[cfg.sensor]
        self.viscous_everywhere = False

    def parameters(self):
        return self.net.parameters()

    def primitive(self, xi, eta):
        """(rho, u, v, p) in the S5 scaling at (xi, eta)."""
        out = self.net(xi, eta)
        if self.cfg.closure == "primitive":
            return out.unbind(-1)
        rho = torch.exp(out[..., 0])
        u, v = self.w_inf[1] + out[..., 1], self.w_inf[2] + out[..., 2]
        nx, ny = wall_normal(xi, self.case.geometry)
        blend = (1.0 - eta) ** 4
        un = u * nx + v * ny
        u, v = u - blend * un * nx, v - blend * un * ny
        h_inf = G / (G - 1) + 0.5 * G * self.case.mach ** 2  # rho_inf = p_inf = 1
        return rho, u, v, (G - 1) / G * rho * (h_inf - 0.5 * (u * u + v * v))

    def _terms(self, xi, eta):
        return metrics(xi, eta, self.case.geometry, self.case.radius, self.ratio)[2:]

    def residual(self, xi, eta):
        """div F - D at (xi, eta): (n, 4), or (n, 3) for the enthalpy closure."""
        xi = xi.detach().clone().requires_grad_(True)
        eta = eta.detach().clone().requires_grad_(True)
        terms = self._terms(xi, eta)
        rho, u, v, p = self.primitive(xi, eta)
        w = conserved(rho, u, v, p)
        f, g = fluxes(rho, u, v, p)
        rows = 3 if self.cfg.closure == "enthalpy" else 4
        r = torch.stack([divergence(f[k], g[k], w[k], xi, eta, terms) for k in range(rows)], -1)
        if self.cfg.viscosity == "none":
            return r
        return r - self.viscous(xi, eta, terms, u, v, p, w)[:, :rows]

    def viscous(self, xi, eta, terms, u, v, p, w):
        """D at every point, (n, 4): zero where the sensor is at or below cfg.sensor_floor."""
        p_x, p_y = physical_grad(p, xi, eta, terms)
        mx_x, mx_y = physical_grad(w[1], xi, eta, terms)
        my_x, my_y = physical_grad(w[2], xi, eta, terms)
        s = sensor(u, v, p_x, p_y, mx_x, mx_y, my_x, my_y, self.k)
        on = torch.ones_like(s, dtype=torch.bool) if self.viscous_everywhere else (
            s > self.cfg.sensor_floor)
        idx = on.nonzero(as_tuple=True)[0]
        d = torch.zeros(len(s), 4, dtype=s.dtype, device=s.device)
        if len(idx) == 0:
            return d
        xm = xi.detach()[idx].clone().requires_grad_(True)
        em = eta.detach()[idx].clone().requires_grad_(True)
        tm = self._terms(xm, em)
        rho_m, u_m, v_m, p_m = self.primitive(xm, em)
        first = [physical_grad(c, xm, em, tm) for c in conserved(rho_m, u_m, v_m, p_m)]
        w_xx = torch.stack([physical_grad(cx, xm, em, tm)[0] for cx, _ in first], -1)
        w_yy = torch.stack([physical_grad(cy, xm, em, tm)[1] for _, cy in first], -1)
        dm = viscous_term(w_xx, w_yy, rho_m, u_m, v_m, p_m, s[idx], self.cfg.nu,
                          self.cfg.viscosity)
        return d.index_put((idx,), dm)


def losses(pinn, batch, bnd, cfg):
    """The S5 loss terms (design §2.3) and the weighted total.

    batch: {"xi", "eta", "w"}, residual points and their weights (mean 1 over the residual set);
    bnd: {"xi_ring", "eta_lr"}: the wall (eta = 0) and outer (eta = 1) rows at xi_ring, and the
    periodic pairs (xi = 0, xi = 1) at eta_lr. Squared norms are summed over the components.
    """
    r = pinn.residual(batch["xi"], batch["eta"])
    res = (batch["w"] * (r * r).sum(-1)).mean()
    xr = bnd["xi_ring"]
    one, zero = torch.ones_like(xr), torch.zeros_like(xr)
    far = sum((a - b) ** 2 for a, b in zip(pinn.primitive(xr, one), pinn.w_inf)).mean()
    _, uw, vw, _ = pinn.primitive(xr, zero)
    nx, ny = wall_normal(xr, pinn.case.geometry)
    wall = ((uw * nx + vw * ny) ** 2).mean()
    el = bnd["eta_lr"]
    left, right = pinn.primitive(torch.zeros_like(el), el), pinn.primitive(torch.ones_like(el), el)
    periodic = sum((a - b) ** 2 for a, b in zip(left, right)).mean()
    total = cfg.lambda_res * res + cfg.w_boundary * (far + wall + periodic)
    return {"residual": res, "far": far, "wall": wall, "periodic": periodic, "total": total}
