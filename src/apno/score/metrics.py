"""What is scored (spec 8.2). One function per quantity; score_case runs them all on one case.

The near field is nodes within 2.5 chords of mid-chord, weighted by dual-cell area. Drag is in
counts and never a percentage. Shocks are located from wall pressure, per surface, only where that
surface goes supersonic. Physics checks run on every contestant, SU2 included.
"""

import numpy as np

from apno.geometry.ogrid import cell_areas
from apno.physics.flow import (
    entropy, freestream_q, isentropic_mach_from_cp, pressure_coefficient, total_enthalpy_ratio,
)
from apno.score.forces import vertex_normals, wall_forces

NEAR_RADIUS = 2.5
FLUX_LINES = (8, 24, 48, 72)
OVERSHOOT_WINDOW = 0.05


def node_areas(X, Y):
    A = cell_areas(X, Y)
    w = np.zeros_like(X)
    w[:, :-1] += A / 4 + np.roll(A, 1, 0) / 4
    w[:, 1:] += A / 4 + np.roll(A, 1, 0) / 4
    return w


def _surfaces(ni):
    le = ni // 2
    return {"lower": np.arange(le, -1, -1), "upper": np.r_[np.arange(le, ni), 0]}  # LE -> TE


def shocks(xw, cp, mach):
    """Per surface, the steepest pressure rise downstream of the suction peak whose upstream node is
    supersonic (spec 8.2 as amended 2026-10-01), or None if that surface has no such segment.

    The supersonic-upstream restriction is what makes this a shock rule. Without it the steepest
    rise on a cosine-clustered wall is the trailing-edge recompression into the closing stagnation
    point (the last interval is about 1e-4 chord, so its dCp/dx is 100-2000 against 50-70 for a
    smeared captured shock), and every shock lands at x/c = 0.9999 with a subsonic Mach ahead. The
    leading-edge node is shared by both surfaces and counts for the upper one only, so a supersonic
    LE node cannot make the lower surface report its LE-to-stagnation compression as a shock.
    """
    out = {}
    for name, idx in _surfaces(len(xw)).items():
        x, c = xw[idx], cp[idx]
        with np.errstate(invalid="ignore"):  # Cp above stagnation -> NaN Mach, never supersonic
            m_loc = isentropic_mach_from_cp(c, mach)
            sup = m_loc > 1.0
        if name == "lower":
            sup[0] = False  # the shared LE node belongs to the upper surface
        if not sup.any():
            out[name] = None
            continue
        k0 = int(np.argmin(c))
        dx = np.diff(x[k0:])
        ok = (dx > 0) & sup[k0:-1]
        if dx.size == 0 or not ok.any():
            out[name] = None
            continue
        slope = np.where(ok, np.diff(c[k0:]) / np.where(ok, dx, 1), -np.inf)
        k = k0 + int(np.argmax(slope))
        out[name] = {"x": float(0.5 * (x[k] + x[k + 1])), "mach_ahead": float(m_loc[k])}
    return out


def _mass_flux(q, X, Y, j):
    n = vertex_normals(X[:, j], Y[:, j])
    rho, u, v = q[:, j, 0], q[:, j, 1], q[:, j, 2]
    return float(np.sum(rho * (u * n[:, 0] + v * n[:, 1])))


def physics_checks(q, X, Y, mach, near, flux_lines=FLUX_LINES):
    w = node_areas(X, Y)[near]
    s = entropy(q[..., 0], q[..., 3])[near]
    h = total_enthalpy_ratio(*(q[..., k] for k in range(4)), mach)[near]
    n = vertex_normals(X[:, 0], Y[:, 0])
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    un = q[:, 0, 1] * n[:, 0] + q[:, 0, 2] * n[:, 1]
    return {"entropy_max": float(np.max(s)), "entropy_mean": float(np.sum(w * s) / np.sum(w)),
            "enthalpy_max_dev": float(np.max(np.abs(h - 1))),
            "wall_normal_rms": float(np.sqrt(np.mean(un * un))),
            "mass_flux": {j: _mass_flux(q, X, Y, j) for j in flux_lines}}


def score_case(pred_q, true_q, X, Y, mach, alpha_deg, truth_regime, flux_lines=FLUX_LINES):
    """flux_lines are j indices of THIS grid: FLUX_LINES on the training grid, j // 2 at 128x49."""
    if pred_q is None:
        return {"status": "missing"}
    if not np.isfinite(pred_q).all():
        return {"status": "non-finite"}
    xw, yw = X[:, 0], Y[:, 0]
    cp_p = pressure_coefficient(pred_q[:, 0, 3], mach)
    cp_t = pressure_coefficient(true_q[:, 0, 3], mach)
    fp, ft = wall_forces(xw, yw, cp_p, alpha_deg), wall_forces(xw, yw, cp_t, alpha_deg)
    near = np.hypot(X - 0.5, Y) <= NEAR_RADIUS
    w = node_areas(X, Y)[near][:, None]
    q_inf = freestream_q(alpha_deg)
    dp, dt = pred_q[near] - q_inf, true_q[near] - q_inf
    den = np.sqrt(np.sum(w * dt * dt))
    out = {"status": "ok",
           "cl_err": fp[0] - ft[0], "cd_err_counts": 1e4 * (fp[1] - ft[1]), "cm_err": fp[2] - ft[2],
           "cl_true": ft[0], "cd_true_counts": 1e4 * ft[1], "cm_true": ft[2],
           "cp_l2": float(np.sqrt(np.mean((cp_p - cp_t) ** 2))),
           "cp_max": float(np.abs(cp_p - cp_t).max()),
           "near_field_rel_l2": float(np.sqrt(np.sum(w * (dp - dt) ** 2)) / den) if den else 0.0,
           "physics": physics_checks(pred_q, X, Y, mach, near, flux_lines)}
    if truth_regime == "supercritical":
        sp, st = shocks(xw, cp_p, mach), shocks(xw, cp_t, mach)
        res = {}
        for surf, t in st.items():
            if t is None:
                continue
            idx = _surfaces(len(xw))[surf]
            win = idx[np.abs(xw[idx] - t["x"]) <= OVERSHOOT_WINDOW]
            p = sp[surf]
            res[surf] = {"x_true": t["x"], "x_err": None if p is None else p["x"] - t["x"],
                         "missed": p is None, "mach_ahead_true": t["mach_ahead"],
                         "mach_ahead_err": None if p is None else p["mach_ahead"] - t["mach_ahead"],
                         "overshoot": float(np.abs(cp_p[win] - cp_t[win]).max())}
        out["shock"] = res
    return out
