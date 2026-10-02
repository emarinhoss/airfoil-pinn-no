import csv

import numpy as np
import pytest

from apno.physics.flow import nondimensionalise, pressure_coefficient
from apno.score.forces import vertex_normals, wall_forces


def _circle(n=256):
    th = -2 * np.pi * np.arange(n) / n
    return 0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th), th


def test_uniform_pressure_gives_no_force():
    x, y, _ = _circle()
    assert np.allclose(wall_forces(x, y, np.full_like(x, 0.7), 3.0), 0.0, atol=1e-14)


def test_potential_cylinder_has_no_lift_or_drag():
    x, y, th = _circle()
    cl, cd, cm = wall_forces(x, y, 1 - 4 * np.sin(th) ** 2, 0.0)
    assert abs(cl) < 1e-13 and abs(cd) < 1e-13


def test_orientation_does_not_matter():
    x, y, th = _circle(64)
    cp = np.cos(th) + 0.3 * np.sin(th)
    a = wall_forces(x, y, cp, 2.0)
    b = wall_forces(x[::-1], y[::-1], cp[::-1], 2.0)
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_normals_point_outward():
    x, y, _ = _circle(64)
    n = vertex_normals(x, y)
    assert np.all((x - 0.5) * n[:, 0] + y * n[:, 1] > 0)


def test_reproduces_su2_on_the_spike(spike_dir):
    with open(spike_dir / "surface.csv") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    head = [h.strip('"') for h in rows[0]]
    d = np.array([[float(v) for v in r] for r in rows[1:] if r])
    d = d[np.argsort(d[:, head.index("PointID")])]
    col = {h: d[:, k] for k, h in enumerate(head)}
    *_, p_h = nondimensionalise(col["Density"], col["Momentum_x"], col["Momentum_y"],
                                col["Energy"], 0.8)
    cl, cd, cm = wall_forces(col["x"], col["y"], pressure_coefficient(p_h, 0.8), 1.25)
    assert cl == pytest.approx(0.2819562904, abs=1e-9)
    assert cd == pytest.approx(0.01969418273, abs=1e-9)
    assert cm == pytest.approx(0.01812806891, abs=1e-9)
