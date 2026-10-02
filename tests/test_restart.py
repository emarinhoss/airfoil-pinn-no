import numpy as np
import pytest

from apno.geometry.ogrid import COARSE, Section, grid
from apno.su2.restart import on_grid, read_restart, write_restart

NAMES = ("x", "y", "Density", "Momentum_x", "Momentum_y", "Energy")


def _grid_restart(path, X, Y):
    ni, nj = X.shape
    xs, ys = X.T.reshape(-1), Y.T.reshape(-1)  # node id j*NI + i
    data = np.column_stack([xs, ys, np.ones_like(xs), xs * 2, ys * 3, np.full_like(xs, 7.0)])
    write_restart(path, NAMES, data)
    return data


def test_round_trip(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    data = _grid_restart(tmp_path / "r.dat", X, Y)
    r = read_restart(tmp_path / "r.dat")
    assert r.names == NAMES and r.npoints == X.size
    np.testing.assert_array_equal(r.data, data)
    f = on_grid(r, X, Y)
    np.testing.assert_array_equal(f["Momentum_x"], 2 * X)


def test_shuffled_nodes_fail_the_coordinate_check(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    data = _grid_restart(tmp_path / "r.dat", X, Y)
    write_restart(tmp_path / "s.dat", NAMES, data[np.random.default_rng(0).permutation(len(data))])
    with pytest.raises(ValueError, match="do not match"):
        on_grid(read_restart(tmp_path / "s.dat"), X, Y)


def test_bad_magic_and_truncation(tmp_path):
    X, Y = grid(Section(0.12, 0.0), COARSE)
    _grid_restart(tmp_path / "r.dat", X, Y)
    raw = (tmp_path / "r.dat").read_bytes()
    (tmp_path / "t.dat").write_bytes(raw[:-8])
    with pytest.raises(ValueError, match="size"):
        read_restart(tmp_path / "t.dat")
    (tmp_path / "m.dat").write_bytes(b"\x00" * 4 + raw[4:])
    with pytest.raises(ValueError, match="magic"):
        read_restart(tmp_path / "m.dat")


def test_reads_the_real_spike_restart(spike_dir):
    r = read_restart(spike_dir / "restart.dat")
    assert r.names == NAMES and r.npoints == 24832
    x, y = r.field("x"), r.field("y")
    assert np.hypot(x[0] - 0.5, y[0]) == pytest.approx(0.5, abs=1e-12)  # trailing edge
    assert np.hypot(x[-1] - 0.5, y[-1]) == pytest.approx(50.0, rel=1e-9)  # spike's far field
