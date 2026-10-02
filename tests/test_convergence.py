import numpy as np
import pytest

from apno.su2.convergence import judge, read_history


def _hist(n=2000, drop=6.0, cl=lambda it: 0.3 + 0 * it, cd=lambda it: 0.01 + 0 * it):
    it = np.arange(n, dtype=float)
    return {"Inner_Iter": it, "rms[Rho]": -1.0 - drop * it / (n - 1), "CL": cl(it), "CD": cd(it)}


def test_steady_forces_with_stalled_residual_converges():
    h = _hist(drop=6.0)
    h["rms[Rho]"][1000:] = h["rms[Rho]"][1000]  # stalls at a 3-order drop ...
    assert not judge(h).converged
    h = _hist(drop=6.0)
    h["rms[Rho]"][1800:] = h["rms[Rho]"][1800]  # ... stalls after 5.40 orders: fine
    v = judge(h)
    assert v.converged and v.reason == "converged"


def test_oscillating_forces_fail():
    v = judge(_hist(cl=lambda it: 0.3 + 1e-4 * np.sin(it / 10)))
    assert not v.converged and "CL varied" in v.reason


def test_residual_drops_but_forces_drift():
    v = judge(_hist(cd=lambda it: 0.01 + 1e-8 * it))
    assert not v.converged and "CD varied" in v.reason


def test_too_few_iterations():
    v = judge(_hist(n=300))
    assert not v.converged and "window" in v.reason


def test_nan_fails():
    h = _hist()
    h["CL"][-3] = np.nan
    v = judge(h)
    assert not v.converged and "non-finite" in v.reason


def test_missing_and_empty_history(tmp_path):
    assert judge(read_history(tmp_path / "absent.csv")).reason == "no history"
    (tmp_path / "empty.csv").write_text("")
    assert judge(read_history(tmp_path / "empty.csv")).reason == "no history"
    (tmp_path / "nocol.csv").write_text('"Inner_Iter","CL"\n0,0.1\n')
    v = judge(read_history(tmp_path / "nocol.csv"))
    assert not v.converged and "missing column" in v.reason


def test_spike_history_converges_narrowly(spike_dir):
    v = judge(read_history(spike_dir / "history.csv"))
    assert v.converged
    assert v.residual_drop == pytest.approx(6.126, abs=1e-3)
    assert v.cl_range == pytest.approx(8.896e-7, rel=1e-3)
    assert v.iterations == 3999
