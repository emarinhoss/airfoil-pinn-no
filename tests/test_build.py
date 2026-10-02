import json

import h5py
import numpy as np
import pytest

from apno.data.build import build_level, regime, showcase
from apno.geometry.ogrid import COARSE, Section, grid
from apno.physics.flow import freestream
from apno.su2.restart import write_restart
from apno.su2.run import CaseSpec, case_dir


def _fake_case(root, spec, converged=True):
    d = case_dir(root, spec)
    d.mkdir(parents=True)
    X, Y = grid(Section(spec.t, spec.m), spec.shape)
    rho, U, p = freestream(spec.mach)
    a = np.radians(spec.alpha_deg)
    n = X.size
    data = np.column_stack([X.T.ravel(), Y.T.ravel(), np.full(n, rho),
                            np.full(n, rho * U * np.cos(a)), np.full(n, rho * U * np.sin(a)),
                            np.full(n, p / 0.4 + 0.5 * rho * U * U)])
    write_restart(d / "restart.dat", ("x", "y", "Density", "Momentum_x", "Momentum_y", "Energy"),
                  data)
    (d / "result.json").write_text(json.dumps({
        "converged": converged, "reason": "converged" if converged else "CL varied",
        "cl": 0.1, "cd": 0.0, "cm": 0.0, "max_surface_mach": spec.mach}))


def test_build_writes_converged_cases_and_lists_failures(tmp_path):
    specs = {"train": [CaseSpec(f"train-{k:04d}", 0.4 + 0.1 * k, 1.0, 0.12, 0.0, COARSE)
                       for k in range(3)]}
    for k, s in enumerate(specs["train"]):
        _fake_case(tmp_path, s, converged=(k != 1))
    fails = build_level(specs, tmp_path, tmp_path / "out.h5", np.float32)
    assert [f["case_id"] for f in fails["train"]] == ["train-0001"]
    with h5py.File(tmp_path / "out.h5") as f:
        g = f["train"]
        assert g["q"].shape == (2, 128, 49, 4) and g["q"].dtype == np.float32
        assert list(g["case_id"].asstr()[...]) == ["train-0000", "train-0002"]
        np.testing.assert_allclose(g["q"][0, :, :, 0], 1.0, rtol=1e-6)
        np.testing.assert_allclose(g["q"][0, :, :, 1], np.cos(np.radians(1.0)), rtol=1e-6)


def test_regime():
    assert regime(1.0001) == "supercritical" and regime(0.9999) == "subcritical"


def test_showcase_rule_on_64():
    ids = [f"c{k}" for k in range(64)]
    mach = np.linspace(0.5, 1.4, 64)[::-1]  # deliberately unsorted input
    picked = showcase(ids, mach)
    order = [ids[k] for k in np.argsort(mach, kind="stable")]
    assert picked == [order[round(p * 63)] for p in (0.10, 0.30, 0.50, 0.70, 0.85, 0.95)]


def test_showcase_on_survivors():
    ids = [f"c{k}" for k in range(10)]
    picked = showcase(ids, np.arange(10.0))
    assert picked == ["c1", "c3", "c4", "c6", "c8", "c9"]
    with pytest.raises(ValueError, match="at least 6"):
        showcase(ids[:5], np.arange(5.0))
