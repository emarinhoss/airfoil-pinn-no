"""The leading/trailing-edge locator: R^2 is right on known lines, the candidates and columns are
where they should be, and the probe runs end to end on the stand-in dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import probe_fno_drag_locate as lo  # noqa: E402
import probe_fno_libraries as probe  # noqa: E402
from apno.data.sampling import case_specs  # noqa: E402
from apno.geometry.ogrid import TRAIN, Section, grid  # noqa: E402
from apno.models.fno import FNO2d  # noqa: E402
from apno.models.targets import features  # noqa: E402
from apno.models.train_fno import FnoConfig  # noqa: E402
from apno.physics.flow import pressure_coefficient  # noqa: E402


def test_r2():
    x = np.linspace(0, 1, 50)
    assert lo.r2(3 * x + 1, [x]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert lo.r2(rng.standard_normal(500), [rng.standard_normal(500)]) < 0.05
    assert np.isnan(lo.r2(np.ones(5), [x[:5]]))


def test_candidates_and_columns_on_the_real_grid():
    s = case_specs("val", TRAIN)[0]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    q = probe.standin_fields(X, Y, s.mach, s.alpha_deg, s.t)
    xy = np.stack([X, Y], -1)[None]
    feat = features(xy, np.array([[s.mach, s.alpha_deg, s.t, s.m]]))[0]
    c = lo.candidates(q, feat, s.mach)
    cp = pressure_coefficient(q[:, :, 3], s.mach)
    assert set(c) == set(lo.CANDIDATES)
    np.testing.assert_allclose(c["cp_wall"], cp[:, 0])
    np.testing.assert_allclose(c["cp_wall_minus_far"], cp[:, 0] - cp[:, -1])
    col = lo.columns(X[:, 0])
    assert col["trailing edge"] == 0 and abs(col["leading edge"] - 128) <= 2
    assert c["dx_k"][col["leading edge"]] == pytest.approx(-np.log1p(0.5) * 2 * 0.5, abs=0.01)
    assert c["dx_k"][0] == pytest.approx(np.log1p(0.5), abs=0.01)
    assert X[col["mid-chord lower"], 0] == pytest.approx(0.5, abs=0.02)
    assert col["mid-chord upper"] > 128 > col["mid-chord lower"]


def test_the_probe_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h5 = probe.write_standin(tmp_path / "d.h5", {"train": 2, "val": 4})
    torch.manual_seed(0)
    model = FNO2d(width=8, modes_i=4, modes_j=3, layers=2)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_complex():
                p.mul_(50.0)
    Path("runs/fno/fno-n4-s0").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "std": {"mean": [0.0] * 4, "std": [0.1] * 4},
                "cfg": FnoConfig(width=8, modes_i=4, modes_j=3, layers=2).__dict__},
               "runs/fno/fno-n4-s0/best.pt")
    assert lo.main(["runs/fno/*/best.pt", "--h5", str(h5), "--device", "cpu", "--out", "o"]) == 0
    rec = json.loads(Path("o/locate.json").read_text())["records"][0]
    assert set(rec["r2_median"]) == {*lo.CANDIDATES, "all five"}
    best_single = max(v["all"] for k, v in rec["r2_median"].items() if k != "all five")
    assert rec["r2_median"]["all five"]["all"] >= best_single - 1e-9
    assert set(rec["profile_mean_abs_change"]) == {"leading edge", "mid-chord lower",
                                                   "mid-chord upper", "trailing edge"}
    assert len(rec["profile_mean_abs_change"]["leading edge"]["p"]) == len(lo.PROFILE_ROWS)
    assert 0 < rec["ends_share_of_abs_change"]["share_of_wall_nodes"] < 0.5
    assert "| all five |" in Path("o/locate.md").read_text() and Path("o/locate.png").exists()
    assert lo.main(["--h5", str(tmp_path / "none.h5")]) == 2


def test_untrained_control_has_the_checkpoints_sizes(tmp_path, monkeypatch):
    m = FNO2d(width=8, modes_i=4, modes_j=3, layers=2)
    u = lo.untrained_like(m)
    assert [p.shape for p in u.parameters()] == [p.shape for p in m.parameters()]
    test_the_probe_runs_end_to_end(tmp_path, monkeypatch)
    h5 = tmp_path / "d.h5"
    assert lo.main(["runs/fno/*/best.pt", "--h5", str(h5), "--device", "cpu", "--out", "o2",
                    "--untrained"]) == 0
    tags = [r["tag"] for r in json.loads(Path("o2/locate.json").read_text())["records"]]
    assert tags == ["fno-n4-s0", "untrained"]


def test_shift_fit_recovers_a_radial_shift():
    """A field moved outward by eps rows changes by eps * dq/dj to first order, in every channel."""
    ni, nj, eps = 64, 40, 0.3
    i = np.arange(ni)[:, None]
    for f in (lambda j: np.exp(-j / 6.0) * (1 + 0.5 * np.cos(2 * np.pi * i / ni)),
              lambda j: np.sin(j / 9.0) * np.sin(2 * np.pi * i / ni) + 0.1 * j):
        q = f(np.arange(nj)[None, :] + 0.0)
        q_moved = f(np.arange(nj)[None, :] + eps)
        for j in (0, 2, 8):
            b, rsq = lo.shift_fit(q_moved[:, j] - q[:, j], lo.radial_difference(q, j))
            assert b == pytest.approx(eps, rel=0.1) and rsq > 0.99


def test_the_report_carries_the_shift_table(tmp_path, monkeypatch):
    test_the_probe_runs_end_to_end(tmp_path, monkeypatch)
    rec = json.loads(Path("o/locate.json").read_text())["records"][0]
    assert set(rec["shift"]) == set(lo.CHANNELS)
    assert len(rec["shift"]["p"]["b_median"]) == len(lo.SHIFT_ROWS)
    assert "| p | " in Path("o/locate.md").read_text()
