"""The drag-offset probe: its forces are S4's, the per-node drag shares sum to the drag change, the
line fit finds a known crossing, the regions cover the wall, and the probe runs end to end on the
stand-in dataset."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import probe_fno_drag as dr  # noqa: E402
import probe_fno_libraries as probe  # noqa: E402
from apno.data.sampling import case_specs  # noqa: E402
from apno.geometry.ogrid import TRAIN, Section, grid  # noqa: E402
from apno.models.fno import FNO2d  # noqa: E402
from apno.models.train_fno import FnoConfig  # noqa: E402
from apno.physics.flow import pressure_coefficient  # noqa: E402
from apno.score.forces import wall_forces  # noqa: E402
from apno.score.metrics import score_case  # noqa: E402


def _cases(n=3):
    specs = case_specs("val", TRAIN)[:n]
    inputs = np.array([[s.mach, s.alpha_deg, s.t, s.m] for s in specs])
    true = np.stack([probe.standin_fields(*grid(Section(s.t, s.m), TRAIN), s.mach, s.alpha_deg,
                                          s.t) for s in specs])
    return specs, inputs, true


def test_drag_is_score_cases_at_both_grids():
    specs, inputs, true = _cases()
    pred = true * (1 + 0.01 * np.random.default_rng(0).standard_normal(true.shape))
    for sub in (1, 2):
        cd_p = dr.drag_counts(pred[:, ::sub, ::sub], dr.walls(inputs, sub), inputs)
        cd_t = dr.drag_counts(true[:, ::sub, ::sub], dr.walls(inputs, sub), inputs)
        for k, s in enumerate(specs):
            X, Y = (a[::sub, ::sub] for a in grid(Section(s.t, s.m), TRAIN))
            ref = score_case(pred[k, ::sub, ::sub], true[k, ::sub, ::sub], X, Y, s.mach,
                             s.alpha_deg, "subcritical",
                             tuple(j // sub for j in (8, 24, 48, 72)))["cd_err_counts"]
            assert cd_p[k] - cd_t[k] == pytest.approx(ref, rel=1e-10, abs=1e-9)


def test_node_shares_sum_to_the_drag_change_and_regions_cover_the_wall():
    specs, inputs, true = _cases(1)
    xw, yw = dr.walls(inputs)[0]
    mach, alpha = inputs[0, :2]
    cp = pressure_coefficient(true[0, :, 0, 3], mach)
    dcp = 0.01 * np.sin(np.linspace(0, 7, len(cp)))
    share = dr.drag_by_node(dcp, xw, yw, alpha)
    change = 1e4 * (wall_forces(xw, yw, cp + dcp, alpha)[1] - wall_forces(xw, yw, cp, alpha)[1])
    assert share.sum() == pytest.approx(change, rel=1e-10)
    reg = dr.region_of(xw)
    assert set(reg) == set(dr.REGIONS) and len(reg) == len(xw)
    assert reg[0] == "trailing edge" and reg[len(xw) // 2] == "leading edge"
    assert reg[len(xw) // 4] == "lower" and reg[3 * len(xw) // 4] == "upper"
    # a uniform Cp change on the closed wall has no drag
    assert dr.drag_by_node(np.full(len(xw), 0.3), xw, yw, alpha).sum() == pytest.approx(0, abs=1e-9)


def test_line_fit_finds_slope_crossing_and_residual():
    rows = [14, 15, 16, 17, 18]
    offs = np.array([[-33.0 * (r - 16.5) for r in rows],
                     [-10.0 * (r - 17.2) + (1.0 if r == 16 else 0.0) for r in rows]])
    slope, cross, resid = dr.line_fit(rows, offs)
    assert slope[0] == pytest.approx(-33.0) and cross[0] == pytest.approx(16.5)
    assert resid[0] == pytest.approx(0, abs=1e-9)
    assert cross[1] == pytest.approx(17.2, abs=0.05) and 0.5 < resid[1] < 1.0


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
                "cfg": FnoConfig(width=8, modes_i=4, modes_j=3, layers=2, n_train=4).__dict__},
               "runs/fno/fno-n4-s0/best.pt")
    args = ["runs/fno/*/best.pt", "--h5", str(h5), "--device", "cpu", "--rows", "15-18",
            "--fit", "15-18", "--out", "out"]
    assert dr.main(args) == 0
    out = json.loads(Path("out/drag.json").read_text())
    rec = out["records"][0]
    assert rec["tag"] == "fno-n4-s0" and sorted(rec["rows"], key=int) == ["15", "16", "17", "18"]
    o = rec["_offsets"]
    avg = rec["avg_16_17"]["drag_offset_median"]  # forces are linear in the wall pressure
    assert avg == pytest.approx(float(np.median((np.array(o["16"]) + np.array(o["17"])) / 2)),
                                rel=1e-6, abs=1e-6)
    ch = rec["change_16_to_17"]
    assert sum(ch["by_region_mean"].values()) == pytest.approx(ch["drag_counts_mean"], rel=1e-9)
    assert ch["drag_counts_mean"] == pytest.approx(
        float(np.mean(np.array(o["17"]) - np.array(o["16"]))), rel=1e-6, abs=1e-6)
    md = Path("out/drag.md").read_text()
    assert "## fno-n4-s0" in md and "| mean of 16 and 17 |" in md
    assert Path("out/drag.png").exists()


def test_missing_inputs_exit_2(tmp_path):
    assert dr.main(["--h5", str(tmp_path / "none.h5")]) == 2
    probe.write_standin(tmp_path / "d.h5", {"train": 2, "val": 2})
    assert dr.main([str(tmp_path / "nothing/*.pt"), "--h5", str(tmp_path / "d.h5")]) == 2


def _scores_text(e256, e128):
    """A scores.json with one FNO run on two test_in cases at both grids."""
    rows = []
    for grid_, errs in (("256x97", e256), ("128x49", e128)):
        for k, e in enumerate(errs):
            rows.append({"tag": "fno-n4-s0", "method": "fno", "grid": grid_,
                         "case_id": f"test_in-{k:04d}",
                         "score": {"status": "ok", "cd_err_counts": e}})
    return json.dumps({"rows": rows})


def test_from_scores_finds_the_crossing(tmp_path):
    # drag linear in the row count, crossing the 128x49 value at 16.5 and 16.25
    new = tmp_path / "new.json"
    new.write_text(_scores_text([-16.5 + 2.0, -10.0 + 1.0], [2.0, 1.0]))
    old = _scores_text([16.5 + 2.0, 30.0 + 1.0], [2.0, 1.0])
    (rec,) = dr.from_scores(new, old)
    assert rec["tag"] == "fno-n4-s0" and rec["split"] == "test_in" and rec["n"] == 2
    assert rec["crossing_rows"]["median"] == pytest.approx((16.5 + 16.25) / 2)
    assert rec["avg_minus_128x49"]["median"] == pytest.approx(5.0)  # (+0 + 10) / 2
    assert rec["neg_16"] == 2 and rec["neg_17"] == 0
    assert "| fno-n4-s0 | test_in | 2 |" in dr.report_from_scores([rec])


def test_layer_transfer_is_exact_for_a_wrap_smooth_field_at_17_rows_only():
    torch.manual_seed(0)
    conv = dr.SpectralConv2d(1, 1, modes_i=4, modes_j=8).double()
    assert dr._layer_transfer(conv, None, 17, True) < 1e-12
    assert dr._layer_transfer(conv, None, 16, True) > 1e-2
    flat = dr._layer_transfer(conv, lambda e: torch.ones_like(e), 16, False)
    assert flat < dr._layer_transfer(conv, lambda e: torch.ones_like(e), 17, False)


def test_vertex_and_synthetic_report():
    rows = [14, 15, 16, 17, 18, 19]
    v = np.abs(np.array(rows) - 16.4) * np.where(np.array(rows) < 16.4, 2.0, 3.0)
    assert dr._vertex(rows, v) == pytest.approx(16.4)
    assert dr._vertex(rows, np.arange(6.0)) is None  # minimum at an end
    md = dr.report_synthetic({"rows": rows, "one_layer": {"a": v.tolist()},
                              "untrained_fno2d": {"b": v.tolist()}})
    assert "| one spectral layer: a |" in md and "16.40 |" in md
