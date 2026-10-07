import numpy as np

from apno.score.report import (
    PHYSICS_METRICS, aggregate, aggregate_physics, aggregate_shocks, break_even, markdown_table,
    plot_break_even,
)


def _row(method, val, status="ok"):
    score = {"status": status}
    if status == "ok":
        score.update({"cl_err": val, "near_field_rel_l2": abs(val)})
    return {"method": method, "variant": "", "n_train": 896, "seed": 0, "split": "test_in",
            "regime": "subcritical", "case_id": "x", "score": score}


def test_aggregate_counts_failures_and_keeps_signs():
    out = aggregate([_row("fno", -0.2), _row("fno", 0.1), _row("fno", 0, "non-finite")])
    (g,) = out
    assert g["n_ok"] == 2 and g["n_non-finite"] == 1
    assert np.isclose(g["cl_err_mean"], -0.05) and np.isclose(g["cl_err_absmax"], 0.2)


def test_break_even():
    costs = {"su2": {"fixed": 0.0, "per": 10.0}, "fno": {"fixed": 1000.0, "per": 0.01}}
    curves = break_even(costs, 200)
    assert curves["su2"][100] == 1000.0
    assert np.argmax(curves["fno"] < curves["su2"]) == 101


def test_break_even_plot_keeps_coincident_curves_visible(tmp_path):
    # FNO and POD share the campaign, so their curves coincide on the log axis; the one drawn first
    # must stay visible under the other (review 2, M10): distinct line styles, the first wider
    costs = {"su2": {"fixed": 0.0, "per": 10.0}, "fno": {"fixed": 1000.0, "per": 0.01},
             "pod": {"fixed": 1000.0, "per": 0.01}, "pinn": {"fixed": 0.0, "per": 1e3}}
    lines = plot_break_even(break_even(costs, 50), tmp_path / "b.png")
    assert (tmp_path / "b.png").exists() and set(lines) == set(costs)
    styles = [(v["ls"], v["lw"]) for v in lines.values()]
    assert len(set(styles)) == len(styles)
    assert lines["fno"]["lw"] > lines["pod"]["lw"] and lines["fno"]["ls"] != lines["pod"]["ls"]


def _phys(h=1e-3, un=1e-4):
    return {"entropy_max": 1e-3, "entropy_mean": 1e-4, "enthalpy_max_dev": h,
            "wall_normal_rms": un, "mass_flux": {8: 1e-5, 24: -2e-5, 48: 3e-5, 72: 4e-5}}


def _sup(method, shock, phys=None):
    score = {"status": "ok", "physics": phys or _phys(), "shock": shock}
    return {"method": method, "variant": "", "grid": "256x97", "n_train": 896, "seed": 0,
            "split": "test_in", "regime": "supercritical", "case_id": "x", "score": score}


def test_aggregate_shocks_counts_misses_and_pools_surfaces():
    hit = {"x_true": 0.6, "x_err": -0.02, "missed": False, "mach_ahead_true": 1.2,
           "mach_ahead_err": 0.05, "overshoot": 0.3}
    miss = {"x_true": 0.35, "x_err": None, "missed": True, "mach_ahead_true": 1.1,
            "mach_ahead_err": None, "overshoot": 0.5}
    (g,) = aggregate_shocks([_sup("fno", {"upper": hit, "lower": miss}),
                             _sup("fno", {"upper": dict(hit, x_err=0.04)}),
                             _row("fno", 0.1)])  # a subcritical row is not in the shock table
    assert g["n_supercritical"] == 2 and g["n_true_shocks"] == 3 and g["n_missed"] == 1
    assert np.isclose(g["abs_x_err_median"], 0.03) and np.isclose(g["abs_x_err_max"], 0.04)
    assert np.isclose(g["overshoot_max"], 0.5) and np.isclose(g["mach_ahead_err_absmax"], 0.05)


def test_physics_by_construction_is_labelled_not_scored():
    groups = aggregate_physics([_sup("pinn", {}, _phys(h=1e-12, un=1e-13)),
                                _sup("fno", {}, _phys())])
    pinn, fno = sorted(groups, key=lambda g: g["method"], reverse=True)
    assert pinn["method"] == "pinn" and pinn["enthalpy_max_dev_max"] == "by construction"
    assert pinn["wall_normal_rms_median"] == "by construction"
    assert np.isclose(pinn["entropy_max_max"], 1e-3)
    assert np.isclose(fno["enthalpy_max_dev_max"], 1e-3)
    assert fno["mass_flux_line1_j"] == 24 and np.isclose(fno["mass_flux_line1_max"], 2e-5)
    md = markdown_table(groups, PHYSICS_METRICS)
    assert "by construction" in md and md.count("\n") == 3


def test_physics_labels_follow_the_pinn_formulation():
    """Spec 7.8: potential wall tangency is a result; enthalpy and entropy are not."""
    euler = dict(_sup("pinn", {}, _phys(h=1e-12, un=1e-13)), formulation="euler")
    pot = dict(_sup("pinn", {}, _phys(h=1e-12, un=2e-3)), formulation="potential")
    groups = aggregate_physics([euler, pot])
    e = next(g for g in groups if g["wall_normal_rms_median"] == "by construction")
    p = next(g for g in groups if g is not e)
    assert e["wall_normal_rms_median"] == "by construction"
    assert e["entropy_max_max"] == 1e-3
    assert p["wall_normal_rms_median"] == 2e-3 and p["wall_normal_rms_max"] == 2e-3
    for m in ("enthalpy_max_dev", "entropy_max", "entropy_mean"):
        assert p[f"{m}_median"] == p[f"{m}_max"] == "by construction"


def test_markdown_table_formats_and_leaves_missing_cells_empty():
    md = markdown_table(aggregate([_row("fno", -0.2), _row("pod", 0.1, "missing")]))
    lines = md.splitlines()
    assert lines[0].startswith("| method | variant | grid |") and len(lines) == 4
    assert "| fno |" in lines[2] and "0.2 |" in lines[2]
    assert lines[3].endswith("|  |  |  |  |")  # the pod group has no ok case


def test_physics_entropy_skips_undefined_cases_and_counts_them():
    bad = dict(_phys(), entropy_max=None, entropy_mean=None, n_nonpositive=12)
    good = dict(_phys(), n_nonpositive=0)
    (g,) = aggregate_physics([_sup("pod", {}, bad), _sup("pod", {}, good)])
    assert g["n_ok"] == 2 and g["n_nonpositive_state"] == 1
    assert g["entropy_max_median"] == g["entropy_max_max"] == 1e-3
    assert np.isclose(g["enthalpy_max_dev_max"], 1e-3)
    assert "n_nonpositive_state" in PHYSICS_METRICS
    md = markdown_table([g], PHYSICS_METRICS)
    assert "nan" not in md
