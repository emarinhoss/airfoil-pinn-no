import numpy as np

from apno.score.report import aggregate, break_even


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
