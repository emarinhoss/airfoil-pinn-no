import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "s5_score.py"


@pytest.fixture
def sc():
    spec = importlib.util.spec_from_file_location("s5_score", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verdict_rules(sc):
    ok = {"status": "ok", "passed": True, "collapsed": False}
    assert sc.verdict(None) == "not run"
    assert sc.verdict({"diverged": True, "score": {"status": "diverged"}}) == "needed"
    assert sc.verdict({"diverged": False, "score": ok}) == "not needed"
    assert sc.verdict({"diverged": False, "score": {**ok, "collapsed": True}}) == "needed"
    assert sc.verdict({"diverged": False, "score": {**ok, "passed": False}}) == "needed"
    assert sc.verdict({"diverged": False, "score": {"status": "missing truth"}}) == "needed"


def test_ablation_lists_every_arm(sc):
    runs = {"A1": {"diverged": False, "score": {"status": "ok", "passed": True,
                                               "collapsed": False}}}
    got = sc.ablation(runs)
    assert list(got) == ["A1", "A2", "A3", "A4", "A5", "A6a", "A6b"]
    assert got["A1"]["verdict"] == "not needed" and got["A2"]["verdict"] == "not run"


def _table(sc):
    def row(nf):
        return {"status": "ok", "near_field_rel_l2": nf}

    table = {}
    for c in sc.STEP3_IN:
        table[c] = {"floor": row(0.01), "fno-256": row(0.02), "fno-128": {"status": "missing"},
                    "pod": row(0.03), "pinn-pure": row(0.5), "euler_av": row(0.05)}
    for c in sc.STEP3_EXT:
        table[c] = {"floor": row(0.05), "fno-256": row(0.2), "fno-128": {"status": "missing"},
                    "pod": row(0.5), "pinn-pure": {"status": "missing"}, "euler_av": row(0.1)}
    return table


def test_group_ratios_and_degradation(sc):
    g = sc.group_ratios(_table(sc))
    assert g["fno-256"]["in"] == pytest.approx(2.0) and g["fno-256"]["ext"] == pytest.approx(4.0)
    assert g["euler_av"]["in"] == pytest.approx(5.0) and g["euler_av"]["ext"] == pytest.approx(2.0)
    assert g["fno-128"]["in"] is None and g["fno-128"]["n_in"] == 0
    assert g["pinn-pure"]["n_ext"] == 0
    d = sc.degradation(g)
    assert d["fno-256"] == pytest.approx(2.0) and d["euler_av"] == pytest.approx(0.4)
    assert d["fno-128"] is None and "pinn-pure" not in d


def test_missing_truth_or_file_is_a_status(sc, monkeypatch):
    specs = sc.all_specs()
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (None, None, None))
    assert sc.score_column("euler_av", "test_ext-0005", specs) == {"status": "missing truth"}
    q = np.ones((256, 97, 4))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (q, q, 1.2))
    monkeypatch.setattr(sc, "method_q", lambda m, cid: None)
    assert sc.score_column("euler_av", "test_ext-0005", specs) == {"status": "missing"}


def test_costs_against_su2(sc):
    runs = {c: {"wall_s": 3600.0, "cpu_s": 1800.0, "device": "mps"}
            for c in sc.STEP3_IN + sc.STEP3_EXT}
    s4 = {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}
    c = sc.costs(runs, s4)
    assert c["euler_av"]["per_wall"] == 3600.0 and c["euler_av"]["n_runs"] == 6
    assert c["ratio_to_su2_cpu"] == {"wall": pytest.approx(18.0), "cpu": pytest.approx(9.0)}
    assert sc.costs({}, s4)["euler_av"] is None


def test_main_writes_its_outputs_when_nothing_has_run(sc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("results/s4").mkdir(parents=True)
    Path("results/s4/costs.json").write_text(json.dumps(
        {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (None, None, None))
    assert sc.main([]) == 0
    scores = json.loads(Path("results/s5/scores.json").read_text())
    assert scores["table"]["test_ext-0005"]["euler_av"] == {"status": "missing truth"}
    assert Path("results/s5/scores.md").exists() and Path("results/s5/ablation.json").exists()


S4_FULL = {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0},
                     "fno": {"fixed": 1.0e5, "per": 0.05}}}


def test_costs_record_each_run_and_cost_it_by_the_s4_rule(sc):
    gpu = {c: {"wall_s": 3600.0, "cpu_s": 1800.0, "device": "mps", "eval_device": "cpu"}
           for c in sc.STEP3_IN + sc.STEP3_EXT}
    c = sc.costs(gpu, S4_FULL)["euler_av"]
    assert c["runs"]["test_in-0004"] == {"wall_s": 3600.0, "cpu_s": 1800.0, "device": "mps",
                                         "eval_device": "cpu", "diverged": False,
                                         "cost_s": 3600.0}
    assert c["per"] == 3600.0 and "wall" in c["basis"]
    cpu = {k: {**r, "device": "cpu"} for k, r in gpu.items()}
    c = sc.costs(cpu, S4_FULL)["euler_av"]
    assert c["per"] == 1800.0 and c["basis"] == "cpu"
    assert sc.costs(gpu, S4_FULL)["fno"] == {"fixed": 1.0e5, "per": 0.05}
    minimal = {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}
    assert sc.costs(gpu, minimal)["fno"] == {"fixed": None, "per": 0.05}


def test_break_even_beside_su2_and_the_fno(sc):
    runs = {c: {"wall_s": 3600.0, "cpu_s": 1800.0, "device": "mps"}
            for c in sc.STEP3_IN + sc.STEP3_EXT}
    curves = sc.break_even_curves(sc.costs(runs, S4_FULL), 10)
    assert list(curves) == ["su2", "fno", "euler_av"]
    assert curves["su2"][10] == pytest.approx(2000.0)
    assert curves["fno"][0] == pytest.approx(1.0e5) and curves["euler_av"][10] == 36000.0
    minimal = {"costs": {"su2": {"per_cpu": 200.0, "per_wall": 220.0}, "fno": {"per": 0.05}}}
    assert list(sc.break_even_curves(sc.costs(runs, minimal), 10)) == ["su2", "euler_av"]
    assert sc.break_even_curves(sc.costs({}, S4_FULL), 10) is None


def test_group_ratios_report_cl_and_cd_errors_beside(sc):
    table = _table(sc)
    for i, c in enumerate(sc.STEP3_IN):
        table[c]["euler_av"].update(cl_err=(-1) ** i * 0.02, cd_err_counts=(-1) ** i * 30.0)
    g = sc.group_ratios(table)
    assert g["euler_av"]["cl_err_in"] == pytest.approx(0.02)
    assert g["euler_av"]["cd_err_counts_in"] == pytest.approx(30.0)
    assert g["euler_av"]["cl_err_ext"] is None and g["fno-128"]["cd_err_counts_in"] is None
    assert g["euler_av"]["in"] == pytest.approx(5.0) and g["euler_av"]["n_in"] == 3
    md = sc.markdown(None, sc.ablation({}), table, g, sc.degradation(g), {})
    assert "CL err" in md.split("## Costs")[0].split("| method |")[1]


def test_ablation_record_carries_scores_losses_and_costs(sc):
    s4 = {"near_field_rel_l2": 0.0123, "cl_err": -0.0456}
    runs = {"A1": {"diverged": False, "final": {"total": 1.5e-3}, "stalled": 2, "func_evals": 900,
                   "wall_s": 7200.0, "cpu_s": 3600.0, "device": "mps",
                   "score": {"status": "ok", "passed": True, "collapsed": False,
                             "rmae": {"cp": 0.0071, "mach": 0.0052}, "s4": s4}},
            "A6a": {"diverged": False, "final": {"total": 2e-4}, "stalled": 0, "func_evals": 10,
                    "wall_s": 360.0, "cpu_s": 300.0, "device": "cpu",
                    "score": {"status": "ok", "passed": False, "collapsed": True,
                              "max_cp_error": 0.876, "cl": 0.0321, "cd": 0.0}}}
    abl = sc.ablation(runs)
    a1 = abl["A1"]
    assert (a1["final"], a1["stalled"], a1["func_evals"]) == ({"total": 1.5e-3}, 2, 900)
    assert (a1["wall_s"], a1["cpu_s"], a1["device"], a1["diverged"]) == (7200.0, 3600.0, "mps",
                                                                         False)
    assert abl["A2"]["wall_s"] is None and abl["A2"]["verdict"] == "not run"
    md = sc.markdown(None, abl, {}, {}, {}, {})
    row = {ln.split("|")[1].strip(): ln for ln in md.splitlines() if ln.startswith("| A")}
    header = next(ln for ln in md.splitlines() if ln.startswith("| arm |"))
    assert all(ln.count("|") == header.count("|") == 12 for ln in row.values())
    for v in ("0.0071", "0.0052", "0.0123", "-0.0456", "2.00", "ok", "False"):
        assert v in row["A1"]
    for v in ("0.876", "0.0321", "True"):
        assert v in row["A6a"]
    assert "not run" in row["A2"] and "—" in row["A2"]


def test_diverged_step3_run_is_a_status_and_left_out_of_the_cost(sc, monkeypatch):
    specs = sc.all_specs()
    q = np.ones((256, 97, 4))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (q, q, 1.2))
    monkeypatch.setattr(sc, "method_q", lambda m, cid: None)
    runs = {"test_ext-0005": {"diverged": True, "wall_s": 300.0, "cpu_s": 200.0,
                              "device": "mps"},
            "test_in-0004": {"diverged": False, "wall_s": 50000.0, "cpu_s": 40000.0,
                             "device": "mps"},
            "test_in-0031": {"diverged": False, "wall_s": 60000.0, "cpu_s": 58000.0,
                             "device": "mps"}}
    assert sc.score_column("euler_av", "test_ext-0005", specs, runs) == {"status": "diverged"}
    assert sc.score_column("euler_av", "test_in-0004", specs, runs) == {"status": "missing"}
    c = sc.costs(runs, S4_FULL)["euler_av"]
    assert c["n_runs"] == 2 and c["n_diverged"] == 1
    assert c["per_wall"] == pytest.approx(55000.0) and c["per"] == pytest.approx(55000.0)
    assert c["runs"]["test_ext-0005"]["diverged"] is True
    only_diverged = sc.costs({"test_ext-0005": runs["test_ext-0005"]}, S4_FULL)
    assert only_diverged["euler_av"]["n_runs"] == 0 and only_diverged["euler_av"]["per"] is None
    assert sc.break_even_curves(only_diverged, 10) is None


def test_main_marks_a_diverged_run_and_draws_the_break_even(sc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("results/s4").mkdir(parents=True)
    Path("results/s4/costs.json").write_text(json.dumps(S4_FULL))
    Path("results/s5").mkdir(parents=True)
    Path("results/s5/runs.json").write_text(json.dumps({
        "test_ext-0005": {"diverged": True, "wall_s": 300.0, "cpu_s": 200.0, "device": "mps",
                          "score": {"status": "diverged", "passed": False, "collapsed": None}},
        "test_in-0004": {"diverged": False, "wall_s": 50000.0, "cpu_s": 40000.0,
                         "device": "mps", "score": {"status": "missing truth"}}}))
    monkeypatch.setattr(sc, "truth_and_floor", lambda cid: (None, None, None))
    assert sc.main([]) == 0
    scores = json.loads(Path("results/s5/scores.json").read_text())
    assert scores["table"]["test_ext-0005"]["euler_av"] == {"status": "diverged"}
    assert scores["table"]["test_in-0004"]["euler_av"] == {"status": "missing truth"}
    cost = json.loads(Path("results/s5/costs.json").read_text())
    assert cost["euler_av"]["n_runs"] == 1 and cost["euler_av"]["per"] == 50000.0
    assert Path("results/s5/break_even.png").exists()


def _method_row(md, method):
    return next(ln for ln in md.split("| method |")[1].splitlines()
                if ln.startswith(f"| {method} |"))


def test_degradation_needs_every_case_and_marks_a_diverged_one(sc):
    """A method short of a case (here euler_av diverged on two extrapolation cases and has no
    result on one in-range case) gets no degradation: the figure is defined on the six cases
    (design §5.3), and one over the cases left would not be like-for-like with the FNO's."""
    table = _table(sc)
    for c in sc.STEP3_EXT[:2]:
        table[c]["euler_av"] = {"status": "diverged"}
    g = sc.group_ratios(table)
    assert (g["euler_av"]["n_ext"], g["euler_av"]["n_diverged_ext"]) == (1, 2)
    assert (g["euler_av"]["n_in"], g["euler_av"]["n_diverged_in"]) == (3, 0)
    d = sc.degradation(g)
    assert d["euler_av"] is None and d["fno-256"] == pytest.approx(2.0)
    md = sc.markdown(None, sc.ablation({}), table, g, d, {})
    cell = _method_row(md, "euler_av").split("|")[4].strip()
    assert cell == "— (ext 1/3, 2 diverged)"
    assert _method_row(md, "fno-256").split("|")[4].strip() == "2"
    table[sc.STEP3_IN[0]]["euler_av"] = {"status": "missing"}
    g = sc.group_ratios(table)
    assert sc.degradation(g)["euler_av"] is None
    md = sc.markdown(None, sc.ablation({}), table, g, sc.degradation(g), {})
    assert _method_row(md, "euler_av").split("|")[4].strip() == "— (in 2/3; ext 1/3, 2 diverged)"


def _synthetic(case_id, sc):
    from apno.geometry.ogrid import TRAIN, Section, grid
    s = sc.all_specs()[case_id]
    X, Y = grid(Section(s.t, s.m), TRAIN)
    a = np.radians(s.alpha_deg)
    q = np.zeros(X.shape + (4,))
    q[..., 0] = 1 + 0.05 * np.sin(X)
    q[..., 1] = np.cos(a) + 0.1 * np.cos(Y)
    q[..., 2] = np.sin(a)
    q[..., 3] = 1 + 0.1 * np.cos(X) * np.exp(-Y ** 2)
    return q


def test_identity_checks_are_labelled_by_construction_not_printed_as_values(
        sc, tmp_path, monkeypatch):
    from apno.data.predictions import write_predictions
    monkeypatch.chdir(tmp_path)
    cid = sc.STEP3_IN[0]
    q = _synthetic(cid, sc)
    monkeypatch.setattr(sc, "truth_and_floor", lambda c: (q, q, 1.2))
    write_predictions(Path("runs/predictions/pinn-pure-test_in-0004.h5"), [cid], q[None],
                      {"method": "pinn", "formulation": "potential"})
    write_predictions(Path("runs/predictions_s5/pinn-euler_av-test_in-0004.h5"), [cid], q[None],
                      {"method": "pinn", "formulation": "euler_av"})
    specs = sc.all_specs()
    pot = sc.score_column("pinn-pure", cid, specs)["physics"]
    for k in ("enthalpy_max_dev", "entropy_max", "entropy_mean"):
        assert pot[k] == "by construction"
    assert isinstance(pot["wall_normal_rms"], float)  # soft for the potential PINN: a result
    av = sc.score_column("euler_av", cid, specs)["physics"]
    assert all(isinstance(av[k], float) for k in ("enthalpy_max_dev", "entropy_max",
                                                  "wall_normal_rms"))
    assert isinstance(sc.score_column("floor", cid, specs)["physics"]["entropy_max"], float)
    md = sc.markdown(None, sc.ablation({}), {cid: {m: sc.score_column(m, cid, specs)
                                                   for m in sc.METHODS}},
                     {}, {}, {"euler_av": None})
    assert "## Physics checks" in md and "by construction" in md
