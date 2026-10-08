import math

import numpy as np
import pytest
import torch

import apno.models.train_pinn_av as tr
from apno.geometry.ogrid import Section, grid
from apno.models.pinn import PinnCase
from apno.models.pinn_euler_av import AvConfig
from apno.models.train_pinn_av import (
    boundary_points, evaluate_on_grid, projected_upper_s, residual_nodes, time_closure, train,
)

CASE = PinnCase(0.72, 3.0, Section(0.12, 0.0))
TINY = AvConfig(hidden=8, depth=2, n_freq=4, grid_shape=(16, 9), batch=40, epochs=1,
                lbfgs_max_iter=2, lbfgs_max_eval=3, lbfgs_history=5, device="cpu")


def test_residual_nodes_drop_the_trailing_edge_node_and_normalise_the_weights():
    nodes = residual_nodes(CASE, TINY)
    assert len(nodes["xi"]) == 16 * 9 - 1
    assert not ((nodes["xi"] == 0) & (nodes["eta"] == 0)).any()
    assert nodes["w"].mean().item() == pytest.approx(1.0)
    assert nodes["xi"].dtype == torch.float64


def test_uniform_weighting_and_the_trailing_edge_exclusion():
    from dataclasses import replace
    assert torch.all(residual_nodes(CASE, replace(TINY, weighting="uniform"))["w"] == 1)
    cfg = replace(TINY, grid_shape=(64, 25), te_exclusion=0.02)
    nodes = residual_nodes(CASE, cfg)
    X, Y = grid(CASE.geometry, (64, 25))
    i = np.rint(nodes["xi"].numpy() * 64).astype(int)
    j = np.rint(nodes["eta"].numpy() * 24).astype(int)
    assert np.all(np.hypot(X[i, j] - 1.0, Y[i, j]) >= 0.02)
    assert len(nodes["xi"]) < 64 * 25 - 1


def test_boundary_points_cover_the_ring_and_the_periodic_pairs():
    b = boundary_points(TINY)
    assert torch.equal(b["xi_ring"], torch.arange(16, dtype=torch.float64) / 16)
    assert torch.equal(b["eta_lr"], torch.arange(9, dtype=torch.float64) / 8)


def test_train_runs_and_records_every_outer_iteration():
    pinn, rec = train(CASE, TINY, log=lambda *a: None)
    assert not rec["diverged"] and rec["device"] == "cpu"
    assert len(rec["history"]) == math.ceil((16 * 9 - 1) / 40)
    assert rec["func_evals"] > 0 and rec["wall_s"] > 0 and "total" in rec["final"]
    q = evaluate_on_grid(pinn, (8, 5))
    assert q.shape == (8, 5, 4) and np.isfinite(q).all()


def test_the_adam_then_lbfgs_variant():
    from dataclasses import replace
    cfg = replace(TINY, optimizer="adam_lbfgs", adam_steps=3, final_lbfgs_iters=2,
                  final_lbfgs_history=3)
    _, rec = train(CASE, cfg, log=lambda *a: None)
    assert not rec["diverged"] and rec["history"][-1]["epoch"] == "final"


def test_the_random_collocation_variant():
    from dataclasses import replace
    _, rec = train(CASE, replace(TINY, collocation="random"), log=lambda *a: None)
    assert not rec["diverged"] and len(rec["history"]) == math.ceil((16 * 9 - 1) / 40)


def test_a_non_finite_loss_stops_the_run(monkeypatch):
    def nan_losses(*a, **k):
        t = torch.tensor(float("nan"), requires_grad=True)
        return {"residual": t, "far": t, "wall": t, "periodic": t, "total": t}

    monkeypatch.setattr(tr, "losses", nan_losses)
    _, rec = train(CASE, TINY, log=lambda *a: None)
    assert rec["diverged"] and "non-finite" in rec["reason"]


def test_time_closure_and_the_upper_bound():
    t = time_closure(CASE, TINY, n=1)
    assert t["device"] == "cpu" and t["closure_s"] > 0
    full = AvConfig()
    assert projected_upper_s(full, 98_815, 0.01) == pytest.approx(40 * 20 * 1250 * 0.01)
    from dataclasses import replace
    adam = replace(full, optimizer="adam_lbfgs")
    assert projected_upper_s(adam, 98_815, 0.01) == pytest.approx((30000 + 6250) * 0.01)


def test_evaluation_lands_exactly_on_the_collocation_nodes():
    xi = torch.arange(8, dtype=torch.float64) / 8  # the 8x5 scoring nodes of a 16x9 grid
    assert torch.equal(16 * xi, 2 * torch.arange(8, dtype=torch.float64))
    eta = torch.arange(5, dtype=torch.float64) / 4
    assert torch.equal(8 * eta, 2 * torch.arange(5, dtype=torch.float64))


def test_a_divergence_records_where_it_happened(monkeypatch):
    from dataclasses import replace

    def nan_losses(*a, **k):
        t = torch.tensor(float("nan"), requires_grad=True)
        return {"residual": t, "far": t, "wall": t, "periodic": t, "total": t}

    monkeypatch.setattr(tr, "losses", nan_losses)
    _, rec = train(CASE, TINY, log=lambda *a: None)
    assert rec["reason"].startswith("epoch 0, batch 0: non-finite loss")
    cfg = replace(TINY, optimizer="adam_lbfgs", adam_steps=3, final_lbfgs_iters=2,
                  final_lbfgs_history=3)
    _, rec = train(CASE, cfg, log=lambda *a: None)
    assert rec["reason"].startswith("adam step 1: non-finite loss")


def test_time_closure_also_times_one_lbfgs_iteration():
    t = time_closure(CASE, TINY, n=1)
    assert set(t) == {"device", "closure_s", "lbfgs_iter_s"}
    assert t["device"] == "cpu" and t["lbfgs_iter_s"] > 0


def test_the_lbfgs_iteration_is_timed_at_the_full_history_on_the_training_device(monkeypatch):
    from dataclasses import replace
    calls = []
    monkeypatch.setattr(tr, "time_two_loop",
                        lambda n_params, history, device, n=3: calls.append(
                            (n_params, history, device, n)) or 0.5)
    n_params = sum(p.numel() for p in tr.make_pinn(CASE, TINY).parameters())
    assert time_closure(CASE, TINY, n=1)["lbfgs_iter_s"] == 0.5
    adam = replace(TINY, optimizer="adam_lbfgs", final_lbfgs_history=3)
    time_closure(CASE, adam, n=2)
    assert calls == [(n_params, 5, "cpu", 1), (n_params, 3, "cpu", 2)]


def test_the_timed_recursion_is_torch_lbfgs_own():
    gen = torch.Generator().manual_seed(0)
    m = torch.randn(20, 20, generator=gen, dtype=torch.float64)
    A, b = m @ m.T + torch.eye(20, dtype=torch.float64), torch.randn(20, generator=gen,
                                                                     dtype=torch.float64)
    x = torch.zeros(20, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=8, history_size=3, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = 0.5 * x @ A @ x - b @ x
        f.backward()
        return f

    opt.step(closure)
    s = opt.state[x]
    assert len(s["old_dirs"]) == 3  # the history is full and has shifted
    d = tr._two_loop(s["prev_flat_grad"], s["old_dirs"], s["old_stps"], s["ro"], s["H_diag"])
    assert torch.allclose(d, s["d"], rtol=1e-12, atol=1e-14)


def test_time_two_loop_at_a_full_history():
    assert tr.time_two_loop(1000, 4, "cpu", n=1) > 0


def test_the_upper_bound_counts_the_lbfgs_iterations_own_cost():
    from dataclasses import replace
    full = AvConfig()
    assert projected_upper_s(full, 98_815, 0.01, 0.02) == pytest.approx(
        40 * 20 * (1250 * 0.01 + 1000 * 0.02))
    adam = replace(full, optimizer="adam_lbfgs")
    assert projected_upper_s(adam, 98_815, 0.01, lbfgs_iter_s=0.02) == pytest.approx(
        30000 * 0.01 + 5000 * (1.25 * 0.01 + 0.02))


def test_the_default_log_flushes_every_line(monkeypatch):
    import io
    import sys

    class Stream(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushed = []  # the lines written so far, at each flush

        def flush(self):
            self.flushed.append(self.getvalue().count("\n"))

    out = Stream()
    monkeypatch.setattr(sys, "stdout", out)
    _, rec = train(CASE, TINY)
    lines = out.getvalue().count("\n")
    assert lines == len(rec["history"]) > 1
    assert out.flushed == list(range(1, lines + 1))


@pytest.mark.parametrize("finite_calls", [2, 6])
def test_a_divergence_counts_the_evaluations_of_its_outer_iteration(monkeypatch, finite_calls):
    real, calls, backwards = tr.losses, [0], [0]

    def failing_losses(*a, **k):
        calls[0] += 1
        if calls[0] > finite_calls:
            t = torch.tensor(float("nan"), requires_grad=True)
            return {"residual": t, "far": t, "wall": t, "periodic": t, "total": t}
        L = real(*a, **k)
        L["total"].register_hook(lambda g: backwards.__setitem__(0, backwards[0] + 1))
        return L

    monkeypatch.setattr(tr, "losses", failing_losses)
    _, rec = train(CASE, TINY, log=lambda *a: None)
    assert rec["diverged"]
    recorded = sum(h["func_evals"] for h in rec["history"])
    assert backwards[0] > recorded  # the diverging outer iteration completed closures
    assert rec["func_evals"] == backwards[0]


def test_time_epoch_trains_one_epoch_of_the_run_itself(monkeypatch):
    from dataclasses import replace
    seen = []

    def one_epoch(case, cfg, log):
        seen.append(cfg)
        return None, {"device": "cpu", "wall_s": 12.5, "func_evals": 7, "stalled": 1,
                      "diverged": False}

    monkeypatch.setattr(tr, "train", one_epoch)
    full = replace(TINY, epochs=40)
    assert tr.time_epoch(CASE, full) == {"device": "cpu", "epoch_s": 12.5, "func_evals": 7,
                                         "stalled": 1, "diverged": False}
    assert seen == [replace(full, epochs=1)]


def test_the_timed_epoch_is_the_runs_first_epoch():
    quiet = lambda *a: None  # noqa: E731
    t = tr.time_epoch(CASE, TINY, log=quiet)
    _, rec = train(CASE, TINY, log=quiet)
    assert t["epoch_s"] > 0 and t["func_evals"] == rec["func_evals"] and t["device"] == "cpu"


def test_an_epoch_is_an_lbfgs_pass():
    from dataclasses import replace
    with pytest.raises(ValueError, match="adam_lbfgs"):
        tr.time_epoch(CASE, replace(TINY, optimizer="adam_lbfgs"))
