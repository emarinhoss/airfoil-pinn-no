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
