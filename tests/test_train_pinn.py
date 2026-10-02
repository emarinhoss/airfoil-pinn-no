import numpy as np
import pytest
import torch

from apno.geometry.ogrid import Cylinder, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import (
    PinnConfig, balance_weights, evaluate_on_grid, projected_s, time_steps, train,
)


def test_tiny_training_reduces_the_loss_and_evaluates():
    cfg = PinnConfig(hidden=16, depth=2, n_interior=256, resample_every=50, n_outer=64,
                     n_wall=64, adam_steps=100, lbfgs_iters=10)
    pinn, rec = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, log=lambda *a: None)
    hist = rec["history"]
    assert hist[-1]["total"] < hist[0]["total"]
    assert np.isfinite(rec["final"]["total"]) and rec["cpu_s"] > 0
    q = evaluate_on_grid(pinn, (32, 9))
    assert q.shape == (32, 9, 4) and np.isfinite(q).all()


def test_zero_circulation_variant_runs():
    cfg = PinnConfig(hidden=8, depth=2, n_interior=128, resample_every=50, n_outer=32, n_wall=32,
                     adam_steps=20, lbfgs_iters=2, kutta=False, zero_circulation=True)
    _, rec = train(PinnCase(0.1, 0.0, Cylinder()), cfg, log=lambda *a: None)
    assert "circulation" in rec["final"] and rec["final"]["kutta"] == 0.0


def test_balance_weights_is_the_wang_teng_perdikaris_update():
    p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = {"residual": (p * p).sum(), "outer": (3 * p).sum(), "kutta": (0 * p).sum(),
         "circulation": torch.zeros(())}
    w = balance_weights(L, [p], {"outer": 10.0, "kutta": 10.0}, alpha=0.9)
    # max|grad res| = 6, mean|grad outer| = 3: lambda_hat = 2, so 0.1 * 10 + 0.9 * 2 = 2.8
    assert w["outer"] == pytest.approx(2.8) and w["kutta"] == 10.0  # zero gradient: unchanged


def _tiny(**kw):
    return PinnConfig(hidden=8, depth=2, n_interior=128, resample_every=50, n_outer=32, n_wall=32,
                      adam_steps=30, lbfgs_iters=2, **kw)


def test_grad_balanced_variant_runs_and_records_its_weights():
    cfg = _tiny(grad_balance=True, balance_every=10)
    _, rec = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, log=lambda *a: None)
    assert rec["weights"]["outer"] != 10.0 and np.isfinite(rec["weights"]["outer"])
    assert np.isfinite(rec["final"]["total"]) and "w_outer" in rec["history"][0]
    _, fixed = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), _tiny(), log=lambda *a: None)
    assert fixed["weights"] == {"outer": 10.0, "kutta": 10.0}


def test_time_steps_and_projection():
    cfg = _tiny(grad_balance=True, balance_every=10)
    rec = time_steps(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, n_adam=2, n_closure=1)
    assert rec["adam_s"] > 0 and rec["closure_s"] > 0 and rec["balance_s"] > 0
    want = 30 * rec["adam_s"] + 2 * rec["closure_s"] + 3 * rec["balance_s"]
    assert projected_s(cfg, rec) == pytest.approx(want)
