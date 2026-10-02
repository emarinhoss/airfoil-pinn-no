import numpy as np

from apno.geometry.ogrid import Cylinder, Section
from apno.models.pinn import PinnCase
from apno.models.train_pinn import PinnConfig, evaluate_on_grid, train


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
