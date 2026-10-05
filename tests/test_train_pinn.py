import numpy as np
import pytest
import torch

from apno.geometry.ogrid import Cylinder, Section, stretch_ratio
from apno.models.pinn import Pinn, PinnCase, PinnNet
from apno.models.train_pinn import (
    PinnConfig, _points, balance_weights, evaluate_on_grid, losses, projected_s, resolve_device,
    time_steps, train,
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
    # the Euler PINN's weights (outer, Kutta); the potential PINN's (wall, Kutta) are tested in
    # test_pinn_potential.py, now that "potential" is the default formulation (2026-10-04)
    cfg = _tiny(grad_balance=True, balance_every=10, formulation="euler")
    _, rec = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, log=lambda *a: None)
    assert rec["weights"]["outer"] != 10.0 and np.isfinite(rec["weights"]["outer"])
    assert np.isfinite(rec["final"]["total"]) and "w_outer" in rec["history"][0]
    _, fixed = train(PinnCase(0.3, 2.0, Section(0.12, 0.0)), _tiny(formulation="euler"),
                     log=lambda *a: None)
    assert fixed["weights"] == {"outer": 10.0, "kutta": 10.0}


def test_time_steps_and_projection():
    cfg = _tiny(grad_balance=True, balance_every=10)
    rec = time_steps(PinnCase(0.3, 2.0, Section(0.12, 0.0)), cfg, n_adam=2, n_closure=1)
    assert rec["adam_s"] > 0 and rec["closure_s"] > 0 and rec["balance_s"] > 0
    want = 30 * rec["adam_s"] + 2 * rec["closure_s"] + 3 * rec["balance_s"]
    assert projected_s(cfg, rec) == pytest.approx(want)


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU (MPS)")
_CASE = PinnCase(0.3, 2.0, Section(0.12, 0.02))


def _losses_on(device, cfg):
    torch.manual_seed(0)
    pinn = Pinn(_CASE, PinnNet(cfg.hidden, cfg.depth).float().to(device))
    pts = _points(cfg, torch.Generator().manual_seed(0), torch.float32, device)
    return losses(pinn, pts, cfg, stretch_ratio(_CASE.radius))


def test_losses_follow_the_device_of_their_inputs_on_the_cpu():
    for cfg in (_tiny(), _tiny(kutta=False, zero_circulation=True), _tiny(viscosity=1e-3)):
        L = _losses_on("cpu", cfg)
        assert all(v.device.type == "cpu" for v in L.values())


@MPS
def test_losses_on_mps_match_the_cpu():
    for cfg in (_tiny(), _tiny(kutta=False, zero_circulation=True), _tiny(viscosity=1e-3)):
        cpu, mps = _losses_on("cpu", cfg), _losses_on("mps", cfg)
        for k in cpu:
            a, b = float(cpu[k].detach()), float(mps[k].detach().cpu())
            assert mps[k].device.type == "mps" and np.isfinite(b)
            assert b == pytest.approx(a, rel=1e-4, abs=1e-12), k


@MPS
def test_the_same_seed_draws_the_same_points_on_cpu_and_mps():
    cfg = _tiny()
    a = _points(cfg, torch.Generator().manual_seed(7), torch.float32, "cpu")
    b = _points(cfg, torch.Generator().manual_seed(7), torch.float32, "mps")
    for k in a:
        assert b[k].device.type == "mps" and torch.equal(a[k], b[k].cpu())


def test_device_resolution():
    want = "mps" if torch.backends.mps.is_available() else "cpu"
    assert resolve_device("auto") == want and PinnConfig().device == "auto"
    assert resolve_device("cpu") == "cpu" and resolve_device("mps") == "mps"


def test_time_steps_and_training_record_their_devices():
    cfg = _tiny(device="cpu")
    rec = time_steps(_CASE, cfg, n_adam=1, n_closure=1)
    assert rec["adam_device"] == "cpu" and rec["lbfgs_device"] == "cpu"
    pinn, tr = train(_CASE, cfg, log=lambda *a: None)
    assert tr["adam_device"] == "cpu"
    assert next(pinn.net.parameters()).dtype == torch.float64


@MPS
def test_training_on_mps_runs_adam_on_the_gpu_and_lbfgs_on_the_cpu_in_float64():
    cfg = _tiny(device="mps")
    rec = time_steps(_CASE, cfg, n_adam=1, n_closure=1)
    assert rec["adam_device"] == "mps" and rec["lbfgs_device"] == "cpu" and rec["adam_s"] > 0
    pinn, tr = train(_CASE, cfg, log=lambda *a: None)
    p = next(pinn.net.parameters())
    assert tr["adam_device"] == "mps" and p.device.type == "cpu" and p.dtype == torch.float64
    assert np.isfinite(tr["final"]["total"])
    q = evaluate_on_grid(pinn, (16, 5))
    assert q.shape == (16, 5, 4) and np.isfinite(q).all()
