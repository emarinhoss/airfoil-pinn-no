"""The transfer ablation's model: all switches off is FNO2d, all on is NeuralOperator's FNO as the
probe builds it (weights copied across, outputs compared at both resolutions), every combination
keeps the operator's contract, each switch changes what its docstring says, and the planned
analysis reads a scores file as the scorer writes it. Skips where NeuralOperator is not installed.
"""

import itertools
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("neuralop")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import fno_ablation as ab  # noqa: E402
import probe_fno_libraries as probe  # noqa: E402
from apno.models.fno import FNO2d, radial_pad_period_matched  # noqa: E402
from apno.models.train_fno import FnoConfig  # noqa: E402

SMALL = FnoConfig(width=8, modes_i=4, modes_j=3, layers=3)
ALL = ["".join(c) for k in range(5) for c in itertools.combinations(ab.FACTORS, k)]


def _close(a, b, x):
    a.eval(), b.eval()
    with torch.no_grad():
        ya, yb = a(x), b(x)
    return float((ya - yb).abs().max() / yb.abs().max())


def test_canonical_order_and_names():
    assert ab.canonical("lim") == "MIL" and ab.canonical("none") == "" and ab.canonical("") == ""
    assert ab.variant_name("") == "abl-none" and ab.variant_name("pm") == "abl-MP"
    for bad in ("X", "MM", "MPx"):
        with pytest.raises(ValueError):
            ab.canonical(bad)


@pytest.mark.parametrize("ni, nj", [(128, 49), (256, 97)])
def test_all_off_is_fno2d(ni, nj):
    """FNO2d as the ablation ran, with the 17-row rule at 256x97 (spec 6.3 until 2026-10-26)."""
    torch.manual_seed(0)
    f = FNO2d(width=8, modes_i=4, modes_j=3, layers=3)
    f.pad_rule = radial_pad_period_matched
    a = ab.AblationFno(SMALL, "")
    sd = {"lift.fcs.0.weight": f.lift.weight[..., 0], "lift.fcs.0.bias": f.lift.bias,
          "proj.fcs.0.weight": f.proj[0].weight[..., 0], "proj.fcs.0.bias": f.proj[0].bias,
          "proj.fcs.1.weight": f.proj[2].weight[..., 0], "proj.fcs.1.bias": f.proj[2].bias}
    for k in range(3):
        sd[f"convs.{k}.weight.tensor"] = torch.cat([f.spectral[k].w_neg, f.spectral[k].w_pos], 2)
        sd[f"skips.{k}.conv.weight"] = f.pointwise[k].weight[..., 0]
        sd[f"skips.{k}.conv.bias"] = f.pointwise[k].bias
    a.load_state_dict({k: v.detach().clone() for k, v in sd.items()})
    assert _close(a, f, torch.randn(2, 6, ni, nj)) < 1e-5


@pytest.mark.parametrize("ni, nj", [(128, 49), (256, 97)])
def test_all_on_is_neuralops_fno(ni, nj):
    torch.manual_seed(0)
    n = probe.NeuralopFno(SMALL)
    a = ab.AblationFno(SMALL, "MPIL")
    rename = {"net.lifting.": "lift.", "net.projection.": "proj.",
              "net.fno_blocks.convs.": "convs.", "net.fno_blocks.fno_skips.": "skips.",
              "net.fno_blocks.channel_mlp_skips.": "gates.",
              "net.fno_blocks.channel_mlp.": "mlps."}
    sd = {}
    for k, v in n.state_dict().items():
        if torch.is_tensor(v):
            prefix = next(p for p in rename if k.startswith(p))
            sd[rename[prefix] + k[len(prefix):]] = v
    a.load_state_dict(sd)
    assert _close(a, n, torch.randn(2, 6, ni, nj)) < 1e-5


def test_parameter_counts_match_both_ends():
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    cfg = FnoConfig(width=48, modes_i=24, modes_j=8, layers=4)
    assert count(ab.AblationFno(cfg, "")) == count(probe.build_model("apno", cfg)) == 3555476
    on = count(ab.AblationFno(cfg, "MPIL"))
    assert on == count(probe.build_model("neuralop", cfg)) == 3568468


@pytest.mark.parametrize("factors", ALL)
def test_every_combination_keeps_the_operator_contract(factors):
    """Both resolutions, and equivariant to a roll around the airfoil (no position input)."""
    torch.manual_seed(0)
    m = ab.AblationFno(SMALL, factors).eval()
    with torch.no_grad():
        for ni, nj in ((128, 49), (256, 97)):
            assert m(torch.randn(1, 6, ni, nj)).shape == (1, 4, ni, nj)
        x = torch.randn(1, 6, 32, 17)
        torch.testing.assert_close(m(torch.roll(x, 5, dims=2)), torch.roll(m(x), 5, dims=2),
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("nj, rows", [(49, (4, 4)), (97, (8, 8))])
def test_p_pads_both_radial_ends_and_off_pads_the_far_end(nj, rows):
    x = torch.ones(1, 1, 16, nj)
    on = ab.AblationFno(SMALL, "P").padding.pad(x)
    assert on.shape[-1] == nj + sum(rows)
    assert on[..., : rows[0]].abs().sum() == 0 and on[..., -rows[1]:].abs().sum() == 0
    off = ab.AblationFno(SMALL, "").padding.pad(x)
    assert off.shape[-1] == nj + radial_pad_period_matched(nj) and off[..., :nj].min() == 1


def test_i_sets_the_spectral_scale_and_moves_the_bias():
    torch.manual_seed(0)
    cfg = FnoConfig(width=48, modes_i=24, modes_j=8, layers=1)
    on, off = ab.AblationFno(cfg, "I"), ab.AblationFno(cfg, "")
    rms = lambda m: float(m.convs[0].weight.tensor.detach().abs().pow(2).mean().sqrt())  # noqa: E731
    assert rms(on) == pytest.approx((2 / 96) ** 0.5, rel=0.01)
    assert rms(off) == pytest.approx(1 / 48**2, rel=0.01)
    f = FNO2d(width=48, modes_i=24, modes_j=8, layers=1)  # the same draw as FNO2d's
    fno2d_rms = float(f.spectral[0].w_pos.detach().abs().pow(2).mean().sqrt())
    assert fno2d_rms == pytest.approx(rms(off), rel=0.01)
    assert on.convs[0].bias is not None and on.skips[0].conv.bias is None
    assert off.convs[0].bias is None and off.skips[0].conv.bias is not None


def test_m_and_l_add_neuralops_modules():
    cfg = FnoConfig(width=48, modes_i=24, modes_j=8, layers=4)
    m = ab.AblationFno(cfg, "M")
    assert len(m.mlps) == 4 and m.mlps[0].fcs[0].out_channels == 24 and len(m.gates) == 4
    assert not hasattr(ab.AblationFno(cfg, ""), "mlps")
    lifted = ab.AblationFno(cfg, "L")
    assert [fc.out_channels for fc in lifted.lift.fcs] == [96, 48]
    assert [fc.out_channels for fc in lifted.proj.fcs] == [96, 4]
    assert [fc.out_channels for fc in ab.AblationFno(cfg, "").proj.fcs] == [128, 4]


def test_the_fft_normalisation_is_not_a_difference():
    """NeuralOperator's 'forward' and FNO2d's 'backward' scale the transform pair by the same
    1/N in total, so one spectral layer gives the same output under both."""
    from neuralop.layers.spectral_convolution import SpectralConv
    torch.manual_seed(0)
    fwd = SpectralConv(8, 8, (8, 6), fft_norm="forward")
    bwd = SpectralConv(8, 8, (8, 6), fft_norm="backward")
    bwd.load_state_dict(fwd.state_dict())
    assert _close(fwd, bwd, torch.randn(2, 8, 32, 17)) < 1e-6


# ---------------------------------------------------------------- the planned analysis

def _scores(configs):
    """A scores.json as the scorer writes it, for {code: [(transfer, best_val, sub, sup), ...]}."""
    runs, ratios = [], []
    for code, seeds in configs.items():
        for s, (tr, bv, sub, sup) in enumerate(seeds):
            runs.append({"tag": f"abl-{code}-n896-s{s}", "transfer_val_rel_l2": tr, "best_val": bv})
            for regime, x in (("subcritical", sub), ("supercritical", sup)):
                ratios.append({"variant": f"abl-{code}", "seed": s, "n_train": 896,
                               "grid": "256x97", "split": "test_in", "regime": regime,
                               "x_floor": x})
    runs.append({"tag": "neuralop-n896-s0-nopatience", "transfer_val_rel_l2": 0.0})  # ignored
    return {"runs": runs, "floor_ratios": ratios}


def test_permutation_p():
    assert ab.permutation_p([1, 2, 3], [10, 11, 12]) == pytest.approx(2 / 20)
    assert ab.permutation_p([1, 2, 3], [1, 2, 3]) == 1.0


def test_summary_reports_configurations_and_effects():
    low, high = [(0.025 + 1e-4 * s, 0.012, 2.3, 1.8) for s in range(3)], \
                [(0.037 + 1e-4 * s, 0.0125, 5.3, 2.5) for s in range(3)]
    text = ab.summary(_scores({"none": high, "M": low, "MPIL": low, "PIL": high}))
    assert "| abl-none | 0, 1, 2 | 0.0371 |" in text
    m_add = next(line for line in text.splitlines() if line.startswith("| M | add-one"))
    m_remove = next(line for line in text.splitlines() if line.startswith("| M | remove-one"))
    for line in (m_add, m_remove):
        cells = [c.strip() for c in line.split("|")]
        assert cells[6] == "-0.012" and cells[7] == "0.100"  # transfer effect and p
    assert "| P | add-one" not in text  # abl-P not run
    assert "Not yet run:" in text and "abl-P," in text and "abl-M," not in text


def test_summary_cli_writes_its_file(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(_scores({"none": [(0.03, 0.01, 5.0, 2.0)] * 2})))
    assert ab.main(["summary", "--scores", str(path), "--out", str(tmp_path / "a.md")]) == 0
    assert (tmp_path / "a.md").read_text().startswith("# FNO transfer ablation")
    assert ab.main(["summary", "--scores", str(tmp_path / "missing.json")]) == 2


# ---------------------------------------------------------------- the probe's ablate option

def test_the_probe_trains_an_ablation_under_its_own_tag(tmp_path, monkeypatch):
    import h5py
    monkeypatch.chdir(tmp_path)
    path = probe.write_standin(tmp_path / "d.h5", {"train": 4, "val": 2, "test_in": 2})
    args = ["--library", "ablate", "--factors", "pm", "--h5", str(path), "--n-train", "4",
            "--epochs", "1", "--width", "8", "--modes-i", "4", "--modes-j", "3", "--layers", "2",
            "--batch", "2", "--device", "cpu", "--runs", str(tmp_path / "runs")]
    assert probe.main(args) == 0
    rec = json.loads((tmp_path / "runs/abl-MP-n4-s0/train.json").read_text())
    assert rec["library"] == "ablate" and rec["factors"] == "MP"
    with h5py.File("runs/predictions_lib/abl-MP-n4-s0.h5") as f:
        assert f.attrs["variant"] == "abl-MP"


@pytest.mark.parametrize("extra", [["--library", "ablate"],
                                   ["--library", "ablate", "--factors", "Q"],
                                   ["--library", "apno", "--factors", "M"]])
def test_the_probe_refuses_bad_factor_arguments(tmp_path, extra):
    with pytest.raises(SystemExit):
        probe.main([*extra, "--h5", str(tmp_path / "d.h5"), "--device", "cpu"])


def test_summary_flags_runs_scored_without_a_training_record():
    scores = _scores({"none": [(0.03, 0.01, 5.0, 2.0)] * 2})
    for r in scores["runs"]:
        if r["tag"] == "abl-none-n896-s1":
            del r["transfer_val_rel_l2"], r["best_val"]
    text = ab.summary(scores)
    assert "**1 run(s) have test scores but no training record**" in text and "abl-none-s1" in text
    assert "no training record" not in ab.summary(_scores({"none": [(0.03, 0.01, 5.0, 2.0)] * 2}))


class _Rows(torch.nn.Module):
    """One-sided radial padding with a fixed row count."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def pad(self, x):
        self.nj = x.shape[-1]
        return torch.nn.functional.pad(x, (0, self.rows))

    def unpad(self, x):
        return x[..., : self.nj]


def test_p_is_a_circular_shift_of_one_sided_padding_with_the_same_row_count():
    """Every layer commutes with a circular shift of the padded radial column, so padding 4 + 4
    rows is the same operator as 8 rows at the far end, and 8 + 8 the same as 16. P therefore
    changes nothing at the 128x49 training grid (FNO2d pads 8 there too) and, at 256x97, only
    the row count: 16 against FNO2d's 17. docs/findings.md, 2026-10-25."""
    torch.manual_seed(0)
    none = ab.AblationFno(SMALL, "").eval()
    torch.manual_seed(0)
    pad = ab.AblationFno(SMALL, "P").eval()
    for a, b in zip(none.state_dict().values(), pad.state_dict().values()):
        assert torch.equal(a, b)  # P adds no parameters, so the same draws
    with torch.no_grad():
        for p in pad.convs:  # spectral weights of a trained scale, so the padding matters
            p.weight.tensor.mul_(100.0)
        for p in none.convs:
            p.weight.tensor.mul_(100.0)
        x = torch.randn(2, 6, 128, 49)
        assert _close(pad, none, x) < 1e-5  # 4 + 4 against 8: the training grid
        x = torch.randn(2, 6, 256, 97)
        assert _close(pad, none, x) > 1e-3  # 8 + 8 against 17
        none.padding = _Rows(16)
        assert _close(pad, none, x) < 1e-5  # 8 + 8 against 16
