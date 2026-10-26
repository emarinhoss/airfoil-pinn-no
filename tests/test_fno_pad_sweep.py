"""The padding sweep: the override changes only the row count at 256x97 and gives each model's own
output at its own rows; checkpoints of every kind load; at the native rows the sweep reproduces
what the probe recorded in train.json; the cache is reused. Library models skip without the
library."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "probes"))

import probe_fno_libraries as probe  # noqa: E402
import probe_fno_pad_sweep as sweep  # noqa: E402
from apno.models.fno import FNO2d  # noqa: E402
from apno.models.train_fno import FnoConfig  # noqa: E402

SMALL = FnoConfig(width=8, modes_i=4, modes_j=3, layers=2)
KINDS = [("apno", ""), ("ablate", ""), ("ablate", "P"), ("ablate", "MPIL"), ("neuralop", ""),
         ("physicsnemo", "")]


def _model(library, factors):
    if library != "apno":
        pytest.importorskip("physicsnemo" if library == "physicsnemo" else "neuralop")
    torch.manual_seed(0)
    m = probe.build_model(library, SMALL, factors).eval()
    with torch.no_grad():  # spectral weights of a trained scale, so the padding matters
        for p in m.parameters():
            if p.is_complex() or p.dim() == 5:  # complex, or PhysicsNeMo's (..., 2) pairs
                p.mul_(50.0)
    return m


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_parse_rows():
    assert sweep.parse_rows("0-3") == [0, 1, 2, 3]
    assert sweep.parse_rows("17,8-9,16,8") == [8, 9, 16, 17]
    for bad in ("", "-1", "a"):
        with pytest.raises(ValueError):
            sweep.parse_rows(bad)


@pytest.mark.parametrize("library, factors", KINDS)
def test_own_rows_reproduce_the_model_and_other_rows_change_only_256x97(library, factors):
    m = _model(library, factors)
    own = sweep.native_rows(m)
    assert own == (16 if library == "neuralop" or "P" in factors else 17)
    fine, coarse = torch.randn(1, 6, 256, 97), torch.randn(1, 6, 128, 49)
    with torch.no_grad():
        ref_fine, ref_coarse = m(fine), m(coarse)
        with sweep.rows_at(m, own):  # one-sided where the model pads both ends: a shift
            assert _rel(m(fine), ref_fine) < 1e-5
        for rows in (0, own - 3, own + 5):
            with sweep.rows_at(m, rows):
                assert _rel(m(fine), ref_fine) > 1e-4  # the 256x97 output moves
                assert _rel(m(coarse), ref_coarse) < 1e-6  # 128x49 keeps its own 8
        assert _rel(m(fine), ref_fine) < 1e-6  # restored


def test_fno2d_at_16_rows_is_the_padding_factor():
    """FNO2d evaluated with 16 rows at 256x97 is the ablation's P, from the same weights."""
    pytest.importorskip("neuralop")
    torch.manual_seed(0)
    p = _model("ablate", "P")
    f = FNO2d(width=8, modes_i=4, modes_j=3, layers=2).eval()
    sd = {"lift.weight": p.lift.fcs[0].weight[..., None], "lift.bias": p.lift.fcs[0].bias,
          "proj.0.weight": p.proj.fcs[0].weight[..., None], "proj.0.bias": p.proj.fcs[0].bias,
          "proj.2.weight": p.proj.fcs[1].weight[..., None], "proj.2.bias": p.proj.fcs[1].bias}
    for k in range(2):
        w = p.convs[k].weight.tensor
        sd[f"spectral.{k}.w_neg"], sd[f"spectral.{k}.w_pos"] = w[:, :, :4], w[:, :, 4:]
        sd[f"pointwise.{k}.weight"] = p.skips[k].conv.weight[..., None]
        sd[f"pointwise.{k}.bias"] = p.skips[k].conv.bias
    f.load_state_dict({k: v.detach().clone() for k, v in sd.items()})
    x = torch.randn(1, 6, 256, 97)
    with torch.no_grad(), sweep.rows_at(f, 16):
        y = f(x)
    with torch.no_grad():
        assert float((y - p(x)).abs().max() / p(x).abs().max()) < 1e-5


def _trained(tmp_path, library, factors, tag):
    """A probe checkpoint and its train.json, from the stand-in dataset."""
    path = probe.write_standin(tmp_path / "d.h5", {"train": 4, "val": 3})
    cfg = FnoConfig(width=8, modes_i=4, modes_j=3, layers=2, epochs=2, batch=2, n_train=4)
    _, _, rec = probe.train(library, cfg, path, tmp_path / "runs/fno_lib" / tag,
                            torch.device("cpu"), factors=factors)
    return path, rec


def test_the_sweep_reproduces_the_probe_record_at_native_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h5, rec_apno = _trained(tmp_path, "apno", "", "apno-n4-s0")
    torch.save({"model": FNO2d(width=8, modes_i=4, modes_j=3, layers=2).state_dict(),
                "std": json.loads(json.dumps({"mean": [0.0] * 4, "std": [1.0] * 4})),
                "cfg": json.loads(json.dumps(SMALL.__dict__))},
               (tmp_path / "s2.pt"))
    Path("runs/fno/fno-n4-s0").mkdir(parents=True)
    (tmp_path / "s2.pt").rename("runs/fno/fno-n4-s0/best.pt")
    args = ["runs/fno_lib/*/best.pt", "runs/fno/*/best.pt", "--rows", "15-18", "--h5", str(h5),
            "--device", "cpu", "--out", str(tmp_path / "out")]
    assert sweep.main(args) == 0
    out = json.loads((tmp_path / "out/pad_sweep.json").read_text())
    recs = {r["tag"]: r for r in out["records"]}
    assert recs["fno-n4-s0"]["source"] == "S2" and recs["fno-n4-s0"]["group"] == "S2 fno-n4"
    a = recs["apno-n4-s0"]
    assert a["native_rows"] == 17 and sorted(a["rows"]) == ["15", "16", "17", "18"]  # JSON keys
    assert a["rows"]["17"]["val_rel_l2_256x97"] == pytest.approx(rec_apno["val_rel_l2_256x97"],
                                                               rel=1e-6)
    assert a["rows"]["17"]["transfer"] == pytest.approx(rec_apno["transfer_val_rel_l2"], rel=1e-6)
    assert a["val_rel_l2_128x49"] == pytest.approx(rec_apno["val_rel_l2_128x49"], rel=1e-6)
    md = (tmp_path / "out/pad_sweep.md").read_text()
    assert "| apno-n4 | 1 | 17 |" in md and (tmp_path / "out/pad_sweep.png").exists()


def test_an_ablation_checkpoint_loads_with_factors_and_the_cache_is_reused(tmp_path, monkeypatch,
                                                                            capsys):
    pytest.importorskip("neuralop")
    monkeypatch.chdir(tmp_path)
    h5, rec = _trained(tmp_path, "ablate", "P", "abl-P-n4-s0")
    args = ["runs/fno_lib/*/best.pt", "--rows", "16,17", "--h5", str(h5), "--device", "cpu",
            "--out", str(tmp_path / "out")]
    assert sweep.main(args) == 0
    r = json.loads((tmp_path / "out/pad_sweep.json").read_text())["records"][0]
    assert r["native_rows"] == 16 and r["group"] == "abl-P-n4"
    assert r["rows"]["16"]["transfer"] == pytest.approx(rec["transfer_val_rel_l2"], rel=1e-6)
    capsys.readouterr()
    assert sweep.main(args) == 0 and "abl-P-n4-s0 (ablate): cached" in capsys.readouterr().out
    assert sweep.main([*args[:2], "15-17", *args[3:]]) == 0  # a new row count: swept again
    assert "cached" not in capsys.readouterr().out


def test_missing_inputs_exit_2(tmp_path):
    assert sweep.main(["--h5", str(tmp_path / "none.h5")]) == 2
    probe.write_standin(tmp_path / "d.h5", {"train": 2, "val": 2})
    assert sweep.main([str(tmp_path / "nothing/*.pt"), "--h5", str(tmp_path / "d.h5")]) == 2
