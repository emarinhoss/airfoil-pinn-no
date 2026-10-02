import numpy as np

from apno.data.sampling import (
    BOX, EXTRAP_MACH, campaign_specs, case_specs, design, prefix_indices, sobol_index,
)


def test_deterministic_and_sized():
    a, b = design(), design()
    assert {k: v.shape[0] for k, v in a.items()} == {"train": 896, "val": 128, "test_in": 64,
                                                     "test_ext": 16}
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_inside_the_box():
    d = design()
    lo = np.array([BOX[k][0] for k in ("mach", "alpha_deg", "t", "m")])
    hi = np.array([BOX[k][1] for k in ("mach", "alpha_deg", "t", "m")])
    for split in ("train", "val", "test_in"):
        assert np.all(d[split] >= lo) and np.all(d[split] <= hi)
    ext = d["test_ext"]
    assert np.all((ext[:, 0] >= EXTRAP_MACH[0]) & (ext[:, 0] <= EXTRAP_MACH[1]))
    assert np.all(ext[:, 1:] >= lo[1:]) and np.all(ext[:, 1:] <= hi[1:])


def test_splits_disjoint():
    d = design()
    rows = np.concatenate(list(d.values()))
    assert len(np.unique(rows.round(12), axis=0)) == len(rows)


def test_prefix_of_64_is_balanced():
    u = (design()["train"][:64] - [0.30, -2.0, 0.08, 0.0]) / [0.5, 8.0, 0.08, 0.04]
    assert np.all(np.abs(u.mean(0) - 0.5) < 0.05)


def test_case_specs():
    specs = case_specs("test_ext", (512, 193))
    assert specs[3].case_id == "test_ext-0003" and specs[3].level == "512x193"
    assert len(campaign_specs("train")) == 896 + 128 + 64 + 16
    assert len(campaign_specs("fine")) == 64 + 16


def test_prefix_is_the_sobol_prefix_minus_failures():
    survivors = [f"train-{k:04d}" for k in range(100) if k not in (3, 70)]
    assert sobol_index("test_in-0063") == 63
    idx = prefix_indices(survivors, 64)
    assert len(idx) == 63 and max(sobol_index(survivors[k]) for k in idx) == 63
    assert len(prefix_indices(survivors, 896)) == 98
