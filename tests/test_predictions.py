import numpy as np

from apno.data.predictions import datasets, read_predictions, write_predictions


def test_round_trip(tmp_path):
    q = np.random.default_rng(0).standard_normal((2, 4, 3, 4))
    write_predictions(tmp_path / "p.h5", ["a", "b"], q, {"method": "fno", "seed": 1})
    got, attrs = read_predictions(tmp_path / "p.h5")
    np.testing.assert_array_equal(got["b"], q[1])
    assert attrs["method"] == "fno" and attrs["seed"] == 1


def test_write_predictions_accepts_str_path(tmp_path):
    p = str(tmp_path / "sub" / "p.h5")
    write_predictions(p, ["a"], np.zeros((1, 2, 2)), {"seed": 1})
    assert "a" in read_predictions(p)[0]


def test_extra_datasets_round_trip(tmp_path):
    q = np.zeros((2, 4, 3, 4))
    write_predictions(tmp_path / "p.h5", ["a", "b"], q, {"method": "fno"},
                      extra={"q_128x49": np.ones((2, 2, 2, 4))})
    assert datasets(tmp_path / "p.h5") == ["q", "q_128x49"]
    coarse, attrs = read_predictions(tmp_path / "p.h5", "q_128x49")
    assert coarse["b"].shape == (2, 2, 4) and attrs["method"] == "fno"
