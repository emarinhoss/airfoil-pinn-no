import numpy as np

from apno.data.predictions import read_predictions, write_predictions


def test_round_trip(tmp_path):
    q = np.random.default_rng(0).standard_normal((2, 4, 3, 4))
    write_predictions(tmp_path / "p.h5", ["a", "b"], q, {"method": "fno", "seed": 1})
    got, attrs = read_predictions(tmp_path / "p.h5")
    np.testing.assert_array_equal(got["b"], q[1])
    assert attrs["method"] == "fno" and attrs["seed"] == 1
