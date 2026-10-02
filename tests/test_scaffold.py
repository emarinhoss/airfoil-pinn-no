from apno import constants


def test_constants():
    assert constants.GAMMA == 1.4
    assert constants.R_GAS == 287.058
    assert constants.P_INF == 101325.0
    assert constants.T_INF == 288.15
