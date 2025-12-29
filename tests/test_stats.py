import numpy as np

from pyguide.stats import calc_curvature_p_value


def test_calc_curvature_p_value_categorical_dependent():
    # Strong association
    x = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = calc_curvature_p_value(x, z, is_categorical=True)
    assert p < 0.05


def test_calc_curvature_p_value_categorical_independent():
    # Weak/no association
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    z = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = calc_curvature_p_value(x, z, is_categorical=True)
    assert p > 0.05


def test_calc_curvature_p_value_numerical():
    # Numerical x with clear split point for z
    x = np.array([1, 2, 3, 4, 10, 11, 12, 13])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Even if not implemented yet, we define the expected behavior
    p = calc_curvature_p_value(x, z, is_categorical=False)
    assert p < 0.05
