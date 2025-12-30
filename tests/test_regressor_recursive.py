import numpy as np

from pyguide import GuideTreeRegressor


def test_regressor_recursive_growth():
    # Create a deeper relationship
    # If x < 5:
    #    if x < 2.5: y = 0
    #    else: y = 10
    # else:
    #    if x < 7.5: y = 20
    #    else: y = 30
    X = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 10, 10, 20, 20, 30, 30], dtype=float)

    # max_depth=2 should allow 4 leaves
    reg = GuideTreeRegressor(max_depth=2, significance_threshold=1.0)
    reg.fit(X, y)

    y_pred = reg.predict(X)
    assert np.array_equal(y_pred, y)


def test_regressor_interaction_detection():
    # Simple XOR-like regression pattern
    # y = x1 * x2 where x1, x2 in {-1, 1}
    # x1  x2  y
    # -1  -1  1
    # -1   1 -1
    #  1  -1 -1
    #  1   1  1
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]] * 10, dtype=float)
    y = np.array([1, -1, -1, 1] * 10, dtype=float)

    # Variable selection with main effects (curvature) might fail at root
    # because x1 and x2 are uncorrelated with y.
    # Interaction detection should find (x1, x2).

    reg = GuideTreeRegressor(
        max_depth=2, interaction_depth=1, significance_threshold=0.05
    )
    reg.fit(X, y)

    # Should be able to predict perfectly if interaction is found
    y_pred = reg.predict(X)
    assert np.allclose(y_pred, y)


def test_regressor_categorical_recursive():
    X = np.array(["A", "A", "B", "B", "C", "C", "D", "D"]).reshape(-1, 1)
    y = np.array([1, 1, 10, 10, 100, 100, 1000, 1000], dtype=float)

    reg = GuideTreeRegressor(max_depth=3, significance_threshold=1.0)
    reg.fit(X, y)

    y_pred = reg.predict(X)
    assert np.allclose(y_pred, y)
