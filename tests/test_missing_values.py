import numpy as np

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_classifier_fit_with_nan():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [0.0, 1.0], [1.0, np.nan]])
    y = np.array([0, 1, 0, 1])

    clf = GuideTreeClassifier()
    # Currently check_X_y will likely fail with ValueError due to NaNs
    # unless we explicitly allow them or handle them.
    clf.fit(X, y)

    # Check that we can predict even with NaNs in test data
    X_test = np.array([[np.nan, 2.0], [1.0, np.nan]])
    y_pred = clf.predict(X_test)
    assert len(y_pred) == 2


def test_regressor_fit_with_nan():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [0.0, 1.0], [1.0, np.nan]])
    y = np.array([10.0, 20.0, 10.0, 20.0])

    reg = GuideTreeRegressor()
    reg.fit(X, y)

    X_test = np.array([[np.nan, 2.0], [1.0, np.nan]])
    y_pred = reg.predict(X_test)
    assert len(y_pred) == 2


def test_missing_value_routing():
    # Test that missing values are routed consistently
    # X0 is predictive, X1 has a missing value that should be routed to
    # the side that matches its Y value.
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.1],
            [0.0, 0.9],
            [0.0, 1.0],
            [np.nan, 0.5],  # Missing value in X0
        ]
    )
    y = np.array([1, 1, 0, 0, 1])  # The missing value sample has y=1

    clf = GuideTreeClassifier(max_depth=1)
    clf.fit(X, y)

    # Predict on a sample with missing value
    X_new = np.array([[np.nan, 0.5]])
    assert clf.predict(X_new)[0] == 1
