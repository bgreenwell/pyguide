import numpy as np

from pyguide import GuideTreeClassifier


def test_predict_proba_shape_and_sum():
    X = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
    y = np.array([0, 0, 1, 1])

    clf = GuideTreeClassifier(max_depth=2)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_predict_matches_proba():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    clf = GuideTreeClassifier(max_depth=2)
    clf.fit(X, y)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)

    # Check consistency
    pred_from_proba = clf.classes_[np.argmax(proba, axis=1)]
    assert np.array_equal(pred, pred_from_proba)


def test_predict_proba_impure_leaf():
    # Leaf has mixed classes, proba should reflect that
    # Create a situation where split is forbidden (min_samples_split)
    X = np.array([[1], [1], [1], [1]])
    y = np.array([0, 0, 0, 1])  # 3 zeros, 1 one

    clf = GuideTreeClassifier(min_samples_split=10)  # Prevent split
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    # Should be [0.75, 0.25] for all samples
    expected = np.array([0.75, 0.25])
    assert np.allclose(proba[0], expected)
