import time

import numpy as np

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_high_cardinality_regression():
    # 50 categories
    categories = np.array([f"cat_{i}" for i in range(50)])
    X = np.random.choice(categories, size=1000).reshape(-1, 1)

    # Target depends on a subset of categories
    # cat_0 to cat_24 -> y=10, cat_25 to cat_49 -> y=100
    y = np.zeros(1000)
    for i in range(50):
        mask = X.flatten() == f"cat_{i}"
        if i < 25:
            y[mask] = 10.0 + np.random.normal(0, 1, np.sum(mask))
        else:
            y[mask] = 100.0 + np.random.normal(0, 1, np.sum(mask))

    reg = GuideTreeRegressor(max_depth=1)

    start = time.time()
    reg.fit(X, y)
    duration = time.time() - start
    print(f"\nRegression fit duration (50 cats): {duration:.4f}s")

    # With one-vs-rest, it's unlikely to find the perfect split in one step
    # but it should still find some gain.
    assert reg.tree_.split_feature == 0
    assert reg.tree_.split_threshold is not None


def test_high_cardinality_classification():
    categories = np.array([f"cat_{i}" for i in range(50)])
    X = np.random.choice(categories, size=1000).reshape(-1, 1)

    y = np.zeros(1000, dtype=int)
    for i in range(50):
        mask = X.flatten() == f"cat_{i}"
        if i < 25:
            y[mask] = 0
        else:
            y[mask] = 1

    clf = GuideTreeClassifier(max_depth=1)

    start = time.time()
    clf.fit(X, y)
    duration = time.time() - start
    print(f"\nClassification fit duration (50 cats): {duration:.4f}s")

    assert clf.tree_.split_feature == 0
    assert clf.tree_.split_threshold is not None
