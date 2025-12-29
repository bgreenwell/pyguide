import numpy as np

from pyguide import GuideTreeClassifier
from pyguide.interactions import calc_interaction_p_value


def test_interaction_xor_pattern():
    # XOR pattern:
    # x1  x2  y
    # 0   0   0
    # 0   1   1
    # 1   0   1
    # 1   1   0
    # Main effects are 0 (uncorrelated individually).
    # Interaction is strong.

    x1 = np.array([0, 0, 1, 1] * 10)
    x2 = np.array([0, 1, 0, 1] * 10)
    y = np.array([0, 1, 1, 0] * 10)

    # Check interaction between x1 and x2
    p = calc_interaction_p_value(x1, x2, y, is_cat1=True, is_cat2=True)
    assert p < 0.05


def test_no_interaction():
    # Random data
    np.random.seed(42)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = np.random.randint(0, 2, 100)

    p = calc_interaction_p_value(x1, x2, y, is_cat1=False, is_cat2=False)
    assert p > 0.05


def test_classifier_interaction_xor():
    # XOR pattern
    # x1  x2  y
    # 0   0   0
    # 0   1   1
    # 1   0   1
    # 1   1   0

    # Repeat to have enough samples for significance
    x1 = np.array([0, 0, 1, 1] * 20)
    x2 = np.array([0, 1, 0, 1] * 20)
    y = np.array([0, 1, 1, 0] * 20)

    # Feature 3: noise
    x3 = np.random.rand(80)

    X = np.column_stack([x1, x2, x3])

    # Default interaction_depth=1 should find this
    clf = GuideTreeClassifier(max_depth=3, interaction_depth=1)
    clf.fit(X, y)

    # Should be able to fit and predict accurately
    # With a simple tree (depth 3), it should solve XOR
    # First split one variable, then the other.
    acc = clf.score(X, y)
    assert acc > 0.9
