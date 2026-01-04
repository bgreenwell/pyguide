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
    X_sub = np.column_stack([x1, x2])
    p = calc_interaction_p_value(X_sub, y, categorical_mask=np.array([True, True]))
    assert p < 0.05


def test_no_interaction():
    # Random data
    np.random.seed(42)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = np.random.randint(0, 2, 100)

    X_sub = np.column_stack([x1, x2])
    p = calc_interaction_p_value(X_sub, y, categorical_mask=np.array([False, False]))
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


def test_interaction_importance():
    # Pure interaction: y = XOR(X0 > 0.5, X1 > 0.5)
    # Neither X0 nor X1 have a main effect, only an interaction.
    np.random.seed(42)
    X = np.random.rand(500, 5)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)

    # We MUST enable interaction detection
    clf = GuideTreeClassifier(interaction_depth=1)

    # Using bias correction
    scores = clf.compute_guide_importance(X, y, n_permutations=20, random_state=42)

    # X0 and X1 should be the most important
    assert scores[0] > 2.0
    assert scores[1] > 2.0
    # They should be much more important than noise X2-X4
    assert scores[0] > scores[2] * 2
    assert scores[1] > scores[3] * 2
