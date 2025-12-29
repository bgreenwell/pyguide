import numpy as np

from pyguide.splitting import find_best_split


def test_find_best_split_numerical():
    # Perfectly separable case
    # x: [1, 2, 8, 9] -> should split between 2 and 8 (e.g. 5)
    # y: [0, 0, 1, 1]
    x = np.array([1, 2, 8, 9])
    y = np.array([0, 0, 1, 1])

    threshold, gain = find_best_split(x, y, is_categorical=False)

    assert 2 <= threshold <= 8
    assert gain > 0


def test_find_best_split_no_gain():
    # All same class, no gain possible
    x = np.array([1, 2, 3, 4])
    y = np.array([0, 0, 0, 0])

    threshold, gain = find_best_split(x, y, is_categorical=False)
    # It might return a threshold now (e.g. 1.5), but gain must be 0
    assert gain == 0
    # Threshold is not None because we allow finding 0-gain splits for interaction lookahead
    assert threshold is not None


def test_find_best_split_categorical_simple():
    # Only two categories, split is trivial
    # x: [0, 0, 1, 1]
    # y: [0, 0, 1, 1]
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 1, 1])

    threshold, gain = find_best_split(x, y, is_categorical=True)
    # For binary categorical, threshold usually separates one category from others
    # Here, 0 goes left, 1 goes right.
    # threshold might be represented as a set {0} or similar.
    # For now, let's assume it returns the category to go left, or a mask.
    assert threshold is not None
    assert gain > 0
