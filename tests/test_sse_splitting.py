import numpy as np

from pyguide.splitting import find_best_split


def test_sse_split_numerical():
    # Simple clear split point
    # x: [1, 2, 3, 4]
    # y: [10, 10, 100, 100]
    # Split between 2 and 3 (threshold 2.5) should minimize SSE
    x = np.array([1, 2, 3, 4], dtype=float)
    y = np.array([10, 10, 100, 100], dtype=float)

    threshold, _, gain = find_best_split(x, y, is_categorical=False, criterion="mse")

    assert threshold == 2.5
    # SSE Total: sum((y-55)**2) = 4 * (45**2) = 8100
    # SSE Left: sum((y_left-10)**2) = 0
    # SSE Right: sum((y_right-100)**2) = 0
    # Gain: 8100 - (0 + 0) = 8100
    assert gain == 8100.0


def test_sse_split_categorical():
    # x: [A, A, B, B]
    # y: [10, 10, 100, 100]
    x = np.array(["A", "A", "B", "B"])
    y = np.array([10, 10, 100, 100], dtype=float)

    best_cat, _, gain = find_best_split(x, y, is_categorical=True, criterion="mse")

    # Either 'A' or 'B' as the "go left" category would result in 0 residual SSE
    assert best_cat in ["A", "B"]
    assert gain == 8100.0


def test_sse_split_no_gain():
    # Constant y
    x = np.array([1, 2, 3, 4], dtype=float)
    y = np.array([10, 10, 10, 10], dtype=float)

    threshold, _, gain = find_best_split(x, y, is_categorical=False, criterion="mse")
    assert gain == 0.0
