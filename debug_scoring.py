import numpy as np

from pyguide.selection import select_split_variable
from pyguide.stats import calc_curvature_p_value


def debug_scoring():
    print("--- Debugging Scoring ---")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    
    # Check calc_curvature_p_value directly for X[0]
    p0 = calc_curvature_p_value(X[:, 0], y)
    print(f"X[0] p-value (direct): {p0}")
    
    # Check select_split_variable
    best_idx, best_p, p_values = select_split_variable(X, y)
    print(f"select_split_variable p_values: {p_values}")
    
    if p_values[0] > 0.9:
        print("FAIL: X[0] p-value is too high.")
    else:
        print("SUCCESS: X[0] p-value is low.")

if __name__ == "__main__":
    debug_scoring()
