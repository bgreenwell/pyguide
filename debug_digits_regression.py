from unittest.mock import patch

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from pyguide.splitting import find_best_split
from pyguide.stats import calc_curvature_test


def debug_digits():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Check selection statistics for all features
    print("--- Checking Selection Stats ---")
    mismatches = 0
    for i in range(X_train.shape[1]):
        x_col = X_train[:, i]
        
        with patch("pyguide.stats.HAS_RUST", False):
            stat_py, p_py = calc_curvature_test(x_col, y_train)
            
        stat_rust, p_rust = calc_curvature_test(x_col, y_train)
        
        if abs(stat_py - stat_rust) > 1e-5 or abs(p_py - p_rust) > 1e-5:
            if mismatches < 5:
                print(f"Feat {i}: Py(stat={stat_py:.4f}, p={p_py:.4f}) vs Rust(stat={stat_rust:.4f}, p={p_rust:.4f})")
            mismatches += 1
            
    print(f"Total Selection Mismatches: {mismatches}")

    # Check split optimization for all features
    print("\n--- Checking Split Optimization ---")
    split_mismatches = 0
    for i in range(X_train.shape[1]):
        x_col = X_train[:, i]
        
        with patch("pyguide.splitting.HAS_RUST", False):
            t_py, l_py, g_py = find_best_split(x_col, y_train, criterion="gini")
            
        t_rust, l_rust, g_rust = find_best_split(x_col, y_train, criterion="gini")
        
        # Handle None thresholds
        if t_py is None and t_rust is None:
            continue
            
        if (t_py is None) != (t_rust is None) or abs(g_py - g_rust) > 1e-5:
            if split_mismatches < 5:
                print(f"Feat {i}: Py(t={t_py}, g={g_py:.4f}) vs Rust(t={t_rust}, g={g_rust:.4f})")
            split_mismatches += 1
            
    print(f"Total Split Mismatches: {split_mismatches}")

if __name__ == "__main__":
    debug_digits()