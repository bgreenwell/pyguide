"""
Comparison of different variable importance metrics in pyguide.
"""
import numpy as np
import pandas as pd

from pyguide import GuideTreeClassifier


def importance_demo():
    print("--- pyguide Variable Importance Demo ---")
    np.random.seed(42)
    
    n_samples = 500
    n_features = 10
    X = np.random.rand(n_samples, n_features)
    
    # y depends on:
    # 1. Main effects: X0, X1
    # 2. Interaction: X2 * X3
    # 3. Noise: X4-X9
    
    y_cont = 5 * X[:, 0] + 5 * X[:, 1] + 10 * (X[:, 2] > 0.5) * (X[:, 3] > 0.5)
    # Add some noise to y
    y_cont += np.random.normal(0, 1, n_samples)
    
    # Classification target
    y = (y_cont > np.median(y_cont)).astype(int)
    
    feature_names = [f"X{i}" for i in range(n_features)]
    feature_names[0] += " (Main)"
    feature_names[1] += " (Main)"
    feature_names[2] += " (Int1)"
    feature_names[3] += " (Int2)"
    
    # 2. Fit GUIDE Tree
    clf = GuideTreeClassifier(
        interaction_depth=1, 
        significance_threshold=0.05,
        max_depth=4,
        max_features=None,
        random_state=42
    )
    clf.fit(X, y)
    
    print(f"Tree structure: {clf.n_nodes_} nodes, {clf.n_leaves_} leaves")
    
    # 3. Collect importances
    results = pd.DataFrame(index=feature_names)
    
    # Standard scikit-learn compatible (impurity-based)
    results["Impurity (Std)"] = clf.feature_importances_
    
    # Interaction-aware impurity
    results["Impurity (Int-Aware)"] = clf.interaction_importances_
    
    # True GUIDE scores (Chi-square based)
    # Normalize GUIDE scores to sum to 1 for comparison
    g_scores = clf.guide_importances_
    if g_scores.sum() > 0:
        g_scores /= g_scores.sum()
    results["GUIDE (Chi-Sq, Norm)"] = g_scores
    
    # 4. Display
    print("Comparison of Importance Scores (Normalized):")
    print(results.sort_values("Impurity (Std)", ascending=False).to_markdown())
    print("\nNote:")
    print("- Impurity (Std) only credits the feature used for the split.")
    print("- Impurity (Int-Aware) distributes credit among interaction partners.")
    print("- GUIDE (Chi-Sq) considers all variables at all nodes.")

if __name__ == "__main__":
    importance_demo()
