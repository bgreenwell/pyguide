"""
Demonstration of GUIDE's interaction detection capabilities.
Shows how GUIDE can solve a 3-way XOR problem that requires higher-order search.
"""
import numpy as np

from pyguide import GuideTreeClassifier


def interaction_demo():
    print("--- Interaction Detection Demo (3-way XOR) ---")
    np.random.seed(42)
    n_samples = 1000
    # Create 5 binary features
    X = (np.random.rand(n_samples, 5) > 0.5).astype(int)
    
    # 3-way XOR interaction between X0, X1, and X2
    # y = X0 ^ X1 ^ X2
    y = X[:, 0] ^ X[:, 1] ^ X[:, 2]
    
    print("Individual features have NO linear correlation with the target.")
    print("Pairwise interactions also have NO correlation with the target.")
    print("Only the triplet (X0, X1, X2) defines the target.")

    # 1. Standard Interaction Search (Depth 1 - Pairs)
    print("\nTraining with interaction_depth=1 (Exhaustive Pair Search)...")
    clf_p = GuideTreeClassifier(max_depth=3, interaction_depth=1, significance_threshold=0.05)
    clf_p.fit(X, y)
    print(f"Accuracy (Pairs): {clf_p.score(X, y):.4f}")

    # 2. Higher-order Interaction Search (Depth 2 - Triplets)
    print("\nTraining with interaction_depth=2 (Higher-order Search)...")
    clf_t = GuideTreeClassifier(max_depth=3, interaction_depth=2, significance_threshold=0.05)
    clf_t.fit(X, y)
    print(f"Accuracy (Triplets): {clf_t.score(X, y):.4f}")

    if clf_t.score(X, y) > 0.99:
        print("\nSUCCESS: GUIDE correctly identified the 3-way interaction!")

if __name__ == "__main__":
    interaction_demo()
