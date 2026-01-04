import numpy as np
from pyguide import GuideTreeClassifier

def verify_interaction_importance():
    print("--- Verifying Interaction Importance ---")
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] * X[:, 1] > 0.25).astype(int)
    
    clf = GuideTreeClassifier(interaction_depth=1, significance_threshold=0.05)
    clf.fit(X, y)
    
    importances = clf.interaction_importances_
    print(f"Interaction Importances: {importances}")
    
    assert np.isclose(importances.sum(), 1.0), "Importances must sum to 1.0"
    
    # We expect both features to have non-zero importance due to the interaction
    # If the root split was an interaction, they should share the importance.
    # If it was a main effect, standard importance applies.
    # Given the data generation, interaction is highly likely.
    
    if clf._root.split_type == "interaction":
        print("Root split was interaction.")
        assert importances[0] > 0
        assert importances[1] > 0
    else:
        print("Root split was main effect (fallback).")
        assert importances.sum() > 0

    print("Verification successful!")

if __name__ == "__main__":
    verify_interaction_importance()
