import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def verify_classifier_metadata():
    print("--- Verifying Classifier Metadata (XOR) ---")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
    y = np.array([0, 1, 1, 0] * 10)
    
    clf = GuideTreeClassifier(interaction_depth=1, significance_threshold=0.05)
    clf.fit(X, y)
    
    print(f"Root Split Type: {clf._root.split_type}")
    print(f"Root Interaction Group: {clf._root.interaction_group}")
    
    assert clf._root.split_type == "interaction"
    assert set(clf._root.interaction_group) == {0, 1}
    print("Classifier metadata verification successful!")
    print()

def verify_regressor_metadata():
    print("--- Verifying Regressor Metadata (XOR) ---")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
    y = np.array([0.0, 1.0, 1.0, 0.0] * 10)
    
    reg = GuideTreeRegressor(interaction_depth=1, significance_threshold=0.05)
    reg.fit(X, y)
    
    print(f"Root Split Type: {reg._root.split_type}")
    print(f"Root Interaction Group: {reg._root.interaction_group}")
    
    assert reg._root.split_type == "interaction"
    assert set(reg._root.interaction_group) == {0, 1}
    print("Regressor metadata verification successful!")
    print()

if __name__ == "__main__":
    verify_classifier_metadata()
    verify_regressor_metadata()
    print("All metadata verifications passed!")
