import numpy as np
import pytest
from sklearn.datasets import load_iris
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def test_compute_guide_importance_exists():
    X, y = load_iris(return_X_y=True)
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    
    assert hasattr(clf, "compute_guide_importance")
    
    reg = GuideTreeRegressor()
    X_reg = np.random.rand(100, 2)
    y_reg = X_reg[:, 0]
    reg.fit(X_reg, y_reg)
    
    assert hasattr(reg, "compute_guide_importance")

def test_compute_guide_importance_output():
    X, y = load_iris(return_X_y=True)
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    
    # Test unadjusted
    scores = clf.compute_guide_importance(X, y, bias_correction=False)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (X.shape[1],)
    assert np.all(scores >= 0)
    
    # Test adjusted
    scores_adj = clf.compute_guide_importance(X, y, bias_correction=True, n_permutations=5)
    assert isinstance(scores_adj, np.ndarray)
    assert scores_adj.shape == (X.shape[1],)

def test_compute_guide_importance_logic():
    # Simple XOR-like or main effect
    np.random.seed(42)
    X = np.random.rand(500, 5) # Increased samples
    y = (X[:, 0] > 0.5).astype(int)
    
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    
    # Increased permutations for better stabilization
    scores = clf.compute_guide_importance(X, y, bias_correction=True, n_permutations=50)
    
    # X0 should be the most important
    assert np.argmax(scores) == 0
    # Signal should be significantly larger than 1.0
    assert scores[0] > 5.0
    
    # Noise variables (X1-X4) should have scores near 1.0 on average
    noise_scores = scores[1:]
    mean_noise = np.mean(noise_scores)
    assert 0.5 < mean_noise < 1.5
    
    # Each noise score shouldn't be outrageously high (e.g., < 4.0 for 1-df chi2)
    for s in noise_scores:
        assert s < 4.0
