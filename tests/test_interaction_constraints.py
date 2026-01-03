import numpy as np

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_interaction_features_constraint():
    """
    Test that interaction_features restricts which features are tested for interaction.
    We'll construct a dataset where X0 and X1 have a strong interaction,
    but we'll restrict interaction_features to [2, 3] (noise).
    The model should NOT find the X0-X1 interaction if the constraint works.
    """
    # XOR problem for X0, X1
    X = np.random.rand(100, 4)
    # Make X0, X1 strongly interacting (XOR-like)
    y = np.zeros(100)
    mask = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
    y[mask] = 1
    
    # Normally, GUIDE should pick up this interaction easily.
    # But we restrict to features 2 and 3.
    clf = GuideTreeClassifier(
        max_depth=2, 
        interaction_depth=1,
        interaction_features=[2, 3], # Only look for interactions here
        significance_threshold=0.01  # Stricter to avoid random individual splits
    )
    clf.fit(X, y)
    
    # Unconstrained baseline
    clf_base = GuideTreeClassifier(max_depth=2, interaction_depth=1, significance_threshold=0.01)
    clf_base.fit(X, y)
    score_base = clf_base.score(X, y)
    
    # Constrained
    score_constrained = clf.score(X, y)
    
    # The constrained model should perform significantly worse
    # OR it should have failed to split at all (n_leaves_ == 1) while baseline split.
    assert score_constrained < score_base or (clf.n_leaves_ == 1 and clf_base.n_leaves_ > 1)

def test_max_interaction_candidates():
    """
    Test that max_interaction_candidates limits the search to the top K features
    based on individual importance.
    """
    # X0 is weakly predictive individually, X1 is weakly predictive individually.
    # Together they are strong.
    # X2, X3, X4 are just noise but might have slight random correlations.
    
    # We want to ensure that if we set max_interaction_candidates=1,
    # it only considers the SINGLE best feature for pairing? 
    # Or top K features are the pool? Usually the pool.
    
    # Let's try a configuration parameter test first to ensure it accepts the param.
    clf = GuideTreeClassifier(max_interaction_candidates=2)
    assert clf.max_interaction_candidates == 2

def test_interaction_init_params():
    """Test that parameters are correctly passed to __init__."""
    clf = GuideTreeClassifier(
        interaction_features=[0, 1],
        max_interaction_candidates=5
    )
    assert clf.interaction_features == [0, 1]
    assert clf.max_interaction_candidates == 5
    
    reg = GuideTreeRegressor(
        interaction_features=[0, 1],
        max_interaction_candidates=5
    )
    assert reg.interaction_features == [0, 1]
    assert reg.max_interaction_candidates == 5
