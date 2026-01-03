import numpy as np

from pyguide import GuideTreeClassifier


def test_triplet_interaction():
    """
    Test that interaction_depth=2 can detect a 3-way interaction.
    We'll use a 3-way XOR problem: y = X0 ^ X1 ^ X2.
    Individual and pairwise signals are weak/non-existent.
    """
    np.random.seed(42)
    n_samples = 1000
    X = (np.random.rand(n_samples, 5) > 0.5).astype(int)
    
    # 3-way XOR
    y = X[:, 0] ^ X[:, 1] ^ X[:, 2]
    
    # Model with interaction_depth=2 (triplets)
    clf = GuideTreeClassifier(
        max_depth=3,
        interaction_depth=2,
        significance_threshold=0.05,
        max_interaction_candidates=5
    )
    clf.fit(X, y)
    score = clf.score(X, y)
    print(f"3-way interaction score: {score}")
    
    # Baseline: depth=1 (only pairs)
    clf_base = GuideTreeClassifier(
        max_depth=3,
        interaction_depth=1,
        significance_threshold=0.05,
        max_interaction_candidates=5
    )
    clf_base.fit(X, y)
    score_base = clf_base.score(X, y)
    print(f"Pairwise interaction score: {score_base}")
    
    # Depth 2 should perform better on a 3-way interaction task
    assert score > score_base
    assert score > 0.9 # Should almost perfectly fit XOR with 1000 samples and no noise