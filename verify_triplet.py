import numpy as np
from pyguide import GuideTreeClassifier

def verify():
    np.random.seed(42)
    n_samples = 1000
    X = (np.random.rand(n_samples, 5) > 0.5).astype(int)
    # 3-way XOR: y = X0 ^ X1 ^ X2
    y = X[:, 0] ^ X[:, 1] ^ X[:, 2]
    
    print("Testing 3-way XOR (interaction_depth=2)...")
    clf = GuideTreeClassifier(max_depth=3, interaction_depth=2, significance_threshold=0.05)
    clf.fit(X, y)
    score = clf.score(X, y)
    print(f"Depth 2 Accuracy: {score}")
    
    print("\nTesting 3-way XOR (interaction_depth=1)...")
    clf_base = GuideTreeClassifier(max_depth=3, interaction_depth=1, significance_threshold=0.05)
    clf_base.fit(X, y)
    score_base = clf_base.score(X, y)
    print(f"Depth 1 Accuracy: {score_base}")
    
    if score > 0.9 and score > score_base:
        print("\nSUCCESS: Higher-order interactions correctly identified and used!")
    else:
        print("\nFAILURE: Higher-order interactions NOT working as expected.")

if __name__ == "__main__":
    verify()
