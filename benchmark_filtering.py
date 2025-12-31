import time
import numpy as np
from pyguide import GuideTreeClassifier

def benchmark():
    # Create a dataset with many features (100) and some interaction
    np.random.seed(42)
    n_samples = 500
    n_features = 500
    X = np.random.rand(n_samples, n_features)
    
    # Interaction between X0 and X1
    y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
    y = y.astype(int)
    
    print(f"Benchmarking with {n_samples} samples and {n_features} features...")
    
    # 1. Baseline: No filtering
    start = time.time()
    clf_base = GuideTreeClassifier(max_depth=2, interaction_depth=1, significance_threshold=0.0)
    clf_base.fit(X, y)
    end = time.time()
    time_base = end - start
    print(f"Baseline time: {time_base:.4f}s")
    
    # 2. Filtered: Top 10 candidates
    start = time.time()
    clf_filtered = GuideTreeClassifier(max_depth=2, interaction_depth=1, max_interaction_candidates=10, significance_threshold=0.0)
    clf_filtered.fit(X, y)
    end = time.time()
    time_filtered = end - start
    print(f"Filtered time: {time_filtered:.4f}s")
    
    print(f"Speedup: {time_base / time_filtered:.2f}x")
    
if __name__ == "__main__":
    benchmark()
