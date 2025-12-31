import time
import cProfile
import pstats
import io
import numpy as np
from sklearn.datasets import make_classification, make_regression
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def run_benchmark():
    print("--- GUIDE Performance Benchmark ---")
    
    # 1. Classification Benchmark
    X_clf, y_clf = make_classification(n_samples=2000, n_features=20, n_informative=5, n_redundant=2, random_state=42)
    clf = GuideTreeClassifier(max_depth=5)
    
    start = time.time()
    clf.fit(X_clf, y_clf)
    duration_clf = time.time() - start
    print(f"Classification Fit (2000 samples, 20 features): {duration_clf:.4f}s")
    
    # 2. Regression Benchmark
    X_reg, y_reg = make_regression(n_samples=2000, n_features=20, n_informative=5, noise=0.1, random_state=42)
    reg = GuideTreeRegressor(max_depth=5)
    
    start = time.time()
    reg.fit(X_reg, y_reg)
    duration_reg = time.time() - start
    print(f"Regression Fit (2000 samples, 20 features): {duration_reg:.4f}s")
    
    return duration_clf, duration_reg

def profile_fit():
    print("\n--- Profiling GuideTreeClassifier.fit ---")
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    clf = GuideTreeClassifier(max_depth=3)
    
    pr = cProfile.Profile()
    pr.enable()
    clf.fit(X, y)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == "__main__":
    run_benchmark()
    profile_fit()
