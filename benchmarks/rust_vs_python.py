import time
from unittest.mock import patch

from sklearn.datasets import load_diabetes, load_digits, load_iris
from tabulate import tabulate

from pyguide import (
    GuideRandomForestRegressor,
    GuideTreeClassifier,
    GuideTreeRegressor,
)


def benchmark_model(model_class, X, y, name, **params):
    # Benchmark Python
    with patch("pyguide.stats.HAS_RUST", False), \
         patch("pyguide.splitting.HAS_RUST", False): 
        
        start = time.perf_counter()
        clf_py = model_class(**params)
        clf_py.fit(X, y)
        py_time = time.perf_counter() - start
        
    # Benchmark Rust
    start = time.perf_counter()
    clf_rust = model_class(**params)
    clf_rust.fit(X, y)
    rust_time = time.perf_counter() - start
    
    speedup = py_time / rust_time if rust_time > 0 else 0
    return py_time, rust_time, speedup

def run_benchmarks():
    data_tasks = [
        ("Iris (Clf)", load_iris, GuideTreeClassifier, {"max_depth": 3}),
        ("Digits (Clf)", load_digits, GuideTreeClassifier, {"max_depth": 5}),
        ("Diabetes (Reg)", load_diabetes, GuideTreeRegressor, {"max_depth": 5}),
        ("Diabetes (RF)", load_diabetes, GuideRandomForestRegressor, {"n_estimators": 10, "max_depth": 5}),
    ]
    
    results = []
    print("Running Python vs Rust Benchmarks...")
    
    for label, data_func, model_class, params in data_tasks:
        X, y = data_func(return_X_y=True)
        py_t, rust_t, speedup = benchmark_model(model_class, X, y, label, **params)
        results.append([label, f"{py_t:.4f}s", f"{rust_t:.4f}s", f"{speedup:.2f}x"])
        print(f"Completed {label}")

    print("\nBenchmark Results:")
    headers = ["Task", "Python Time", "Rust Time", "Speedup"]
    print(tabulate(results, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    run_benchmarks()
