# Spec: Benchmarking & Optimization

## Overview
This track aims to rigorously evaluate the performance of `pyguide` against standard implementations (like scikit-learn's `DecisionTreeClassifier`) and identify key areas for optimization. The goal is to make `pyguide` competitive for medium-sized datasets even while running in pure Python, before eventually moving to compiled extensions.

## Goals
- **Benchmark Suite:** Create a reproducible benchmark script (`benchmarks/main_benchmark.py`) using standard datasets (e.g., OpenML).
- **Profiling:** Use `cProfile` and `snakeviz` (or similar) to identify hotspots in the `fit` and `predict` loops.
- **Optimization:** Implement algorithmic or NumPy-based optimizations to reduce runtime.
- **Completeness:** Implement the missing `cost_complexity_pruning_path` method to enable full pruning analysis.

## Key Components

### 1. Benchmark Suite
- **Datasets:**
  - Classification: Digits (easy), Covertype (large), Synthetic (interactions).
  - Regression: California Housing, Synthetic (interactions).
- **Metrics:** Training time, Inference time, Accuracy/R2, Peak Memory.
- **Comparison:** `pyguide` vs `sklearn.tree.DecisionTreeClassifier`.

### 2. Feature Completeness
- Implement `GuideTreeClassifier.cost_complexity_pruning_path` and `GuideTreeRegressor.cost_complexity_pruning_path`.
- This involves calculating the effective alpha for every node and sorting them to produce the pruning sequence.

### 3. Optimization Targets
- **Vectorization:** Ensure all Chi-square and SSE calculations are fully vectorized.
- **Memory Management:** Avoid unnecessary copying of X/y at each split.
- **Interaction Search:** Refine the candidate filtering to be even more efficient.

## Success Criteria
- Benchmark script produces a clear report (CSV/Markdown).
- `cost_complexity_pruning_path` is fully implemented and tested.
- At least one significant bottleneck is identified and optimized (e.g., >10% speedup).
