# Spec: Pruning, Visualization, and Optimization

## Overview
This track focuses on making `pyguide` production-ready by addressing three key areas:
1.  **Pruning:** Implementing minimal Cost-Complexity Pruning (CCP) to prevent overfitting and align with scikit-learn's pruning API.
2.  **Visualization:** Enabling compatibility with `sklearn.tree.plot_tree` and `export_graphviz`. This likely requires mimicking the `tree_` attribute structure of scikit-learn or providing a compatible exporter.
3.  **Optimization:** Profiling and optimizing the critical path (Chi-square tests and split searching) to ensure competitive training times.

## Goals
- **Pruning Support:** Add `ccp_alpha` parameter and implement pruning logic.
- **Visualization Support:** Allow users to visualize `pyguide` trees using standard tools.
- **Performance:** Reduce `fit` time for medium-sized datasets (e.g., 10k samples, 50 features) by at least 20%.

## Key Components

### 1. Pruning (Cost-Complexity)
- **Attribute:** `ccp_alpha` (non-negative float).
- **Mechanism:**
  - Calculate the effective alpha for each node.
  - Recursively prune subtrees where the cost-complexity of the subtree is greater than the cost-complexity of the node as a leaf.
- **API:** `cost_complexity_pruning_path` method (optional but good for parity).

### 2. Visualization Compatibility
- **Strategy:** Instead of fully mimicking the internal Cython `Tree` structure (which is complex and private), we will implement a `export_graphviz` compatible method or property.
- **Alternative:** Implement a `to_sklearn_tree()` converter that builds a dummy `sklearn.tree.Tree` object populated with our nodes. This allows `plot_tree` to work natively.

### 3. Optimization
- **Profiling:** Use `cProfile` to identify bottlenecks.
- **Vectorization:** Ensure `select_split_variable` and `find_best_split` use pure numpy operations where possible.
- **Caching:** Cache Chi-square results if features are reused (though GUIDE usually recalculates on subsets).

## Success Criteria
- `GuideTreeClassifier(ccp_alpha=0.01)` produces a smaller tree than `ccp_alpha=0.0`.
- `sklearn.tree.plot_tree(model)` (or a wrapper) successfully renders a GUIDE tree.
- Benchmarks show improved `fit` times compared to the current baseline.
