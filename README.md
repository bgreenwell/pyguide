# pyguide

[![CI](https://github.com/bgreenwell/pyguide/actions/workflows/ci.yml/badge.svg)](https://github.com/bgreenwell/pyguide/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python Version](https://img.shields.io/pypi/pyversions/pyguide.svg)](https://pypi.org/project/pyguide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Disclaimer:** `pyguide` is currently a **work in progress** and is not yet production-ready. The current implementation is written in pure Python/NumPy for clarity and algorithm verification. Future versions will identify performance bottlenecks and replace them with optimized, compiled code (Rust or C).

A high-quality, scikit-learn compatible Python implementation of the **GUIDE** (Generalized, Unbiased, Interaction Detection and Estimation) algorithm.

## Why GUIDE?

Standard decision tree implementations (like CART or scikit-learn's `DecisionTreeClassifier`) suffer from **variable selection bias**: they tend to favor features with many unique values (high cardinality), even if they are noise. They also often fail to detect complex feature interactions unless they are deep.

`pyguide` fills this gap by:
- **Unbiased Selection:** Separating variable selection from split optimization using Chi-square tests.
- **Built-in Interaction Detection:** Explicitly searching for multi-variable interactions (pairs, triplets, etc.).
- **Scikit-learn Compatibility:** Full parity with the scikit-learn estimator API, including structural attributes (`n_leaves_`, `max_depth_`) and diagnostic methods (`apply`, `decision_path`).

## Key Features

- **Unbiased Variable Selection:** Prevents bias towards high-cardinality features.
- **Advanced Interaction Detection:** Configure `interaction_depth` to find complex relationships.
- **Scalable Search:** Use `max_interaction_candidates` to speed up training on high-dimensional data by orders of magnitude.
- **Handling Missing Values:** Native support for NaNs using impurity-based routing.
- **Pruning:** Minimal Cost-Complexity Pruning support via `ccp_alpha`.
- **Visualization:** Integrated with scikit-learn's `plot_tree` for easy interpretation.

## Installation

```bash
pip install pyguide  # Note: Replace with actual install command when published
```

## Quick Start

### Classification

```python
from pyguide import GuideTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = GuideTreeClassifier(max_depth=3, interaction_depth=1)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Number of leaves: {clf.n_leaves_}")
```

### Regression

```python
from pyguide import GuideTreeRegressor
import numpy as np

# XOR-like interaction problem
X = np.random.rand(500, 2)
y = (X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)
y = y.astype(float) + np.random.normal(0, 0.1, 500)

reg = GuideTreeRegressor(max_depth=2, interaction_depth=1)
reg.fit(X, y)

print(f"R2 Score: {reg.score(X, y):.4f}")
```

## Advanced Usage

### Scalable Interaction Search

For data sets with hundreds or thousands of features, exhaustive interaction search is slow. Use `max_interaction_candidates` to restrict the search to the most promising features:

```python
clf = GuideTreeClassifier(
    interaction_depth=1,
    max_interaction_candidates=10,  # Only test interactions among top 10 features
    significance_threshold=0.05
)
```

### Visualization

```python
import matplotlib.pyplot as plt
from pyguide.visualization import plot_guide_tree

clf.fit(X, y)
plt.figure(figsize=(12, 8))
plot_guide_tree(clf, feature_names=iris.feature_names)
plt.show()
```

## References

- Loh, W.-Y. (2002). *Regression trees with unbiased variable selection and interaction detection*. Statistica Sinica, 361-386.
- Loh, W.-Y. (2009). *Improving the precision of classification trees*. Annals of Applied Statistics, 3(4), 1710-1737.

## Roadmap

- [ ] **Tree Ensembles:** Random Forest and Gradient Boosting wrappers using GUIDE as the base learner.
- [ ] **Variable Importance Mode:** Enhanced diagnostics and standalone importance scores.
- [ ] **Performance Optimization:** Porting core splitting and selection logic to Rust/C for production-scale performance.
- [ ] **Extended Interaction Support:** Automated search for arbitrary-depth interactions with better pruning.
