# pyguide

[![CI](https://github.com/bgreenwell/pyguide/actions/workflows/ci.yml/badge.svg)](https://github.com/bgreenwell/pyguide/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python Version](https://img.shields.io/pypi/pyversions/pyguide.svg)](https://pypi.org/project/pyguide/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source, scikit-learn compatible Python implementation of the **GUIDE** (Generalized, Unbiased, Interaction Detection and Estimation) decision tree algorithm.

## Why GUIDE?

Standard decision tree implementations (like CART or scikit-learn's `DecisionTreeClassifier`) suffer from **variable selection bias**: they tend to favor features with many unique values (high cardinality), even if they are noise. They also often fail to detect complex feature interactions unless they are deep.

`pyguide` fills this gap by:
- **Unbiased Selection:** Separating variable selection from split optimization using Chi-square tests.
- **Built-in Interaction Detection:** Explicitly searching for multi-variable interactions (pairs, triplets, etc.).
- **Scikit-learn Compatibility:** Full parity with the scikit-learn estimator API, including structural attributes (`n_leaves_`, `max_depth_`) and diagnostic methods (`apply`, `decision_path`).
- **High Performance:** Core algorithms are implemented in Rust for speed and efficiency.

## Key Features

- **Unbiased Variable Selection:** Prevents bias towards high-cardinality features.
- **Strict GUIDE Importance:** Implementation of Loh & Zhou (2021) unbiased importance scores with bias correction.
- **Tree Ensembles:** Random Forest implementation using GUIDE trees for improved accuracy and robustness.
- **Advanced Interaction Detection:** Configure `interaction_depth` to find complex relationships.
- **Scalable Search:** Use `max_interaction_candidates` to speed up training on high-dimensional data by orders of magnitude.
- **Handling Missing Values:** Native support for NaNs using impurity-based routing.
- **Pruning:** Minimal Cost-Complexity Pruning support via `ccp_alpha`.
- **Visualization:** Integrated with scikit-learn's `plot_tree` for easy interpretation.

## Installation

```bash
pip install pyguide  # Note: Replace with actual install command when published
```

### Building from Source

To build from source, you will need the Rust toolchain installed (e.g., via `rustup`).

```bash
# Clone the repository
git clone https://github.com/bgreenwell/pyguide.git
cd pyguide

# Install maturin
pip install maturin

# Build and install
maturin develop --release
```

## Quick Start

### Classification (Gradient Boosting)

```python
from pyguide import GuideGradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = GuideGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

### Classification (Random Forest)

```python
from pyguide import GuideRandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = GuideRandomForestClassifier(n_estimators=100, max_features="sqrt")
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

### Classification (Single Tree)

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

### Strict Variable Importance

```python
from pyguide import GuideTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Calculate unbiased importance scores (Loh & Zhou, 2021)
clf = GuideTreeClassifier(interaction_depth=1)
scores = clf.compute_guide_importance(X, y, bias_correction=True)

# scores[i] > 1.0 indicates importance greater than noise
print("Feature Importances:", scores)
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

## Benchmarks

Some (very) preliminary benchmarks are shown below:

```bash
uv run python benchmarks/main_benchmark.py
--- Benchmarking Classifier: Iris (150 samples, 4 features) ---
| Model                       |   Train Time (s) |   Test Time (s) |   Accuracy |
|:----------------------------|-----------------:|----------------:|-----------:|
| sklearn (CART)              |      0.000684977 |     0.000187159 |          1 |
| pyguide (GUIDE)             |      0.00245309  |     0.000161886 |          1 |
| sklearn (Random Forest)     |      0.00682116  |     0.000476837 |          1 |
| pyguide (Random Forest)     |      0.042968    |     0.00081706  |          1 |
| sklearn (Gradient Boosting) |      0.0229151   |     0.000649929 |          1 |
| pyguide (Gradient Boosting) |      0           |     0           |          0 |

--- Benchmarking Classifier: Digits (1797 samples, 64 features) ---
| Model                       |   Train Time (s) |   Test Time (s) |   Accuracy |
|:----------------------------|-----------------:|----------------:|-----------:|
| sklearn (CART)              |       0.00626874 |     0.000355721 |   0.663889 |
| pyguide (GUIDE)             |       0.0897272  |     0.000550985 |   0.708333 |
| sklearn (Random Forest)     |       0.0162752  |     0.000712872 |   0.938889 |
| pyguide (Random Forest)     |       0.467017   |     0.00838208  |   0.925    |
| sklearn (Gradient Boosting) |       0.613889   |     0.00105691  |   0.930556 |
| pyguide (Gradient Boosting) |       0          |     0           |   0        |

--- Benchmarking Classifier: Breast Cancer (569 samples, 30 features) ---
| Model                       |   Train Time (s) |   Test Time (s) |   Accuracy |
|:----------------------------|-----------------:|----------------:|-----------:|
| sklearn (CART)              |       0.00244403 |     0.000133991 |   0.947368 |
| pyguide (GUIDE)             |       0.024884   |     0.000295877 |   0.973684 |
| sklearn (Random Forest)     |       0.0109897  |     0.000590086 |   0.964912 |
| pyguide (Random Forest)     |       0.376772   |     0.00233197  |   0.964912 |
| sklearn (Gradient Boosting) |       0.0363209  |     0.000236988 |   0.95614  |
| pyguide (Gradient Boosting) |       3.26811    |     0.00185609  |   0.95614  |

--- Benchmarking Regressor: Diabetes (442 samples, 10 features) ---
| Model                       |   Train Time (s) |   Test Time (s) |   R2 Score |
|:----------------------------|-----------------:|----------------:|-----------:|
| sklearn (CART)              |      0.000643015 |     0.000144958 |   0.334482 |
| pyguide (GUIDE)             |      0.0127881   |     0.000193119 |   0.314395 |
| sklearn (Random Forest)     |      0.0108769   |     0.000529289 |   0.429393 |
| pyguide (Random Forest)     |      1.3244      |     0.002249    |   0.46877  |
| sklearn (Gradient Boosting) |      0.00865674  |     0.000172138 |   0.450993 |
| pyguide (Gradient Boosting) |      0.866097    |     0.00140405  |   0.484351 |
```
**Note:** The gradient boosting results for the iris and digits data sets are currently 0 because multivariate outcomes are currently not supported.

## References

- Loh, W.-Y. (2002). *Regression trees with unbiased variable selection and interaction detection*. Statistica Sinica, 361-386.
- Loh, W.-Y. (2009). *Improving the precision of classification trees*. Annals of Applied Statistics, 3(4), 1710-1737.
- Loh, W.-Y. and Zhou, P. (2021). *Variable Importance Scores*. Journal of Data Science, 19(4), 569-592.

## Roadmap

- [x] **Tree Ensembles:** Random Forest wrappers using GUIDE as the base learner.
- [x] **Variable Importance Mode:** Enhanced diagnostics and standalone importance scores (Strict GUIDE).
- [x] **Gradient Boosting:** Boosting wrappers using GUIDE as the base learner.
- [x] **Performance Optimization:** Porting core splitting and selection logic to Rust/C for production-scale performance.
- [ ] **Extended Interaction Support:** Automated search for arbitrary-depth interactions with better pruning.
