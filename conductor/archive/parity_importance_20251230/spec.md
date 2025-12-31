# Spec: Scikit-Learn Parity and Variable Importance

## Overview
This track aims to complete the scikit-learn estimator interface for `pyguide` by implementing missing structural attributes, diagnostic methods, and variable importance scores. This ensures that `pyguide` models can be used as drop-in replacements in complex scikit-learn workflows (e.g., pipelines with feature selection).

## Goals
- **Full Parity:** Implement `n_leaves_`, `max_depth_`, `apply()`, and `decision_path()`.
- **Variable Importance:** Implement `feature_importances_` using GUIDE's importance scoring logic.
- **Consistency:** Ensure both `GuideTreeClassifier` and `GuideTreeRegressor` support these features identically.

## Key Components

### 1. Structural Attributes
- **`n_leaves_`**: Total number of leaf nodes in the fitted tree.
- **`max_depth_`**: The actual maximum depth of the tree (computed after fitting/pruning).

### 2. Variable Importance (`feature_importances_`)
- **Strategy:** GUIDE typically calculates importance based on the weighted impurity reduction across all nodes where a variable was chosen for splitting.
- **Alternative:** Incorporate the Chi-square/interaction statistics used during selection.
- **Requirement:** The values must be normalized to sum to 1.0.

### 3. Diagnostic Methods
- **`apply(X)`**: Returns the index of the leaf that each sample is predicted as.
- **`decision_path(X)`**: Returns the decision path of each sample in the tree (as a CSR sparse matrix).

## Success Criteria
- `model.n_leaves_` returns the correct count.
- `model.feature_importances_` has shape `(n_features,)` and sums to 1.0.
- `model.apply(X)` returns valid leaf indices.
- Standard scikit-learn utilities like `SelectFromModel` work with `pyguide` estimators.
