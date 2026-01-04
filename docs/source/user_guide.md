# User Guide: The GUIDE Algorithm

`pyguide` is a Python implementation of the **GUIDE** (Generalized, Unbiased, Interaction Detection and Estimation) algorithm, originally developed by Wei-Yin Loh.

## Core Philosophy: The Two-Step Process

Most decision tree algorithms (like CART) use an exhaustive search to find the best split. At each node, they iterate through every possible variable and every possible split point to find the one that minimizes impurity (like Gini or MSE).

**The Problem with CART:**
Exhaustive search is biased towards **high-cardinality features** (features with many unique values). A feature like "Customer ID" or a noisy continuous variable provides many more opportunities for a lucky split that reduces impurity by chance, even if it has no true predictive power.

**The GUIDE Solution:**
GUIDE separates variable selection from split point optimization:
1.  **Step 1: Variable Selection:** Use statistical tests (Chi-square) to rank variables based on their relationship with the target, independent of any split points.
2.  **Step 2: Split Point Optimization:** Only after the best variable is chosen does the algorithm search for the optimal threshold.

---

## How it Works

### 1. Variable Selection (Curvature Tests)
At each node, GUIDE performs a Chi-square test of independence between each feature and the target.
- **For Classification:** It creates a contingency table of feature bins vs. class labels.
- **For Regression:** It calculates residuals from the node mean and performs a Chi-square test on the *signs* of the residuals.

The variable with the lowest p-value (most significant relationship) is selected.

### 2. Interaction Detection
If no single variable is significant (p-value > `significance_threshold`), GUIDE enters its interaction detection mode. It tests pairs (or higher-order groups) of variables to see if their *combination* is significant.

### 3. Split Point Optimization
Once a variable is selected, GUIDE finds the split point $s$ that minimizes the impurity of the resulting children. For numerical variables, it searches all unique values. For categorical variables, it uses an optimized $O(K)$ search based on the target mean/distribution.

---

## Key Parameters

- `significance_threshold`: The alpha level for the Chi-square tests. Higher values allow more splits; lower values make the tree more conservative.
- `interaction_depth`: Controls the maximum order of interaction search (1 for pairs, 2 for triplets).
- `max_interaction_candidates`: Limits interaction search to the top $K$ features to improve performance on wide datasets.

`pyguide` trees naturally handle missing values by routing observations based on the impurity reduction of the non-missing data. No imputation is required.

## Variable Importance

`pyguide` provides three distinct ways to measure the importance of predictor variables, each offering a different perspective on the model.

### 1. Standard Impurity Importance (`feature_importances_`)

This is the standard scikit-learn compatible importance. It measures the total weighted reduction of the impurity criterion (Gini index or SSE) brought by each feature. 

**Note:** In GUIDE, only one variable is used for the physical split at each node. This metric only credits that single variable.

### 2. Interaction-Aware Importance (`interaction_importances_`)

When a split is identified via interaction detection (e.g., between $X_i$ and $X_j$), both variables contribute to the discovery of the split. This metric distributes the impurity reduction of such a split equally among all members of the interaction group.

This is often more robust than standard importance for detecting variables that primarily act through interactions.

### 3. GUIDE Importance Scores (`guide_importances_`)

Following the methodology in **Loh & Zhou (2021)**, these scores are calculated by summing the statistical significance (Chi-square statistics) of all features across all intermediate nodes in the tree.

$v(X_k) = \sum_{t} \sqrt{n_t} \chi_1^2(k, t)$

Unlike impurity-based metrics, GUIDE importance:
- Is **unbiased**: It doesn't favor variables with many unique values.
- Is **associative**: It captures the potential of every variable at every node, even if the variable was not chosen for the split.
- Uses **raw statistics**: Does not depend on the specific split point or threshold chosen.

### 4. Strict GUIDE Variable Importance (`compute_guide_importance`)

The `compute_guide_importance` method implements the "Strict" version of the algorithm described in **Loh & Zhou (2021)**. This is the recommended method for feature selection and importance ranking.

**Key Features:**
- **Standalone API:** Can be called on an unfitted estimator: `scores = GuideTreeClassifier().compute_guide_importance(X, y)`.
- **Auxiliary Trees:** Grows a short (depth 4) unpruned tree to capture robust associations without overfitting.
- **Bias Correction:** Performs 300 permutations of the target variable to normalize scores. A score of **1.0** represents the expected importance of a random noise variable.
- **Strict Interaction Capture:** Automatically incorporates interaction signals into the scores of the involved features.

```python
# Standard way to get unbiased, calibrated importance
clf = GuideTreeClassifier(interaction_depth=1)
scores = clf.compute_guide_importance(X, y)

# scores[i] > 1.0 indicates a variable more important than noise.
```

## Visualization

