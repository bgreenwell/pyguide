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

The `compute_guide_importance` method implements the "Strict" variable importance algorithm as detailed in **Loh & Zhou (2021)**. This approach is designed to be the gold standard for unbiased feature ranking and selection.

#### Why "Strict"?
Standard impurity-based importance scores (like those in Random Forest or CART) often suffer from two main issues:
1.  **Bias towards high-cardinality:** Variables with many unique values or categories can appear artificially important.
2.  **Lack of a null distribution:** It is difficult to know if a score of "0.05" is significant or just noise.

Strict GUIDE Importance addresses these by using:
- **Unbiased Chi-Square Statistics:** Importance is derived from the statistical strength of association, not impurity reduction.
- **Normalization via Permutation:** Scores are calibrated such that a score of **1.0** represents the expected importance of a random noise variable.

#### Algorithm Details
1.  **Auxiliary Tree:** An unpruned, depth-limited (default `max_depth=4`) GUIDE tree is grown on the data.
2.  **Raw Importance Calculation:** At each node $t$, the association between every feature $X_k$ and the target $Y$ is measured using a 1-degree-of-freedom Chi-square statistic, $\chi^2_1(k, t)$. The raw importance $v(X_k)$ is the sum of these statistics weighted by the square root of the sample size at each node:
    $$v(X_k) = \sum_{t} \sqrt{n_t} \chi^2_1(k, t)$$
3.  **Interaction Handling:** If an interaction is detected at a node, the high Chi-square statistic of the interaction is attributed to all participating variables, ensuring associative signals are not lost.
4.  **Bias Correction (Normalization):** The target variable $Y$ is permuted $B$ times (default 300). For each permutation, the raw importance scores are recalculated. The final **Strict Importance Score** ($VI$) is the ratio of the original score to the average score from the permuted runs:
    $$VI(X_k) = \frac{v(X_k)}{\bar{v}_{perm}(X_k)}$$

#### Interpretation
- **$VI(X_k) \approx 1.0$:** The variable behaves like noise.
- **$VI(X_k) \gg 1.0$:** The variable has a significant association with the target.
- **$VI(X_k) < 1.0$:** The variable performs worse than random noise (rare, but possible due to sampling variance).

#### Usage Example

```python
from pyguide import GuideTreeClassifier

# Initialize a clean estimator
clf = GuideTreeClassifier(interaction_depth=1)

# Compute Strict Importance (computationally intensive due to permutations)
vi_scores = clf.compute_guide_importance(
    X, y,
    max_depth=4,            # Recommended depth from paper
    bias_correction=True,   # Enable permutation normalization
    n_permutations=300      # Recommended number of permutations
)

# Filter for significant features
significant_features = [
    feature_names[i] 
    for i, score in enumerate(vi_scores) 
    if score > 1.2  # A conservative threshold above noise
]
```

**Reference:**
Loh, W.-Y. and Zhou, P. (2021). "Variable Importance Scores." *Journal of Data Science*, 19(4), 569-592.

## Visualization

