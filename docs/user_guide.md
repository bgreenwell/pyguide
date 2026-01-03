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

## Handling Missing Values (NaNs)
GUIDE has a native, elegant way to handle missing values. During variable selection, "missingness" is treated as a separate category in the Chi-square tests. During splitting, if a variable has NaNs, GUIDE evaluates whether sending them to the left or right child results in lower total impurity.
