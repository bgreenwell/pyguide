# Spec: Strict GUIDE Variable Importance

## Overview
This track implements the formal variable importance scoring algorithm defined in "Variable Importance Scores" (Loh & Zhou, 2021). While the current implementation provides scores based on the predictive model, the formal GUIDE method uses an auxiliary "short" tree and permutation tests to produce unbiased, debiased importance scores.

## Goals
- **Method Implementation:** Add `compute_guide_importance()` to `GuideTreeClassifier` and `GuideTreeRegressor`.
- **Auxiliary Tree Logic:** Growth of a shallow (default 4-level) unpruned tree specifically for importance.
- **Bias Correction:** Permutation-based estimation of importance under the null hypothesis to normalize scores.
- **Thresholding (Bonus):** Support for the threshold scores mentioned in Section 6 of the paper.

## Key Components

### 1. `compute_guide_importance` Method
- **Signature:** `compute_guide_importance(X, y, max_depth=4, bias_correction=True, n_permutations=100, random_state=None)`
- **Logic:**
  1. Calculate unadjusted importance $v(X_k)$ by fitting a tree with depth `max_depth` and summing $\sqrt{n_t} \chi_1^2(k, t)$ over intermediate nodes.
  2. If `bias_correction` is True:
     - For $b$ in $1..n\_permutations$:
       - Permute $y$.
       - Fit auxiliary tree and calculate $v^*(X_k)$.
     - Compute $\bar{v}(X_k)$ as the average of $v^*(X_k)$.
     - Final score $VI(X_k) = v(X_k) / \bar{v}(X_k)$.

### 2. Efficiency Considerations
- Bias correction is $O(B \cdot \text{fit_time(shallow_tree)})$. 
- Fitting a 4-level tree is very fast, but 100+ permutations will still take noticeable time.
- Implementation should reuse optimized `_fit_node` logic but allow bypassing prediction-related overhead (e.g., probability calculation) if possible.

## Success Criteria
- Matches the logic described in Section 2 and Equations (1) and (2) of Loh & Zhou (2021).
- Correctly identifies important features in XOR and other complex interaction datasets.
- Normalized scores ($VI(X_k)$) are close to 1.0 for noise variables.
