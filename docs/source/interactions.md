# Interaction Detection in pyguide

One of the most powerful features of GUIDE is its explicit support for identifying feature interactions. This allows the model to capture relationships that are invisible to standard CART trees.

## What is an Interaction?

An interaction occurs when the effect of one feature on the target depends on the value of another feature. A classic example is the **XOR problem**:
- If $X_0=0$ and $X_1=0$, $y=0$
- If $X_0=0$ and $X_1=1$, $y=1$
- If $X_0=1$ and $X_1=0$, $y=1$
- If $X_0=1$ and $X_1=1$, $y=0$

In this case, neither $X_0$ nor $X_1$ is individually predictive of $y$. A standard decision tree might fail to split at the root because no single variable reduces impurity.

## The GUIDE Search Strategy

If no single variable is individually significant (based on `significance_threshold`), `pyguide` falls back to interaction detection:

1.  **Candidate Selection:** All features (or a subset restricted by `interaction_features`) are considered.
2.  **Pairwise/Higher-order Tests:** The algorithm tests combinations of variables (size determined by `interaction_depth + 1`).
3.  **Chi-square Interaction Test:** For each group, it bins the variables and performs a multi-way Chi-square test against the target.
4.  **Variable Selection from Interaction:** If a significant interaction is found, GUIDE must still choose **one** variable from the group to split on.
    - **For Pairs:** It uses a "look-ahead" strategy, choosing the variable that yields the highest gain after a potential second-level split.
    - **For Triplets+:** It chooses the variable with the lowest individual p-value.

---

## Tuning Interactions

### `interaction_depth`
- `interaction_depth=0`: Disable interaction search. Fastest training.
- `interaction_depth=1` (Default): Search for pairwise interactions.
- `interaction_depth=2`: Search for triplets. Use with caution on large datasets.

### `max_interaction_candidates`
On high-dimensional datasets (e.g., 500+ features), searching all $\binom{500}{2} = 124,750$ pairs is extremely slow. 

Setting `max_interaction_candidates=10` tells the model to:
1.  Rank all features individually.
2.  Only test interactions among the **top 10** features.

This can result in speedups of **100x to 1000x** with minimal impact on accuracy, as interacting variables usually show *some* individual signal.

## Example: Forcing Interaction Search

If you know your data has strong interactions but also a lot of noise, you might want to set a very low `significance_threshold` (even `0.0`) to force the model into interaction detection mode at every node.

```python
clf = GuideTreeClassifier(
    interaction_depth=1,
    significance_threshold=0.0  # Force interaction fallback
)
```
