# Tree Ensembles

`pyguide` provides Random Forest ensemble models that leverage the unbiased splitting power of GUIDE trees. By combining multiple trees trained on bootstrap samples with random feature selection, these ensembles often achieve higher predictive accuracy than single trees while reducing overfitting.

## Random Forest

The `GuideRandomForestClassifier` and `GuideRandomForestRegressor` classes are drop-in replacements for their scikit-learn counterparts, but use `GuideTreeClassifier` and `GuideTreeRegressor` as the base estimators.

### Key Benefits

1.  **Unbiased Base Learners:** Unlike standard Random Forests which use CART trees (biased towards high-cardinality features), `GuideRandomForest` uses unbiased GUIDE trees. This reduces the bias of the ensemble as a whole.
2.  **Interaction Detection:** Each tree in the forest can still detect interactions if configured (though typically Random Forests disable interaction search for speed, you can enable it).
3.  **Familiar API:** Uses scikit-learn's `Bagging` infrastructure, so all standard parameters (`n_estimators`, `max_samples`, `n_jobs`) work as expected.

### Usage

```python
from pyguide import GuideRandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a Random Forest with 100 trees
rf = GuideRandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features="sqrt", # Standard RF heuristic
    n_jobs=-1,           # Use all cores
    random_state=42
)
rf.fit(X_train, y_train)

print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
```

### Parameters

- `n_estimators`: Number of trees (default=100).
- `max_features`: Number of features to consider at each split. Defaults to "sqrt" for classifiers and 1.0 for regressors.
- `bootstrap`: Whether to use bootstrap sampling (default=True).
- `significance_threshold`: Passed to the underlying GUIDE trees.
- `interaction_depth`: Passed to the underlying GUIDE trees (default=1).

## Gradient Boosting (Planned)

Support for Gradient Boosting using GUIDE trees as weak learners is planned for a future release.
