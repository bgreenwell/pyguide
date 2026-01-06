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

## Gradient Boosting

The `GuideGradientBoostingClassifier` and `GuideGradientBoostingRegressor` classes implement Gradient Boosting machines using GUIDE trees as the base weak learners.

### Overview

Gradient Boosting builds an additive model in a forward stage-wise fashion. At each stage, a regression tree is fit on the negative gradient of the loss function (e.g., deviance for classification, least squares for regression).

**Why use GUIDE for Boosting?**
Standard GBM implementations (like XGBoost, LightGBM, scikit-learn) use CART-like greedy splitting. `pyguide`'s implementation ensures that **every split in every tree is unbiased**. This is particularly valuable when your dataset contains a mix of categorical variables with varying cardinality or continuous variables, where standard boosting might over-select high-cardinality features.

### Usage

```python
from pyguide import GuideGradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a Gradient Boosting Regressor
gbm = GuideGradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,  # Stochastic Gradient Boosting
    random_state=42
)
gbm.fit(X_train, y_train)

print(f"R2 Score: {gbm.score(X_test, y_test):.4f}")
```

### Parameters

- `n_estimators`: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
- `learning_rate`: learning rate shrinks the contribution of each tree by `learning_rate`.
- `max_depth`: maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
- `subsample`: The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. `subsample` interacts with the parameter `n_estimators`. Choosing `subsample < 1.0` leads to a reduction of variance and an increase in bias.
- `significance_threshold`, `interaction_depth`, `max_interaction_candidates`: Passed to the underlying GUIDE trees.

### Current Limitations
- **Binary Classification Only:** `GuideGradientBoostingClassifier` currently supports only binary classification (Log Loss).
- **Speed:** While optimized with Rust kernels, the Python-driven boosting loop is slower than highly optimized C++ libraries like XGBoost or LightGBM. It is prioritized for statistical rigor and unbiased selection rather than raw speed.
