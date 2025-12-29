This is an ambitious and valuable project. Implementing GUIDE (Generalized, Unbiased, Interaction Detection and Estimation) for `scikit-learn` fills a significant gap: standard Decision Trees (CART) in sklearn are prone to **variable selection bias** (favoring features with more unique values) and generally lack built-in **interaction detection**.

Based on the uploaded papers, here is a comprehensive implementation plan.

### **Core Philosophy: The "GUIDE" Difference**

Unlike CART (which greedily searches every split point of every variable to maximize impurity reduction), GUIDE separates the process into two distinct steps to ensure unbiasedness:

1. **Variable Selection:** Use statistical tests (Chi-square) to rank variables and select the best one.
2. **Split Point Selection:** Once the variable is chosen, find the best cut-point (minimizing Gini/SSE) *only* for that variable.

---

### **Phase 1: Architecture & Scikit-Learn Compliance**

Your implementation should inherit from `BaseEstimator` and `ClassifierMixin` (or `RegressorMixin`).

**Class Structure:**

```python
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GuideTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, 
                 significance_threshold=0.05, interaction_depth=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.significance_threshold = significance_threshold
        # GUIDE specific: Control how deep to look for interactions
        self.interaction_depth = interaction_depth 
    
    def fit(self, X, y):
        # Implementation of the GUIDE growing algorithm
        pass

```

---

### **Phase 2: The Splitting Mechanism (The Hard Part)**

This is where GUIDE diverges from CART. You need to implement the statistical testing framework defined in Loh (2002) and Loh (2009).

#### **1. Residual Definition (The "Target")**

GUIDE transforms the problem into a classification problem to apply Chi-square tests, even for regression.

* **For Classification:** The target  is already categorical. You work directly with the class labels.


* **For Regression:** 1. Fit a constant (mean) to the node.
2. Calculate residuals: .
3. Create a temporary "class" variable . For example,  if  and  otherwise.
4. You now perform tests between predictor  and temporary target .



#### **2. Variable Selection (Unbiased)**

You need a helper function `select_split_variable(X, Z)` that iterates through all features and returns the index of the best one.

* **Handling Continuous Variables:** GUIDE does *not* test every cut-point during selection. Instead, it discretizes continuous variables into groups (usually 3 or 4 quartiles) to form a contingency table against .


* *Implementation:* Use `pandas.qcut` or `numpy.percentile` to bin continuous features on the fly inside the node.


* 
**The Test:** Compute the Pearson Chi-square statistic () for the contingency table of  vs .


* **Correction:** Convert the  value to a probability or score. GUIDE often uses the Wilson-Hilferty approximation to normalize these values.


* **Result:** The variable with the most significant p-value (smallest) is selected.

#### **3. Interaction Detection**

If the main effects (univariate tests above) are weak, GUIDE looks for interactions.

* **Logic:** If no main effect p-value is below `significance_threshold`, run interaction tests.
* 
**The Test:** For every pair of variables , partition the space (e.g., quadrants based on medians) and form a contingency table against .


* 
**Selection:** If an interaction is found, GUIDE adopts a specific look-ahead strategy (splitting on  then , vs  then ) to decide the immediate split.



---

### **Phase 3: Split Point Optimization**

Once the variable (say, `age`) is selected via the statistical test:

1. **Search:** Now you act like CART, but *only* for `age`.
2. **Criterion:**
* 
**Classification:** Minimize Gini Impurity or Entropy.


* 
**Regression:** Minimize Sum of Squared Errors (SSE).




3. **Missing Values:** GUIDE treats "Missing" as a category during the statistical test. During the split search, it allows a "missing" group to go Left or Right, depending on which reduces impurity more. This is distinct from CART's surrogate splits.



---

### **Phase 4: Advanced Features (To Add Later)**

1. **Categorical Splits:** GUIDE is famous for handling categorical variables without one-hot encoding. It groups categories (e.g.,  go left,  goes right).
* *Implementation:* If the selected variable is categorical, perform a search over subsets of categories (heuristic or exhaustive depending on cardinality).


2. **Pruning:** GUIDE grows a large tree and prunes back.
* *Implementation:* Implement Cost-Complexity Pruning (standard in sklearn now). You can use `sklearn.model_selection.cross_val_score` to help choose the alpha parameter.



---

### **Step-by-Step Implementation Roadmap**

**Step 1: The Curvature Test Helper**
Create a function that takes a feature vector `x` and target `z` and returns a Chi-square p-value.

```python
import numpy as np
from scipy.stats import chi2_contingency

def calc_curvature_p_value(x, z, is_categorical=False):
    # 1. Discretize if continuous
    if not is_categorical:
        # GUIDE typically uses 3 or 4 groups based on sample size
        n_bins = 4 if len(x) >= 40 else 3
        try:
            # simple binning for illustration
            x_binned = np.array(pd.qcut(x, n_bins, labels=False, duplicates='drop'))
        except:
             return 1.0 # Fallback if binning fails
    else:
        x_binned = x
        
    # 2. Contingency Table
    # Create matrix of counts: rows=x_binned, cols=z
    contingency = pd.crosstab(x_binned, z)
    
    # 3. Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    return p

```

**Step 2: The Node Class**

```python
class GuideNode:
    def __init__(self, depth, is_leaf=False, prediction=None):
        self.depth = depth
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = None
        self.split_threshold = None # Or set of categories for categorical splits
        self.left = None
        self.right = None

```

**Step 3: The Fit Loop**

1. Check stopping criteria (depth, min_samples).
2. **Construct "Z" target:**
* If Regressor: `residual = y - mean(y)`; `z = (residual > 0).astype(int)`
* If Classifier: `z = y`


3. **Variable Selection:**
* Loop over all columns in `X`.
* Call `calc_curvature_p_value(X[:, col], z)`.
* `best_feature_idx = argmin(p_values)`.


4. **Split Search:**
* Using *only* `X[:, best_feature_idx]`, find the value `t` that minimizes Impurity (Gini/SSE) of `y`.


5. **Recursion:**
* Split `X`, `y` into sets `L` and `R`.
* `node.left = fit(X_L, y_L)`
* `node.right = fit(X_R, y_R)`



### **Key References for Logic Details**

* 
**Variable Selection Logic:** See *Section 2.1* in `classification.pdf`.


* 
**Regression Residuals:** See *Section 2* in `variable-importance-scores.pdf`.


* 
**Missing Values:** See *Section 2* in `survey.pdf`, noting GUIDE sends missing values to the node yielding greater impurity reduction.
