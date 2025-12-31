# Initial Concept
I'd like to plan out and build the package, starting from the resources and @PLAN.md info.

# Product Guide: pyguide

## Vision
To provide a high-quality, scikit-learn compatible Python implementation of the GUIDE (Generalized, Unbiased, Interaction Detection and Estimation) algorithm, filling the gap in standard decision tree implementations regarding variable selection bias and interaction detection.

## Target Audience
- **Data Scientists and Machine Learning Researchers:** Seeking unbiased decision trees for more accurate feature importance and model interpretation.
- **Statisticians:** Needing models that reliably detect interactions between features.
- **Software Engineers:** Requiring a robust, easy-to-integrate library that follows established scikit-learn patterns.

## Core Features (MVP - GuideTreeClassifier)
- **Unbiased Variable Selection:** Employs Chi-square tests to rank and select variables independently of the split point optimization, preventing bias towards features with many unique values.
- **Separated Splitting Process:** Implements a two-step process: first selecting the variable, then optimizing the cut-point.
- **Interaction Detection:** Built-in support for detecting feature interactions with configurable depth, enhancing the model's predictive power and interpretability.

## Current Capabilities
- **GuideTreeClassifier:** Fully functional scikit-learn compatible classifier.
  - Unbiased variable selection via Chi-square and Fisher's Exact tests.
  - Interaction detection using the full GUIDE look-ahead strategy.
  - Support for numerical and categorical features.
- **GuideTreeRegressor:** Fully functional scikit-learn compatible regressor.
  - Residual-based variable selection for unbiased feature ranking.
  - SSE (Sum of Squared Errors) split optimization.
  - Full parity with Classifier features (interactions, categorical support).
- **Advanced Data Support:**
  - Native support for missing values (NaNs) using impurity-based routing.
  - Optimized categorical splitting using ordered categories for $O(K)$ search instead of $O(2^K)$.
- **Model Management and Visualization:**
  - Integrated support for Minimal Cost-Complexity Pruning via `ccp_alpha`.
  - Compatibility with scikit-learn visualization tools (e.g., `plot_tree`).
  - Highly optimized training path using vectorized contingency tables and cumulative statistics.
- **Advanced Diagnostics and Parity:**
  - Full structural parity with scikit-learn (e.g., `n_leaves_`, `max_depth_`).
  - Variable importance scores (`feature_importances_`) based on weighted impurity reduction.
  - Diagnostic methods for sample tracking (`apply`, `decision_path`).

## Roadmap
- [Planned] Tree ensembles (Random Forest, Boosting) using GUIDE as the base learner.