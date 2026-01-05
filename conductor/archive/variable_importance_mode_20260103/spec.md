# Spec: Variable Importance Mode

## Overview
Currently, `pyguide` provides standard feature importance based on weighted impurity reduction. However, one of GUIDE's key strengths is its ability to detect interactions. This track aims to implement a more sophisticated variable importance scoring system that can distinguish between a feature's contribution as a main effect versus its contribution within an interaction.

## Goals
- **Enhanced Importance Attribute:** Introduce `interaction_importances_` (or similar) to the estimator.
- **Detailed Scoring:** Score features based on their role in identifying significant interactions, even if they aren't the primary split variable.
- **Diagnostic Method:** A dedicated method (e.g., `get_importance_report()`) that returns a detailed breakdown.

## Key Components

### 1. Importance Calculation Logic
- **Split Role:** Track whether a split was chosen due to a main effect test (Chi-square) or an interaction test.
- **Interaction Attribution:** If a split was chosen via interaction (e.g., pair `(x1, x2)`), attribute importance to *both* variables, not just the one used for the physical split.
- **Scoring:**
  - `main_effect_importance`: Standard impurity reduction when selected as a main effect.
  - `interaction_importance`: Impurity reduction distributed among interacting variables when selected via interaction search.

### 2. API Extensions
- Update `GuideNode` to store `split_type` ("main" or "interaction") and `interaction_group` (list of feature indices).
- Update `_compute_feature_importances` to use this new metadata.

## Success Criteria
- In a pure XOR problem, standard importance might be low/confused, but `interaction_importance` should be high for the relevant features.
- `feature_importances_` (standard) remains compatible with scikit-learn (sum to 1).
- A new attribute or method provides the detailed breakdown.
