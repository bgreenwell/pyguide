# Specification: Documentation & Benchmarks Update

## Overview
Update benchmarks to include a binary classification example and add detailed Jupyter notebooks to the documentation.

## Goals
1.  **Benchmarks:** Add Breast Cancer dataset (binary classification) to `main_benchmark.py` to properly showcase `GuideGradientBoostingClassifier`.
2.  **Documentation:** Add detailed example notebooks for Regression and Binary Classification.
3.  **Integration:** Ensure notebooks are rendered in the Sphinx documentation.

## Implementation Details
- **Benchmarks:** Use `sklearn.datasets.load_breast_cancer`.
- **Notebooks:**
    - `examples/notebooks/regression.ipynb`: Diabetes dataset, comparing Single Tree, RF, and GBM. Highlight feature importance and interactions.
    - `examples/notebooks/classification.ipynb`: Breast Cancer dataset, focusing on Log Loss, probability calibration, and unbiased selection.
- **Sphinx:** Use `nbsphinx` to execute and render notebooks.

## Dependencies
- `nbsphinx`
- `ipykernel`
- `pandoc` (system dependency, might be needed for nbsphinx)
