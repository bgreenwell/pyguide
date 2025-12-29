# Tech Stack: pyguide

## Core Language & Environment
- **Python 3.9+**: Ensuring modern language features and long-term support.
- **uv**: For ultra-fast Python package management and environment handling.
- **Poetry**: Used for packaging and dependency definition (compatible with `uv`).

## Data Science & Machine Learning
- **NumPy**: For efficient numerical array operations and vectorization.
- **SciPy**: Specifically for `scipy.stats` (Chi-square tests, probability distributions).
- **Scikit-Learn**: For the `BaseEstimator` and `ClassifierMixin` foundations, and validation utilities.
- **Pandas**: Primarily used to ensure compatibility with `DataFrame` inputs and for occasional contingency table operations where performance permits.

## Tooling & Quality
- **Ruff**: For high-performance linting and code formatting.
- **Pytest**: The primary testing framework.
- **Coverage.py**: To ensure high test coverage as defined in the workflow.
