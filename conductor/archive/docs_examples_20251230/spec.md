# Spec: Documentation & Examples

## Overview
This track aims to produce high-quality, developer-friendly documentation for `pyguide`. As the library has grown in complexity (advanced interaction detection, pruning, handling of missing values), having clear examples and API references is crucial for adoption.

## Goals
- **API Reference:** Generate clear, readable docstrings for all public classes and methods.
- **User Guide:** Create a `docs/` directory with markdown files explaining core concepts (Unbiased Selection, Interactions).
- **Examples:** Provide executable Python scripts or notebooks demonstrating key features.
- **README Update:** Refresh the main `README.md` to reflect the latest capabilities.

## Key Components

### 1. API Documentation
- Ensure all docstrings in `classifier.py`, `regressor.py`, and `visualization.py` follow the NumPy/Scikit-Learn style.
- Highlight unique parameters like `interaction_depth`, `max_interaction_candidates`, and `significance_threshold`.

### 2. Usage Examples (`examples/`)
- **`basic_classification.py`**: Simple Iris/Titanic example.
- **`interaction_demo.py`**: Synthetic XOR example showing how GUIDE finds interactions where CART fails.
- **`bias_demo.py`**: Demonstration of variable selection bias in standard CART vs. GUIDE on random data with high cardinality.

### 3. User Guides (`docs/`)
- **`user_guide.md`**: Detailed explanation of the GUIDE algorithm steps (Chi-square selection -> Split optimization).
- **`interactions.md`**: Deep dive into how the interaction detection works and how to tune it.

## Success Criteria
- All public API methods have complete docstrings.
- At least 3 working example scripts in `examples/`.
- `README.md` contains a "Quick Start" and "Features" section that matches the current code state.
