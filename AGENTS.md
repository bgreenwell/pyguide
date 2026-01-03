# AGENTS.md for pyguide

A high-quality, scikit-learn compatible Python implementation of the **GUIDE** (Generalized, Unbiased, Interaction Detection and Estimation) algorithm.

**Tech Stack:** Python >= 3.11, uv, ruff, pytest, Sphinx, MyST-Parser, NumPy, SciPy, scikit-learn, Pandas

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# Run Tests
uv run pytest

# Run Lint
uv run ruff check .
```

## File Structure

```
pyguide/
├── src/pyguide/         # Source code
│   ├── __init__.py
│   ├── classifier.py    # GuideTreeClassifier
│   ├── regressor.py     # GuideTreeRegressor
│   ├── node.py          # Tree node structure
│   ├── interactions.py  # Interaction detection logic
│   ├── selection.py     # Unbiased variable selection
│   ├── splitting.py     # Split optimization
│   └── visualization.py # plot_guide_tree
├── tests/               # Tests with pytest
├── docs/                # Sphinx documentation (source/)
├── examples/            # Usage examples
├── conductor/           # Project management & specs
├── pyproject.toml       # Config and dependencies
├── uv.lock              # Locked dependencies
└── README.md
```

## Common Commands

```bash
# Dependencies
uv add [package]              # Add runtime dependency
uv add --dev [package]        # Add dev dependency
uv sync                       # Sync environment

# Testing
uv run pytest                 # All tests
uv run pytest -v              # Verbose
uv run pytest --cov=src/pyguide # Coverage

# Code quality
uv run ruff check .           # Check
uv run ruff check --fix .     # Fix auto-fixable
uv run ruff format .          # Format (if enabled)

# Documentation
uv run sphinx-build -b html docs/source docs/build
```

## Code Style

- **Style:** PEP 8, 88 char lines (Ruff default)
- **Lint:** `ruff check --fix .`
- **Docstrings:** NumPy/Scikit-Learn style required for public API.

**pyproject.toml config:**
```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "B", "I"]
ignore = ["E501"]

[tool.pytest.ini_options]
addopts = "--cov=src/pyguide --cov-report=term-missing"
testpaths = ["tests"]
```

## Common Patterns

**Scikit-learn Estimator Pattern:**
All estimators must inherit from `BaseEstimator` and appropriate Mixin (`ClassifierMixin`, `RegressorMixin`).
They must implement `fit(X, y)` and `predict(X)`.
Use `check_X_y`, `check_array`, and `check_is_fitted` for validation.

**Unbiased Splitting:**
Variable selection (`selection.py`) is decoupled from split optimization (`splitting.py`).
1. `select_split_variable`: Returns best feature index and p-value.
2. `find_best_split`: Returns optimal threshold for that feature.

## Development Workflow

**Commits:** Conventional format (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)

**Branching:** `feature/[name]`, `fix/[name]`, `docs/[name]`

**PR Checklist:**
- [ ] Tests pass: `uv run pytest`
- [ ] No lint errors: `uv run ruff check .`
- [ ] Documentation updated (if public API changed)
- [ ] Coverage maintained (>80%)

## Project-Specific Notes

- **Conductor:** This project uses the `conductor` framework for task management. All work is tracked in `conductor/tracks/` and `conductor/tracks.md`.
- **Interaction Detection:** Key differentiator. See `src/pyguide/interactions.py` for Chi-square interaction tests.
- **Categorical Handling:** Native support without one-hot encoding. `_get_categorical_mask` helper in estimators.
- **Missing Values:** Handled via `missing_go_left` flag in `GuideNode`.
