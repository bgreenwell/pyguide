# Spec: Packaging & Publishing

## Overview
This track focuses on polishing the `pyguide` repository for public release. It involves cleaning up temporary files, finalizing the project metadata, setting up GitHub Actions for CI/CD, and adding professional touches like badges to the README.

## Goals
- **Clean Repository:** Remove all junk files, temporary scripts, and build artifacts from the root directory.
- **Professional Appearance:** Add CI/CD status, coverage, and PyPI badges to `README.md`.
- **Release Readiness:** Ensure `pyproject.toml` has all necessary metadata (license, classifiers, urls).
- **CI/CD:** Implement a GitHub Actions workflow for running tests, linting, and building documentation on every push.

## Key Components

### 1. Cleanup
- Identify and remove files like `benchmark.py`, `prototype_viz.py`, `iris_guide_tree.png`, etc., that are leftover from development.
- Move useful scripts to `examples/` or `benchmarks/` if worth keeping.

### 2. Project Metadata
- Update `pyproject.toml` with:
  - `license = { file = "LICENSE" }`
  - `classifiers` (e.g., "Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License")
  - `urls` (Homepage, Repository, Documentation)

### 3. CI/CD Pipeline
- Create `.github/workflows/ci.yml`:
  - Run `uv sync`
  - Run `ruff check .`
  - Run `pytest`
  - Build Sphinx docs

### 4. README Polish
- Add badges:
  - Tests (GitHub Actions)
  - Code Style (Ruff)
  - Python Version

## Success Criteria
- Root directory contains only standard project files (`src`, `tests`, `docs`, `examples`, `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, `.github`).
- `pyproject.toml` validates successfully.
- GitHub Actions workflow is created and (theoretically) functional.
- `README.md` looks professional.
