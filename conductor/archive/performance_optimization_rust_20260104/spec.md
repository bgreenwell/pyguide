# Specification: Performance Optimization with Rust

## Overview
Optimize the core computational kernels of `pyguide` by replacing pure Python/NumPy implementations with high-performance Rust extensions using `pyo3` and `maturin`.

## Goals
- Significantly reduce training time for `GuideTreeClassifier` and `GuideTreeRegressor`.
- Maintain 100% functional parity with the current Python implementation.
- Establish a seamless build process using `uv` and `maturin`.

## Targeted Bottlenecks (Hypothesis)
1.  **Contingency Table Construction:** `crosstab` or manual loops in `stats.py` during variable selection.
2.  **Split Optimization:** Iterating through sorted values to find best split (`find_best_split`) in `splitting.py`.
3.  **Interaction Detection:** Combinatorial checks for interactions.

## Architecture
- **Tooling:** `maturin` as the build backend, `pyo3` for Python bindings.
- **Structure:**
    - New Rust crate at `src/rust_core` (or similar).
    - Compiled extension module `pyguide._core`.
- **Integration:**
    - Python modules (`stats.py`, `splitting.py`) will import from `pyguide._core` if available, falling back to Python (optional, but good for dev) or replacing it entirely.

## Implementation Steps
1.  **Profiling:** Use `cProfile` and `snakeviz` on `benchmarks/main_benchmark.py` to confirm targets.
2.  **Scaffolding:** Initialize `pyproject.toml` integration with `maturin`.
3.  **Porting:**
    - `calc_curvature_test`: Chi-square logic.
    - `find_best_split`: Gini/SSE optimization loops.
4.  **Verification:** reuse existing `tests/` to ensure no regression in logic.

## Dependencies
- `maturin` (build dependency)
- `pyo3` (Rust dependency)
- Rust toolchain (cargo, rustc)
