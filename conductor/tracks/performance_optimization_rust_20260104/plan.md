# Plan: Performance Optimization (Rust)

## Phase 1: Profiling & Setup
- [ ] Task: Profile current benchmarks to identify top 3 hotspots
- [ ] Task: Add `maturin` to `pyproject.toml` and configure build system
- [ ] Task: Initialize Rust crate structure
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Setup' (Protocol in workflow.md)

## Phase 2: Core Kernels
- [ ] Task: Port `calc_curvature_test` (Chi-square stats) to Rust
- [ ] Task: Port `find_best_split` (Split optimization) to Rust
- [ ] Task: Integrate Rust extension into Python modules
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Kernels' (Protocol in workflow.md)

## Phase 3: Benchmarking & Finalization
- [ ] Task: Run benchmarks comparing Pure Python vs Rust
- [ ] Task: Ensure CI/CD builds wheels correctly
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking' (Protocol in workflow.md)
