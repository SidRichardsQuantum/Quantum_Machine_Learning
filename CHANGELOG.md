# CHANGELOG.md

## [0.1.5]

### Added

- Multiple dataset support via `qml.data`

  - classification datasets:

    - `moons`
    - `circles`
    - `blobs`
    - `xor`
  - regression datasets:

    - `linear`
    - `sine`
    - `polynomial`
- Dataset selection exposed across public APIs:

  - `run_vqc(dataset=...)`
  - `run_vqr(dataset=...)`
  - `run_quantum_kernel_classifier(dataset=...)`
  - `run_trainable_quantum_kernel_classifier(dataset=...)`
  - `compare_classification_models(dataset=...)`
  - `compare_regression_models(dataset=...)`
- CLI support for dataset selection:

  ```bash
  python -m qml vqc --dataset circles
  python -m qml regression --dataset sine
  python -m qml benchmark classification --dataset xor
  ```
- Dataset smoke tests ensuring end-to-end compatibility
- Deterministic dataset generation with seeded NumPy RNG

### Changed

- Benchmark framework updated to support model-specific kwargs alongside dataset selection
- Classification and regression runners now return consistent `"dataset"` metadata
- Improved separation between dataset specification and data tensors
- Benchmark dispatch filters unsupported kwargs for classical baselines

### Fixed

- Finite-shot determinism preserved across datasets
- Regression benchmark default dataset corrected to `"linear"`
- Removed dataset shadowing bug where dataset dict replaced dataset name
- CLI dataset argument now correctly propagates to runners

---

## 0.1.4

### Added
- Noise-aware benchmark support via per-model `model_kwargs`
- Finite-shot benchmark smoke tests
- Deterministic finite-shot benchmark execution with fixed seeds
- Extended dataset utilities for multiple classification and regression dataset types

### Changed
- Updated benchmark dispatch to support model-specific kwargs cleanly
- Refined README, USAGE, and THEORY documentation to reflect current package capabilities
- Generalised `qml.data` dataset generation with lightweight dispatch helpers

---

## 0.1.2

### Added

- benchmark CLI workflow
- multi-seed comparison utilities
- benchmark smoke tests
- documentation for benchmarking workflows

### Improved

- consistency of classical vs quantum comparisons
