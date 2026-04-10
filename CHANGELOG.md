# CHANGELOG.md

## [0.1.10] - 10-04-2026

### Fixed

- `metric-learning` now honors `--save` in the CLI and API
- Added JSON/plot artifact saving for quantum metric learning
- Normalised VQR artifact output paths to `results/vqr/` and `images/vqr/`
- Updated API docs to reflect that metric learning returns a dataclass result

### Added

- Regression tests for metric-learning artifact saving
- Regression tests for VQR default artifact path selection

### Maintenance

- Ignored local `.codex` file and `.codex/` directory
- Removed stale local build artifacts before the next release cut

---

## [0.1.9] - 06-04-2026

### Added

#### Quantum metric learning
- Implemented supervised **quantum metric learning** using contrastive loss
- Trainable data re-uploading embedding circuits
- Nearest-centroid classification in learned quantum feature space
- CLI workflow:

```bash
python -m qml metric-learning --samples 200 --layers 2 --steps 50 --plot
```

- Notebook:

```
notebooks/quantum_metric_learning.ipynb
```

- Documentation:

```
docs/qml/metric_learning.md
```

#### Visualisation support

- Added `plot_metric_learning_embeddings(...)` to `qml.visualize`
- Standardised plotting via shared visualisation utilities
- Automatic embedding plots when `plot=True`

#### Benchmark integration

- Added `quantum_metric_learning` to classification benchmark framework
- Supports multi-seed comparison with VQC, quantum kernel, and classical baselines
- Compatible with per-model hyperparameter overrides

#### CLI integration

- Added `metric-learning` subcommand
- Consistent interface with other QML workflows

#### Testing

- Added smoke tests for:

  - API workflow
  - CLI execution
- Ensures reproducibility with small-step configurations

#### Documentation updates

- README feature list updated
- USAGE.md includes API and CLI usage examples
- THEORY.md extended with contrastive learning formulation
- Added dedicated docs page:

```
docs/qml/metric_learning.md
```

### Internal improvements

- Unified plotting interface across models
- Improved result dataclass structure for embedding-based workflows
- Added label outputs (`y_train`, `y_test`) to metric learning results
- Improved compatibility of benchmark framework with dataclass-based outputs

### Summary

New core QML capability:

- variational quantum classification (VQC)
- variational quantum regression (VQR)
- quantum kernel methods
- trainable quantum kernels
- quantum metric learning

Metric learning provides a flexible representation-learning approach compatible with classical classifiers and similarity-based workflows.

---

## [0.1.7] - 06-04-2026

### Added
- unified training loop via `qml.training.run_training_loop`
- shared utilities in `qml.utils`
- centralised path handling via `qml.io_utils.ensure_dir`

### Refactored
- removed duplicated optimisation loops across VQC, VQR, and kernel workflows
- improved package modularity and internal consistency
- simplified experiment output handling

### Removed
- deprecated `qml.datasets`
- redundant local helper functions

---

## [0.1.5] - 06-04-2026

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

## 0.1.4 - 06-04-2026

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
