# CHANGELOG.md

## [0.2.3] - 22-05-2026

### Changed

- Added rendered diagnostic plot outputs across the tutorial notebooks so the
  generated tutorial result page now includes dataset, loss, prediction,
  kernel-matrix, embedding, and forecasting figures where relevant.
- Expanded the quantum metric-learning tutorial with loss-history and
  dataset-comparison plots.
- Removed the archived QAOA notebook and dropped the `notebooks/archive/`
  workflow from generated result pages and GitHub Pages navigation.
- Bumped package metadata to `0.2.3`.

### Removed

- Removed `notebooks/archive/01-qaoa-max-cut.ipynb`.
- Removed `RESULTS_ARCHIVE.md` and the extracted archive plot assets.

### Validation

- Executed all tutorial notebooks from `notebooks/tutorials/`.
- Regenerated `RESULTS_TUTORIALS.md` and extracted tutorial plot assets.
- Verified generated Pages site builds from the project virtualenv.

---

## [0.2.2] - 22-05-2026

### Added

- Added `docs/qml/api_reference.md` as a compact public API reference for
  top-level imports, workflow functions, estimator APIs, benchmark helpers,
  classical baselines, reporting helpers, and version access.
- Added benchmark runtime tracking to classification and regression benchmark
  run records and aggregate summaries.
- Added benchmark train/test generalization-gap tracking:
  - classification gap is `train_accuracy - test_accuracy`
  - regression gap is `test_mse - train_mse`
- Added a `best_model` summary field to benchmark outputs:
  - classification selects the highest mean `test_accuracy`
  - regression selects the lowest mean `test_mse`

### Changed

- Updated benchmark documentation to match the current model registry,
  including trainable quantum kernels and quantum metric learning.
- Expanded benchmark guidance to require classical baselines, explicit seed
  lists, runtime reporting, train/test gap reporting, and clear non-advantage
  framing for release-quality comparisons.
- Wired the API reference into the generated GitHub Pages site and primary
  navigation.
- Updated `README.md` and `USAGE.md` to describe the stronger benchmark output
  contract and interpretation limits.
- Bumped package metadata to `0.2.2`.

### Validation

- Added benchmark smoke-test assertions for runtime summaries,
  generalization-gap summaries, and `best_model` metadata.

---

## [0.2.1] - 22-05-2026

### Fixed

- Fixed packaging metadata tests on Python 3.10 by falling back from the
  standard-library `tomllib` module to `tomli`.

### Maintenance

- Added the conditional development dependency `tomli>=2; python_version < '3.11'`.
- Marked the finite-shot trainable-kernel dataset smoke test as slow so
  `pytest -m "not slow"` remains focused on the fast CI subset.

### Validation

- Verified `pytest -m "not slow"` passes with 58 tests selected and 4 slow tests deselected.

---

## [0.2.0] - 21-05-2026

### Added

- Added implementation contracts documenting the model family, objective, metric
  semantics, and behavioral checks expected for each advertised algorithm.
- Added dataset-agnostic estimator APIs:
  - `qml.estimators.QuantumClassifier`
  - `qml.estimators.QuantumRegressor`
  - `qml.kernels.QuantumKernel`
  - `qml.kernels.QuantumKernelClassifier`
  - `qml.kernels.QuantumKernelRegressor`
  - `qml.preprocessing.make_sequence_windows`
- Added `qml.reporting` helpers for human-readable notebook and CLI tables:
  - `format_table(...)`
  - `print_table(...)`
  - `print_section(...)`
- Added tutorial notebooks for the new estimator and preprocessing APIs:
  - `notebooks/tutorials/06-quantum-kernel-estimators.ipynb`
  - `notebooks/tutorials/07-variational-quantum-estimators.ipynb`
  - `notebooks/tutorials/08-sequence-window-quantum-forecasting.ipynb`
- Added real-example notebooks for small reproducible physics and dynamical-system tasks:
  - `notebooks/real_examples/01-rabi-oscillation-parameter-inference.ipynb`
  - `notebooks/real_examples/02-ising-correlation-temperature-classifier.ipynb`
  - `notebooks/real_examples/03-lorenz-regime-classifier.ipynb`
  - `notebooks/real_examples/04-condensed-matter-tfim-phase-classifier.ipynb`
  - `notebooks/real_examples/05-pendulum-trajectory-surrogate.ipynb`
  - `notebooks/real_examples/06-damped-oscillator-parameter-inference.ipynb`
- Added `notebooks/README.md` to document tutorial and real-example notebooks.
- Added generated notebook result pages:
  - `RESULTS_TUTORIALS.md`
  - `RESULTS_REAL_EXAMPLES.md`

### Changed

- Moved importable package code from root-level `qml/` to `src/qml/`.
- Updated packaging and local test configuration for the `src/` layout.
- Reworked the QCNN workflow into a defensible four-qubit QCNN with trainable
  convolution blocks, trainable pooling blocks, and active-wire reduction from
  four to two to one wire.
- Consolidated the quantum-kernel classifier workflow onto the reusable
  `qml.kernels.QuantumKernel` implementation.
- Added sklearn-style `get_params` and `set_params` methods to variational and
  quantum-kernel estimators.
- Updated classical baselines and benchmark helpers so selected datasets are
  propagated consistently across quantum and classical models.
- Moved algorithm walkthrough notebooks into `notebooks/tutorials/`.
- Renamed notebooks to numbered kebab-case names for stable file-browser ordering.
- Converted repeated notebook `print_section` helper code to use `qml.reporting.print_section`.
- Updated notebook bootstrap cells so tutorials and real examples run from the repository root,
  `notebooks/`, or their own subdirectories.
- Updated Pages workflow triggers so documentation is rebuilt when `src/**` changes.
- Updated Pages result generation to execute notebooks and publish tutorial and real-example
  result pages with extracted tables and plots.
- Excluded notebooks from Ruff and Black because executable notebook bootstrap cells intentionally
  adjust import paths before importing project modules.

### Fixed

- Corrected `qml.io_utils` repository-root detection after the `src/` layout migration.
- Updated markdown references to the renamed tutorial notebook paths.
- Corrected quantum-autoencoder reconstruction fidelity so it is evaluated after
  compression loss via trash-zero postselection and tied decoding, rather than by
  applying an encoder immediately followed by its inverse.

### Validation

- Executed all tutorial notebooks from `notebooks/tutorials/`.
- Executed the affected real-example notebooks after adopting `qml.reporting`.
- Verified package imports, reporting helpers, estimator APIs, and notebook parsing.
- Added behavioral test coverage for autoencoder reconstruction, QCNN pooling
  structure, analytic kernel positive semidefiniteness, estimator parameter
  APIs, and benchmark dataset consistency.

---

## [0.1.12] - 06-05-2026

### Added

- Implemented a first-class quantum autoencoder workflow in `qml.autoencoder`
- Added `autoencoder` CLI support via `python -m qml autoencoder`
- Added smoke, artifact, CLI, and import coverage for the quantum autoencoder
- Added autoencoder documentation and example notebook support
- Added a GitHub Pages workflow for publishing a custom static documentation site
- Added a Pages site generator and stylesheet matched to the root portfolio site
- Added generated `RESULTS.md` reference outputs and a Results web page
- Added generated result images to `RESULTS.md` and the Results web page
- Added a Pages workflow status badge to `README.md`
- Added documentation for the Pages tooling in `docs/pages/README.md`
- Added README and usage-documentation links to the published Pages site

### Maintenance

- Ignored local static-site build output directories (`_site/` and `site_docs/`)
- Configured pytest to include the repository root on the import path for local test runs
- Regenerate reference results during the Pages workflow before building the site

### Summary

New core QML capability:

- variational quantum classification (VQC)
- variational quantum regression (VQR)
- quantum convolutional neural networks (QCNN)
- quantum autoencoders
- quantum kernel methods
- trainable quantum kernels
- quantum metric learning

---

## [0.1.11] - 10-04-2026

### Added

- Implemented a first-class QCNN workflow in `qml.qcnn`
- Added `qcnn` CLI support via `python -m qml qcnn`
- Added QCNN benchmark support in classification benchmarks
- Added QCNN smoke, CLI, benchmark, and import coverage
- Added QCNN documentation across README, usage docs, theory notes, and a dedicated algorithm page
- Added QCNN example notebook: `notebooks/tutorials/10-quantum-convolutional-neural-network.ipynb`

### Summary

New core QML capability:

- variational quantum classification (VQC)
- variational quantum regression (VQR)
- quantum convolutional neural networks (QCNN)
- quantum kernel methods
- trainable quantum kernels
- quantum metric learning

---

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
notebooks/tutorials/09-quantum-metric-learning.ipynb
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
