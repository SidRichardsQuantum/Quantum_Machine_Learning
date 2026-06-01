# Public API Reference

This page summarizes the import paths that are intended for examples,
notebooks, scripts, and downstream use. The project keeps notebooks as thin
clients over these package APIs.

## Top-Level Imports

The `qml` package lazily exposes the main workflow and estimator APIs:

```python
from qml import (
    QuantumClassifier,
    QuantumKernel,
    QuantumKernelClassifier,
    QuantumGaussianProcessRegressor,
    QuantumKernelRegressor,
    QuantumKernelPCA,
    QuantumOneClassClassifier,
    QuantumReservoirClassifier,
    QuantumReservoirFeatures,
    QuantumReservoirRegressor,
    QuantumRegressor,
    TrainableQuantumKernelRegressor,
    ansatz_parameter_count,
    build_noise_model,
    cross_validate_estimator,
    embedding_parameter_count,
    estimate_circuit_depth,
    format_table,
    kernel_target_alignment,
    list_supported_optimizers,
    make_sequence_windows,
    print_section,
    print_table,
    qcnn_parameter_count,
    run_qcnn,
    run_quantum_autoencoder,
    run_quantum_kernel_classifier,
    run_quantum_metric_learner,
    run_trainable_quantum_kernel_classifier,
    run_trainable_quantum_kernel_regressor,
    run_vqc,
    run_vqr,
    select_best_model,
    train_test_evaluate,
)
```

Use top-level imports for compact examples. Use module imports when you want to
make the implementation area explicit.

## Workflow Functions

| Function | Module | Purpose |
| --- | --- | --- |
| `run_vqc(...)` | `qml.classifiers` | Train a variational quantum classifier on a supported classification dataset. |
| `run_vqr(...)` | `qml.regression` | Train a variational quantum regressor on a supported regression dataset. |
| `run_qcnn(...)` | `qml.qcnn` | Train the four-qubit QCNN classifier. |
| `run_quantum_autoencoder(...)` | `qml.autoencoder` | Train and evaluate a quantum autoencoder on structured state families. |
| `run_quantum_kernel_classifier(...)` | `qml.kernel_methods` | Train an SVM on a PennyLane fidelity kernel. |
| `run_trainable_quantum_kernel_classifier(...)` | `qml.trainable_kernels` | Optimize kernel-target alignment, then fit an SVM. |
| `run_trainable_quantum_kernel_regressor(...)` | `qml.trainable_kernels` | Optimize continuous target alignment, then fit kernel-ridge regression. |
| `run_quantum_metric_learner(...)` | `qml.metric_learning` | Learn a supervised quantum embedding with contrastive loss. |

Workflow functions return dictionaries unless documented otherwise. Metric
learning returns a result dataclass with attributes such as `train_accuracy`,
`test_accuracy`, and `loss_history`.

## Estimator APIs

Use these when data already exists outside the package:

| Class or function | Module | Purpose |
| --- | --- | --- |
| `QuantumClassifier` | `qml.estimators` | Sklearn-style variational quantum classifier for user-supplied arrays. |
| `QuantumRegressor` | `qml.estimators` | Sklearn-style variational quantum regressor for user-supplied arrays. |
| `QuantumKernel` | `qml.kernels` | Reusable fidelity-kernel object. |
| `QuantumKernelClassifier` | `qml.kernels` | Kernel SVM wrapper over `QuantumKernel`. |
| `QuantumKernelRegressor` | `qml.kernels` | Kernel ridge wrapper over `QuantumKernel`. |
| `QuantumKernelPCA` | `qml.kernels` | Kernel PCA with a quantum fidelity kernel. |
| `QuantumOneClassClassifier` | `qml.kernels` | One-class anomaly detector with a quantum fidelity kernel. |
| `QuantumGaussianProcessRegressor` | `qml.kernels` | Gaussian-process regression with a quantum kernel covariance. |
| `TrainableQuantumKernelRegressor` | `qml.trainable_kernels` | Trainable feature-map kernel regressor for user-supplied arrays. |
| `QuantumReservoirFeatures` | `qml.reservoir` | Fixed random quantum reservoir feature map. |
| `QuantumReservoirClassifier` | `qml.reservoir` | Logistic classifier trained on reservoir features. |
| `QuantumReservoirRegressor` | `qml.reservoir` | Ridge regressor trained on reservoir features. |
| `make_sequence_windows` | `qml.preprocessing` | Convert a sequence into fixed-width supervised windows. |
| `cross_validate_estimator(...)` | `qml.model_selection` | Cross-validate estimator-style models with deterministic sklearn splitters. |
| `train_test_evaluate(...)` | `qml.model_selection` | Fit and score one deterministic train/test split. |
| `select_best_model(...)` | `qml.model_selection` | Cross-validate candidate estimators and optionally refit the best one. |

The estimator classes expose `fit`, `predict`, `score`, `get_params`, and
`set_params` where those operations apply.

Circuit-backed estimators accept an optional `noise_model` dictionary for
depolarizing, amplitude-damping, and readout-error simulation. Use
`qml.noise.build_noise_model(...)` to validate explicit channel probabilities.

## Embeddings and Ansatz Helpers

| Function | Module | Purpose |
| --- | --- | --- |
| `available_embeddings()` | `qml.embeddings` | List canonical embedding names. |
| `apply_angle_embedding(...)` | `qml.embeddings` | Angle encoding with one feature per wire. |
| `apply_amplitude_embedding(...)` | `qml.embeddings` | Amplitude encoding with padding and normalization. |
| `apply_zz_feature_map(...)` | `qml.embeddings` | Second-order ZZ feature map. |
| `apply_iqp_feature_map(...)` | `qml.embeddings` | Compact IQP-style feature map. |
| `apply_data_reuploading_embedding(...)` | `qml.embeddings` | Trainable repeated feature encoding. |
| `apply_hardware_efficient_ansatz(...)` | `qml.ansatz` | Default variational ansatz. |
| `apply_strongly_entangling_ansatz(...)` | `qml.ansatz` | PennyLane strongly entangling template wrapper. |
| `ansatz_parameter_count(...)` | `qml.circuit_metadata` | Count trainable parameters for supported ansatz templates. |
| `embedding_parameter_count(...)` | `qml.circuit_metadata` | Count trainable parameters for supported embeddings. |
| `estimate_circuit_depth(...)` | `qml.circuit_metadata` | Estimate package-template circuit depth for reporting. |
| `qcnn_parameter_count()` | `qml.circuit_metadata` | Count trainable parameters in the default QCNN template. |

Workflow result dictionaries for circuit-backed package models include a
`circuit_metadata` field with qubit count, trainable-parameter count, estimated
depth, embedding/ansatz labels, and a `depth_is_estimate` flag.

## Noise Helpers

| Function | Module | Purpose |
| --- | --- | --- |
| `build_noise_model(...)` | `qml.noise` | Build a validated optional noise-model dictionary. |
| `normalize_noise_model(...)` | `qml.noise` | Canonicalize and validate supported channel probabilities. |
| `noise_model_tag(...)` | `qml.noise` | Build compact tags for filenames and metadata. |

## Benchmark APIs

| Function | Module | Purpose |
| --- | --- | --- |
| `compare_classification_models(...)` | `qml.benchmarks` | Compare quantum and classical classifiers across seed lists. |
| `compare_regression_models(...)` | `qml.benchmarks` | Compare quantum and classical regressors across seed lists. |

Benchmark results include run records, aggregate metric summaries, runtime
summaries, train/test gap summaries, confidence intervals, paired deltas against
the best included classical baseline, tuning metadata, environment metadata, and
a convenience `best_model` field based on the primary test metric.

## Classical Baselines

Classical baselines are first-class package workflows so quantum models can be
compared against standard references under the same dataset settings:

| Function | Module |
| --- | --- |
| `run_logistic_classifier(...)` | `qml.classical_baselines` |
| `run_svm_classifier(...)` | `qml.classical_baselines` |
| `run_mlp_classifier(...)` | `qml.classical_baselines` |
| `run_random_forest_classifier(...)` | `qml.classical_baselines` |
| `run_gradient_boosting_classifier(...)` | `qml.classical_baselines` |
| `run_knn_classifier(...)` | `qml.classical_baselines` |
| `run_gaussian_process_classifier(...)` | `qml.classical_baselines` |
| `run_ridge_regression(...)` | `qml.classical_baselines` |
| `run_mlp_regressor(...)` | `qml.classical_baselines` |
| `run_kernel_ridge_regression(...)` | `qml.classical_baselines` |
| `run_svr_regression(...)` | `qml.classical_baselines` |
| `run_gaussian_process_regressor(...)` | `qml.classical_baselines` |
| `run_random_forest_regressor(...)` | `qml.classical_baselines` |
| `run_gradient_boosting_regressor(...)` | `qml.classical_baselines` |
| `run_knn_regressor(...)` | `qml.classical_baselines` |
| `run_lasso_regression(...)` | `qml.classical_baselines` |
| `run_elasticnet_regression(...)` | `qml.classical_baselines` |

## Reporting Helpers

Notebook and CLI examples should use the shared reporting helpers instead of
duplicating table-formatting code:

| Function | Module | Purpose |
| --- | --- | --- |
| `format_table(...)` | `qml.reporting` | Return a plain-text table. |
| `print_table(...)` | `qml.reporting` | Print a plain-text table. |
| `print_section(...)` | `qml.reporting` | Print a titled block with one or more tables. |

## Versioning

The installed package version is exposed as:

```python
import qml

print(qml.__version__)
```

During editable local development, the value comes from installed package
metadata. Reinstall with `pip install -e .` after version changes.
