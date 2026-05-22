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
    QuantumKernelRegressor,
    QuantumRegressor,
    format_table,
    kernel_target_alignment,
    list_supported_optimizers,
    make_sequence_windows,
    print_section,
    print_table,
    run_qcnn,
    run_quantum_autoencoder,
    run_quantum_kernel_classifier,
    run_quantum_metric_learner,
    run_trainable_quantum_kernel_classifier,
    run_vqc,
    run_vqr,
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
| `make_sequence_windows` | `qml.preprocessing` | Convert a sequence into fixed-width supervised windows. |

The estimator classes expose `fit`, `predict`, `score`, `get_params`, and
`set_params` where those operations apply.

## Benchmark APIs

| Function | Module | Purpose |
| --- | --- | --- |
| `compare_classification_models(...)` | `qml.benchmarks` | Compare quantum and classical classifiers across seed lists. |
| `compare_regression_models(...)` | `qml.benchmarks` | Compare quantum and classical regressors across seed lists. |

Benchmark results include run records, aggregate metric summaries, runtime
summaries, train/test gap summaries, and a convenience `best_model` field based
on the primary test metric.

## Classical Baselines

Classical baselines are first-class package workflows so quantum models can be
compared against standard references under the same dataset settings:

| Function | Module |
| --- | --- |
| `run_logistic_classifier(...)` | `qml.classical_baselines` |
| `run_svm_classifier(...)` | `qml.classical_baselines` |
| `run_mlp_classifier(...)` | `qml.classical_baselines` |
| `run_ridge_regression(...)` | `qml.classical_baselines` |
| `run_mlp_regressor(...)` | `qml.classical_baselines` |

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
