# Model Selection Guide

This guide maps common tasks to the package APIs that are most appropriate for
small, reproducible experiments.

Start with the simplest model that answers the question. Add more expensive QML
models only when the baseline result, dataset size, and interpretation need make
the comparison worthwhile.

## Classification

Use these APIs for binary or small multiclass classification:

| Need | Recommended starting point | When to expand |
| --- | --- | --- |
| Fast classical reference | `qml.classical_baselines.run_logistic_classifier(...)` or `run_svm_classifier(...)` | Always include before interpreting QML results. |
| Trainable circuit classifier | `qml.classifiers.run_vqc(...)` or `qml.estimators.QuantumClassifier` | Use when feature counts are small and optimization behavior matters. |
| Kernel classifier | `qml.kernel_methods.run_quantum_kernel_classifier(...)` or `qml.kernels.QuantumKernelClassifier` | Use when kernel geometry is the main comparison. |
| Trainable kernel classifier | `qml.trainable_kernels.run_trainable_quantum_kernel_classifier(...)` | Use when fixed kernels underfit and alignment optimization is the target. |
| Quantum reservoir classifier | `qml.reservoir.QuantumReservoirClassifier` | Use when fixed quantum features plus a classical readout are enough. |
| Metric-learning workflow | `qml.metric_learning.run_quantum_metric_learner(...)` | Use when learned embedding geometry is the object of study. |
| QCNN classifier | `qml.qcnn.run_qcnn(...)` | Use for compact four-qubit structured classification examples. |

For classification benchmarks, include at least one linear baseline and one
nonlinear classical baseline such as SVM, random forest, kNN, or Gaussian
process classification.

## Regression

Use these APIs for continuous targets:

| Need | Recommended starting point | When to expand |
| --- | --- | --- |
| Fast classical reference | `qml.classical_baselines.run_ridge_regression(...)` or `run_svr_regression(...)` | Always include before interpreting QML results. |
| Trainable circuit regressor | `qml.regression.run_vqr(...)` or `qml.estimators.QuantumRegressor` | Use when a small parameterized circuit is part of the experiment. |
| Kernel ridge regression | `qml.kernels.QuantumKernelRegressor` | Use when comparing fidelity-kernel geometry against RBF or polynomial kernels. |
| Trainable kernel regression | `qml.trainable_kernels.TrainableQuantumKernelRegressor` | Use when target alignment is part of the question. |
| Gaussian-process regression | `qml.kernels.QuantumGaussianProcessRegressor` | Use when uncertainty estimates or smooth surrogate behavior matter. |
| Reservoir regression | `qml.reservoir.QuantumReservoirRegressor` | Use for fixed quantum features and fast classical readout fitting. |

For regression benchmarks, inspect both MSE and MAE when available. If model
rankings change across metrics, report the comparison as metric-sensitive.

## Anomaly Detection And Structure Discovery

Use kernel methods when the task is mostly about geometry:

| Need | Recommended API |
| --- | --- |
| Low-dimensional feature exploration | `qml.kernels.QuantumKernelPCA` |
| One-class anomaly detection | `qml.kernels.QuantumOneClassClassifier` |
| Kernel matrix inspection | `qml.kernels.QuantumKernel` |

These workflows are sensitive to preprocessing and feature scaling. Keep the
feature count small, record the scaling transform, and compare with classical
kernel PCA or one-class SVM references when possible.

## User-Supplied Arrays

Prefer estimator classes when data already exists outside the package:

```python
from qml.estimators import QuantumClassifier, QuantumRegressor
from qml.kernels import QuantumKernelClassifier, QuantumKernelRegressor
from qml.reservoir import QuantumReservoirClassifier, QuantumReservoirRegressor
```

Estimator classes expose `fit`, `predict`, `score`, `get_params`, and
`set_params` where those operations apply. They are better than notebook helper
functions when you need custom splits, preprocessing, or integration with other
Python workflows.

Fitted estimators record `n_features_in_`, and classifiers record `classes_`.
Kernel wrappers retain `kernel_matrix_train_`; reservoir wrappers retain
`feature_matrix_train_`; circuit-backed fitted estimators expose
`circuit_metadata_` where the estimator owns the circuit execution path.

For variational estimators, set `batch_size` to use deterministic mini-batch
optimizer updates. Leave `batch_size=None` for full-batch training, which is the
default and matches earlier package behavior.

Composed estimators expose nested parameters for lightweight tuning:

```python
from qml import QuantumKernel, QuantumKernelClassifier
from qml.reservoir import QuantumReservoirClassifier

kernel_model = QuantumKernelClassifier(QuantumKernel(seed=0))
kernel_model.set_params(kernel__shots=128, kernel__embedding="angle")

reservoir_model = QuantumReservoirClassifier(seed=0)
reservoir_model.set_params(reservoir__n_layers=3, reservoir__noise_model=None)
```

`get_params(deep=True)` includes these nested keys for inspection and manual
tuning. `qml.model_selection.clone_estimator(...)` clones from shallow
constructor parameters so configured kernel and reservoir objects are preserved
without passing nested keys into constructors.

## Cross-Validation Helpers

Use `qml.model_selection` when comparing estimator-style models on
user-supplied arrays:

```python
from qml import QuantumReservoirClassifier, cross_validate_estimator, select_best_model
from sklearn.linear_model import LogisticRegression

cv_result = cross_validate_estimator(
    QuantumReservoirClassifier(seed=0),
    x,
    y,
    cv=3,
    task="classification",
)

selection = select_best_model(
    {
        "logistic": LogisticRegression(max_iter=1000),
        "reservoir": QuantumReservoirClassifier(seed=0),
    },
    x,
    y,
    cv=3,
    task="classification",
)
```

The helpers return dictionaries with fold records, mean/std/95 percent interval
summaries, runtime summaries, scorer metadata, and the refit best estimator when
requested. Defaults are intentionally simple:

| Task | Default scorer | Splitter |
| --- | --- | --- |
| Classification | `accuracy` | `StratifiedKFold` |
| Regression | `neg_mean_squared_error` | `KFold` |

Supported classification scorers are:

```text
accuracy
balanced_accuracy
f1
f1_binary
```

Supported regression scorers are:

```text
neg_mean_squared_error
mean_squared_error
neg_root_mean_squared_error
root_mean_squared_error
neg_mean_absolute_error
mean_absolute_error
r2
```

Use the negative loss variants when you want larger scores to be better across
candidate models.

Model-selection outputs can be rendered with the shared reporting helpers:

```python
from qml.reporting import model_selection_table

print(model_selection_table(selection, title="Cross-validation summary"))
```

For lower-level use, `qml.model_selection.selection_summary_rows(...)` returns
the same normalized rows as dictionaries.

Pass `task` explicitly when floating labels are actually class labels.

The package scorers intentionally avoid estimator-specific score overrides
during cross-validation: predictions are generated with `predict(...)`, then
scored by `score_predictions(...)`. This keeps classifier accuracy and
regression loss semantics consistent across QML and classical estimators.

## Choosing Embeddings

Use the smallest embedding that preserves the task signal:

| Embedding | Use when |
| --- | --- |
| Angle embedding | Features are already scaled and map naturally to rotations. |
| Amplitude embedding | Compact normalized vectors are useful and padding is acceptable. |
| ZZ feature map | Pairwise feature interactions are relevant. |
| IQP feature map | You want a compact fixed feature-map comparison. |
| Data reuploading | Repeated feature injection is part of a trainable circuit design. |

Record feature scaling in notebooks. Embedding comparisons are hard to interpret
without knowing the input range.

## Shots And Noise

Use analytic execution for API smoke tests and initial model comparisons. Add
finite-shot runs when sampling variance is part of the question. Add channel
noise only after the noiseless behavior is understood.

Finite-shot or noisy results are most useful when paired with:

- the matching analytic result
- the same train/test split
- multiple seeds
- runtime summaries
- a short statement about metric degradation

The package-level `noise_model` option supports depolarizing,
amplitude-damping, and readout-error probabilities for circuit-backed QML APIs.
These simulations are robustness checks, not hardware-calibrated claims.

Do not mix analytic and finite-shot results in one ranking unless the execution
mode is clearly labeled.

## Practical Defaults

For a new task:

1. Build deterministic train/test splits.
2. Fit at least one classical baseline.
3. Fit the simplest relevant QML model.
4. Compare test metrics, generalization gaps, and runtime.
5. Add more seeds before interpreting small differences.
6. Add finite-shot or noise-aware execution only after the analytic comparison is understood.

If a QML model does not beat a tuned classical baseline in a small run, it may
still be useful as a circuit, kernel, or feature-map demonstration. The release
notes should describe that scope directly.
