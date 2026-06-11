# Estimator Consistency Checklist

This checklist records the v0.2.12 generalization pass for estimator-style
APIs. Rows marked `Done` are covered by implementation changes and focused
tests at smoke scale.

## Target Classes

| Class | Module | Role | Required surface to audit |
| --- | --- | --- | --- |
| `QuantumClassifier` | `qml.estimators` | Variational classifier for user-supplied arrays | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumRegressor` | `qml.estimators` | Variational regressor for user-supplied arrays | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumKernelClassifier` | `qml.kernels` | SVM classifier over a quantum fidelity kernel | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumKernelRegressor` | `qml.kernels` | Kernel-ridge regressor over a quantum fidelity kernel | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `TrainableQuantumKernelRegressor` | `qml.trainable_kernels` | Alignment-trained kernel regressor | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumReservoirClassifier` | `qml.reservoir` | Logistic classifier over fixed quantum reservoir features | `fit`, `predict`, `predict_proba`, `score`, `get_params`, `set_params` |
| `QuantumReservoirRegressor` | `qml.reservoir` | Ridge regressor over fixed quantum reservoir features | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumGaussianProcessRegressor` | `qml.kernels` | Gaussian-process regressor over a quantum kernel | `fit`, `predict`, `score`, `get_params`, `set_params` |
| `QuantumKernelPCA` | `qml.kernels` | Kernel PCA over a quantum fidelity kernel | `fit`, `transform`, `fit_transform`, `get_params`, `set_params` |
| `QuantumOneClassClassifier` | `qml.kernels` | One-class anomaly detector over a quantum fidelity kernel | `fit`, `predict`, `decision_function`, `score_samples`, `get_params`, `set_params` |

`QuantumReservoirFeatures` and `QuantumKernel` are supporting transformers or
kernel objects rather than full predictive estimators. They should still keep
compatible `get_params` and `set_params` behavior because the classifier and
regressor wrappers compose them.

## Audit Matrix

Use this matrix to track the first pass. A row is complete when both behavior
and tests are updated.

| Check | `QuantumClassifier` | `QuantumRegressor` | `QuantumKernelClassifier` | `QuantumKernelRegressor` | `TrainableQuantumKernelRegressor` | `QuantumReservoirClassifier` | `QuantumReservoirRegressor` | `QuantumGaussianProcessRegressor` | `QuantumKernelPCA` | `QuantumOneClassClassifier` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Constructor arguments round-trip through `get_params` and `set_params` | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| Methods raise clear fitted-state errors before `fit` | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| `fit` records `n_features_in_` when input arrays are two-dimensional | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| Classifiers record `classes_` after fitting | Done | N/A | Done | N/A | N/A | Done | N/A | N/A | N/A | N/A |
| `predict` or `transform` preserves expected sample dimension | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| `score` semantics are documented and tested | Done | Done | Done | Done | Done | Done | Done | Done | N/A | N/A |
| Repeated fits with the same seed are deterministic at smoke scale | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| Invalid shapes and target types raise actionable `ValueError`s | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| Finite-shot and `noise_model` support is accepted, rejected, or documented explicitly | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |
| Fitted attributes use consistent trailing-underscore names | Done | Done | Done | Done | Done | Done | Done | Done | Done | Done |

## v0.2.12 Notes

- Kernel and reservoir wrappers expose useful nested parameters through
  `get_params(deep=True)` and `set_params(...)`, including `kernel__shots`,
  `kernel__embedding`, `reservoir__n_layers`, and `reservoir__noise_model`.
- `qml.model_selection.clone_estimator(...)` clones from shallow constructor
  parameters so composed kernel and reservoir objects keep their configured
  nested state without passing `kernel__...` or `reservoir__...` keys to
  constructors.
- Circuit-backed fitted estimators expose `circuit_metadata_` where the
  estimator owns the circuit execution path. Kernel wrappers also retain
  `kernel_matrix_train_`; reservoir wrappers retain `feature_matrix_train_`.
- `QuantumOneClassClassifier` exposes `score_samples(...)` in addition to
  `decision_function(...)`.

## Expected Attribute Conventions

Use trailing underscores for values learned or inferred during `fit`:

| Attribute | Applies to | Meaning |
| --- | --- | --- |
| `n_features_in_` | All fitted array estimators | Number of input features seen during `fit`. |
| `classes_` | Classifiers | Sorted or model-native class labels used for predictions. |
| `params_` or `trained_params_` | Trainable quantum models | Learned circuit or feature-map parameters. |
| `loss_history_`, `loss_trace_`, or `alignment_trace_` | Optimized models | Training objective history. The naming should be normalized where practical. |
| `model_` or `estimator_` | Wrappers around scikit-learn models | Fitted downstream classical model. |
| `kernel_matrix_train_` | Kernel wrappers | Quantum kernel matrix used to fit the downstream classical model. |
| `feature_matrix_train_` | Reservoir wrappers | Quantum reservoir feature matrix used to fit the downstream classical model. |
| `circuit_metadata_` | Circuit-backed estimators | Qubit count, trainable-parameter count, estimated depth, embedding, ansatz, and execution mode metadata. |

## Test Targets

Focused coverage lives in:

- `tests/test_estimator_consistency.py`
- `tests/test_algorithm_truth_contracts.py`

Keep or expand these test targets before broad refactors:

- parameter round-trip tests for every target class
- fitted-state error tests for prediction, transform, scoring, and decision
  methods
- deterministic seed smoke tests for trainable and random-feature models
- prediction-shape tests for classifiers and regressors
- transform-shape tests for `QuantumKernelPCA`
- probability-shape tests for `QuantumReservoirClassifier.predict_proba`
- invalid input-shape tests with clear error-message expectations
- finite-shot and noise-model compatibility tests for circuit-backed
  estimators

## Documentation Targets

The audit updates landed in:

- `docs/qml/api_reference.md` with the estimator consistency guarantees
- `docs/qml/implementation_contracts.md` with fitted-state, finite-shot, and
  noise-model behavior
- `docs/qml/model_selection.md` with any estimator limitations relevant to
  cross-validation and reporting helpers
