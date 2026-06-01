# Algorithm Documentation Coverage

This page maps the implemented QML algorithm surfaces to their theory notes,
API documentation, tutorials, benchmarks, and generated web pages. The package
documents reusable algorithm families rather than creating separate theory pages
for every thin workflow wrapper.

## Coverage Matrix

| Implementation | Theory / method note | Web page | Tutorial coverage | Benchmark / result coverage |
| --- | --- | --- | --- | --- |
| `run_vqc(...)`, `QuantumClassifier` | `docs/qml/variational_quantum_classifier.md` | `variational-quantum-classifier.html` | `notebooks/tutorials/03-variational-quantum-classifier.ipynb`, `notebooks/tutorials/07-variational-quantum-estimators.ipynb` | classification, capacity, finite-shot, noise-model benchmarks |
| `run_vqr(...)`, `QuantumRegressor` | `docs/qml/variational_regression.md` | `variational-regression.html` | `notebooks/tutorials/04-variational-quantum-regressor.ipynb`, `notebooks/tutorials/07-variational-quantum-estimators.ipynb` | regression, capacity, finite-shot, noise-model benchmarks |
| `run_qcnn(...)` | `docs/qml/qcnn.md` | `qcnn.html` | `notebooks/tutorials/10-quantum-convolutional-neural-network.ipynb` | classification, capacity, finite-shot benchmarks |
| `run_quantum_autoencoder(...)` | `docs/qml/autoencoder.md` | `autoencoder.html` | `notebooks/tutorials/11-quantum-autoencoder.ipynb` | API reference results |
| `run_quantum_kernel_classifier(...)`, `QuantumKernelClassifier` | `docs/qml/quantum_kernels.md` | `quantum-kernels.html` | `notebooks/tutorials/05-quantum-kernel-classifier.ipynb`, `notebooks/tutorials/06-quantum-kernel-estimators.ipynb` | classification, kernel-family, finite-shot, noise-model benchmarks |
| `QuantumKernelRegressor` | `docs/qml/quantum_kernels.md` | `quantum-kernels.html` | `notebooks/tutorials/06-quantum-kernel-estimators.ipynb` | regression, finite-shot, noise-model benchmarks |
| `run_trainable_quantum_kernel_classifier(...)` | `docs/qml/advanced_kernels.md` | `advanced-kernels.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | classification and kernel-family benchmarks |
| `TrainableQuantumKernelRegressor`, `run_trainable_quantum_kernel_regressor(...)` | `docs/qml/advanced_kernels.md` | `advanced-kernels.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | regression benchmarks |
| `QuantumKernelPCA` | `docs/qml/advanced_kernels.md` | `advanced-kernels.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | real-example notebooks |
| `QuantumOneClassClassifier` | `docs/qml/advanced_kernels.md` | `advanced-kernels.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | real-example notebooks |
| `QuantumGaussianProcessRegressor` | `docs/qml/advanced_kernels.md` | `advanced-kernels.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | regression and real-example notebooks |
| `QuantumReservoirFeatures`, `QuantumReservoirClassifier`, `QuantumReservoirRegressor` | `docs/qml/quantum_reservoirs.md` | `quantum-reservoirs.html` | `notebooks/tutorials/12-advanced-quantum-kernel-and-reservoir-models.ipynb` | classification, regression, finite-shot, and real-example notebooks |
| `run_quantum_metric_learner(...)` | `docs/qml/metric_learning.md` | `metric-learning.html` | `notebooks/tutorials/09-quantum-metric-learning.ipynb` | classification benchmarks |

## Supporting Pages

| Support surface | Documentation | Web page |
| --- | --- | --- |
| Embeddings and ansatz helpers | `docs/qml/embeddings.md` | `embeddings.html` |
| Noise models | `docs/qml/noise_models.md` | `noise-models.html` |
| Classical baselines | `docs/qml/classical_baselines.md` | `classical-baselines.html` |
| Benchmark helpers | `docs/qml/benchmarks.md` | `benchmarks.html` |
| Model selection helpers | `docs/qml/model_selection.md` | `model-selection.html` |
| Implementation contracts | `docs/qml/implementation_contracts.md` | `implementation-contracts.html` |

## Documentation Policy

Each advertised QML algorithm should have:

- a theory or method note in `docs/qml/`
- a generated web page in the static site
- API reference coverage
- at least one tutorial, benchmark, or real-example notebook path
- implementation-contract coverage for objective, circuit family, readout, and
  metric semantics

Thin estimator wrappers share the theory page of the underlying algorithm. For
example, `QuantumClassifier` and `QuantumRegressor` share the VQC and VQR theory
pages because they expose sklearn-style `fit`, `predict`, and `score` methods
around the same variational circuit families.
