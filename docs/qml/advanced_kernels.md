# Advanced Quantum Kernel Models

This page documents general-purpose kernel estimators built on
`qml.kernels.QuantumKernel`. They are domain-agnostic and consume user-supplied
arrays.

## Shared Kernel

All advanced kernel estimators use a fidelity kernel:

```python
from qml import QuantumKernel

kernel = QuantumKernel(embedding="zz", seed=123)
```

Supported non-trainable embeddings include `angle`, `amplitude`, `zz`, and
`iqp`. Trainable kernels can use `data_reupload`.

## QuantumKernelPCA

`QuantumKernelPCA` performs kernel PCA with a quantum fidelity kernel matrix.

```python
from qml import QuantumKernel, QuantumKernelPCA

kpca = QuantumKernelPCA(QuantumKernel(embedding="iqp"), n_components=2)
z_train = kpca.fit_transform(x_train)
z_test = kpca.transform(x_test)
```

Use it for unsupervised visualization, dimensionality reduction, and checking
whether a feature map separates regimes before fitting a classifier.

Returned attributes after fitting:

| Attribute | Meaning |
| --- | --- |
| `x_train_` | Training features used as kernel landmarks. |
| `eigenvalues_` | Leading centered-kernel eigenvalues. |
| `alphas_` | Normalized projection coefficients. |
| `embedding_` | Training samples projected into component space. |

## QuantumOneClassClassifier

`QuantumOneClassClassifier` wraps scikit-learn one-class SVM over a precomputed
quantum kernel.

```python
from qml import QuantumKernel, QuantumOneClassClassifier

detector = QuantumOneClassClassifier(QuantumKernel(embedding="zz"), nu=0.1)
detector.fit(x_normal)
labels = detector.predict(x_candidate)
scores = detector.decision_function(x_candidate)
```

Predictions follow the scikit-learn convention:

| Prediction | Meaning |
| --- | --- |
| `1` | Inlier / normal-like sample. |
| `-1` | Outlier / anomalous sample. |

## QuantumGaussianProcessRegressor

`QuantumGaussianProcessRegressor` treats the quantum kernel matrix as a
Gaussian-process covariance matrix.

```python
from qml import QuantumGaussianProcessRegressor, QuantumKernel

model = QuantumGaussianProcessRegressor(
    QuantumKernel(embedding="iqp"),
    alpha=1e-4,
)
model.fit(x_train, y_train)
mean, std = model.predict(x_test, return_std=True)
```

Use it when uncertainty estimates are useful, for example interpolation,
surrogate modeling, or inverse problems with sparse observations.

## TrainableQuantumKernelRegressor

`TrainableQuantumKernelRegressor` optimizes a parameterized feature map by
maximizing normalized alignment between the quantum kernel and the continuous
target kernel:

$$
A(K, yy^T) =
\frac{\langle K, yy^T \rangle_F}
{\|K\|_F \|yy^T\|_F}.
$$

After kernel training, it fits kernel-ridge regression on the learned kernel.

```python
from qml import TrainableQuantumKernelRegressor

model = TrainableQuantumKernelRegressor(
    embedding="data_reupload",
    embedding_layers=1,
    steps=10,
    alpha=1e-3,
    seed=123,
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
```

Important fitted attributes:

| Attribute | Meaning |
| --- | --- |
| `trained_params_` | Learned feature-map parameters. |
| `loss_trace_` | Alignment-loss values during training. |
| `alignment_` | Final continuous target alignment. |
| `kernel_matrix_train_` | Learned training kernel matrix. |

## Workflow Helper

For small package-level smoke workflows:

```python
from qml import run_trainable_quantum_kernel_regressor

result = run_trainable_quantum_kernel_regressor(
    dataset="sine",
    n_samples=40,
    embedding_layers=1,
    steps=5,
)
```

The helper returns a dictionary with train/test predictions, kernel matrices,
loss trace, alignment, and regression metrics.
