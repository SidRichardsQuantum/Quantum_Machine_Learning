"""
qml.kernels
===========

Reusable quantum kernel estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pennylane as qml
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC

from qml.circuit_metadata import circuit_metadata, embedding_parameter_count
from qml.embeddings import apply_angle_embedding, get_embedding
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.noise import (
    apply_noise_channels,
    device_name_for_noise,
    noise_model_cache_key,
    noise_model_to_dict,
)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    return x


def _cache_key(x: np.ndarray, params: np.ndarray | None) -> tuple[Any, ...]:
    x = np.asarray(x, dtype=float).ravel()
    params_array = None if params is None else np.asarray(params, dtype=float).ravel()
    return (
        x.shape,
        x.tobytes(),
        None if params_array is None else params_array.shape,
        None if params_array is None else params_array.tobytes(),
    )


@dataclass
class QuantumKernel:
    """
    Dataset-agnostic quantum fidelity kernel.

    Parameters
    ----------
    embedding
        Embedding name supported by ``qml.embeddings``. ``"angle"`` is the
        default and needs no trainable parameters.
    params
        Optional trainable embedding parameters for embeddings such as
        ``"data_reupload"``.
    shots
        Optional shot count. ``None`` uses analytic simulation.
    seed
        Device seed.
    noise_model
        Optional channel probabilities for noisy simulation. Supported keys are
        ``depolarizing``, ``amplitude_damping``, and ``readout_error``.
    cache
        Whether to cache pairwise kernel evaluations.
    """

    embedding: str = "angle"
    params: np.ndarray | None = None
    shots: int | None = None
    seed: int = 123
    noise_model: dict[str, float] | None = None
    cache: bool = True
    _pair_cache: dict[tuple[Any, ...], float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.noise_model = noise_model_to_dict(self.noise_model)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style composition."""
        return {
            "embedding": self.embedding,
            "params": self.params,
            "shots": self.shots,
            "seed": self.seed,
            "noise_model": self.noise_model,
            "cache": self.cache,
        }

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style composition."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for QuantumKernel.")
            if key == "noise_model":
                value = noise_model_to_dict(value)
            setattr(self, key, value)
        self.clear_cache()
        return self

    def clear_cache(self) -> None:
        """Clear cached pairwise kernel evaluations."""
        self._pair_cache.clear()

    def _apply_embedding(self, x, wires) -> None:
        name = self.embedding.strip().lower()
        if name in {"angle", "angle_embedding"}:
            apply_angle_embedding(x, wires=wires)
            return

        embedding_fn = get_embedding(name)
        if self.params is None:
            embedding_fn(x, wires=wires)
        else:
            embedding_fn(x, self.params, wires=wires)

    def value(self, x1, x2) -> float:
        """Evaluate ``k(x1, x2)``."""
        x1 = np.asarray(x1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        if x1.shape != x2.shape:
            raise ValueError(
                f"Kernel inputs must have the same shape, got {x1.shape} and {x2.shape}."
            )

        params = None if self.params is None else np.asarray(self.params, dtype=float)
        key = (
            _cache_key(x1, params),
            _cache_key(x2, params),
            self.embedding,
            self.shots,
            noise_model_cache_key(self.noise_model),
        )
        reverse_key = (key[1], key[0], key[2], key[3], key[4])
        if self.cache:
            if key in self._pair_cache:
                return self._pair_cache[key]
            if reverse_key in self._pair_cache:
                return self._pair_cache[reverse_key]

        wires = list(range(x1.shape[0]))
        dev = qml.device(device_name_for_noise(self.noise_model), wires=len(wires), seed=self.seed)

        @qml.qnode(dev)
        def circuit_base(a, b):
            self._apply_embedding(a, wires)
            qml.adjoint(self._apply_embedding)(b, wires)
            apply_noise_channels(wires, self.noise_model, readout_wires=wires)
            return qml.probs(wires=wires)

        circuit = (
            qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base
        )
        value = float(circuit(x1, x2)[0])

        if self.cache:
            self._pair_cache[key] = value
        return value

    def evaluate(self, x, y=None) -> np.ndarray:
        """
        Evaluate the kernel matrix.

        If ``y`` is omitted, returns the square matrix ``K(X, X)``.
        Otherwise returns ``K(X, Y)``.
        """
        x = _as_2d(x)
        y = x if y is None else _as_2d(y)
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"Feature dimensions must match, got {x.shape[1]} and {y.shape[1]}.")

        matrix = np.empty((x.shape[0], y.shape[0]), dtype=float)
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                matrix[i, j] = self.value(xi, yj)
        return matrix


def _split_nested_params(
    params: dict[str, Any], prefix: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    nested_prefix = f"{prefix}__"
    local: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith(nested_prefix):
            nested[key[len(nested_prefix) :]] = value
        else:
            local[key] = value
    return local, nested


def _kernel_circuit_metadata(model: str, kernel: QuantumKernel, n_features: int) -> dict[str, Any]:
    params = None if kernel.params is None else np.asarray(kernel.params)
    trainable_parameters = 0 if params is None else int(params.size)
    try:
        embedding_layers = int(params.shape[0]) if params is not None and params.ndim > 0 else 1
        if trainable_parameters == 0:
            trainable_parameters = embedding_parameter_count(
                kernel.embedding,
                n_layers=embedding_layers,
                n_qubits=n_features,
            )
    except ValueError:
        embedding_layers = 1
    return circuit_metadata(
        model=model,
        n_qubits=n_features,
        n_layers=1,
        embedding=kernel.embedding,
        embedding_layers=embedding_layers,
        ansatz=None,
        template="kernel",
        trainable_parameters=trainable_parameters,
        extra={
            "shots": kernel.shots,
            "noise_model": kernel.noise_model,
        },
    )


class QuantumKernelClassifier:
    """SVM classifier backed by a precomputed quantum kernel."""

    def __init__(
        self,
        kernel: QuantumKernel | None = None,
        *,
        c: float = 1.0,
        seed: int = 123,
        **svc_kwargs,
    ) -> None:
        self.kernel = kernel if kernel is not None else QuantumKernel(seed=seed)
        self.c = c
        self.seed = seed
        self.svc_kwargs = svc_kwargs
        self.model_: SVC | None = None
        self.x_train_: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style model selection."""
        params = {
            "kernel": self.kernel,
            "c": self.c,
            "seed": self.seed,
            **self.svc_kwargs,
        }
        if deep and hasattr(self.kernel, "get_params"):
            params.update(
                {f"kernel__{key}": value for key, value in self.kernel.get_params().items()}
            )
        return params

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
        params, kernel_params = _split_nested_params(params, "kernel")
        if kernel_params:
            self.kernel.set_params(**kernel_params)
        for key, value in params.items():
            if key == "kernel":
                self.kernel = value
            elif key == "c":
                self.c = value
            elif key == "seed":
                self.seed = value
            elif key in self.svc_kwargs:
                self.svc_kwargs[key] = value
            else:
                self.svc_kwargs[key] = value
        return self

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y)
        k_train = self.kernel.evaluate(x)
        self.model_ = SVC(kernel="precomputed", C=self.c, **self.svc_kwargs)
        self.model_.fit(k_train, y)
        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self.classes_ = self.model_.classes_
        self.kernel_matrix_train_ = k_train
        self.circuit_metadata_ = _kernel_circuit_metadata(
            "quantum_kernel_classifier", self.kernel, self.n_features_in_
        )
        return self

    def predict(self, x) -> np.ndarray:
        if self.model_ is None or self.x_train_ is None:
            raise ValueError("QuantumKernelClassifier must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        return self.model_.predict(k_test)

    def score(self, x, y) -> float:
        return accuracy_score(np.asarray(y), self.predict(x))


class QuantumKernelRegressor:
    """Kernel-ridge regressor backed by a precomputed quantum kernel."""

    def __init__(
        self,
        kernel: QuantumKernel | None = None,
        *,
        alpha: float = 1.0,
        seed: int = 123,
        **kernel_ridge_kwargs,
    ) -> None:
        self.kernel = kernel if kernel is not None else QuantumKernel(seed=seed)
        self.alpha = alpha
        self.seed = seed
        self.kernel_ridge_kwargs = kernel_ridge_kwargs
        self.model_: KernelRidge | None = None
        self.x_train_: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style model selection."""
        params = {
            "kernel": self.kernel,
            "alpha": self.alpha,
            "seed": self.seed,
            **self.kernel_ridge_kwargs,
        }
        if deep and hasattr(self.kernel, "get_params"):
            params.update(
                {f"kernel__{key}": value for key, value in self.kernel.get_params().items()}
            )
        return params

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
        params, kernel_params = _split_nested_params(params, "kernel")
        if kernel_params:
            self.kernel.set_params(**kernel_params)
        for key, value in params.items():
            if key == "kernel":
                self.kernel = value
            elif key == "alpha":
                self.alpha = value
            elif key == "seed":
                self.seed = value
            else:
                self.kernel_ridge_kwargs[key] = value
        return self

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y, dtype=float)
        k_train = self.kernel.evaluate(x)
        self.model_ = KernelRidge(
            alpha=self.alpha, kernel="precomputed", **self.kernel_ridge_kwargs
        )
        self.model_.fit(k_train, y)
        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self.kernel_matrix_train_ = k_train
        self.circuit_metadata_ = _kernel_circuit_metadata(
            "quantum_kernel_regressor", self.kernel, self.n_features_in_
        )
        return self

    def predict(self, x) -> np.ndarray:
        if self.model_ is None or self.x_train_ is None:
            raise ValueError("QuantumKernelRegressor must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        return np.asarray(self.model_.predict(k_test), dtype=float)

    def score(self, x, y) -> float:
        y = np.asarray(y, dtype=float)
        pred = self.predict(x)
        return -mean_squared_error(y, pred)

    def mean_absolute_error(self, x, y) -> float:
        return mean_absolute_error(y, self.predict(x))


class QuantumKernelPCA:
    """Kernel PCA using a precomputed quantum fidelity kernel."""

    def __init__(
        self,
        kernel: QuantumKernel | None = None,
        *,
        n_components: int = 2,
        seed: int = 123,
    ) -> None:
        self.kernel = kernel if kernel is not None else QuantumKernel(seed=seed)
        self.n_components = n_components
        self.seed = seed

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"kernel": self.kernel, "n_components": self.n_components, "seed": self.seed}
        if deep and hasattr(self.kernel, "get_params"):
            params.update(
                {f"kernel__{key}": value for key, value in self.kernel.get_params().items()}
            )
        return params

    def set_params(self, **params):
        params, kernel_params = _split_nested_params(params, "kernel")
        if kernel_params:
            self.kernel.set_params(**kernel_params)
        for key, value in params.items():
            if key not in {"kernel", "n_components", "seed"}:
                raise ValueError(f"Invalid parameter {key!r} for QuantumKernelPCA.")
            setattr(self, key, value)
        return self

    def _center_train_kernel(self, k_train: np.ndarray) -> np.ndarray:
        self.train_row_mean_ = k_train.mean(axis=1)
        self.train_mean_ = float(k_train.mean())
        return (
            k_train
            - self.train_row_mean_[:, None]
            - self.train_row_mean_[None, :]
            + self.train_mean_
        )

    def fit(self, x, y=None):
        x = _as_2d(x)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if self.n_components > x.shape[0]:
            raise ValueError("n_components cannot exceed the number of training samples.")

        k_train = self.kernel.evaluate(x)
        k_centered = self._center_train_kernel(k_train)
        eigvals, eigvecs = np.linalg.eigh(k_centered)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        positive = eigvals > 1e-12
        eigvals = eigvals[positive][: self.n_components]
        eigvecs = eigvecs[:, positive][:, : self.n_components]
        if eigvals.shape[0] < self.n_components:
            pad = self.n_components - eigvals.shape[0]
            eigvals = np.pad(eigvals, (0, pad))
            eigvecs = np.pad(eigvecs, ((0, 0), (0, pad)))

        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self.eigenvalues_ = eigvals
        self.alphas_ = eigvecs / np.sqrt(np.where(eigvals > 1e-12, eigvals, 1.0))
        self.embedding_ = k_centered @ self.alphas_
        self.kernel_matrix_train_ = k_train
        self.circuit_metadata_ = _kernel_circuit_metadata(
            "quantum_kernel_pca", self.kernel, self.n_features_in_
        )
        return self

    def transform(self, x) -> np.ndarray:
        if not hasattr(self, "x_train_"):
            raise ValueError("QuantumKernelPCA must be fitted before transform.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        row_mean = k_test.mean(axis=1)
        k_centered = k_test - row_mean[:, None] - self.train_row_mean_[None, :] + self.train_mean_
        return k_centered @ self.alphas_

    def fit_transform(self, x, y=None) -> np.ndarray:
        return self.fit(x).embedding_


class QuantumOneClassClassifier:
    """One-class anomaly detector backed by a precomputed quantum kernel."""

    def __init__(
        self,
        kernel: QuantumKernel | None = None,
        *,
        nu: float = 0.1,
        seed: int = 123,
        **svm_kwargs,
    ) -> None:
        self.kernel = kernel if kernel is not None else QuantumKernel(seed=seed)
        self.nu = nu
        self.seed = seed
        self.svm_kwargs = svm_kwargs

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"kernel": self.kernel, "nu": self.nu, "seed": self.seed, **self.svm_kwargs}
        if deep and hasattr(self.kernel, "get_params"):
            params.update(
                {f"kernel__{key}": value for key, value in self.kernel.get_params().items()}
            )
        return params

    def set_params(self, **params):
        params, kernel_params = _split_nested_params(params, "kernel")
        if kernel_params:
            self.kernel.set_params(**kernel_params)
        for key, value in params.items():
            if key == "kernel":
                self.kernel = value
            elif key == "nu":
                self.nu = value
            elif key == "seed":
                self.seed = value
            else:
                self.svm_kwargs[key] = value
        return self

    def fit(self, x, y=None):
        x = _as_2d(x)
        k_train = self.kernel.evaluate(x)
        self.model_ = OneClassSVM(kernel="precomputed", nu=self.nu, **self.svm_kwargs)
        self.model_.fit(k_train)
        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self.kernel_matrix_train_ = k_train
        self.circuit_metadata_ = _kernel_circuit_metadata(
            "quantum_one_class_classifier", self.kernel, self.n_features_in_
        )
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumOneClassClassifier must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        return self.model_.predict(k_test)

    def decision_function(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumOneClassClassifier must be fitted before scoring.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        return self.model_.decision_function(k_test)

    def score_samples(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumOneClassClassifier must be fitted before scoring.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        return self.model_.score_samples(k_test)


class QuantumGaussianProcessRegressor:
    """Gaussian process regressor using a quantum kernel covariance matrix."""

    def __init__(
        self,
        kernel: QuantumKernel | None = None,
        *,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        seed: int = 123,
    ) -> None:
        self.kernel = kernel if kernel is not None else QuantumKernel(seed=seed)
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.seed = seed

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "kernel": self.kernel,
            "alpha": self.alpha,
            "normalize_y": self.normalize_y,
            "seed": self.seed,
        }
        if deep and hasattr(self.kernel, "get_params"):
            params.update(
                {f"kernel__{key}": value for key, value in self.kernel.get_params().items()}
            )
        return params

    def set_params(self, **params):
        params, kernel_params = _split_nested_params(params, "kernel")
        if kernel_params:
            self.kernel.set_params(**kernel_params)
        for key, value in params.items():
            if key not in {"kernel", "alpha", "normalize_y", "seed"}:
                raise ValueError(f"Invalid parameter {key!r} for QuantumGaussianProcessRegressor.")
            setattr(self, key, value)
        return self

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("y length must match the number of samples.")

        self.y_mean_ = float(y.mean()) if self.normalize_y else 0.0
        y_centered = y - self.y_mean_
        k_train = self.kernel.evaluate(x)
        eye = np.eye(k_train.shape[0])
        jitter = float(self.alpha)
        for _ in range(6):
            regularized = k_train + jitter * eye
            try:
                self.cholesky_ = np.linalg.cholesky(regularized)
                self.effective_alpha_ = jitter
                break
            except np.linalg.LinAlgError:
                jitter = max(1e-8, jitter * 10.0)
        else:
            self.cholesky_ = np.linalg.cholesky(k_train + jitter * eye)
            self.effective_alpha_ = jitter
        tmp = np.linalg.solve(self.cholesky_, y_centered)
        self.dual_coef_ = np.linalg.solve(self.cholesky_.T, tmp)
        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self.kernel_matrix_train_ = k_train
        self.circuit_metadata_ = _kernel_circuit_metadata(
            "quantum_gaussian_process_regressor", self.kernel, self.n_features_in_
        )
        return self

    def predict(self, x, return_std: bool = False):
        if not hasattr(self, "dual_coef_"):
            raise ValueError("QuantumGaussianProcessRegressor must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = self.kernel.evaluate(x, self.x_train_)
        mean = k_test @ self.dual_coef_ + self.y_mean_
        if not return_std:
            return np.asarray(mean, dtype=float)

        v = np.linalg.solve(self.cholesky_, k_test.T)
        k_self = np.asarray([self.kernel.value(sample, sample) for sample in x], dtype=float)
        variance = np.maximum(k_self - np.sum(v * v, axis=0), 0.0)
        return np.asarray(mean, dtype=float), np.sqrt(variance)

    def score(self, x, y) -> float:
        return -mean_squared_error(y, self.predict(x))


def kernel_target_alignment(kernel_matrix, labels) -> float:
    """Compute normalized kernel-target alignment for classification labels."""
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)
    labels = np.asarray(labels)
    if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
        raise ValueError("kernel_matrix must be square.")
    if kernel_matrix.shape[0] != labels.shape[0]:
        raise ValueError("labels length must match kernel_matrix size.")

    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("kernel_target_alignment currently expects binary labels.")

    y_pm = np.where(labels == classes[0], -1.0, 1.0)
    target = np.outer(y_pm, y_pm)
    numerator = float(np.sum(kernel_matrix * target))
    kernel_norm = float(np.sqrt(np.sum(kernel_matrix * kernel_matrix) + 1e-12))
    target_norm = float(np.sqrt(np.sum(target * target) + 1e-12))
    return numerator / (kernel_norm * target_norm + 1e-12)
