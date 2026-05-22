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
from sklearn.svm import SVC

from qml.embeddings import apply_angle_embedding, get_embedding
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error


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
    cache
        Whether to cache pairwise kernel evaluations.
    """

    embedding: str = "angle"
    params: np.ndarray | None = None
    shots: int | None = None
    seed: int = 123
    cache: bool = True
    _pair_cache: dict[tuple[Any, ...], float] = field(default_factory=dict, init=False)

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
        key = (_cache_key(x1, params), _cache_key(x2, params), self.embedding, self.shots)
        reverse_key = (key[1], key[0], key[2], key[3])
        if self.cache:
            if key in self._pair_cache:
                return self._pair_cache[key]
            if reverse_key in self._pair_cache:
                return self._pair_cache[reverse_key]

        wires = list(range(x1.shape[0]))
        dev = qml.device("default.qubit", wires=len(wires), seed=self.seed)

        @qml.qnode(dev)
        def circuit_base(a, b):
            self._apply_embedding(a, wires)
            qml.adjoint(self._apply_embedding)(b, wires)
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
        return {
            "kernel": self.kernel,
            "c": self.c,
            "seed": self.seed,
            **self.svc_kwargs,
        }

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
        for key, value in params.items():
            if key == "kernel":
                self.kernel = value
            elif key == "c":
                self.c = value
            elif key == "seed":
                self.seed = value
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
        self.classes_ = self.model_.classes_
        return self

    def predict(self, x) -> np.ndarray:
        if self.model_ is None or self.x_train_ is None:
            raise ValueError("QuantumKernelClassifier must be fitted before prediction.")
        k_test = self.kernel.evaluate(_as_2d(x), self.x_train_)
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
        return {
            "kernel": self.kernel,
            "alpha": self.alpha,
            "seed": self.seed,
            **self.kernel_ridge_kwargs,
        }

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
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
        return self

    def predict(self, x) -> np.ndarray:
        if self.model_ is None or self.x_train_ is None:
            raise ValueError("QuantumKernelRegressor must be fitted before prediction.")
        k_test = self.kernel.evaluate(_as_2d(x), self.x_train_)
        return np.asarray(self.model_.predict(k_test), dtype=float)

    def score(self, x, y) -> float:
        y = np.asarray(y, dtype=float)
        pred = self.predict(x)
        return -mean_squared_error(y, pred)

    def mean_absolute_error(self, x, y) -> float:
        return mean_absolute_error(y, self.predict(x))


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
