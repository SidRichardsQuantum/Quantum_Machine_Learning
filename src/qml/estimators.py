"""
qml.estimators
==============

Dataset-agnostic variational quantum estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qml.ansatz import apply_hardware_efficient_ansatz, parameter_shape
from qml.embeddings import apply_angle_embedding
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.optimizers import get_optimizer
from qml.training import run_training_loop


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    return x


@dataclass
class _TrainedCircuit:
    params: np.ndarray
    loss_history: list[float]


class QuantumRegressor:
    """
    Variational quantum regressor with a sklearn-like API.

    Multi-output targets are handled by fitting one independent variational
    readout per target column.
    """

    def __init__(
        self,
        *,
        n_layers: int = 2,
        steps: int = 50,
        step_size: float = 0.1,
        optimizer: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        seed: int = 123,
        shots: int | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.steps = steps
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.seed = seed
        self.shots = shots

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style model selection."""
        return {
            "n_layers": self.n_layers,
            "steps": self.steps,
            "step_size": self.step_size,
            "optimizer": self.optimizer,
            "optimizer_kwargs": dict(self.optimizer_kwargs),
            "seed": self.seed,
            "shots": self.shots,
        }

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for QuantumRegressor.")
            setattr(self, key, value)
        return self

    def _fit_single(self, x: np.ndarray, y: np.ndarray, seed: int) -> _TrainedCircuit:
        n_qubits = x.shape[1]
        wires = list(range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits, seed=seed)

        @qml.qnode(dev, interface="autograd")
        def circuit_base(sample, params):
            apply_angle_embedding(sample, wires=wires)
            apply_hardware_efficient_ansatz(params, wires=wires)
            return qml.expval(qml.PauliZ(wires[0]))

        circuit = (
            qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base
        )

        def predict_batch(samples, params):
            return pnp.array([circuit(sample, params) for sample in samples])

        def cost(params):
            preds = predict_batch(x, params)
            return pnp.mean((preds - pnp.asarray(y, dtype=float)) ** 2)

        rng = np.random.default_rng(seed)
        init = 0.01 * rng.standard_normal(parameter_shape(self.n_layers, n_qubits))
        params = pnp.array(init, requires_grad=True)
        opt = get_optimizer(self.optimizer, stepsize=self.step_size, **self.optimizer_kwargs)

        def step_fn(current_params):
            return opt.step_and_cost(cost, current_params)

        params, loss_history = run_training_loop(step_fn, params, self.steps)
        return _TrainedCircuit(params=np.asarray(params, dtype=float), loss_history=loss_history)

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y, dtype=float)
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
        if y_2d.ndim != 2 or y_2d.shape[0] != x.shape[0]:
            raise ValueError("y must have shape (n_samples,) or (n_samples, n_outputs).")

        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y_2d.shape[1]
        self.models_ = [
            self._fit_single(x, y_2d[:, output], self.seed + output)
            for output in range(self.n_outputs_)
        ]
        self.loss_history_ = [model.loss_history for model in self.models_]
        return self

    def _predict_single(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        n_qubits = x.shape[1]
        wires = list(range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits, seed=self.seed)

        @qml.qnode(dev)
        def circuit_base(sample, current_params):
            apply_angle_embedding(sample, wires=wires)
            apply_hardware_efficient_ansatz(current_params, wires=wires)
            return qml.expval(qml.PauliZ(wires[0]))

        circuit = (
            qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base
        )
        return np.asarray([circuit(sample, params) for sample in x], dtype=float)

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("QuantumRegressor must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        preds = np.column_stack([self._predict_single(x, model.params) for model in self.models_])
        return preds.ravel() if self.n_outputs_ == 1 else preds

    def score(self, x, y) -> float:
        return -mean_squared_error(y, self.predict(x))

    def mean_absolute_error(self, x, y) -> float:
        return mean_absolute_error(y, self.predict(x))


class QuantumClassifier:
    """
    Variational quantum classifier with binary and one-vs-rest multiclass support.
    """

    def __init__(
        self,
        *,
        n_layers: int = 2,
        steps: int = 50,
        step_size: float = 0.1,
        optimizer: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        seed: int = 123,
        shots: int | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.steps = steps
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.seed = seed
        self.shots = shots

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style model selection."""
        return {
            "n_layers": self.n_layers,
            "steps": self.steps,
            "step_size": self.step_size,
            "optimizer": self.optimizer,
            "optimizer_kwargs": dict(self.optimizer_kwargs),
            "seed": self.seed,
            "shots": self.shots,
        }

    def set_params(self, **params):
        """Set constructor parameters for sklearn-style model selection."""
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for QuantumClassifier.")
            setattr(self, key, value)
        return self

    def _fit_binary(self, x: np.ndarray, y_binary: np.ndarray, seed: int) -> _TrainedCircuit:
        n_qubits = x.shape[1]
        wires = list(range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits, seed=seed)

        @qml.qnode(dev, interface="autograd")
        def circuit_base(sample, params):
            apply_angle_embedding(sample, wires=wires)
            apply_hardware_efficient_ansatz(params, wires=wires)
            return qml.expval(qml.PauliZ(wires[0]))

        circuit = (
            qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base
        )

        def predict_proba_batch(samples, params):
            return pnp.array([0.5 * (1.0 - circuit(sample, params)) for sample in samples])

        def cost(params):
            eps = 1e-8
            probs = pnp.clip(predict_proba_batch(x, params), eps, 1.0 - eps)
            targets = pnp.asarray(y_binary, dtype=float)
            return -pnp.mean(targets * pnp.log(probs) + (1.0 - targets) * pnp.log(1.0 - probs))

        rng = np.random.default_rng(seed)
        init = 0.01 * rng.standard_normal(parameter_shape(self.n_layers, n_qubits))
        params = pnp.array(init, requires_grad=True)
        opt = get_optimizer(self.optimizer, stepsize=self.step_size, **self.optimizer_kwargs)

        def step_fn(current_params):
            return opt.step_and_cost(cost, current_params)

        params, loss_history = run_training_loop(step_fn, params, self.steps)
        return _TrainedCircuit(params=np.asarray(params, dtype=float), loss_history=loss_history)

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must have shape (n_samples,).")

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required.")

        self.n_features_in_ = x.shape[1]
        self.models_ = []
        for idx, cls in enumerate(self.classes_):
            if len(self.classes_) == 2 and idx == 0:
                continue
            binary = (y == cls).astype(float)
            self.models_.append(self._fit_binary(x, binary, self.seed + idx))
        self.loss_history_ = [model.loss_history for model in self.models_]
        return self

    def _predict_proba_binary(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        n_qubits = x.shape[1]
        wires = list(range(n_qubits))
        dev = qml.device("default.qubit", wires=n_qubits, seed=self.seed)

        @qml.qnode(dev)
        def circuit_base(sample, current_params):
            apply_angle_embedding(sample, wires=wires)
            apply_hardware_efficient_ansatz(current_params, wires=wires)
            return qml.expval(qml.PauliZ(wires[0]))

        circuit = (
            qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base
        )
        return np.asarray([0.5 * (1.0 - circuit(sample, params)) for sample in x], dtype=float)

    def predict_proba(self, x) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise ValueError("QuantumClassifier must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")

        if len(self.classes_) == 2:
            positive = self._predict_proba_binary(x, self.models_[0].params)
            return np.column_stack([1.0 - positive, positive])

        scores = np.column_stack(
            [self._predict_proba_binary(x, model.params) for model in self.models_]
        )
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        return scores / row_sums

    def predict(self, x) -> np.ndarray:
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, x, y) -> float:
        return accuracy_score(np.asarray(y), self.predict(x))
