"""
qml.reservoir
=============

Quantum reservoir estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression, Ridge

from qml.circuit_metadata import circuit_metadata
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.noise import apply_noise_channels, device_name_for_noise, noise_model_to_dict


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    return x


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


def _reservoir_circuit_metadata(
    model: str, reservoir: "QuantumReservoirFeatures"
) -> dict[str, Any]:
    return circuit_metadata(
        model=model,
        n_qubits=reservoir.n_qubits_,
        n_layers=reservoir.n_layers,
        embedding="reservoir_angle",
        embedding_layers=1,
        ansatz=None,
        template="variational",
        trainable_parameters=0,
        extra={
            "fixed_random_parameters": int(reservoir.weights_.size),
            "shots": reservoir.shots,
            "noise_model": reservoir.noise_model,
            "input_scale": reservoir.input_scale,
            "weight_scale": reservoir.weight_scale,
        },
    )


class QuantumReservoirFeatures:
    """
    Fixed random quantum reservoir feature map.

    Classical inputs are angle encoded, evolved through fixed random rotations
    and entanglers, then mapped to Pauli-Z expectation features.
    """

    def __init__(
        self,
        *,
        n_qubits: int | None = None,
        n_layers: int = 2,
        seed: int = 123,
        shots: int | None = None,
        noise_model: dict[str, float] | None = None,
        input_scale: float = 1.0,
        weight_scale: float = 1.0,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        self.shots = shots
        self.noise_model = noise_model_to_dict(noise_model)
        self.input_scale = input_scale
        self.weight_scale = weight_scale

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "shots": self.shots,
            "noise_model": self.noise_model,
            "input_scale": self.input_scale,
            "weight_scale": self.weight_scale,
        }

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for QuantumReservoirFeatures.")
            if key == "noise_model":
                value = noise_model_to_dict(value)
            setattr(self, key, value)
        return self

    def fit(self, x, y=None):
        x = _as_2d(x)
        n_qubits = self.n_qubits or x.shape[1]
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        rng = np.random.default_rng(self.seed)
        self.n_features_in_ = x.shape[1]
        self.n_qubits_ = n_qubits
        self.weights_ = self.weight_scale * rng.standard_normal((self.n_layers, n_qubits, 3))
        return self

    def _scaled_input(self, sample: np.ndarray) -> np.ndarray:
        repeated = np.resize(sample, self.n_qubits_)
        return self.input_scale * np.pi * np.tanh(repeated)

    def _make_qnode(self):
        wires = list(range(self.n_qubits_))
        dev = qml.device(
            device_name_for_noise(self.noise_model), wires=self.n_qubits_, seed=self.seed
        )

        @qml.qnode(dev)
        def circuit_base(sample, weights):
            angles = self._scaled_input(sample)
            for wire, angle in zip(wires, angles):
                qml.RY(angle, wires=wire)

            for layer in range(weights.shape[0]):
                for i, wire in enumerate(wires):
                    qml.RX(weights[layer, i, 0], wires=wire)
                    qml.RY(weights[layer, i, 1], wires=wire)
                    qml.RZ(weights[layer, i, 2], wires=wire)

                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
                if len(wires) > 2:
                    qml.CNOT(wires=[wires[-1], wires[0]])

            apply_noise_channels(wires, self.noise_model, readout_wires=wires)
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        return qml.set_shots(circuit_base, self.shots) if self.shots is not None else circuit_base

    def transform(self, x) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise ValueError("QuantumReservoirFeatures must be fitted before transform.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        circuit = self._make_qnode()
        return np.asarray([circuit(sample, self.weights_) for sample in x], dtype=float)

    def fit_transform(self, x, y=None) -> np.ndarray:
        return self.fit(x).transform(x)


class QuantumReservoirRegressor:
    """Ridge regressor trained on fixed quantum reservoir features."""

    def __init__(
        self,
        reservoir: QuantumReservoirFeatures | None = None,
        *,
        alpha: float = 1.0,
        seed: int = 123,
        **ridge_kwargs,
    ) -> None:
        self.reservoir = reservoir if reservoir is not None else QuantumReservoirFeatures(seed=seed)
        self.alpha = alpha
        self.seed = seed
        self.ridge_kwargs = ridge_kwargs

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "reservoir": self.reservoir,
            "alpha": self.alpha,
            "seed": self.seed,
            **self.ridge_kwargs,
        }
        if deep and hasattr(self.reservoir, "get_params"):
            params.update(
                {f"reservoir__{key}": value for key, value in self.reservoir.get_params().items()}
            )
        return params

    def set_params(self, **params):
        params, reservoir_params = _split_nested_params(params, "reservoir")
        if reservoir_params:
            self.reservoir.set_params(**reservoir_params)
        for key, value in params.items():
            if key == "reservoir":
                self.reservoir = value
            elif key == "alpha":
                self.alpha = value
            elif key == "seed":
                self.seed = value
            else:
                self.ridge_kwargs[key] = value
        return self

    def fit(self, x, y):
        features = self.reservoir.fit_transform(x)
        self.model_ = Ridge(alpha=self.alpha, **self.ridge_kwargs)
        self.model_.fit(features, np.asarray(y, dtype=float))
        self.n_features_in_ = self.reservoir.n_features_in_
        self.feature_matrix_train_ = features
        self.circuit_metadata_ = _reservoir_circuit_metadata(
            "quantum_reservoir_regressor", self.reservoir
        )
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumReservoirRegressor must be fitted before prediction.")
        return np.asarray(self.model_.predict(self.reservoir.transform(x)), dtype=float)

    def score(self, x, y) -> float:
        return -mean_squared_error(y, self.predict(x))

    def mean_absolute_error(self, x, y) -> float:
        return mean_absolute_error(y, self.predict(x))


class QuantumReservoirClassifier:
    """Logistic classifier trained on fixed quantum reservoir features."""

    def __init__(
        self,
        reservoir: QuantumReservoirFeatures | None = None,
        *,
        c: float = 1.0,
        seed: int = 123,
        max_iter: int = 1000,
        **logistic_kwargs,
    ) -> None:
        self.reservoir = reservoir if reservoir is not None else QuantumReservoirFeatures(seed=seed)
        self.c = c
        self.seed = seed
        self.max_iter = max_iter
        self.logistic_kwargs = logistic_kwargs

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "reservoir": self.reservoir,
            "c": self.c,
            "seed": self.seed,
            "max_iter": self.max_iter,
            **self.logistic_kwargs,
        }
        if deep and hasattr(self.reservoir, "get_params"):
            params.update(
                {f"reservoir__{key}": value for key, value in self.reservoir.get_params().items()}
            )
        return params

    def set_params(self, **params):
        params, reservoir_params = _split_nested_params(params, "reservoir")
        if reservoir_params:
            self.reservoir.set_params(**reservoir_params)
        for key, value in params.items():
            if key == "reservoir":
                self.reservoir = value
            elif key == "c":
                self.c = value
            elif key == "seed":
                self.seed = value
            elif key == "max_iter":
                self.max_iter = value
            else:
                self.logistic_kwargs[key] = value
        return self

    def fit(self, x, y):
        features = self.reservoir.fit_transform(x)
        self.model_ = LogisticRegression(C=self.c, max_iter=self.max_iter, **self.logistic_kwargs)
        self.model_.fit(features, np.asarray(y))
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.reservoir.n_features_in_
        self.feature_matrix_train_ = features
        self.circuit_metadata_ = _reservoir_circuit_metadata(
            "quantum_reservoir_classifier", self.reservoir
        )
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumReservoirClassifier must be fitted before prediction.")
        return self.model_.predict(self.reservoir.transform(x))

    def predict_proba(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("QuantumReservoirClassifier must be fitted before prediction.")
        return self.model_.predict_proba(self.reservoir.transform(x))

    def score(self, x, y) -> float:
        return accuracy_score(np.asarray(y), self.predict(x))
