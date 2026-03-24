"""
qml.ansatz
==========

Parameterized circuit templates for quantum machine learning models.
"""

from __future__ import annotations

from collections.abc import Sequence

import pennylane as qml
from pennylane import numpy as pnp


def validate_parameter_vector(params, n_params: int | None = None):
    """
    Validate and return a 1D parameter vector.

    This helper is intended for ordinary array validation outside traced
    autodiff/QNode execution paths.
    """
    params = pnp.ravel(pnp.asarray(params))
    if n_params is not None and params.shape[0] != n_params:
        raise ValueError(f"Expected parameter vector of length {n_params}, got {params.shape[0]}.")
    return params


def parameter_shape(n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    """
    Return the parameter shape for the default hardware-efficient ansatz.

    Each qubit in each layer gets two trainable angles: RY and RZ.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    return (n_layers, n_qubits, 2)


def apply_hardware_efficient_ansatz(params, wires: Sequence[int]) -> None:
    """
    Apply a simple hardware-efficient ansatz.

    Parameters
    ----------
    params
        Array-like of shape ``(n_layers, n_qubits, 2)``.
    wires
        Circuit wires.
    """
    wires = list(wires)
    shape = qml.math.shape(params)

    if len(shape) != 3:
        raise ValueError(
            f"Expected params with rank 3 and shape (n_layers, {len(wires)}, 2), got {shape}."
        )

    if shape[1] != len(wires) or shape[2] != 2:
        raise ValueError(f"Expected params with shape (n_layers, {len(wires)}, 2), got {shape}.")

    n_layers = shape[0]
    n_qubits = len(wires)

    for layer in range(n_layers):
        for i, wire in enumerate(wires):
            qml.RY(params[layer, i, 0], wires=wire)
            qml.RZ(params[layer, i, 1], wires=wire)

        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

            if n_qubits > 2:
                qml.CNOT(wires=[wires[-1], wires[0]])
