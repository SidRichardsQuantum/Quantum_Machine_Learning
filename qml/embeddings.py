"""
qml.embeddings
==============

Feature-embedding utilities for encoding classical data into quantum circuits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pennylane as qml


def validate_feature_vector(x, n_features: int | None = None) -> np.ndarray:
    """
    Validate and return a 1D feature vector.

    Parameters
    ----------
    x
        Input feature vector.
    n_features
        Expected feature dimension, if enforced.

    Returns
    -------
    np.ndarray
        Validated 1D NumPy array.
    """
    x = np.asarray(x, dtype=float).ravel()
    if n_features is not None and x.shape[0] != n_features:
        raise ValueError(f"Expected feature vector of length {n_features}, got {x.shape[0]}.")
    return x


def apply_angle_embedding(x, wires: Sequence[int], rotation: str = "Y") -> None:
    """
    Apply an angle embedding.

    Parameters
    ----------
    x
        Input feature vector.
    wires
        Wires on which to apply the embedding.
    rotation
        Rotation basis passed to ``qml.AngleEmbedding``.
    """
    wires = list(wires)
    x = validate_feature_vector(x, n_features=len(wires))
    qml.AngleEmbedding(features=x, wires=wires, rotation=rotation)


def apply_data_reuploading_embedding(
    x,
    weights,
    wires: Sequence[int],
) -> None:
    wires = list(wires)
    x = validate_feature_vector(x, n_features=len(wires))

    if weights.ndim != 3:
        raise ValueError("weights must have shape (n_layers, n_qubits, 3)")

    if weights.shape[1:] != (len(wires), 3):
        raise ValueError(f"Expected weights shape (n_layers, {len(wires)}, 3), got {weights.shape}")

    for layer_weights in weights:

        for i, wire in enumerate(wires):
            qml.RY(x[i], wires=wire)
            qml.Rot(
                layer_weights[i, 0],
                layer_weights[i, 1],
                layer_weights[i, 2],
                wires=wire,
            )

        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def embedding_parameter_shape(name: str, n_layers: int, n_qubits: int) -> tuple[int, ...]:
    """
    Return the trainable parameter shape for a named embedding.

    Parameters
    ----------
    name
        Embedding name.
    n_layers
        Number of embedding layers.
    n_qubits
        Number of qubits / wires.

    Returns
    -------
    tuple[int, ...]
        Parameter shape for the embedding.

    Raises
    ------
    ValueError
        If the embedding name is unknown.
    """
    key = name.strip().lower()

    if key in {"angle", "angle_embedding"}:
        return ()

    if key in {"data_reupload", "data_reuploading", "data_reuploading_embedding"}:
        return (n_layers, n_qubits, 3)

    raise ValueError(f"Unknown embedding '{name}'.")


def get_embedding(name: str):
    """
    Return an embedding callable by name.

    Parameters
    ----------
    name
        Embedding name.

    Returns
    -------
    callable
        Embedding function.

    Raises
    ------
    ValueError
        If the embedding name is unknown.
    """
    key = name.strip().lower()

    if key in {"angle", "angle_embedding"}:
        return apply_angle_embedding

    if key in {"data_reupload", "data_reuploading", "data_reuploading_embedding"}:
        return apply_data_reuploading_embedding

    raise ValueError(f"Unknown embedding '{name}'.")


def available_embeddings() -> list[str]:
    """
    Return the list of canonical embedding names.
    """
    return ["angle", "data_reupload"]
