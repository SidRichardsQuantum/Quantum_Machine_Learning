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


def apply_amplitude_embedding(x, wires: Sequence[int]) -> None:
    """
    Apply amplitude embedding with automatic normalization and padding.

    The input length may be smaller than ``2 ** len(wires)``. Longer inputs are
    rejected because truncation would silently discard information.
    """
    wires = list(wires)
    x = validate_feature_vector(x)
    max_dim = 2 ** len(wires)
    if x.shape[0] > max_dim:
        raise ValueError(
            f"Amplitude embedding on {len(wires)} wires accepts at most {max_dim} features, "
            f"got {x.shape[0]}."
        )
    qml.AmplitudeEmbedding(features=x, wires=wires, pad_with=0.0, normalize=True)


def apply_zz_feature_map(x, wires: Sequence[int]) -> None:
    """
    Apply a simple second-order ZZ feature map.

    Single-qubit rotations encode each feature and nearest-neighbor ZZ phases
    encode pairwise products.
    """
    wires = list(wires)
    x = validate_feature_vector(x, n_features=len(wires))

    for feature, wire in zip(x, wires):
        qml.Hadamard(wires=wire)
        qml.RZ(feature, wires=wire)

    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.RZ(x[i] * x[i + 1], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def apply_iqp_feature_map(x, wires: Sequence[int]) -> None:
    """
    Apply a compact IQP-style feature map.
    """
    wires = list(wires)
    x = validate_feature_vector(x, n_features=len(wires))

    for wire in wires:
        qml.Hadamard(wires=wire)

    for feature, wire in zip(x, wires):
        qml.RZ(feature, wires=wire)

    for i in range(len(wires)):
        j = (i + 1) % len(wires)
        if i == j:
            continue
        qml.IsingZZ(x[i] * x[j], wires=[wires[i], wires[j]])


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

    if key in {"angle", "angle_embedding", "amplitude", "amplitude_embedding"}:
        return ()

    if key in {"zz", "zz_feature_map", "iqp", "iqp_feature_map"}:
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

    if key in {"amplitude", "amplitude_embedding"}:
        return apply_amplitude_embedding

    if key in {"zz", "zz_feature_map"}:
        return apply_zz_feature_map

    if key in {"iqp", "iqp_feature_map"}:
        return apply_iqp_feature_map

    if key in {"data_reupload", "data_reuploading", "data_reuploading_embedding"}:
        return apply_data_reuploading_embedding

    raise ValueError(f"Unknown embedding '{name}'.")


def available_embeddings() -> list[str]:
    """
    Return the list of canonical embedding names.
    """
    return ["angle", "amplitude", "zz", "iqp", "data_reupload"]
