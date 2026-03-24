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


def apply_angle_embedding(x, wires: Sequence[int]) -> None:
    """
    Apply a simple angle embedding using Y rotations.

    Parameters
    ----------
    x
        Input feature vector.
    wires
        Wires on which to apply the embedding.
    """
    wires = list(wires)
    x = validate_feature_vector(x, n_features=len(wires))
    qml.AngleEmbedding(features=x, wires=wires, rotation="Y")
