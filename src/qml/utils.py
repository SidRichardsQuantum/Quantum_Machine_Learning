"""
qml.utils
=========

General utilities for the qml package.
"""

from __future__ import annotations


import numpy as np


def set_random_seed(seed: int | None) -> None:
    """Set the random seed for reproducibility."""
    if seed is not None:
        set_random_seed(seed)


def small_random(shape, scale: float = 0.01, seed: int | None = None):
    """Generate small random numbers with a given shape and scale."""
    if seed is not None:
        set_random_seed(seed)
    return scale * np.random.randn(*shape)
