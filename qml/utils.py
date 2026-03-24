"""
qml.utils
=========

General utilities for the qml package.
"""

from __future__ import annotations

import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)
