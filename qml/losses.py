"""
qml.losses
==========

Loss functions for supervised quantum machine learning workflows.
"""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    """Compute mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))