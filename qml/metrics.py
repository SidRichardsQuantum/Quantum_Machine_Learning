"""
qml.metrics
===========

Evaluation metrics for quantum machine learning models.
"""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true, y_pred) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(y_true == y_pred))