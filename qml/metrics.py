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


def mean_squared_error(y_true, y_pred) -> float:
    """Compute mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred) -> float:
    """Compute mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(np.abs(y_true - y_pred)))
