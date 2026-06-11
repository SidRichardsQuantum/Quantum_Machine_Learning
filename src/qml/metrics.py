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


def balanced_accuracy_score(y_true, y_pred) -> float:
    """Compute mean per-class recall for classification labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    classes = np.unique(y_true)
    if classes.size == 0:
        raise ValueError("y_true must contain at least one class.")

    recalls = []
    for label in classes:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls))


def f1_score(y_true, y_pred, *, positive_label=1) -> float:
    """Compute binary F1 score for the selected positive label."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    true_positive = float(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    false_positive = float(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    false_negative = float(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    denominator = 2.0 * true_positive + false_positive + false_negative
    if denominator == 0.0:
        return 0.0
    return float(2.0 * true_positive / denominator)


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


def root_mean_squared_error(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Compute coefficient of determination for regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)
