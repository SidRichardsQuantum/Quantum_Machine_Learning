"""
qml.preprocessing
=================

General preprocessing helpers for QML workflows.
"""

from __future__ import annotations

import numpy as np


def make_sequence_windows(
    values,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a sequence into supervised learning windows.

    Parameters
    ----------
    values
        Array of shape ``(n_steps,)`` or ``(n_steps, n_features)``.
    window_size
        Number of consecutive steps in each input window.
    horizon
        Number of steps ahead to predict. ``1`` predicts the next step.
    stride
        Step between successive windows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(x, y)`` where ``x`` has shape
        ``(n_windows, window_size * n_features)`` and ``y`` has shape
        ``(n_windows,)`` for scalar series or ``(n_windows, n_features)`` for
        multivariate series.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values_2d = values.reshape(-1, 1)
        scalar = True
    elif values.ndim == 2:
        values_2d = values
        scalar = False
    else:
        raise ValueError(f"values must be 1D or 2D, got shape {values.shape}.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    stop = values_2d.shape[0] - window_size - horizon + 1
    if stop <= 0:
        raise ValueError("Sequence is too short for the requested window_size and horizon.")

    x_rows = []
    y_rows = []
    for start in range(0, stop, stride):
        end = start + window_size
        target = end + horizon - 1
        x_rows.append(values_2d[start:end].ravel())
        y_rows.append(values_2d[target])

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    if scalar:
        y = y.ravel()
    return x, y
