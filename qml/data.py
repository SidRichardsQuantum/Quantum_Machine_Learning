"""
qml.data
========

Dataset generation and preprocessing utilities for quantum machine learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression


def to_numpy(x: Any) -> np.ndarray:
    """Convert input to a NumPy array."""
    return np.asarray(x)


def standardize_features(x: np.ndarray) -> np.ndarray:
    """
    Standardize features columnwise.

    Parameters
    ----------
    x
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Standardized feature matrix.
    """
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std


def make_moons_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized binary classification dataset based on two moons.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Noise level passed to ``sklearn.datasets.make_moons``.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return {
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
    }


def make_regression_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized 2D regression dataset.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Noise level passed to ``sklearn.datasets.make_regression``.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    x, y = make_regression(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_targets=1,
        noise=noise,
        random_state=seed,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    y_train = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(np.asarray(y_test).reshape(-1, 1)).ravel()

    return {
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
    }
