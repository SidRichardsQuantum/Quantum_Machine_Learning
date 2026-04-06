"""
qml.data
========

Dataset generation and preprocessing utilities for quantum machine learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def _split_and_scale_classification(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Split and standardize a binary classification dataset.
    """
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


def _split_and_scale_regression(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Split and standardize a regression dataset.
    """
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
    return _split_and_scale_classification(x=x, y=y, test_size=test_size, seed=seed)


def make_circles_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    factor: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized binary classification dataset based on concentric circles.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Noise level passed to ``sklearn.datasets.make_circles``.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.
    factor
        Scale factor between inner and outer circle.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    x, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
    )
    return _split_and_scale_classification(x=x, y=y, test_size=test_size, seed=seed)


def make_blobs_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    centers: int = 2,
    cluster_std: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized binary classification dataset based on Gaussian blobs.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Fallback cluster standard deviation when ``cluster_std`` is ``None``.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.
    centers
        Number of blob centers. Must be 2 for binary classification workflows.
    cluster_std
        Standard deviation of blob clusters.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    if centers != 2:
        raise ValueError("make_blobs_dataset currently supports binary classification only.")

    std = noise if cluster_std is None else cluster_std

    x, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=std,
        random_state=seed,
    )
    return _split_and_scale_classification(x=x, y=y, test_size=test_size, seed=seed)


def make_xor_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized binary XOR-style classification dataset.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Standard deviation of Gaussian perturbations added to the sampled points.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = ((x[:, 0] > 0.0) ^ (x[:, 1] > 0.0)).astype(int)

    if noise > 0.0:
        x = x + rng.normal(0.0, noise, size=x.shape)

    return _split_and_scale_classification(x=x, y=y, test_size=test_size, seed=seed)


def make_classification_dataset(
    dataset: str = "moons",
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    """
    Generate a standardized binary classification dataset by name.

    Supported datasets are ``"moons"``, ``"circles"``, ``"blobs"``, and ``"xor"``.

    Parameters
    ----------
    dataset
        Dataset name.
    n_samples
        Total number of samples.
    noise
        Dataset noise level.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    dataset = dataset.lower()

    if dataset == "moons":
        return make_moons_dataset(
            n_samples=n_samples,
            noise=noise,
            test_size=test_size,
            seed=seed,
        )
    if dataset == "circles":
        return make_circles_dataset(
            n_samples=n_samples,
            noise=noise,
            test_size=test_size,
            seed=seed,
        )
    if dataset == "blobs":
        return make_blobs_dataset(
            n_samples=n_samples,
            noise=noise,
            test_size=test_size,
            seed=seed,
        )
    if dataset == "xor":
        return make_xor_dataset(
            n_samples=n_samples,
            noise=noise,
            test_size=test_size,
            seed=seed,
        )

    raise ValueError(
        f"Unknown classification dataset: {dataset}. "
        "Available datasets: moons, circles, blobs, xor."
    )


def make_regression_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "linear",
) -> dict[str, np.ndarray]:
    """
    Generate a standardized regression dataset.

    Supported datasets are ``"linear"``, ``"sine"``, and ``"polynomial"``.

    Parameters
    ----------
    n_samples
        Total number of samples.
    noise
        Noise level for the generated targets.
    test_size
        Fraction of samples reserved for the test split.
    seed
        Random seed for reproducibility.
    dataset
        Regression dataset name.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing train/test splits.
    """
    dataset = dataset.lower()
    rng = np.random.default_rng(seed)

    if dataset == "linear":
        x, y = make_regression(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_targets=1,
            noise=noise,
            random_state=seed,
        )
        return _split_and_scale_regression(x=x, y=y, test_size=test_size, seed=seed)

    if dataset == "sine":
        x1 = rng.uniform(-np.pi, np.pi, size=n_samples)
        x2 = rng.uniform(-1.0, 1.0, size=n_samples)
        x = np.column_stack([x1, x2])
        y = np.sin(x1) + rng.normal(0.0, noise, size=n_samples)
        return _split_and_scale_regression(x=x, y=y, test_size=test_size, seed=seed)

    if dataset == "polynomial":
        x1 = rng.uniform(-2.0, 2.0, size=n_samples)
        x2 = rng.uniform(-1.0, 1.0, size=n_samples)
        x = np.column_stack([x1, x2])
        y = x1**2 + 0.5 * x1 - 0.25 * x2 + rng.normal(0.0, noise, size=n_samples)
        return _split_and_scale_regression(x=x, y=y, test_size=test_size, seed=seed)

    raise ValueError(
        f"Unknown regression dataset: {dataset}. " "Available datasets: linear, sine, polynomial."
    )
