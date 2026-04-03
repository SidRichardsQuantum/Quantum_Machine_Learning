"""
qml.datasets
============

Dataset registry and convenience wrappers for qml workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qml.data import make_moons_dataset, make_regression_dataset

DatasetBuilder = Callable[..., dict[str, Any]]


_DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "moons": make_moons_dataset,
    "two_moons": make_moons_dataset,
    "classification_moons": make_moons_dataset,
    "regression": make_regression_dataset,
    "regression_1d": make_regression_dataset,
    "synthetic_regression": make_regression_dataset,
}


def available_datasets() -> list[str]:
    """
    Return the list of registered dataset names.
    """
    return sorted(_DATASET_REGISTRY)


def get_dataset_builder(name: str) -> DatasetBuilder:
    """
    Return the dataset builder associated with a registered name.

    Parameters
    ----------
    name
        Dataset name or alias.

    Returns
    -------
    DatasetBuilder
        Callable dataset constructor.

    Raises
    ------
    ValueError
        If the dataset name is not registered.
    """
    key = name.strip().lower()
    if key not in _DATASET_REGISTRY:
        available = ", ".join(available_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}.")
    return _DATASET_REGISTRY[key]


def make_dataset(name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Build a dataset by registered name.

    Parameters
    ----------
    name
        Dataset name or alias.
    **kwargs
        Keyword arguments forwarded to the underlying dataset builder.

    Returns
    -------
    dict[str, Any]
        Dataset dictionary returned by the underlying builder.
    """
    builder = get_dataset_builder(name)
    return builder(**kwargs)


def is_classification_dataset(name: str) -> bool:
    """
    Return whether a dataset name maps to a classification dataset.
    """
    key = name.strip().lower()
    return key in {"moons", "two_moons", "classification_moons"}


def is_regression_dataset(name: str) -> bool:
    """
    Return whether a dataset name maps to a regression dataset.
    """
    key = name.strip().lower()
    return key in {"regression", "regression_1d", "synthetic_regression"}
