"""
qml.optimizers
"""

from __future__ import annotations

from typing import Any

import pennylane as qml


def get_optimizer(
    name: str,
    *,
    stepsize: float = 0.1,
    **kwargs: Any,
):
    """
    Return a PennyLane optimizer instance by name.

    Parameters
    ----------
    name
        Optimizer name. Supported values:
        ``"adam"``, ``"gd"``, ``"gradient_descent"``, ``"spsa"``.
    stepsize
        Optimizer step size / learning rate.
    **kwargs
        Additional optimizer-specific keyword arguments.

    Returns
    -------
    object
        Instantiated PennyLane optimizer.

    Raises
    ------
    ValueError
        If the optimizer name is not supported.
    """
    key = name.strip().lower()

    if key == "adam":
        return qml.AdamOptimizer(stepsize=stepsize, **kwargs)

    if key in {"gd", "gradient_descent", "sgd"}:
        return qml.GradientDescentOptimizer(stepsize=stepsize, **kwargs)

    if key == "spsa":
        return qml.SPSAOptimizer(stepsize=stepsize, **kwargs)

    raise ValueError(
        f"Unsupported optimizer '{name}'. "
        "Supported optimizers are: adam, gd, gradient_descent, spsa."
    )


def list_supported_optimizers() -> list[str]:
    """
    Return the canonical supported optimizer names.
    """
    return ["adam", "gradient_descent", "spsa"]
