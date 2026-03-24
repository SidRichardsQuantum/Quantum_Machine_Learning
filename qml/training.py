"""
qml.training
============

Shared training-loop utilities for hybrid quantum-classical models.
"""

from __future__ import annotations

from typing import Any, Callable


def run_training_loop(
    step_fn: Callable[..., Any],
    n_steps: int,
    *args,
    **kwargs,
) -> Any:
    """
    Run a minimal generic training loop.

    Parameters
    ----------
    step_fn
        Callable representing one optimization step.
    n_steps
        Number of steps to execute.

    Returns
    -------
    Any
        Final output returned by the last step.
    """
    result = None
    for _ in range(n_steps):
        result = step_fn(*args, **kwargs)
    return result