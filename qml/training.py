from __future__ import annotations

from typing import Any, Callable


def run_training_loop(
    step_fn: Callable[[Any], tuple[Any, Any]],
    params: Any,
    n_steps: int,
    log_every: int | None = None,
) -> tuple[Any, list[float]]:
    """
    Run a minimal generic training loop.

    Parameters
    ----------
    step_fn
        Callable representing one optimisation step. Must accept the current
        parameters and return ``(new_params, loss)``.
    params
        Initial parameter tensor/array.
    n_steps
        Number of optimisation steps.
    log_every
        Optional logging interval.

    Returns
    -------
    tuple[Any, list[float]]
        Final parameters and loss history.
    """
    losses: list[float] = []

    for step in range(n_steps):
        params, loss = step_fn(params)
        losses.append(float(loss))

        if log_every is not None and step % log_every == 0:
            print(f"step {step}: loss={float(loss):.6f}")

    return params, losses
