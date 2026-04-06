"""
qml.training
"""

from __future__ import annotations


def run_training_loop(
    step_fn,
    init_params,
    steps: int,
    patience: int | None = None,
    min_delta: float = 0.0,
):
    """
    Generic optimizer loop.

    Returns
    -------
    tuple
        (final_params, loss_trace)
    """

    params = init_params
    loss_trace: list[float] = []

    best_loss = float("inf")
    patience_counter = 0

    for _ in range(steps):

        out = step_fn(params)

        # allow step_fn to return extra metadata safely
        if isinstance(out, tuple):
            params = out[0]
            loss = out[1]
        else:
            raise ValueError("step_fn must return at least (params, loss)")

        loss_trace.append(float(loss))

        if patience is not None:

            if loss < best_loss - min_delta:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    return params, loss_trace
