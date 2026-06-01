"""
qml.training
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def validate_batch_size(batch_size: int | None, n_samples: int) -> int | None:
    """
    Validate a mini-batch size against a training-set size.

    ``None`` keeps full-batch training. Integer values must select at least one
    sample and no more than the full training set.
    """
    if batch_size is None:
        return None

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer or None.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if batch_size < 1 or batch_size > n_samples:
        raise ValueError("batch_size must be between 1 and the number of training samples.")
    return batch_size


def minibatch_indices(
    n_samples: int,
    batch_size: int | None,
    *,
    seed: int,
    shuffle: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield deterministic mini-batch index arrays indefinitely.

    If ``batch_size`` is ``None`` or equal to ``n_samples``, every yield contains
    the full index range. Otherwise, each epoch covers every sample once, with an
    optional deterministic shuffle.
    """
    batch_size = validate_batch_size(batch_size, n_samples)
    full_batch = np.arange(n_samples)
    if batch_size is None or batch_size == n_samples:
        while True:
            yield full_batch.copy()

    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(n_samples) if shuffle else full_batch.copy()
        for start in range(0, n_samples, batch_size):
            yield order[start : start + batch_size]


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
