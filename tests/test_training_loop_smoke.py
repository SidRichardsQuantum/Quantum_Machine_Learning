import pytest

from qml.training import minibatch_indices, run_training_loop, validate_batch_size


def test_run_training_loop_basic():
    def step_fn(x):
        return x + 1, x

    params, losses = run_training_loop(step_fn, 0, 3)

    assert params == 3
    assert len(losses) == 3
    assert losses == [0, 1, 2]


def test_run_training_loop_accepts_extra_outputs():
    def step_fn(x):
        return x + 1, x, {"extra": True}

    params, losses = run_training_loop(step_fn, 0, 2)

    assert params == 2
    assert losses == [0, 1]


def test_minibatch_indices_are_deterministic():
    first = minibatch_indices(5, 2, seed=7)
    second = minibatch_indices(5, 2, seed=7)

    first_batches = [next(first).tolist() for _ in range(4)]
    second_batches = [next(second).tolist() for _ in range(4)]

    assert first_batches == second_batches
    assert sorted(first_batches[0] + first_batches[1] + first_batches[2]) == [0, 1, 2, 3, 4]


def test_validate_batch_size_rejects_invalid_values():
    assert validate_batch_size(None, 3) is None
    assert validate_batch_size(2, 3) == 2

    with pytest.raises(ValueError):
        validate_batch_size(0, 3)
    with pytest.raises(ValueError):
        validate_batch_size(4, 3)
    with pytest.raises(TypeError):
        validate_batch_size(1.5, 3)
