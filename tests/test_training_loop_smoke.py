from qml.training import run_training_loop


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
