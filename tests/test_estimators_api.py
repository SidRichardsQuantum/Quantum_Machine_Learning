import numpy as np

from qml import (
    QuantumClassifier,
    QuantumKernel,
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    QuantumRegressor,
    kernel_target_alignment,
    make_sequence_windows,
)


def test_quantum_kernel_evaluate_and_alignment():
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.2, 0.9],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1])

    kernel = QuantumKernel(seed=0)
    matrix = kernel.evaluate(x)

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-8
    assert -1.0 <= kernel_target_alignment(matrix, y) <= 1.0


def test_quantum_kernel_classifier_smoke():
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.8, -0.9],
            [1.0, 1.0],
            [0.9, 0.7],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1])

    clf = QuantumKernelClassifier(QuantumKernel(seed=1), c=1.0)
    clf.fit(x, y)
    pred = clf.predict(x)

    assert pred.shape == y.shape
    assert 0.0 <= clf.score(x, y) <= 1.0


def test_quantum_kernel_regressor_smoke():
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.5, -0.5],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([-0.8, -0.4, 0.4, 0.8], dtype=float)

    reg = QuantumKernelRegressor(QuantumKernel(seed=2), alpha=1e-3)
    reg.fit(x, y)
    pred = reg.predict(x)

    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    assert np.isfinite(reg.score(x, y))


def test_quantum_regressor_multi_output_smoke():
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [0.6, 0.5],
        ],
        dtype=float,
    )
    y = np.column_stack([x[:, 0], x[:, 1]])

    reg = QuantumRegressor(n_layers=1, steps=1, step_size=0.05, seed=3)
    assert reg.get_params()["n_layers"] == 1
    reg.set_params(step_size=0.04)
    assert reg.step_size == 0.04
    reg.fit(x, y)
    pred = reg.predict(x)

    assert pred.shape == y.shape
    assert len(reg.loss_history_) == 2
    assert np.isfinite(pred).all()


def test_quantum_classifier_multiclass_smoke():
    x = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.2, 1.1],
            [-1.0, 1.0],
            [-1.1, 1.2],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 1, 1, 2, 2])

    clf = QuantumClassifier(n_layers=1, steps=1, step_size=0.05, seed=4)
    assert clf.get_params()["steps"] == 1
    clf.set_params(step_size=0.04)
    assert clf.step_size == 0.04
    clf.fit(x, y)
    probs = clf.predict_proba(x)
    pred = clf.predict(x)

    assert probs.shape == (6, 3)
    assert pred.shape == y.shape
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_make_sequence_windows_scalar_and_multivariate():
    x, y = make_sequence_windows(np.arange(6), window_size=3, horizon=1)
    assert x.shape == (3, 3)
    assert y.tolist() == [3.0, 4.0, 5.0]

    series = np.column_stack([np.arange(6), np.arange(6) + 10])
    x_multi, y_multi = make_sequence_windows(series, window_size=2, horizon=2, stride=2)
    assert x_multi.shape == (2, 4)
    assert y_multi.shape == (2, 2)
