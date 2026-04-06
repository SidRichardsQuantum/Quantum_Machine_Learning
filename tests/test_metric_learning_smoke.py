from __future__ import annotations

import subprocess
import sys

from qml.metric_learning import run_quantum_metric_learner


def test_run_quantum_metric_learner_smoke() -> None:
    """
    Smoke test for the quantum metric learning API.
    """
    result = run_quantum_metric_learner(
        dataset="moons",
        samples=40,
        test_size=0.25,
        seed=123,
        layers=1,
        steps=3,
        stepsize=0.05,
        margin=0.5,
        pairs_per_step=8,
        log_every=0,
        scale_data=True,
        plot=False,
    )

    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= result.test_accuracy <= 1.0
    assert len(result.loss_history) == 3

    n_train = int(40 * (1.0 - 0.25))
    n_test = 40 - n_train
    n_qubits = 2

    assert result.params.shape == (1, n_qubits, 3)
    assert result.train_embeddings.shape == (n_train, n_qubits)
    assert result.test_embeddings.shape == (n_test, n_qubits)
    assert set(result.train_centroids.keys()) == {0, 1}
    assert result.y_train.shape == (n_train,)
    assert result.y_test.shape == (n_test,)


def test_metric_learning_cli_smoke() -> None:
    """
    Smoke test for the quantum metric learning CLI.
    """
    cmd = [
        sys.executable,
        "-m",
        "qml",
        "metric-learning",
        "--samples",
        "40",
        "--steps",
        "3",
        "--layers",
        "1",
        "--log-every",
        "0",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout

    assert "Model: quantum_metric_learning" in stdout
    assert "Dataset: moons" in stdout
    assert "Train accuracy:" in stdout
    assert "Test accuracy:" in stdout
    assert "Final loss:" in stdout
