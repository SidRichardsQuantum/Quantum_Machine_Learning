import numpy as np

from qml.qcnn import run_qcnn


def test_run_qcnn_smoke():
    result = run_qcnn(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        steps=2,
        step_size=0.1,
        plot=False,
        save=False,
    )

    assert result["model"] == "qcnn"
    assert result["dataset"] == "moons"
    assert isinstance(result["loss_history"], list)
    assert len(result["loss_history"]) == 2

    assert isinstance(result["train_accuracy"], float)
    assert isinstance(result["test_accuracy"], float)
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0

    assert isinstance(result["params"], np.ndarray)
    assert result["params"].ndim == 1

    assert isinstance(result["embedding_params"], np.ndarray)
    assert result["embedding_params"].shape == (4, 3)

    assert isinstance(result["conv1_params"], np.ndarray)
    assert result["conv1_params"].shape == (2, 6)

    assert isinstance(result["conv2_params"], np.ndarray)
    assert result["conv2_params"].shape == (1, 6)

    assert isinstance(result["dense_params"], np.ndarray)
    assert result["dense_params"].shape == (2,)

    assert result["y_test"].shape == result["y_test_pred"].shape
    assert result["test_probabilities"].shape == result["y_test"].shape
