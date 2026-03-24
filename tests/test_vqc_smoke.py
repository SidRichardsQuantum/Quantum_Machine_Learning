import numpy as np

from qml.classifiers import run_vqc


def test_run_vqc_smoke():
    result = run_vqc(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=3,
        step_size=0.1,
    )

    assert result["model"] == "vqc"
    assert result["dataset"] == "moons"
    assert isinstance(result["loss_history"], list)
    assert len(result["loss_history"]) == 3

    assert isinstance(result["train_accuracy"], float)
    assert isinstance(result["test_accuracy"], float)
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0

    assert isinstance(result["params"], np.ndarray)
    assert result["params"].ndim == 3

    assert result["y_test"].shape == result["y_test_pred"].shape
    assert result["test_probabilities"].shape == result["y_test"].shape