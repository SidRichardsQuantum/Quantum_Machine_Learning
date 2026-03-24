from pathlib import Path

import numpy as np

from qml.kernel_methods import run_quantum_kernel_classifier


def test_run_quantum_kernel_classifier_smoke():
    result = run_quantum_kernel_classifier(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        plot=False,
        save=False,
    )

    assert result["model"] == "quantum_kernel_classifier"
    assert result["dataset"] == "moons"

    assert isinstance(result["train_accuracy"], float)
    assert isinstance(result["test_accuracy"], float)
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0

    assert isinstance(result["kernel_matrix_train"], np.ndarray)
    assert isinstance(result["kernel_matrix_test"], np.ndarray)
    assert result["kernel_matrix_train"].ndim == 2
    assert result["kernel_matrix_test"].ndim == 2

    assert isinstance(result["x_train"], np.ndarray)
    assert isinstance(result["x_test"], np.ndarray)
    assert isinstance(result["y_train"], np.ndarray)
    assert isinstance(result["y_test"], np.ndarray)

    assert result["kernel_matrix_train"].shape == (
        result["x_train"].shape[0],
        result["x_train"].shape[0],
    )
    assert result["kernel_matrix_test"].shape == (
        result["x_test"].shape[0],
        result["x_train"].shape[0],
    )

    assert result["y_train"].shape == result["y_train_pred"].shape
    assert result["y_test"].shape == result["y_test_pred"].shape


def test_run_quantum_kernel_classifier_save_outputs(tmp_path: Path):
    result = run_quantum_kernel_classifier(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
    )

    assert result["model"] == "quantum_kernel_classifier"
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
