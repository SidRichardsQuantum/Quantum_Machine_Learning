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

    assert result["y_test"].shape == result["y_test_pred"].shape