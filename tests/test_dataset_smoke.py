from qml.classifiers import run_vqc
from qml.kernel_methods import run_quantum_kernel_classifier
from qml.regression import run_vqr
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier


def test_vqc_circles_smoke():
    result = run_vqc(
        dataset="circles",
        n_samples=24,
        n_layers=1,
        steps=2,
        shots=8,
        plot=False,
        save=False,
    )

    assert result["model"] == "vqc"
    assert result["dataset"] == "circles"
    assert "train_accuracy" in result
    assert "test_accuracy" in result


def test_quantum_kernel_xor_smoke():
    result = run_quantum_kernel_classifier(
        dataset="xor",
        n_samples=24,
        shots=8,
        plot=False,
        save=False,
    )

    assert result["model"] == "quantum_kernel_classifier"
    assert result["dataset"] == "xor"
    assert "train_accuracy" in result
    assert "test_accuracy" in result


def test_trainable_kernel_blobs_smoke():
    result = run_trainable_quantum_kernel_classifier(
        dataset="blobs",
        n_samples=24,
        steps=1,
        shots_train=8,
        shots_kernel=8,
        plot=False,
        save=False,
    )

    assert result["model"] == "trainable_quantum_kernel_classifier"
    assert result["dataset"] == "blobs"
    assert "train_accuracy" in result
    assert "test_accuracy" in result
    assert "final_alignment" in result


def test_vqr_sine_smoke():
    result = run_vqr(
        dataset="sine",
        n_samples=24,
        n_layers=1,
        steps=2,
        shots=8,
        plot=False,
        save=False,
    )

    assert result["model"] == "vqr"
    assert result["dataset"] == "sine"
    assert "train_mse" in result
    assert "test_mse" in result
    assert "train_mae" in result
    assert "test_mae" in result
