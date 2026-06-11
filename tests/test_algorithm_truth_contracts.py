import numpy as np

from qml import (
    QuantumClassifier,
    QuantumKernel,
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    run_qcnn,
    run_quantum_autoencoder,
)
from qml.trainable_kernels import TrainableQuantumKernelRegressor


def test_quantum_kernel_estimators_consume_precomputed_quantum_kernel_matrix():
    x = np.asarray([[-1.0, -1.0], [-0.8, -0.9], [0.8, 0.7], [1.0, 1.0]], dtype=float)
    y_cls = np.asarray([0, 0, 1, 1])
    y_reg = np.asarray([-1.0, -0.8, 0.8, 1.0], dtype=float)

    kernel = QuantumKernel(seed=21)
    expected = kernel.evaluate(x)
    assert np.allclose(expected, expected.T)
    assert np.allclose(np.diag(expected), 1.0)
    assert np.linalg.eigvalsh(expected).min() >= -1e-8

    clf = QuantumKernelClassifier(kernel, c=1.0).fit(x, y_cls)
    assert np.allclose(clf.kernel_matrix_train_, expected)
    assert clf.model_.kernel == "precomputed"

    reg = QuantumKernelRegressor(QuantumKernel(seed=21), alpha=1e-3).fit(x, y_reg)
    assert np.allclose(reg.kernel_matrix_train_, expected)
    assert reg.model_.kernel == "precomputed"


def test_trainable_kernel_regressor_exposes_alignment_objective_trace():
    x = np.asarray([[-1.0, -1.0], [-0.3, -0.3], [0.3, 0.3], [1.0, 1.0]], dtype=float)
    y = np.asarray([-0.9, -0.2, 0.2, 0.9], dtype=float)

    reg = TrainableQuantumKernelRegressor(
        embedding_layers=1,
        steps=1,
        step_size=0.05,
        alpha=1e-3,
        seed=22,
    ).fit(x, y)

    assert len(reg.loss_trace_) == 1
    assert reg.trained_params_.shape == (1, 2, 3)
    assert np.isfinite(reg.alignment_)
    assert reg.circuit_metadata_["alignment"] == reg.alignment_


def test_variational_classifier_trains_circuit_loss_and_records_shot_metadata():
    x = np.asarray([[-1.0, -1.0], [-0.8, -0.9], [0.8, 0.7], [1.0, 1.0]], dtype=float)
    y = np.asarray([0, 0, 1, 1])

    clf = QuantumClassifier(n_layers=1, steps=1, shots=32, seed=23).fit(x, y)

    assert len(clf.loss_history_[0]) == 1
    assert np.isfinite(clf.loss_history_[0][0])
    assert clf.circuit_metadata_["shots"] == 32
    assert clf.circuit_metadata_["template"] == "variational"


def test_qcnn_reports_documented_active_wire_reduction():
    result = run_qcnn(n_samples=8, steps=1, seed=24, dataset="moons")

    assert result["circuit_metadata"]["active_wires_by_stage"] == [[0, 1, 2, 3], [1, 3], [3]]
    assert result["circuit_metadata"]["template"] == "qcnn"
    assert result["pool1_params"].shape == (2, 2)
    assert result["pool2_params"].shape == (1, 2)


def test_autoencoder_reconstruction_uses_postselected_compression_path():
    result = run_quantum_autoencoder(
        n_samples=8,
        steps=1,
        seed=25,
        family="correlated",
    )

    assert result["reconstruction_method"] == "trash_zero_postselection_tied_decoder"
    assert result["circuit_metadata"]["latent_qubits"] == 2
    assert result["circuit_metadata"]["trash_qubits"] == 2
    assert np.all((0.0 <= result["test_reconstruction_scores"]))
    assert np.all((result["test_reconstruction_scores"] <= 1.0))
