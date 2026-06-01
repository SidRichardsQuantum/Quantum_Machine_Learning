from qml import (
    ansatz_parameter_count,
    embedding_parameter_count,
    estimate_circuit_depth,
    qcnn_parameter_count,
    run_qcnn,
    run_trainable_quantum_kernel_regressor,
    run_vqc,
    run_vqr,
)
from qml.autoencoder import run_quantum_autoencoder
from qml.circuit_metadata import circuit_metadata


def test_parameter_count_helpers() -> None:
    assert ansatz_parameter_count(2, 3) == 12
    assert ansatz_parameter_count(2, 3, ansatz="strongly_entangling") == 18
    assert embedding_parameter_count("angle", 2, 3) == 0
    assert embedding_parameter_count("data_reupload", 2, 3) == 18
    assert qcnn_parameter_count() == 38


def test_depth_estimates_are_positive_and_template_specific() -> None:
    variational_depth = estimate_circuit_depth(
        n_qubits=2,
        n_layers=1,
        embedding="angle",
        template="vqc",
    )
    kernel_depth = estimate_circuit_depth(
        n_qubits=2,
        embedding="data_reupload",
        embedding_layers=2,
        ansatz=None,
        template="trainable_kernel",
    )

    assert variational_depth > 0
    assert kernel_depth > variational_depth
    assert estimate_circuit_depth(n_qubits=4, template="qcnn") > 0


def test_circuit_metadata_builds_json_friendly_record() -> None:
    metadata = circuit_metadata(
        model="example",
        n_qubits=2,
        n_layers=1,
        embedding="data_reupload",
        embedding_layers=2,
        template="vqc",
    )

    assert metadata["model"] == "example"
    assert metadata["trainable_parameters"] == 16
    assert metadata["depth_is_estimate"] is True


def test_workflow_results_include_circuit_metadata() -> None:
    vqc = run_vqc(n_samples=12, n_layers=1, steps=1, seed=0)
    vqr = run_vqr(n_samples=12, n_layers=1, steps=1, seed=0)
    qcnn = run_qcnn(n_samples=12, steps=1, seed=0)
    autoencoder = run_quantum_autoencoder(n_samples=12, n_layers=1, steps=1, seed=0)
    trainable_regressor = run_trainable_quantum_kernel_regressor(
        n_samples=10,
        embedding_layers=1,
        steps=1,
        seed=0,
        alpha=1e-3,
    )

    for result in (vqc, vqr, qcnn, autoencoder, trainable_regressor):
        metadata = result["circuit_metadata"]
        assert metadata["n_qubits"] > 0
        assert metadata["trainable_parameters"] > 0
        assert metadata["estimated_depth"] > 0
