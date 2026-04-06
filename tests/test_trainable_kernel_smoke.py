from qml.trainable_kernels import run_trainable_quantum_kernel_classifier


def test_trainable_kernel_data_reupload_runs():
    result = run_trainable_quantum_kernel_classifier(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=123,
        embedding="data_reupload",
        embedding_layers=1,
        steps=2,
        step_size=0.1,
        plot=False,
        save=False,
    )

    assert result["model"] == "trainable_quantum_kernel_classifier"
    assert result["dataset"] == "moons"
    assert result["embedding"] == "data_reupload"
    assert result["embedding_layers"] == 1

    n_train = result["x_train"].shape[0]
    n_test = result["x_test"].shape[0]
    n_qubits = result["n_qubits"]

    assert result["kernel_matrix_train"].shape == (n_train, n_train)
    assert result["kernel_matrix_test"].shape == (n_test, n_train)

    assert "train_accuracy" in result
    assert "test_accuracy" in result
    assert "final_alignment" in result
    assert "final_loss" in result

    assert len(result["alignment_trace"]) >= 1
    assert len(result["loss_trace"]) >= 1

    assert result["trained_params"].shape == (1, n_qubits, 3)


def test_trainable_kernel_angle_runs():
    result = run_trainable_quantum_kernel_classifier(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=123,
        embedding="angle",
        embedding_layers=1,
        steps=0,
        plot=False,
        save=False,
    )

    assert result["model"] == "trainable_quantum_kernel_classifier"
    assert result["embedding"] == "angle"

    n_train = result["x_train"].shape[0]
    n_test = result["x_test"].shape[0]

    assert result["kernel_matrix_train"].shape == (n_train, n_train)
    assert result["kernel_matrix_test"].shape == (n_test, n_train)

    assert len(result["alignment_trace"]) == 1
    assert len(result["loss_trace"]) == 1
