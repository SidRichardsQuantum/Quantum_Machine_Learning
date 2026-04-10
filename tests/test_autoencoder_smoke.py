from pathlib import Path

import numpy as np

from qml.autoencoder import run_quantum_autoencoder


def test_run_quantum_autoencoder_smoke():
    result = run_quantum_autoencoder(
        n_samples=24,
        noise=0.02,
        test_size=0.25,
        seed=0,
        n_layers=1,
        latent_qubits=2,
        steps=2,
        step_size=0.1,
        plot=False,
        save=False,
        family="correlated",
    )

    assert result["model"] == "quantum_autoencoder"
    assert result["family"] == "correlated"
    assert result["latent_qubits"] == 2
    assert isinstance(result["loss_history"], list)
    assert len(result["loss_history"]) == 2

    assert isinstance(result["params"], np.ndarray)
    assert result["params"].shape == (1, 4, 2)

    assert 0.0 <= result["train_compression_fidelity"] <= 1.0
    assert 0.0 <= result["test_compression_fidelity"] <= 1.0
    assert 0.0 <= result["train_reconstruction_fidelity"] <= 1.0
    assert 0.0 <= result["test_reconstruction_fidelity"] <= 1.0


def test_run_quantum_autoencoder_save_outputs(tmp_path: Path):
    result = run_quantum_autoencoder(
        n_samples=24,
        noise=0.02,
        test_size=0.25,
        seed=0,
        n_layers=1,
        latent_qubits=2,
        steps=2,
        step_size=0.1,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
        family="hybrid",
    )

    assert result["model"] == "quantum_autoencoder"
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
