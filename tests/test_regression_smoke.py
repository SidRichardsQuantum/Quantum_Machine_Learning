from pathlib import Path

import numpy as np

from qml.regression import run_vqr


def test_run_vqr_smoke():
    result = run_vqr(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=3,
        step_size=0.1,
        plot=False,
        save=False,
    )

    assert result["model"] == "vqr"
    assert result["dataset"] == "regression"

    assert isinstance(result["loss_history"], list)
    assert len(result["loss_history"]) == 3

    assert isinstance(result["train_mse"], float)
    assert isinstance(result["test_mse"], float)
    assert isinstance(result["train_mae"], float)
    assert isinstance(result["test_mae"], float)

    assert result["train_mse"] >= 0.0
    assert result["test_mse"] >= 0.0
    assert result["train_mae"] >= 0.0
    assert result["test_mae"] >= 0.0

    assert isinstance(result["params"], np.ndarray)
    assert result["params"].ndim == 3

    assert result["y_test"].shape == result["y_test_pred"].shape


def test_run_vqr_save_outputs(tmp_path: Path):
    result = run_vqr(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=3,
        step_size=0.1,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
    )

    assert result["model"] == "vqr"
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
