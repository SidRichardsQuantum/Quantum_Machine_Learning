from pathlib import Path

from qml.regression import run_vqr


def test_run_vqr_smoke():
    result = run_vqr(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=2,
        step_size=0.1,
        plot=False,
        save=False,
    )

    assert result["model"] == "vqr"
    assert result["dataset"] == "linear"
    assert "train_mse" in result
    assert "test_mse" in result
    assert "train_mae" in result
    assert "test_mae" in result


def test_run_vqr_save_outputs(tmp_path: Path):
    result = run_vqr(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=2,
        step_size=0.1,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
    )

    assert result["model"] == "vqr"
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
