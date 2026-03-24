from pathlib import Path

from qml.classifiers import run_vqc


def test_run_vqc_save_outputs(tmp_path: Path):
    result = run_vqc(
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

    assert result["model"] == "vqc"
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
