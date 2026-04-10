from pathlib import Path

import qml.regression as regression
from qml.regression import run_vqr


def test_run_vqr_default_artifact_paths_use_vqr(monkeypatch, tmp_path: Path):
    seen_result_modules: list[str] = []
    seen_image_modules: list[str] = []

    def fake_results_path(module: str, filename: str) -> Path:
        seen_result_modules.append(module)
        path = tmp_path / "results" / module / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fake_images_path(module: str, filename: str) -> Path:
        seen_image_modules.append(module)
        path = tmp_path / "images" / module / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(regression, "results_path", fake_results_path)
    monkeypatch.setattr(regression, "images_path", fake_images_path)

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
    )

    assert result["model"] == "vqr"
    assert seen_result_modules == ["vqr"]
    assert seen_image_modules == ["vqr", "vqr", "vqr"]
