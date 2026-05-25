from pathlib import Path

import qml.classifiers as classifiers
from qml.classifiers import run_vqc


def test_run_vqc_save_outputs(monkeypatch, tmp_path: Path):
    saved_json_paths: list[Path] = []
    saved_image_paths: list[Path] = []

    def fake_save_json(_result, path: Path) -> None:
        saved_json_paths.append(path)

    def fake_plot(*_args, save_path: Path | None = None, **_kwargs) -> None:
        if save_path is not None:
            saved_image_paths.append(save_path)

    monkeypatch.setattr(classifiers, "save_json", fake_save_json)
    monkeypatch.setattr(classifiers, "plot_dataset_2d", fake_plot)
    monkeypatch.setattr(classifiers, "plot_loss_curve", fake_plot)
    monkeypatch.setattr(classifiers, "plot_decision_boundary", fake_plot)

    result = run_vqc(
        n_samples=8,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        steps=0,
        step_size=0.1,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
    )

    assert result["model"] == "vqc"
    assert len(saved_json_paths) == 1
    assert len(saved_image_paths) == 3
    assert saved_json_paths[0].parent == tmp_path / "results"
    assert {path.parent for path in saved_image_paths} == {tmp_path / "images"}
    assert saved_json_paths[0].suffix == ".json"
    assert {path.suffix for path in saved_image_paths} == {".png"}
