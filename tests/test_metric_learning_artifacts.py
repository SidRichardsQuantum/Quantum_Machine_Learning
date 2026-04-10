from pathlib import Path

from qml.metric_learning import run_quantum_metric_learner


def test_run_quantum_metric_learner_save_outputs(tmp_path: Path):
    result = run_quantum_metric_learner(
        dataset="moons",
        samples=24,
        test_size=0.25,
        seed=0,
        layers=1,
        steps=2,
        stepsize=0.05,
        log_every=0,
        plot=False,
        save=True,
        results_dir=tmp_path / "results",
        images_dir=tmp_path / "images",
    )

    assert 0.0 <= result.train_accuracy <= 1.0
    assert any((tmp_path / "results").glob("*.json"))
    assert any((tmp_path / "images").glob("*.png"))
