from qml.benchmarks import (
    compare_classification_models,
    compare_regression_models,
)


def test_classification_benchmark_runs():
    result = compare_classification_models(
        models=["vqc", "qcnn", "logistic_regression"],
        seeds=[0],
        n_samples=20,
        model_kwargs={
            "vqc": {
                "n_layers": 1,
                "steps": 2,
            },
            "qcnn": {
                "steps": 2,
            },
        },
        save=False,
        dataset="circles",
    )
    assert "summary" in result
    assert result["dataset"] == "circles"
    assert {run["dataset"] for run in result["runs"]} == {"circles"}
    assert result["best_model"]["metric"] == "test_accuracy"
    assert result["best_model"]["higher_is_better"] is True

    for run in result["runs"]:
        assert "generalization_gap" in run
        assert run["runtime_seconds"] >= 0.0

    for model_summary in result["summary"].values():
        assert "generalization_gap" in model_summary
        assert "runtime_seconds" in model_summary


def test_regression_benchmark_runs():
    result = compare_regression_models(
        models=["vqr", "ridge_regression"],
        seeds=[0],
        n_samples=20,
        model_kwargs={
            "vqr": {
                "n_layers": 1,
                "steps": 2,
            },
        },
        save=False,
        dataset="sine",
    )
    assert "summary" in result
    assert result["dataset"] == "sine"
    assert {run["dataset"] for run in result["runs"]} == {"sine"}
    assert result["best_model"]["metric"] == "test_mse"
    assert result["best_model"]["higher_is_better"] is False

    for run in result["runs"]:
        assert "generalization_gap" in run
        assert run["runtime_seconds"] >= 0.0

    for model_summary in result["summary"].values():
        assert "generalization_gap" in model_summary
        assert "runtime_seconds" in model_summary
