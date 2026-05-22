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
