from qml.benchmarks import (
    compare_classification_models,
    compare_regression_models,
)


def test_classification_benchmark_runs():
    result = compare_classification_models(
        models=["vqc", "logistic_regression"],
        seeds=[0],
        n_samples=20,
        model_kwargs={
            "vqc": {
                "n_layers": 1,
                "steps": 2,
            },
        },
        save=False,
    )
    assert "summary" in result


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
    )
    assert "summary" in result
