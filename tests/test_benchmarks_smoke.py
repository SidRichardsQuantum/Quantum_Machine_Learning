from qml.benchmarks import (
    compare_classification_models,
    compare_regression_models,
)


def test_classification_benchmark_runs():

    result = compare_classification_models(
        models=["vqc", "logistic_regression"],
        seeds=[0, 1],
        n_samples=40,
    )

    assert result["benchmark_type"] == "classification"

    assert "vqc" in result["summary"]

    assert "logistic_regression" in result["summary"]


def test_regression_benchmark_runs():

    result = compare_regression_models(
        models=["vqr", "ridge_regression"],
        seeds=[0, 1],
        n_samples=40,
    )

    assert result["benchmark_type"] == "regression"

    assert "vqr" in result["summary"]

    assert "ridge_regression" in result["summary"]


def test_invalid_model_raises():

    import pytest

    with pytest.raises(ValueError):

        compare_classification_models(
            models=["not_a_model"],
            seeds=[0],
        )


def test_classification_benchmark_blobs_smoke():
    result = compare_classification_models(
        models=["vqc", "quantum_kernel"],
        seeds=[0],
        dataset="blobs",
        n_samples=24,
        save=False,
    )

    assert result["benchmark_type"] == "classification"
    assert result["models"] == ["vqc", "quantum_kernel"]
    assert result["summary"]["vqc"]["n_runs"] == 1
    assert result["summary"]["quantum_kernel"]["n_runs"] == 1


def test_regression_benchmark_polynomial_smoke():
    result = compare_regression_models(
        models=["vqr", "ridge_regression"],
        seeds=[0],
        dataset="polynomial",
        n_samples=24,
        save=False,
    )

    assert result["benchmark_type"] == "regression"
    assert result["models"] == ["vqr", "ridge_regression"]
    assert result["summary"]["vqr"]["n_runs"] == 1
    assert result["summary"]["ridge_regression"]["n_runs"] == 1
