"""
qml.benchmarks
==============

Benchmark helpers for comparing quantum and classical models across multiple seeds.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, pstdev
from typing import Any

from qml.classical_baselines import (
    run_logistic_classifier,
    run_mlp_classifier,
    run_mlp_regressor,
    run_ridge_regression,
    run_svm_classifier,
)
from qml.classifiers import run_vqc
from qml.io_utils import results_path, save_json
from qml.kernel_methods import run_quantum_kernel_classifier
from qml.regression import run_vqr

ClassificationRunner = Callable[..., dict[str, Any]]
RegressionRunner = Callable[..., dict[str, Any]]


_CLASSIFICATION_MODELS: dict[str, ClassificationRunner] = {
    "vqc": run_vqc,
    "quantum_kernel": run_quantum_kernel_classifier,
    "logistic_regression": run_logistic_classifier,
    "svm_classifier": run_svm_classifier,
    "mlp_classifier": run_mlp_classifier,
}

_REGRESSION_MODELS: dict[str, RegressionRunner] = {
    "vqr": run_vqr,
    "ridge_regression": run_ridge_regression,
    "mlp_regressor": run_mlp_regressor,
}


def _mean_std(values: list[float]) -> dict[str, float]:
    """
    Return mean and population standard deviation for a list of floats.
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan")}

    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}

    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
    }


def _validate_models(
    requested_models: list[str] | None,
    available_models: dict[str, Callable[..., dict[str, Any]]],
    benchmark_name: str,
) -> list[str]:
    """
    Validate requested model names against the available registry.
    """
    if requested_models is None:
        return list(available_models.keys())

    unknown = [name for name in requested_models if name not in available_models]
    if unknown:
        available = ", ".join(sorted(available_models))
        missing = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown model(s) for {benchmark_name}: {missing}. " f"Available models: {available}."
        )

    return requested_models


def _run_classification_model(
    model_name: str,
    runner: ClassificationRunner,
    common_kwargs: dict[str, Any],
    model_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Run one classification model with merged kwargs.
    """
    kwargs = dict(common_kwargs)
    kwargs.update(model_kwargs.get(model_name, {}))
    kwargs["plot"] = False
    kwargs["save"] = False
    return runner(**kwargs)


def _run_regression_model(
    model_name: str,
    runner: RegressionRunner,
    common_kwargs: dict[str, Any],
    model_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Run one regression model with merged kwargs.
    """
    kwargs = dict(common_kwargs)
    kwargs.update(model_kwargs.get(model_name, {}))
    kwargs["plot"] = False
    kwargs["save"] = False
    return runner(**kwargs)


def compare_classification_models(
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    save: bool = False,
    filename: str = "classification_benchmark.json",
) -> dict[str, Any]:
    """
    Compare classification models across multiple seeds.

    Parameters
    ----------
    models
        Model names to evaluate. If ``None``, all registered classification models are used.
    seeds
        Random seeds to evaluate. If ``None``, uses ``[123]``.
    n_samples
        Number of dataset samples per run.
    noise
        Dataset noise level.
    test_size
        Fraction reserved for test data.
    model_kwargs
        Optional per-model kwargs, keyed by model name.
    save
        Whether to save the benchmark summary JSON.
    filename
        Output filename when ``save=True``.

    Returns
    -------
    dict[str, Any]
        Benchmark summary including per-run records and aggregated metrics.
    """
    selected_models = _validate_models(
        requested_models=models,
        available_models=_CLASSIFICATION_MODELS,
        benchmark_name="classification benchmark",
    )
    seeds = [123] if seeds is None else seeds
    model_kwargs = {} if model_kwargs is None else model_kwargs

    common_kwargs = {
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
    }

    runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for model_name in selected_models:
        runner = _CLASSIFICATION_MODELS[model_name]
        train_accuracies: list[float] = []
        test_accuracies: list[float] = []

        for seed in seeds:
            result = _run_classification_model(
                model_name=model_name,
                runner=runner,
                common_kwargs={**common_kwargs, "seed": seed},
                model_kwargs=model_kwargs,
            )

            train_accuracy = float(result["train_accuracy"])
            test_accuracy = float(result["test_accuracy"])

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            run_record = {
                "model": model_name,
                "seed": seed,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }

            if "final_loss" in result:
                run_record["final_loss"] = float(result["final_loss"])

            runs.append(run_record)

        summary[model_name] = {
            "train_accuracy": _mean_std(train_accuracies),
            "test_accuracy": _mean_std(test_accuracies),
            "n_runs": len(seeds),
        }

    benchmark = {
        "benchmark_type": "classification",
        "models": selected_models,
        "seeds": list(seeds),
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "runs": runs,
        "summary": summary,
    }

    if save:
        save_json(benchmark, results_path("benchmarks", filename))

    return benchmark


def compare_regression_models(
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    save: bool = False,
    filename: str = "regression_benchmark.json",
) -> dict[str, Any]:
    """
    Compare regression models across multiple seeds.

    Parameters
    ----------
    models
        Model names to evaluate. If ``None``, all registered regression models are used.
    seeds
        Random seeds to evaluate. If ``None``, uses ``[123]``.
    n_samples
        Number of dataset samples per run.
    noise
        Dataset noise level.
    test_size
        Fraction reserved for test data.
    model_kwargs
        Optional per-model kwargs, keyed by model name.
    save
        Whether to save the benchmark summary JSON.
    filename
        Output filename when ``save=True``.

    Returns
    -------
    dict[str, Any]
        Benchmark summary including per-run records and aggregated metrics.
    """
    selected_models = _validate_models(
        requested_models=models,
        available_models=_REGRESSION_MODELS,
        benchmark_name="regression benchmark",
    )
    seeds = [123] if seeds is None else seeds
    model_kwargs = {} if model_kwargs is None else model_kwargs

    common_kwargs = {
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
    }

    runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for model_name in selected_models:
        runner = _REGRESSION_MODELS[model_name]
        train_mse_values: list[float] = []
        test_mse_values: list[float] = []
        train_mae_values: list[float] = []
        test_mae_values: list[float] = []

        for seed in seeds:
            result = _run_regression_model(
                model_name=model_name,
                runner=runner,
                common_kwargs={**common_kwargs, "seed": seed},
                model_kwargs=model_kwargs,
            )

            train_mse = float(result["train_mse"])
            test_mse = float(result["test_mse"])
            train_mae = float(result["train_mae"])
            test_mae = float(result["test_mae"])

            train_mse_values.append(train_mse)
            test_mse_values.append(test_mse)
            train_mae_values.append(train_mae)
            test_mae_values.append(test_mae)

            run_record = {
                "model": model_name,
                "seed": seed,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_mae": train_mae,
                "test_mae": test_mae,
            }

            if "final_loss" in result:
                run_record["final_loss"] = float(result["final_loss"])

            runs.append(run_record)

        summary[model_name] = {
            "train_mse": _mean_std(train_mse_values),
            "test_mse": _mean_std(test_mse_values),
            "train_mae": _mean_std(train_mae_values),
            "test_mae": _mean_std(test_mae_values),
            "n_runs": len(seeds),
        }

    benchmark = {
        "benchmark_type": "regression",
        "models": selected_models,
        "seeds": list(seeds),
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "runs": runs,
        "summary": summary,
    }

    if save:
        save_json(benchmark, results_path("benchmarks", filename))

    return benchmark
