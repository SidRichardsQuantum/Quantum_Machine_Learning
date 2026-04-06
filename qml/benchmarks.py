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
from qml.metric_learning import run_quantum_metric_learner
from qml.regression import run_vqr
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier

ClassificationRunner = Callable[..., dict[str, Any]]
RegressionRunner = Callable[..., dict[str, Any]]


_CLASSIFICATION_MODELS: dict[str, ClassificationRunner] = {
    "vqc": run_vqc,
    "quantum_kernel": run_quantum_kernel_classifier,
    "trainable_quantum_kernel": run_trainable_quantum_kernel_classifier,
    "quantum_metric_learning": run_quantum_metric_learner,
    "logistic_regression": run_logistic_classifier,
    "svm_classifier": run_svm_classifier,
    "mlp_classifier": run_mlp_classifier,
}

_REGRESSION_MODELS: dict[str, RegressionRunner] = {
    "vqr": run_vqr,
    "ridge_regression": run_ridge_regression,
    "mlp_regressor": run_mlp_regressor,
}

_MODEL_NAME_ALIASES: dict[str, str] = {
    "kernel": "quantum_kernel",
    "trainable_kernel": "trainable_quantum_kernel",
    "trainable-kernel": "trainable_quantum_kernel",
    "metric_learning": "quantum_metric_learning",
    "metric-learning": "quantum_metric_learning",
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


def _canonical_model_name(
    model_name: str,
    available_models: dict[str, Callable[..., dict[str, Any]]],
) -> str:
    """
    Return the canonical model name, resolving supported aliases.
    """
    canonical = _MODEL_NAME_ALIASES.get(model_name, model_name)
    if canonical not in available_models:
        available = sorted(set(available_models) | set(_MODEL_NAME_ALIASES))
        raise ValueError(
            f"Unknown model: {model_name}. " f"Available models: {', '.join(available)}."
        )
    return canonical


def _normalize_model_names(
    requested_models: list[str] | None,
    available_models: dict[str, Callable[..., dict[str, Any]]],
) -> list[str]:
    """
    Validate and canonicalize requested model names while preserving order.
    """
    if requested_models is None:
        return list(available_models.keys())

    normalized: list[str] = []
    for name in requested_models:
        canonical = _canonical_model_name(name, available_models)
        if canonical not in normalized:
            normalized.append(canonical)

    return normalized


def _normalize_model_kwargs(
    model_kwargs: dict[str, dict[str, Any]] | None,
    available_models: dict[str, Callable[..., dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """
    Canonicalize per-model kwargs keys and merge alias entries if needed.
    """
    if model_kwargs is None:
        return {}

    normalized: dict[str, dict[str, Any]] = {}

    for model_name, kwargs in model_kwargs.items():
        canonical = _canonical_model_name(model_name, available_models)
        if canonical not in normalized:
            normalized[canonical] = {}
        normalized[canonical].update(kwargs)

    return normalized


def _prepare_runner_kwargs(
    model_name: str,
    common_kwargs: dict[str, Any],
    model_kwargs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge common kwargs with model-specific kwargs and disable plotting/saving.
    """
    kwargs = dict(common_kwargs)
    kwargs.update(model_kwargs.get(model_name, {}))
    kwargs["plot"] = False
    kwargs["save"] = False
    return kwargs


def _validate_models(
    requested_models: list[str] | None,
    available_models: dict[str, Callable[..., dict[str, Any]]],
    benchmark_name: str,
) -> list[str]:
    """
    Validate requested model names against the available registry.
    """
    try:
        return _normalize_model_names(requested_models, available_models)
    except ValueError as exc:
        raise ValueError(f"Invalid model selection for {benchmark_name}: {exc}") from exc


def _run_classification_model(
    model_name: str,
    runner: ClassificationRunner,
    common_kwargs: dict[str, Any],
    model_kwargs: dict[str, dict[str, Any]],
) -> Any:
    """
    Run one classification model with merged kwargs.
    """
    kwargs = _prepare_runner_kwargs(
        model_name=model_name,
        common_kwargs=common_kwargs,
        model_kwargs=model_kwargs,
    )

    if model_name in {
        "logistic_regression",
        "svm_classifier",
        "mlp_classifier",
    }:
        kwargs.pop("dataset", None)

    if model_name == "quantum_metric_learning":
        if "n_samples" in kwargs:
            kwargs["samples"] = kwargs.pop("n_samples")
        if "n_layers" in kwargs:
            kwargs["layers"] = kwargs.pop("n_layers")
        if "step_size" in kwargs:
            kwargs["stepsize"] = kwargs.pop("step_size")
        kwargs.pop("noise", None)
        kwargs.pop("save", None)

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
    kwargs = _prepare_runner_kwargs(
        model_name=model_name,
        common_kwargs=common_kwargs,
        model_kwargs=model_kwargs,
    )
    if model_name in {
        "ridge_regression",
        "mlp_regressor",
    }:
        kwargs.pop("dataset", None)
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
    dataset: str = "moons",
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
    model_kwargs = _normalize_model_kwargs(model_kwargs, _CLASSIFICATION_MODELS)

    common_kwargs = {
        "dataset": dataset,
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

            if isinstance(result, dict):
                train_accuracy = float(result["train_accuracy"])
                test_accuracy = float(result["test_accuracy"])
            else:
                train_accuracy = float(result.train_accuracy)
                test_accuracy = float(result.test_accuracy)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            run_record = {
                "model": model_name,
                "seed": seed,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }

            if isinstance(result, dict):
                if "final_loss" in result:
                    run_record["final_loss"] = float(result["final_loss"])

                if "final_alignment" in result:
                    run_record["final_alignment"] = float(result["final_alignment"])
            else:
                if hasattr(result, "loss_history") and result.loss_history:
                    run_record["final_loss"] = float(result.loss_history[-1])

            runs.append(run_record)

        model_summary = {
            "train_accuracy": _mean_std(train_accuracies),
            "test_accuracy": _mean_std(test_accuracies),
            "n_runs": len(seeds),
        }

        alignment_values = [
            float(run["final_alignment"])
            for run in runs
            if run["model"] == model_name and "final_alignment" in run
        ]
        if alignment_values:
            model_summary["final_alignment"] = _mean_std(alignment_values)

        summary[model_name] = model_summary

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
    dataset: str = "linear",
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
    model_kwargs = _normalize_model_kwargs(model_kwargs, _REGRESSION_MODELS)

    common_kwargs = {
        "dataset": dataset,
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
