"""
qml.benchmarks
==============

Benchmark helpers for comparing quantum and classical models across multiple seeds.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean
from time import perf_counter
from typing import Any

from qml._benchmark_utils import (
    benchmark_metadata as _benchmark_metadata,
    summary_stats as _summary_stats,
    timing_from_result as _timing_from_result,
)
from qml.circuit_metadata import circuit_metadata
from qml.classical_baselines import (
    run_elasticnet_regression,
    run_gaussian_process_classifier,
    run_gaussian_process_regressor,
    run_gradient_boosting_classifier,
    run_gradient_boosting_regressor,
    run_kernel_ridge_regression,
    run_knn_classifier,
    run_knn_regressor,
    run_lasso_regression,
    run_logistic_classifier,
    run_mlp_classifier,
    run_mlp_regressor,
    run_random_forest_classifier,
    run_random_forest_regressor,
    run_ridge_regression,
    run_svr_regression,
    run_svm_classifier,
)
from qml.classifiers import run_vqc
from qml.data import make_classification_dataset, make_regression_dataset
from qml.io_utils import results_path, save_json
from qml.kernels import (
    QuantumGaussianProcessRegressor,
    QuantumKernel,
    QuantumKernelRegressor,
)
from qml.kernel_methods import run_quantum_kernel_classifier
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.metric_learning import run_quantum_metric_learner
from qml.qcnn import run_qcnn
from qml.regression import run_vqr
from qml.reservoir import (
    QuantumReservoirClassifier,
    QuantumReservoirFeatures,
    QuantumReservoirRegressor,
)
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier
from qml.trainable_kernels import run_trainable_quantum_kernel_regressor

ClassificationRunner = Callable[..., dict[str, Any]]
RegressionRunner = Callable[..., dict[str, Any]]


def _run_quantum_reservoir_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "moons",
    n_layers: int = 2,
    shots: int | None = None,
    noise_model: dict[str, float] | None = None,
    input_scale: float = 1.0,
    weight_scale: float = 1.0,
    c: float = 1.0,
    max_iter: int = 1000,
    plot: bool = False,
    save: bool = False,
    **logistic_kwargs,
) -> dict[str, Any]:
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    model = QuantumReservoirClassifier(
        reservoir=QuantumReservoirFeatures(
            n_layers=n_layers,
            seed=seed,
            shots=shots,
            noise_model=noise_model,
            input_scale=input_scale,
            weight_scale=weight_scale,
        ),
        c=c,
        seed=seed,
        max_iter=max_iter,
        **logistic_kwargs,
    )
    fit_start = perf_counter()
    model.fit(data["x_train"], data["y_train"])
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    y_train_pred = model.predict(data["x_train"])
    y_test_pred = model.predict(data["x_test"])
    predict_seconds = perf_counter() - predict_start
    n_qubits = int(data["x_train"].shape[1])
    return {
        "model": "quantum_reservoir_classifier",
        "dataset": dataset,
        "seed": seed,
        "n_qubits": n_qubits,
        "circuit_metadata": circuit_metadata(
            model="quantum_reservoir_classifier",
            n_qubits=n_qubits,
            n_layers=n_layers,
            embedding="reservoir_angle",
            embedding_layers=1,
            ansatz=None,
            template="variational",
            trainable_parameters=0,
            extra={"fixed_reservoir_parameters": int(n_layers * n_qubits * 3)},
        ),
        "train_accuracy": accuracy_score(data["y_train"], y_train_pred),
        "test_accuracy": accuracy_score(data["y_test"], y_test_pred),
        "timing": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": fit_seconds + predict_seconds,
        },
    }


def _run_quantum_kernel_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "sine",
    embedding: str = "angle",
    shots: int | None = None,
    noise_model: dict[str, float] | None = None,
    alpha: float = 1.0,
    plot: bool = False,
    save: bool = False,
    **kernel_ridge_kwargs,
) -> dict[str, Any]:
    data = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
        dataset=dataset,
    )
    model = QuantumKernelRegressor(
        kernel=QuantumKernel(
            embedding=embedding,
            shots=shots,
            seed=seed,
            noise_model=noise_model,
        ),
        alpha=alpha,
        seed=seed,
        **kernel_ridge_kwargs,
    )
    fit_start = perf_counter()
    model.fit(data["x_train"], data["y_train"])
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    y_train_pred = model.predict(data["x_train"])
    y_test_pred = model.predict(data["x_test"])
    predict_seconds = perf_counter() - predict_start
    n_qubits = int(data["x_train"].shape[1])
    return {
        "model": "quantum_kernel_regressor",
        "dataset": dataset,
        "seed": seed,
        "n_qubits": n_qubits,
        "circuit_metadata": circuit_metadata(
            model="quantum_kernel_regressor",
            n_qubits=n_qubits,
            n_layers=1,
            embedding=embedding,
            embedding_layers=1,
            ansatz=None,
            template="kernel",
            trainable_parameters=0,
        ),
        "train_mse": mean_squared_error(data["y_train"], y_train_pred),
        "test_mse": mean_squared_error(data["y_test"], y_test_pred),
        "train_mae": mean_absolute_error(data["y_train"], y_train_pred),
        "test_mae": mean_absolute_error(data["y_test"], y_test_pred),
        "timing": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": fit_seconds + predict_seconds,
        },
    }


def _run_quantum_gaussian_process_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "sine",
    embedding: str = "angle",
    shots: int | None = None,
    noise_model: dict[str, float] | None = None,
    alpha: float = 1e-6,
    normalize_y: bool = True,
    plot: bool = False,
    save: bool = False,
) -> dict[str, Any]:
    data = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
        dataset=dataset,
    )
    model = QuantumGaussianProcessRegressor(
        kernel=QuantumKernel(
            embedding=embedding,
            shots=shots,
            seed=seed,
            noise_model=noise_model,
        ),
        alpha=alpha,
        normalize_y=normalize_y,
        seed=seed,
    )
    fit_start = perf_counter()
    model.fit(data["x_train"], data["y_train"])
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    y_train_pred = model.predict(data["x_train"])
    y_test_pred = model.predict(data["x_test"])
    predict_seconds = perf_counter() - predict_start
    n_qubits = int(data["x_train"].shape[1])
    return {
        "model": "quantum_gaussian_process_regressor",
        "dataset": dataset,
        "seed": seed,
        "n_qubits": n_qubits,
        "circuit_metadata": circuit_metadata(
            model="quantum_gaussian_process_regressor",
            n_qubits=n_qubits,
            n_layers=1,
            embedding=embedding,
            embedding_layers=1,
            ansatz=None,
            template="kernel",
            trainable_parameters=0,
        ),
        "train_mse": mean_squared_error(data["y_train"], y_train_pred),
        "test_mse": mean_squared_error(data["y_test"], y_test_pred),
        "train_mae": mean_absolute_error(data["y_train"], y_train_pred),
        "test_mae": mean_absolute_error(data["y_test"], y_test_pred),
        "timing": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": fit_seconds + predict_seconds,
        },
    }


def _run_quantum_reservoir_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "sine",
    n_layers: int = 2,
    shots: int | None = None,
    noise_model: dict[str, float] | None = None,
    input_scale: float = 1.0,
    weight_scale: float = 1.0,
    alpha: float = 1.0,
    plot: bool = False,
    save: bool = False,
    **ridge_kwargs,
) -> dict[str, Any]:
    data = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
        dataset=dataset,
    )
    model = QuantumReservoirRegressor(
        reservoir=QuantumReservoirFeatures(
            n_layers=n_layers,
            seed=seed,
            shots=shots,
            noise_model=noise_model,
            input_scale=input_scale,
            weight_scale=weight_scale,
        ),
        alpha=alpha,
        seed=seed,
        **ridge_kwargs,
    )
    fit_start = perf_counter()
    model.fit(data["x_train"], data["y_train"])
    fit_seconds = perf_counter() - fit_start
    predict_start = perf_counter()
    y_train_pred = model.predict(data["x_train"])
    y_test_pred = model.predict(data["x_test"])
    predict_seconds = perf_counter() - predict_start
    n_qubits = int(data["x_train"].shape[1])
    return {
        "model": "quantum_reservoir_regressor",
        "dataset": dataset,
        "seed": seed,
        "n_qubits": n_qubits,
        "circuit_metadata": circuit_metadata(
            model="quantum_reservoir_regressor",
            n_qubits=n_qubits,
            n_layers=n_layers,
            embedding="reservoir_angle",
            embedding_layers=1,
            ansatz=None,
            template="variational",
            trainable_parameters=0,
            extra={"fixed_reservoir_parameters": int(n_layers * n_qubits * 3)},
        ),
        "train_mse": mean_squared_error(data["y_train"], y_train_pred),
        "test_mse": mean_squared_error(data["y_test"], y_test_pred),
        "train_mae": mean_absolute_error(data["y_train"], y_train_pred),
        "test_mae": mean_absolute_error(data["y_test"], y_test_pred),
        "timing": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": fit_seconds + predict_seconds,
        },
    }


_CLASSIFICATION_MODELS: dict[str, ClassificationRunner] = {
    "vqc": run_vqc,
    "qcnn": run_qcnn,
    "quantum_kernel": run_quantum_kernel_classifier,
    "trainable_quantum_kernel": run_trainable_quantum_kernel_classifier,
    "quantum_metric_learning": run_quantum_metric_learner,
    "quantum_reservoir": _run_quantum_reservoir_classifier,
    "logistic_regression": run_logistic_classifier,
    "svm_classifier": run_svm_classifier,
    "mlp_classifier": run_mlp_classifier,
    "random_forest_classifier": run_random_forest_classifier,
    "gradient_boosting_classifier": run_gradient_boosting_classifier,
    "knn_classifier": run_knn_classifier,
    "gaussian_process_classifier": run_gaussian_process_classifier,
}

_REGRESSION_MODELS: dict[str, RegressionRunner] = {
    "vqr": run_vqr,
    "quantum_kernel_regressor": _run_quantum_kernel_regressor,
    "trainable_quantum_kernel_regressor": run_trainable_quantum_kernel_regressor,
    "quantum_gaussian_process_regressor": _run_quantum_gaussian_process_regressor,
    "quantum_reservoir_regressor": _run_quantum_reservoir_regressor,
    "ridge_regression": run_ridge_regression,
    "mlp_regressor": run_mlp_regressor,
    "kernel_ridge_regression": run_kernel_ridge_regression,
    "svr_regression": run_svr_regression,
    "gaussian_process_regressor": run_gaussian_process_regressor,
    "random_forest_regressor": run_random_forest_regressor,
    "gradient_boosting_regressor": run_gradient_boosting_regressor,
    "knn_regressor": run_knn_regressor,
    "lasso_regression": run_lasso_regression,
    "elasticnet_regression": run_elasticnet_regression,
}

_CLASSICAL_CLASSIFICATION_MODELS = {
    "logistic_regression",
    "svm_classifier",
    "mlp_classifier",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "knn_classifier",
    "gaussian_process_classifier",
}

_CLASSICAL_REGRESSION_MODELS = {
    "ridge_regression",
    "mlp_regressor",
    "kernel_ridge_regression",
    "svr_regression",
    "gaussian_process_regressor",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "knn_regressor",
    "lasso_regression",
    "elasticnet_regression",
}

_MODEL_NAME_ALIASES: dict[str, str] = {
    "kernel": "quantum_kernel",
    "trainable_kernel": "trainable_quantum_kernel",
    "trainable-kernel": "trainable_quantum_kernel",
    "metric_learning": "quantum_metric_learning",
    "metric-learning": "quantum_metric_learning",
    "reservoir": "quantum_reservoir",
    "reservoir_classifier": "quantum_reservoir",
    "reservoir-classifier": "quantum_reservoir",
    "quantum_kernel_regression": "quantum_kernel_regressor",
    "kernel_regressor": "quantum_kernel_regressor",
    "kernel-regressor": "quantum_kernel_regressor",
    "trainable_kernel_regressor": "trainable_quantum_kernel_regressor",
    "trainable-kernel-regressor": "trainable_quantum_kernel_regressor",
    "quantum_gpr": "quantum_gaussian_process_regressor",
    "gpr_quantum": "quantum_gaussian_process_regressor",
    "reservoir_regressor": "quantum_reservoir_regressor",
    "reservoir-regressor": "quantum_reservoir_regressor",
    "rf_classifier": "random_forest_classifier",
    "gb_classifier": "gradient_boosting_classifier",
    "gpc": "gaussian_process_classifier",
    "rf_regressor": "random_forest_regressor",
    "gb_regressor": "gradient_boosting_regressor",
    "gpr": "gaussian_process_regressor",
    "kernel_ridge": "kernel_ridge_regression",
    "svr": "svr_regression",
    "lasso": "lasso_regression",
    "elasticnet": "elasticnet_regression",
}


def _apply_classical_tuning(
    model_name: str,
    model_kwargs: dict[str, dict[str, Any]],
    classical_models: set[str],
    *,
    tune_classical: bool,
    cv: int,
) -> dict[str, Any]:
    kwargs = dict(model_kwargs.get(model_name, {}))
    if model_name in classical_models:
        kwargs.setdefault("tune", tune_classical)
        kwargs.setdefault("cv", cv)
    return kwargs


def _paired_classical_comparison(
    runs: list[dict[str, Any]],
    selected_models: list[str],
    classical_models: set[str],
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    classical_selected = [model for model in selected_models if model in classical_models]
    if not classical_selected:
        return {"reference_model": None, "metric": metric, "comparisons": {}}

    means = {}
    for model in classical_selected:
        values = [float(run[metric]) for run in runs if run["model"] == model]
        if values:
            means[model] = mean(values)
    if not means:
        return {"reference_model": None, "metric": metric, "comparisons": {}}

    reference_model = max(means, key=means.get) if higher_is_better else min(means, key=means.get)
    reference_by_seed = {
        run["seed"]: float(run[metric])
        for run in runs
        if run["model"] == reference_model and metric in run
    }

    comparisons: dict[str, Any] = {}
    for model in selected_models:
        deltas = []
        wins = 0
        losses = 0
        ties = 0
        for run in runs:
            if run["model"] != model or run["seed"] not in reference_by_seed:
                continue
            delta = float(run[metric]) - reference_by_seed[run["seed"]]
            deltas.append(delta)
            better_delta = delta if higher_is_better else -delta
            if better_delta > 0:
                wins += 1
            elif better_delta < 0:
                losses += 1
            else:
                ties += 1

        comparisons[model] = {
            "mean_delta": _summary_stats(deltas),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "n_pairs": len(deltas),
        }

    return {
        "reference_model": reference_model,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "comparisons": comparisons,
    }


def _best_model_by_summary_metric(
    summary: dict[str, dict[str, Any]],
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    """
    Return the best model according to one aggregated summary metric.
    """
    if not summary:
        return {"model": None, "metric": metric, "value": float("nan")}

    candidates = [
        (model_name, float(model_summary[metric]["mean"]))
        for model_name, model_summary in summary.items()
    ]
    best_name, best_value = (
        max(candidates, key=lambda item: item[1])
        if higher_is_better
        else min(candidates, key=lambda item: item[1])
    )
    return {
        "model": best_name,
        "metric": metric,
        "value": best_value,
        "higher_is_better": higher_is_better,
    }


def _circuit_metadata_from_result(result: Any) -> dict[str, Any] | None:
    """
    Extract JSON-friendly circuit metadata from a workflow result when present.
    """
    if isinstance(result, dict):
        metadata = result.get("circuit_metadata")
    else:
        metadata = getattr(result, "circuit_metadata", None)

    if not isinstance(metadata, dict):
        return None
    return dict(metadata)


def _add_circuit_metadata(run_record: dict[str, Any], result: Any) -> None:
    metadata = _circuit_metadata_from_result(result)
    if metadata is None:
        return
    run_record["circuit_metadata"] = metadata
    if "trainable_parameters" in metadata:
        run_record["trainable_parameters"] = int(metadata["trainable_parameters"])
    if "estimated_depth" in metadata:
        run_record["estimated_depth"] = int(metadata["estimated_depth"])


def _add_circuit_summary(model_summary: dict[str, Any], model_runs: list[dict[str, Any]]) -> None:
    trainable_parameters = [
        float(run["trainable_parameters"]) for run in model_runs if "trainable_parameters" in run
    ]
    estimated_depths = [
        float(run["estimated_depth"]) for run in model_runs if "estimated_depth" in run
    ]
    if trainable_parameters:
        model_summary["trainable_parameters"] = _summary_stats(trainable_parameters)
    if estimated_depths:
        model_summary["estimated_depth"] = _summary_stats(estimated_depths)


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
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(available)}.")
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
    if model_name == "trainable_quantum_kernel_regressor":
        kwargs.pop("plot", None)
        kwargs.pop("save", None)
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
    tune_classical: bool = False,
    cv: int = 3,
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
    tune_classical
        Whether to run supported classical baselines through ``GridSearchCV``.
    cv
        Cross-validation folds used when ``tune_classical=True``.

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
        final_losses: list[float] = []
        runtime_values: list[float] = []

        for seed in seeds:
            start = perf_counter()
            result = _run_classification_model(
                model_name=model_name,
                runner=runner,
                common_kwargs={**common_kwargs, "seed": seed},
                model_kwargs={
                    **model_kwargs,
                    model_name: _apply_classical_tuning(
                        model_name,
                        model_kwargs,
                        _CLASSICAL_CLASSIFICATION_MODELS,
                        tune_classical=tune_classical,
                        cv=cv,
                    ),
                },
            )
            runtime_seconds = perf_counter() - start

            if isinstance(result, dict):
                train_accuracy = float(result["train_accuracy"])
                test_accuracy = float(result["test_accuracy"])
            else:
                train_accuracy = float(result.train_accuracy)
                test_accuracy = float(result.test_accuracy)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            runtime_values.append(runtime_seconds)

            run_record = {
                "model": model_name,
                "seed": seed,
                "dataset": dataset,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "generalization_gap": train_accuracy - test_accuracy,
                "runtime_seconds": runtime_seconds,
                "timing": _timing_from_result(result, runtime_seconds),
            }
            _add_circuit_metadata(run_record, result)

            if isinstance(result, dict):
                if "tuning" in result:
                    run_record["tuning"] = result["tuning"]

                if "final_loss" in result:
                    final_loss = float(result["final_loss"])
                    run_record["final_loss"] = final_loss
                    final_losses.append(final_loss)

                if "final_alignment" in result:
                    run_record["final_alignment"] = float(result["final_alignment"])
            else:
                if hasattr(result, "loss_history") and result.loss_history:
                    final_loss = float(result.loss_history[-1])
                    run_record["final_loss"] = final_loss
                    final_losses.append(final_loss)

            runs.append(run_record)

        model_summary = {
            "train_accuracy": _summary_stats(train_accuracies),
            "test_accuracy": _summary_stats(test_accuracies),
            "generalization_gap": _summary_stats(
                [
                    train_accuracy - test_accuracy
                    for train_accuracy, test_accuracy in zip(
                        train_accuracies,
                        test_accuracies,
                        strict=True,
                    )
                ]
            ),
            "runtime_seconds": _summary_stats(runtime_values),
            "n_runs": len(seeds),
        }

        if final_losses:
            model_summary["final_loss"] = _summary_stats(final_losses)

        alignment_values = [
            float(run["final_alignment"])
            for run in runs
            if run["model"] == model_name and "final_alignment" in run
        ]
        if alignment_values:
            model_summary["final_alignment"] = _summary_stats(alignment_values)

        _add_circuit_summary(
            model_summary,
            [run for run in runs if run["model"] == model_name],
        )
        summary[model_name] = model_summary

    benchmark = {
        "benchmark_type": "classification",
        "models": selected_models,
        "seeds": list(seeds),
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "dataset": dataset,
        "tune_classical": tune_classical,
        "cv": cv,
        "runs": runs,
        "summary": summary,
        "best_model": _best_model_by_summary_metric(
            summary,
            "test_accuracy",
            higher_is_better=True,
        ),
        "paired_vs_best_classical": _paired_classical_comparison(
            runs,
            selected_models,
            _CLASSICAL_CLASSIFICATION_MODELS,
            "test_accuracy",
            higher_is_better=True,
        ),
        "metadata": _benchmark_metadata(),
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
    tune_classical: bool = False,
    cv: int = 3,
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
    tune_classical
        Whether to run supported classical baselines through ``GridSearchCV``.
    cv
        Cross-validation folds used when ``tune_classical=True``.

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
        final_losses: list[float] = []
        runtime_values: list[float] = []

        for seed in seeds:
            start = perf_counter()
            result = _run_regression_model(
                model_name=model_name,
                runner=runner,
                common_kwargs={**common_kwargs, "seed": seed},
                model_kwargs={
                    **model_kwargs,
                    model_name: _apply_classical_tuning(
                        model_name,
                        model_kwargs,
                        _CLASSICAL_REGRESSION_MODELS,
                        tune_classical=tune_classical,
                        cv=cv,
                    ),
                },
            )
            runtime_seconds = perf_counter() - start

            train_mse = float(result["train_mse"])
            test_mse = float(result["test_mse"])
            train_mae = float(result["train_mae"])
            test_mae = float(result["test_mae"])

            train_mse_values.append(train_mse)
            test_mse_values.append(test_mse)
            train_mae_values.append(train_mae)
            test_mae_values.append(test_mae)
            runtime_values.append(runtime_seconds)

            run_record = {
                "model": model_name,
                "seed": seed,
                "dataset": dataset,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "generalization_gap": test_mse - train_mse,
                "runtime_seconds": runtime_seconds,
                "timing": _timing_from_result(result, runtime_seconds),
            }
            _add_circuit_metadata(run_record, result)

            if "tuning" in result:
                run_record["tuning"] = result["tuning"]

            if "final_loss" in result:
                final_loss = float(result["final_loss"])
                run_record["final_loss"] = final_loss
                final_losses.append(final_loss)

            runs.append(run_record)

        model_summary = {
            "train_mse": _summary_stats(train_mse_values),
            "test_mse": _summary_stats(test_mse_values),
            "train_mae": _summary_stats(train_mae_values),
            "test_mae": _summary_stats(test_mae_values),
            "generalization_gap": _summary_stats(
                [
                    test_mse - train_mse
                    for train_mse, test_mse in zip(
                        train_mse_values,
                        test_mse_values,
                        strict=True,
                    )
                ]
            ),
            "runtime_seconds": _summary_stats(runtime_values),
            "n_runs": len(seeds),
        }
        if final_losses:
            model_summary["final_loss"] = _summary_stats(final_losses)

        _add_circuit_summary(
            model_summary,
            [run for run in runs if run["model"] == model_name],
        )
        summary[model_name] = model_summary

    benchmark = {
        "benchmark_type": "regression",
        "models": selected_models,
        "seeds": list(seeds),
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "dataset": dataset,
        "tune_classical": tune_classical,
        "cv": cv,
        "runs": runs,
        "summary": summary,
        "best_model": _best_model_by_summary_metric(
            summary,
            "test_mse",
            higher_is_better=False,
        ),
        "paired_vs_best_classical": _paired_classical_comparison(
            runs,
            selected_models,
            _CLASSICAL_REGRESSION_MODELS,
            "test_mse",
            higher_is_better=False,
        ),
        "metadata": _benchmark_metadata(),
    }

    if save:
        save_json(benchmark, results_path("benchmarks", filename))

    return benchmark
