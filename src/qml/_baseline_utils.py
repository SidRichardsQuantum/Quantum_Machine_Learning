"""
Shared helpers for classical baseline workflows.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV

from qml.data import make_classification_dataset, make_regression_dataset
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def classification_stem(
    model_name: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> str:
    return f"{model_name}_samples{n_samples}_noise{str(noise).replace('.', 'p')}_seed{seed}"


def regression_stem(
    model_name: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> str:
    return f"{model_name}_samples{n_samples}_noise{str(noise).replace('.', 'p')}_seed{seed}"


def classification_data(dataset, n_samples, noise, test_size, seed):
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"]


def regression_data(dataset, n_samples, noise, test_size, seed):
    data = make_regression_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    return data["x_train"], data["x_test"], data["y_train"], data["y_test"]


def decision_function_from_predict_proba(model):
    def predict_proba_grid(x_grid):
        x_grid = np.asarray(x_grid, dtype=float)
        return model.predict_proba(x_grid)[:, 1]

    return predict_proba_grid


def decision_function_from_predict(model):
    def predict_grid(x_grid):
        x_grid = np.asarray(x_grid, dtype=float)
        return model.predict(x_grid)

    return predict_grid


def fit_estimator(
    estimator,
    x_train,
    y_train,
    *,
    tune: bool,
    param_grid: dict[str, list[Any]] | None,
    cv: int,
    scoring: str | None = None,
) -> tuple[Any, dict[str, Any], float]:
    """
    Fit an estimator, optionally through GridSearchCV, and return fit metadata.
    """
    start = perf_counter()
    if tune:
        search = GridSearchCV(
            estimator,
            param_grid=param_grid or {},
            cv=cv,
            scoring=scoring,
            n_jobs=None,
        )
        search.fit(x_train, y_train)
        fit_seconds = perf_counter() - start
        return (
            search.best_estimator_,
            {
                "enabled": True,
                "cv": cv,
                "best_params": search.best_params_,
                "best_score": float(search.best_score_),
            },
            fit_seconds,
        )

    estimator.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    return estimator, {"enabled": False}, fit_seconds


def classifier_result(
    *,
    model_name: str,
    dataset: str,
    seed: int,
    n_samples: int,
    noise: float,
    test_size: float,
    estimator,
    x_train,
    x_test,
    y_train,
    y_test,
    extra: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
    fit_seconds: float = 0.0,
) -> dict[str, Any]:
    start = perf_counter()
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)
    predict_seconds = perf_counter() - start

    result = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
        "timing": {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": float(fit_seconds + predict_seconds),
        },
        "tuning": tuning or {"enabled": False},
    }
    if hasattr(estimator, "predict_proba"):
        result["train_probabilities"] = np.asarray(
            estimator.predict_proba(x_train)[:, 1], dtype=float
        )
        result["test_probabilities"] = np.asarray(
            estimator.predict_proba(x_test)[:, 1], dtype=float
        )
    if extra:
        result.update(extra)
    return result


def regressor_result(
    *,
    model_name: str,
    dataset: str,
    seed: int,
    n_samples: int,
    noise: float,
    test_size: float,
    estimator,
    x_train,
    x_test,
    y_train,
    y_test,
    extra: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
    fit_seconds: float = 0.0,
) -> dict[str, Any]:
    start = perf_counter()
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)
    predict_seconds = perf_counter() - start

    result = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
        "y_train_pred": np.asarray(y_train_pred, dtype=float),
        "y_test_pred": np.asarray(y_test_pred, dtype=float),
        "timing": {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": float(fit_seconds + predict_seconds),
        },
        "tuning": tuning or {"enabled": False},
    }
    if extra:
        result.update(extra)
    return result
