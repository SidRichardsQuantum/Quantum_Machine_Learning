"""
qml.model_selection
===================

Small model-selection helpers for estimator-style QML workflows.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from qml._benchmark_utils import summary_stats
from qml.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

_CLASSIFICATION_SCORERS = {"accuracy", "balanced_accuracy", "f1", "f1_binary"}
_REGRESSION_SCORERS = {
    "mean_squared_error",
    "mse",
    "neg_mean_squared_error",
    "neg_mse",
    "mean_absolute_error",
    "mae",
    "neg_mean_absolute_error",
    "neg_mae",
    "root_mean_squared_error",
    "rmse",
    "neg_root_mean_squared_error",
    "neg_rmse",
    "r2",
}

__all__ = [
    "clone_estimator",
    "cross_validate_estimator",
    "default_scoring",
    "infer_task",
    "score_predictions",
    "scorer_direction",
    "selection_summary_rows",
    "select_best_model",
    "train_test_evaluate",
]


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    return x


def _as_target(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 0:
        raise ValueError("y must contain at least one sample.")
    if y.ndim > 2:
        raise ValueError(f"Expected a 1D or 2D target array, got shape {y.shape}.")
    return y


def _validate_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = _as_2d(x)
    y = _as_target(y)
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"x and y must contain the same number of samples, got {x.shape[0]} and {y.shape[0]}."
        )
    if x.shape[0] < 2:
        raise ValueError("At least two samples are required.")
    return x, y


def infer_task(y, task: str = "auto") -> str:
    """
    Infer or validate a task name.

    ``"auto"`` treats string, boolean, and integer targets as classification
    and floating targets as regression. Pass ``task`` explicitly when floating
    labels represent classes.
    """
    task = task.strip().lower()
    if task in {"classification", "classifier", "classify"}:
        return "classification"
    if task in {"regression", "regressor", "regress"}:
        return "regression"
    if task != "auto":
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")

    y = _as_target(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return "regression"
    if np.issubdtype(y.dtype, np.bool_) or np.issubdtype(y.dtype, np.integer):
        return "classification"
    if np.issubdtype(y.dtype, np.str_) or np.issubdtype(y.dtype, np.object_):
        return "classification"
    return "regression"


def default_scoring(task: str) -> str:
    """Return the default scorer name for a task."""
    task = infer_task([], task)
    return "accuracy" if task == "classification" else "neg_mean_squared_error"


def scorer_direction(scoring: str) -> bool:
    """Return whether larger scores are better for a scorer name."""
    scoring = scoring.strip().lower()
    larger_is_better = {"accuracy", "balanced_accuracy", "f1", "f1_binary", "r2"}
    if scoring.startswith("neg_") or scoring in larger_is_better:
        return True
    if scoring in {
        "mean_squared_error",
        "mse",
        "mean_absolute_error",
        "mae",
        "root_mean_squared_error",
        "rmse",
    }:
        return False
    raise ValueError(f"Unsupported scoring value {scoring!r}.")


def score_predictions(y_true, y_pred, scoring: str) -> float:
    """
    Score predictions with a supported scorer.

    Supported classification values are ``accuracy``, ``balanced_accuracy``,
    and binary ``f1``/``f1_binary``. Supported regression values are
    ``mean_squared_error``/``mse``, ``root_mean_squared_error``/``rmse``,
    ``mean_absolute_error``/``mae``, ``r2``, and negative loss variants.
    """
    scoring = scoring.strip().lower()
    if scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if scoring in {"f1", "f1_binary"}:
        return f1_score(y_true, y_pred)
    if scoring in {"mean_squared_error", "mse"}:
        return mean_squared_error(y_true, y_pred)
    if scoring in {"neg_mean_squared_error", "neg_mse"}:
        return -mean_squared_error(y_true, y_pred)
    if scoring in {"root_mean_squared_error", "rmse"}:
        return root_mean_squared_error(y_true, y_pred)
    if scoring in {"neg_root_mean_squared_error", "neg_rmse"}:
        return -root_mean_squared_error(y_true, y_pred)
    if scoring in {"mean_absolute_error", "mae"}:
        return mean_absolute_error(y_true, y_pred)
    if scoring in {"neg_mean_absolute_error", "neg_mae"}:
        return -mean_absolute_error(y_true, y_pred)
    if scoring == "r2":
        return r2_score(y_true, y_pred)
    raise ValueError(f"Unsupported scoring value {scoring!r}.")


def _validate_scoring(task: str, scoring: str) -> str:
    scoring = scoring.strip().lower()
    valid = _CLASSIFICATION_SCORERS if task == "classification" else _REGRESSION_SCORERS
    if scoring not in valid:
        raise ValueError(f"Scoring {scoring!r} is not valid for {task}.")
    return scoring


def clone_estimator(estimator):
    """
    Clone an estimator from its constructor parameters.

    The helper supports the package's estimator-style classes and most sklearn
    estimators that expose ``get_params``.
    """
    if not hasattr(estimator, "get_params"):
        raise ValueError("Estimator must expose get_params().")
    return estimator.__class__(**copy.deepcopy(estimator.get_params(deep=False)))


def _splitter(task: str, cv: int, *, shuffle: bool, seed: int):
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    if task == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=seed if shuffle else None)
    return KFold(n_splits=cv, shuffle=shuffle, random_state=seed if shuffle else None)


def _fit_predict_score(estimator, x_train, y_train, x_test, y_test, scoring: str) -> dict[str, Any]:
    start = perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    train_pred = estimator.predict(x_train)
    test_pred = estimator.predict(x_test)
    score_seconds = perf_counter() - start

    train_score = score_predictions(y_train, train_pred, scoring)
    test_score = score_predictions(y_test, test_pred, scoring)
    return {
        "estimator": estimator,
        "train_score": float(train_score),
        "test_score": float(test_score),
        "fit_seconds": float(fit_seconds),
        "score_seconds": float(score_seconds),
    }


def cross_validate_estimator(
    estimator,
    x,
    y,
    *,
    cv: int = 3,
    task: str = "auto",
    scoring: str | None = None,
    shuffle: bool = True,
    seed: int = 123,
    return_estimators: bool = False,
) -> dict[str, Any]:
    """
    Cross-validate an estimator-style model on user-supplied arrays.

    Returns fold records and aggregate summaries for train score, test score,
    fit time, and scoring time. Scores are oriented according to ``scoring``:
    negative regression losses are larger-is-better.
    """
    x, y = _validate_xy(x, y)
    task_name = infer_task(y, task)
    scoring_name = _validate_scoring(task_name, scoring or default_scoring(task_name))
    splitter = _splitter(task_name, cv, shuffle=shuffle, seed=seed)

    folds = []
    split_y = y if task_name == "classification" else None
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, split_y), start=1):
        fitted = clone_estimator(estimator)
        result = _fit_predict_score(
            fitted,
            x[train_idx],
            y[train_idx],
            x[test_idx],
            y[test_idx],
            scoring_name,
        )
        fold_record = {
            "fold": fold,
            "train_size": int(train_idx.size),
            "test_size": int(test_idx.size),
            "train_score": result["train_score"],
            "test_score": result["test_score"],
            "fit_seconds": result["fit_seconds"],
            "score_seconds": result["score_seconds"],
        }
        if return_estimators:
            fold_record["estimator"] = result["estimator"]
        folds.append(fold_record)

    return {
        "task": task_name,
        "scoring": scoring_name,
        "higher_is_better": scorer_direction(scoring_name),
        "cv": cv,
        "folds": folds,
        "summary": {
            "train_score": summary_stats([fold["train_score"] for fold in folds]),
            "test_score": summary_stats([fold["test_score"] for fold in folds]),
            "fit_seconds": summary_stats([fold["fit_seconds"] for fold in folds]),
            "score_seconds": summary_stats([fold["score_seconds"] for fold in folds]),
        },
    }


def train_test_evaluate(
    estimator,
    x,
    y,
    *,
    test_size: float = 0.25,
    task: str = "auto",
    scoring: str | None = None,
    seed: int = 123,
    stratify: bool | None = None,
    return_estimator: bool = True,
) -> dict[str, Any]:
    """
    Fit an estimator on one deterministic train/test split and return metrics.
    """
    x, y = _validate_xy(x, y)
    task_name = infer_task(y, task)
    scoring_name = _validate_scoring(task_name, scoring or default_scoring(task_name))
    use_stratify = task_name == "classification" if stratify is None else stratify
    stratify_target = y if use_stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_target,
    )

    fitted = clone_estimator(estimator)
    result = _fit_predict_score(fitted, x_train, y_train, x_test, y_test, scoring_name)
    record = {
        "task": task_name,
        "scoring": scoring_name,
        "higher_is_better": scorer_direction(scoring_name),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
        "train_score": result["train_score"],
        "test_score": result["test_score"],
        "fit_seconds": result["fit_seconds"],
        "score_seconds": result["score_seconds"],
    }
    if return_estimator:
        record["estimator"] = result["estimator"]
    return record


def _candidate_items(candidates) -> list[tuple[str, Any]]:
    if isinstance(candidates, Mapping):
        items = list(candidates.items())
    else:
        if not isinstance(candidates, Iterable):
            raise ValueError("candidates must be a mapping or iterable of estimators.")
        items = [(candidate.__class__.__name__, candidate) for candidate in candidates]
    if not items:
        raise ValueError("At least one candidate estimator is required.")
    return [(str(name), estimator) for name, estimator in items]


def selection_summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Return compact table rows for model-selection helper outputs.

    The function accepts the dictionaries returned by
    ``cross_validate_estimator(...)``, ``train_test_evaluate(...)``, or
    ``select_best_model(...)`` and normalizes their primary score and timing
    fields for use with ``qml.reporting.format_table``.
    """
    if "candidates" in result:
        rows = []
        for candidate in result["candidates"]:
            cv_result = candidate["cv_result"]
            test_summary = cv_result["summary"]["test_score"]
            fit_summary = cv_result["summary"]["fit_seconds"]
            rows.append(
                {
                    "name": candidate["name"],
                    "scoring": result["scoring"],
                    "mean_test_score": float(test_summary["mean"]),
                    "ci95_low": float(test_summary["ci95_low"]),
                    "ci95_high": float(test_summary["ci95_high"]),
                    "fit_seconds": float(fit_summary["mean"]),
                    "best": candidate["name"] == result.get("best_name"),
                }
            )
        return rows

    if "folds" in result:
        return [
            {
                "fold": fold["fold"],
                "train_size": fold["train_size"],
                "test_size": fold["test_size"],
                "train_score": fold["train_score"],
                "test_score": fold["test_score"],
                "fit_seconds": fold["fit_seconds"],
                "score_seconds": fold["score_seconds"],
            }
            for fold in result["folds"]
        ]

    return [
        {
            "task": result["task"],
            "scoring": result["scoring"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "train_score": result["train_score"],
            "test_score": result["test_score"],
            "fit_seconds": result["fit_seconds"],
            "score_seconds": result["score_seconds"],
        }
    ]


def select_best_model(
    candidates,
    x,
    y,
    *,
    cv: int = 3,
    task: str = "auto",
    scoring: str | None = None,
    shuffle: bool = True,
    seed: int = 123,
    refit: bool = True,
) -> dict[str, Any]:
    """
    Cross-validate candidate estimators and optionally refit the best one.

    ``candidates`` may be a mapping of names to estimators or an iterable of
    estimators. The best candidate is selected by mean cross-validation test
    score, respecting whether the scorer is larger-is-better.
    """
    x, y = _validate_xy(x, y)
    task_name = infer_task(y, task)
    scoring_name = _validate_scoring(task_name, scoring or default_scoring(task_name))
    higher_is_better = scorer_direction(scoring_name)

    results = []
    for name, estimator in _candidate_items(candidates):
        cv_result = cross_validate_estimator(
            estimator,
            x,
            y,
            cv=cv,
            task=task_name,
            scoring=scoring_name,
            shuffle=shuffle,
            seed=seed,
        )
        results.append(
            {
                "name": name,
                "estimator": estimator,
                "mean_test_score": cv_result["summary"]["test_score"]["mean"],
                "cv_result": cv_result,
            }
        )

    def mean_test_score(result: dict[str, Any]) -> float:
        return result["mean_test_score"]

    if higher_is_better:
        best = max(results, key=mean_test_score)
    else:
        best = min(results, key=mean_test_score)
    fitted = None
    if refit:
        fitted = clone_estimator(best["estimator"])
        fitted.fit(x, y)

    return {
        "task": task_name,
        "scoring": scoring_name,
        "higher_is_better": higher_is_better,
        "best_name": best["name"],
        "best_score": float(best["mean_test_score"]),
        "best_estimator": fitted,
        "candidates": [
            {
                "name": result["name"],
                "mean_test_score": float(result["mean_test_score"]),
                "cv_result": result["cv_result"],
            }
            for result in results
        ],
    }
