import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from qml import QuantumReservoirClassifier
from qml.model_selection import (
    cross_validate_estimator,
    infer_task,
    score_predictions,
    select_best_model,
    train_test_evaluate,
)


def test_infer_task_and_score_predictions() -> None:
    assert infer_task(np.asarray([0, 1, 1])) == "classification"
    assert infer_task(np.asarray([0.1, 0.2, 0.3])) == "regression"
    assert score_predictions([0, 1], [0, 1], "accuracy") == 1.0
    assert score_predictions([1.0, 2.0], [1.0, 3.0], "mean_squared_error") == 0.5
    assert score_predictions([1.0, 2.0], [1.0, 3.0], "neg_mean_squared_error") == -0.5


def test_cross_validate_estimator_classification_summary() -> None:
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.8, -0.7],
            [-0.9, -1.2],
            [1.0, 1.0],
            [0.8, 0.7],
            [1.2, 0.9],
        ]
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])

    result = cross_validate_estimator(
        LogisticRegression(),
        x,
        y,
        cv=3,
        task="classification",
        scoring="accuracy",
        seed=0,
    )

    assert result["task"] == "classification"
    assert result["higher_is_better"] is True
    assert len(result["folds"]) == 3
    assert result["summary"]["test_score"]["n"] == 3
    assert 0.0 <= result["summary"]["test_score"]["mean"] <= 1.0


def test_train_test_evaluate_regression_returns_negative_mse() -> None:
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = 2.0 * x.ravel() + 1.0

    result = train_test_evaluate(Ridge(alpha=0.1), x, y, task="regression", seed=0)

    assert result["scoring"] == "neg_mean_squared_error"
    assert result["higher_is_better"] is True
    assert result["train_size"] == 9
    assert result["test_size"] == 3
    assert result["test_score"] <= 0.0
    assert hasattr(result["estimator"], "predict")


def test_select_best_model_refits_best_candidate() -> None:
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.8, -0.7],
            [-0.9, -1.2],
            [1.0, 1.0],
            [0.8, 0.7],
            [1.2, 0.9],
        ]
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])

    result = select_best_model(
        {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic": LogisticRegression(),
        },
        x,
        y,
        cv=3,
        task="classification",
        seed=0,
    )

    assert result["best_name"] == "logistic"
    assert result["best_estimator"] is not None
    assert result["best_estimator"].predict(x).shape == y.shape
    assert {candidate["name"] for candidate in result["candidates"]} == {"dummy", "logistic"}


def test_cross_validate_rejects_task_incompatible_scoring() -> None:
    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])

    with pytest.raises(ValueError, match="not valid for classification"):
        cross_validate_estimator(DummyClassifier(), x, y, task="classification", scoring="mse")


def test_cross_validate_package_estimator_smoke() -> None:
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.8, -0.7],
            [1.0, 1.0],
            [0.8, 0.7],
        ]
    )
    y = np.asarray([0, 0, 1, 1])

    result = cross_validate_estimator(
        QuantumReservoirClassifier(seed=0, max_iter=200),
        x,
        y,
        cv=2,
        task="classification",
        seed=0,
    )

    assert len(result["folds"]) == 2
    assert 0.0 <= result["summary"]["test_score"]["mean"] <= 1.0
