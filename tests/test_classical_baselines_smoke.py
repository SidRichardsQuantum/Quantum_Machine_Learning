import math

import warnings

from sklearn.exceptions import ConvergenceWarning


from qml.classical_baselines import (
    run_logistic_classifier,
    run_mlp_classifier,
    run_mlp_regressor,
    run_ridge_regression,
    run_svm_classifier,
)


def test_run_logistic_classifier_smoke():
    result = run_logistic_classifier(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
    )

    assert result["model"] == "logistic_regression"
    assert result["dataset"] == "moons"
    assert math.isfinite(result["train_accuracy"])
    assert math.isfinite(result["test_accuracy"])
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_run_svm_classifier_smoke():
    result = run_svm_classifier(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
    )

    assert result["model"] == "svm_classifier"
    assert result["dataset"] == "moons"
    assert math.isfinite(result["train_accuracy"])
    assert math.isfinite(result["test_accuracy"])
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_run_mlp_classifier_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        result = run_mlp_classifier(
            n_samples=40,
            noise=0.1,
            test_size=0.25,
            seed=0,
            hidden_layer_sizes=(8,),
            max_iter=100,
        )

    assert result["model"] == "mlp_classifier"
    assert result["dataset"] == "moons"
    assert math.isfinite(result["train_accuracy"])
    assert math.isfinite(result["test_accuracy"])
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert len(result["loss_curve"]) > 0


def test_run_ridge_regression_smoke():
    result = run_ridge_regression(
        n_samples=40,
        noise=0.1,
        test_size=0.25,
        seed=0,
        alpha=1.0,
    )

    assert result["model"] == "ridge_regression"
    assert result["dataset"] == "regression"
    assert math.isfinite(result["train_mse"])
    assert math.isfinite(result["test_mse"])
    assert math.isfinite(result["train_mae"])
    assert math.isfinite(result["test_mae"])
    assert result["train_mse"] >= 0.0
    assert result["test_mse"] >= 0.0
    assert result["train_mae"] >= 0.0
    assert result["test_mae"] >= 0.0


def test_run_mlp_regressor_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        result = run_mlp_regressor(
            n_samples=40,
            noise=0.1,
            test_size=0.25,
            seed=0,
            hidden_layer_sizes=(16,),
            max_iter=100,
        )

    assert result["model"] == "mlp_regressor"
    assert result["dataset"] == "regression"
    assert math.isfinite(result["train_mse"])
    assert math.isfinite(result["test_mse"])
    assert math.isfinite(result["train_mae"])
    assert math.isfinite(result["test_mae"])
    assert result["train_mse"] >= 0.0
    assert result["test_mse"] >= 0.0
    assert result["train_mae"] >= 0.0
    assert result["test_mae"] >= 0.0
    assert len(result["loss_curve"]) > 0
