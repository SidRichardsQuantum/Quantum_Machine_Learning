import math
import warnings

from sklearn.exceptions import ConvergenceWarning

from qml.classical_baselines import (
    run_gaussian_process_classifier,
    run_gaussian_process_regressor,
    run_gradient_boosting_classifier,
    run_kernel_ridge_regression,
    run_knn_classifier,
    run_knn_regressor,
    run_logistic_classifier,
    run_mlp_classifier,
    run_mlp_regressor,
    run_random_forest_classifier,
    run_ridge_regression,
    run_svr_regression,
    run_svm_classifier,
)


def test_run_logistic_classifier_smoke():
    result = run_logistic_classifier(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        dataset="circles",
    )

    assert result["model"] == "logistic_regression"
    assert result["dataset"] == "circles"
    assert math.isfinite(result["train_accuracy"])
    assert math.isfinite(result["test_accuracy"])
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_run_svm_classifier_smoke():
    result = run_svm_classifier(
        n_samples=24,
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


def test_run_additional_classical_classifiers_smoke():
    runners = [
        run_random_forest_classifier,
        run_gradient_boosting_classifier,
        run_knn_classifier,
        run_gaussian_process_classifier,
    ]

    for runner in runners:
        result = runner(
            n_samples=28,
            noise=0.1,
            test_size=0.25,
            seed=0,
            dataset="linear",
        )

        assert math.isfinite(result["train_accuracy"])
        assert math.isfinite(result["test_accuracy"])
        assert 0.0 <= result["train_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert result["timing"]["fit_seconds"] >= 0.0


def test_run_mlp_classifier_smoke():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        result = run_mlp_classifier(
            n_samples=24,
            noise=0.1,
            test_size=0.25,
            seed=0,
            hidden_layer_sizes=(4,),
            max_iter=50,
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
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        alpha=1.0,
        dataset="sine",
    )

    assert result["model"] == "ridge_regression"
    assert result["dataset"] == "sine"
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
            n_samples=24,
            noise=0.1,
            test_size=0.25,
            seed=0,
            hidden_layer_sizes=(8,),
            max_iter=50,
            dataset="polynomial",
        )

    assert result["model"] == "mlp_regressor"
    assert result["dataset"] == "polynomial"
    assert math.isfinite(result["train_mse"])
    assert math.isfinite(result["test_mse"])
    assert math.isfinite(result["train_mae"])
    assert math.isfinite(result["test_mae"])
    assert result["train_mse"] >= 0.0
    assert result["test_mse"] >= 0.0
    assert result["train_mae"] >= 0.0
    assert result["test_mae"] >= 0.0
    assert len(result["loss_curve"]) > 0


def test_run_additional_classical_regressors_smoke():
    runners = [
        run_kernel_ridge_regression,
        run_svr_regression,
        run_gaussian_process_regressor,
        run_knn_regressor,
    ]

    for runner in runners:
        result = runner(
            n_samples=28,
            noise=0.1,
            test_size=0.25,
            seed=0,
            dataset="sine",
        )

        assert math.isfinite(result["train_mse"])
        assert math.isfinite(result["test_mse"])
        assert result["train_mse"] >= 0.0
        assert result["test_mse"] >= 0.0
        assert result["timing"]["fit_seconds"] >= 0.0


def test_tuned_classical_baseline_records_search_metadata():
    result = run_logistic_classifier(
        n_samples=30,
        noise=0.1,
        test_size=0.25,
        seed=0,
        dataset="linear",
        tune=True,
        cv=2,
        param_grid={"C": [0.1, 1.0]},
    )

    assert result["tuning"]["enabled"] is True
    assert result["tuning"]["cv"] == 2
    assert "best_params" in result["tuning"]
