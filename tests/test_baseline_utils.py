import math

import numpy as np

from qml._baseline_utils import (
    classification_stem,
    classifier_result,
    decision_function_from_predict,
    decision_function_from_predict_proba,
    regression_stem,
    regressor_result,
)


class BinaryEstimator:
    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return (x[:, 0] > 0.5).astype(int)

    def predict_proba(self, x):
        predictions = self.predict(x)
        return np.column_stack([1 - predictions, predictions])


class MeanRegressor:
    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return x[:, 0]


def test_baseline_stems_encode_noise_without_decimal_points() -> None:
    assert classification_stem("model", 24, 0.125, 7) == "model_samples24_noise0p125_seed7"
    assert regression_stem("reg", 32, 1.0, 2) == "reg_samples32_noise1p0_seed2"


def test_classifier_result_records_predictions_probabilities_and_timing() -> None:
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    x_test = np.array([[0.25, 0.0], [0.75, 1.0]])
    y_train = np.array([0, 1])
    y_test = np.array([0, 1])

    result = classifier_result(
        model_name="binary",
        dataset="toy",
        seed=3,
        n_samples=4,
        noise=0.0,
        test_size=0.5,
        estimator=BinaryEstimator(),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        fit_seconds=0.25,
    )

    assert result["train_accuracy"] == 1.0
    assert result["test_accuracy"] == 1.0
    assert result["tuning"] == {"enabled": False}
    assert result["timing"]["fit_seconds"] == 0.25
    assert result["timing"]["total_seconds"] >= result["timing"]["fit_seconds"]
    assert result["test_probabilities"].tolist() == [0.0, 1.0]


def test_regressor_result_records_error_metrics_and_timing() -> None:
    x_train = np.array([[0.0], [1.0]])
    x_test = np.array([[2.0], [3.0]])
    y_train = np.array([0.0, 1.5])
    y_test = np.array([2.5, 3.0])

    result = regressor_result(
        model_name="mean",
        dataset="toy",
        seed=5,
        n_samples=4,
        noise=0.0,
        test_size=0.5,
        estimator=MeanRegressor(),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        fit_seconds=0.1,
    )

    assert math.isclose(result["train_mse"], 0.125)
    assert math.isclose(result["test_mse"], 0.125)
    assert math.isclose(result["train_mae"], 0.25)
    assert math.isclose(result["test_mae"], 0.25)
    assert result["timing"]["total_seconds"] >= 0.1


def test_plot_decision_function_adapters_normalize_inputs() -> None:
    estimator = BinaryEstimator()

    assert decision_function_from_predict(estimator)([[0.0], [1.0]]).tolist() == [0, 1]
    assert decision_function_from_predict_proba(estimator)([[0.0], [1.0]]).tolist() == [
        0.0,
        1.0,
    ]
