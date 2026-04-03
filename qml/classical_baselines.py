"""
qml.classical_baselines
=======================

Classical baseline workflows for supervised learning experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC

from qml.data import make_moons_dataset, make_regression_dataset
from qml.io_utils import images_path, results_path, save_json
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.visualize import (
    plot_dataset_2d,
    plot_decision_boundary,
    plot_regression_predictions,
)


def _classification_stem(
    model_name: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> str:
    return f"{model_name}_samples{n_samples}_noise{str(noise).replace('.', 'p')}_seed{seed}"


def _regression_stem(
    model_name: str,
    n_samples: int,
    noise: float,
    seed: int,
) -> str:
    return f"{model_name}_samples{n_samples}_noise{str(noise).replace('.', 'p')}_seed{seed}"


def _decision_function_from_predict_proba(model):
    def predict_proba_grid(x_grid):
        x_grid = np.asarray(x_grid, dtype=float)
        return model.predict_proba(x_grid)[:, 1]

    return predict_proba_grid


def _decision_function_from_predict(model):
    def predict_grid(x_grid):
        x_grid = np.asarray(x_grid, dtype=float)
        return model.predict(x_grid)

    return predict_grid


def run_logistic_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """
    Train a logistic regression baseline on the two-moons dataset.
    """
    dataset = make_moons_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(x_train, y_train)

    train_probs = clf.predict_proba(x_train)[:, 1]
    test_probs = clf.predict_proba(x_test)[:, 1]

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    result = {
        "model": "logistic_regression",
        "dataset": "moons",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "max_iter": max_iter,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "coef": np.asarray(clf.coef_, dtype=float),
        "intercept": np.asarray(clf.intercept_, dtype=float),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
        "train_probabilities": np.asarray(train_probs, dtype=float),
        "test_probabilities": np.asarray(test_probs, dtype=float),
    }

    stem = _classification_stem("logistic_regression", n_samples, noise, seed)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="Logistic regression training dataset",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_dataset.png") if save else None
            ),
        )

        plot_decision_boundary(
            _decision_function_from_predict_proba(clf),
            x_train,
            y_train,
            title="Logistic regression decision boundary",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_decision_boundary.png")
                if save
                else None
            ),
        )

    if save:
        save_json(result, results_path("classification_baselines", f"{stem}.json"))

    return result


def run_svm_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    kernel: str = "rbf",
    c: float = 1.0,
    gamma: str | float = "scale",
) -> dict[str, Any]:
    """
    Train a classical SVM baseline on the two-moons dataset.
    """
    dataset = make_moons_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    clf = SVC(kernel=kernel, C=c, gamma=gamma, probability=True, random_state=seed)
    clf.fit(x_train, y_train)

    train_probs = clf.predict_proba(x_train)[:, 1]
    test_probs = clf.predict_proba(x_test)[:, 1]

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    result = {
        "model": "svm_classifier",
        "dataset": "moons",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "kernel": kernel,
        "c": c,
        "gamma": gamma,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
        "train_probabilities": np.asarray(train_probs, dtype=float),
        "test_probabilities": np.asarray(test_probs, dtype=float),
    }

    stem = _classification_stem("svm_classifier", n_samples, noise, seed)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="SVM training dataset",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_dataset.png") if save else None
            ),
        )

        plot_decision_boundary(
            _decision_function_from_predict_proba(clf),
            x_train,
            y_train,
            title="SVM decision boundary",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_decision_boundary.png")
                if save
                else None
            ),
        )

    if save:
        save_json(result, results_path("classification_baselines", f"{stem}.json"))

    return result


def run_mlp_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    hidden_layer_sizes: tuple[int, ...] = (16, 16),
    max_iter: int = 500,
) -> dict[str, Any]:
    """
    Train an MLP classifier baseline on the two-moons dataset.
    """
    dataset = make_moons_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    train_probs = clf.predict_proba(x_train)[:, 1]
    test_probs = clf.predict_proba(x_test)[:, 1]

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    result = {
        "model": "mlp_classifier",
        "dataset": "moons",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "max_iter": max_iter,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "loss_curve": list(clf.loss_curve_),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
        "train_probabilities": np.asarray(train_probs, dtype=float),
        "test_probabilities": np.asarray(test_probs, dtype=float),
    }

    stem = _classification_stem("mlp_classifier", n_samples, noise, seed)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="MLP classifier training dataset",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_dataset.png") if save else None
            ),
        )

        plot_decision_boundary(
            _decision_function_from_predict_proba(clf),
            x_train,
            y_train,
            title="MLP classifier decision boundary",
            show=plot,
            save_path=(
                images_path("classification_baselines", f"{stem}_decision_boundary.png")
                if save
                else None
            ),
        )

    if save:
        save_json(result, results_path("classification_baselines", f"{stem}.json"))

    return result


def run_ridge_regression(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """
    Train a ridge regression baseline on a synthetic regression dataset.
    """
    dataset = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    reg = Ridge(alpha=alpha)
    reg.fit(x_train, y_train)

    y_train_pred = reg.predict(x_train)
    y_test_pred = reg.predict(x_test)

    result = {
        "model": "ridge_regression",
        "dataset": "regression",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "alpha": alpha,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "coef": np.asarray(reg.coef_, dtype=float),
        "intercept": np.asarray(reg.intercept_, dtype=float),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
        "y_train_pred": np.asarray(y_train_pred, dtype=float),
        "y_test_pred": np.asarray(y_test_pred, dtype=float),
    }

    stem = _regression_stem("ridge_regression", n_samples, noise, seed)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="Ridge regression training dataset",
            show=plot,
            save_path=images_path("regression_baselines", f"{stem}_dataset.png") if save else None,
        )

        plot_regression_predictions(
            y_test,
            y_test_pred,
            title="Ridge regression test predictions",
            show=plot,
            save_path=(
                images_path("regression_baselines", f"{stem}_predictions.png") if save else None
            ),
        )

    if save:
        save_json(result, results_path("regression_baselines", f"{stem}.json"))

    return result


def run_mlp_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    hidden_layer_sizes: tuple[int, ...] = (32, 32),
    max_iter: int = 500,
) -> dict[str, Any]:
    """
    Train an MLP regressor baseline on a synthetic regression dataset.
    """
    dataset = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=seed,
    )
    reg.fit(x_train, y_train)

    y_train_pred = reg.predict(x_train)
    y_test_pred = reg.predict(x_test)

    result = {
        "model": "mlp_regressor",
        "dataset": "regression",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "max_iter": max_iter,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "loss_curve": list(reg.loss_curve_),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
        "y_train_pred": np.asarray(y_train_pred, dtype=float),
        "y_test_pred": np.asarray(y_test_pred, dtype=float),
    }

    stem = _regression_stem("mlp_regressor", n_samples, noise, seed)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="MLP regressor training dataset",
            show=plot,
            save_path=images_path("regression_baselines", f"{stem}_dataset.png") if save else None,
        )

        plot_regression_predictions(
            y_test,
            y_test_pred,
            title="MLP regressor test predictions",
            show=plot,
            save_path=(
                images_path("regression_baselines", f"{stem}_predictions.png") if save else None
            ),
        )

    if save:
        save_json(result, results_path("regression_baselines", f"{stem}.json"))

    return result
