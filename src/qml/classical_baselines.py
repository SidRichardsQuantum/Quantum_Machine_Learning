"""
qml.classical_baselines
=======================

Classical baseline workflows for supervised learning experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, SVR

from qml._baseline_utils import (
    classification_data as _classification_data,
    classification_stem as _classification_stem,
    classifier_result as _classifier_result,
    decision_function_from_predict_proba as _decision_function_from_predict_proba,
    fit_estimator as _fit_estimator,
    regression_data as _regression_data,
    regression_stem as _regression_stem,
    regressor_result as _regressor_result,
)
from qml.data import make_classification_dataset, make_regression_dataset
from qml.io_utils import images_path, results_path, save_json
from qml.visualize import (
    plot_dataset_2d,
    plot_decision_boundary,
    plot_regression_predictions,
)


def run_logistic_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    max_iter: int = 1000,
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train a logistic regression baseline on a supported classification dataset.
    """
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    clf, tuning, fit_seconds = _fit_estimator(
        LogisticRegression(max_iter=max_iter, random_state=seed),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"C": [0.1, 1.0, 10.0]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="logistic_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "max_iter": max_iter,
            "coef": np.asarray(clf.coef_, dtype=float),
            "intercept": np.asarray(clf.intercept_, dtype=float),
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )

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
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train a classical SVM baseline on a supported classification dataset.
    """
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    svc = SVC(kernel=kernel, C=c, gamma=gamma, random_state=seed)
    if param_grid is None:
        tuned_grid = {
            "estimator__C": [0.1, 1.0, 10.0],
            "estimator__gamma": ["scale", "auto"],
            "estimator__kernel": [kernel],
        }
    else:
        tuned_grid = {
            key if key.startswith("estimator__") else f"estimator__{key}": value
            for key, value in param_grid.items()
        }

    clf, tuning, fit_seconds = _fit_estimator(
        CalibratedClassifierCV(svc, method="sigmoid", cv=cv, ensemble=False),
        x_train,
        y_train,
        tune=tune,
        param_grid=tuned_grid,
        cv=cv,
        scoring="accuracy",
    )
    fitted_svc = getattr(clf, "estimator", svc)
    result = _classifier_result(
        model_name="svm_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "kernel": getattr(fitted_svc, "kernel", kernel),
            "c": getattr(fitted_svc, "C", c),
            "gamma": getattr(fitted_svc, "gamma", gamma),
            "calibration": "sigmoid",
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )

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
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train an MLP classifier baseline on a supported classification dataset.
    """
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    clf, tuning, fit_seconds = _fit_estimator(
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=seed,
        ),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"hidden_layer_sizes": [hidden_layer_sizes, (8,), (16, 16)], "alpha": [1e-4, 1e-3]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="mlp_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "hidden_layer_sizes": list(clf.hidden_layer_sizes),
            "max_iter": max_iter,
            "loss_curve": list(clf.loss_curve_),
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )

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
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train a ridge regression baseline on a synthetic regression dataset.
    """
    data = make_regression_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    reg, tuning, fit_seconds = _fit_estimator(
        Ridge(alpha=alpha),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"alpha": [0.01, 0.1, 1.0, 10.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="ridge_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "alpha": float(reg.alpha),
            "coef": np.asarray(reg.coef_, dtype=float),
            "intercept": np.asarray(reg.intercept_, dtype=float),
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )

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
            result["y_test_pred"],
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
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train an MLP regressor baseline on a synthetic regression dataset.
    """
    data = make_regression_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    reg, tuning, fit_seconds = _fit_estimator(
        MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=seed,
        ),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"hidden_layer_sizes": [hidden_layer_sizes, (16,), (32, 32)], "alpha": [1e-4, 1e-3]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="mlp_regressor",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "hidden_layer_sizes": list(reg.hidden_layer_sizes),
            "max_iter": max_iter,
            "loss_curve": list(reg.loss_curve_),
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )

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
            result["y_test_pred"],
            title="MLP regressor test predictions",
            show=plot,
            save_path=(
                images_path("regression_baselines", f"{stem}_predictions.png") if save else None
            ),
        )

    if save:
        save_json(result, results_path("regression_baselines", f"{stem}.json"))

    return result


def run_random_forest_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_estimators: int = 100,
    max_depth: int | None = None,
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a random forest classifier baseline."""
    x_train, x_test, y_train, y_test = _classification_data(
        dataset, n_samples, noise, test_size, seed
    )
    clf, tuning, fit_seconds = _fit_estimator(
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        ),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"n_estimators": [50, 100], "max_depth": [None, 3, 6]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="random_forest_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"n_estimators": clf.n_estimators, "max_depth": clf.max_depth},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _classification_stem("random_forest_classifier", n_samples, noise, seed)
        save_json(result, results_path("classification_baselines", f"{stem}.json"))
    return result


def run_gradient_boosting_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a gradient boosting classifier baseline."""
    x_train, x_test, y_train, y_test = _classification_data(
        dataset, n_samples, noise, test_size, seed
    )
    clf, tuning, fit_seconds = _fit_estimator(
        GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
        ),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="gradient_boosting_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "n_estimators": clf.n_estimators,
            "learning_rate": clf.learning_rate,
            "max_depth": clf.max_depth,
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _classification_stem("gradient_boosting_classifier", n_samples, noise, seed)
        save_json(result, results_path("classification_baselines", f"{stem}.json"))
    return result


def run_knn_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_neighbors: int = 5,
    weights: str = "uniform",
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a k-nearest-neighbors classifier baseline."""
    x_train, x_test, y_train, y_test = _classification_data(
        dataset, n_samples, noise, test_size, seed
    )
    clf, tuning, fit_seconds = _fit_estimator(
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="knn_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"n_neighbors": clf.n_neighbors, "weights": clf.weights},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _classification_stem("knn_classifier", n_samples, noise, seed)
        save_json(result, results_path("classification_baselines", f"{stem}.json"))
    return result


def run_kernel_ridge_regression(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    alpha: float = 1.0,
    kernel: str = "rbf",
    gamma: float | None = None,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a kernel ridge regression baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"alpha": [0.01, 0.1, 1.0, 10.0], "kernel": [kernel], "gamma": [None, 0.1, 1.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="kernel_ridge_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"alpha": reg.alpha, "kernel": reg.kernel, "gamma": reg.gamma},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("kernel_ridge_regression", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_svr_regression(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    kernel: str = "rbf",
    c: float = 1.0,
    gamma: str | float = "scale",
    epsilon: float = 0.1,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a support-vector regression baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        SVR(kernel=kernel, C=c, gamma=gamma, epsilon=epsilon),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="svr_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"kernel": reg.kernel, "c": reg.C, "gamma": reg.gamma, "epsilon": reg.epsilon},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("svr_regression", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_gaussian_process_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a Gaussian-process regression baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    base_kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=1e-3)
    reg, tuning, fit_seconds = _fit_estimator(
        GaussianProcessRegressor(kernel=base_kernel, random_state=seed, normalize_y=True),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"kernel": [RBF(0.5) + WhiteKernel(1e-3), RBF(1.0) + WhiteKernel(1e-3)]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="gaussian_process_regressor",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"kernel": str(reg.kernel)},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("gaussian_process_regressor", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_random_forest_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_estimators: int = 100,
    max_depth: int | None = None,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a random forest regressor baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"n_estimators": [50, 100], "max_depth": [None, 3, 6]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="random_forest_regressor",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"n_estimators": reg.n_estimators, "max_depth": reg.max_depth},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("random_forest_regressor", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_gradient_boosting_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a gradient boosting regressor baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
        ),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid
        or {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="gradient_boosting_regressor",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "n_estimators": reg.n_estimators,
            "learning_rate": reg.learning_rate,
            "max_depth": reg.max_depth,
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("gradient_boosting_regressor", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_knn_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    n_neighbors: int = 5,
    weights: str = "uniform",
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a k-nearest-neighbors regressor baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="knn_regressor",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"n_neighbors": reg.n_neighbors, "weights": reg.weights},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("knn_regressor", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_lasso_regression(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    alpha: float = 0.01,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a Lasso regression baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        Lasso(alpha=alpha, random_state=seed, max_iter=5000),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"alpha": [0.001, 0.01, 0.1, 1.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="lasso_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"alpha": reg.alpha, "coef": np.asarray(reg.coef_, dtype=float)},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("lasso_regression", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_elasticnet_regression(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    dataset: str = "linear",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train an ElasticNet regression baseline."""
    del plot
    x_train, x_test, y_train, y_test = _regression_data(dataset, n_samples, noise, test_size, seed)
    reg, tuning, fit_seconds = _fit_estimator(
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=5000),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    result = _regressor_result(
        model_name="elasticnet_regression",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=reg,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={
            "alpha": reg.alpha,
            "l1_ratio": reg.l1_ratio,
            "coef": np.asarray(reg.coef_, dtype=float),
        },
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _regression_stem("elasticnet_regression", n_samples, noise, seed)
        save_json(result, results_path("regression_baselines", f"{stem}.json"))
    return result


def run_gaussian_process_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    dataset: str = "moons",
    tune: bool = False,
    cv: int = 3,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Train a Gaussian-process classifier baseline."""
    x_train, x_test, y_train, y_test = _classification_data(
        dataset, n_samples, noise, test_size, seed
    )
    clf, tuning, fit_seconds = _fit_estimator(
        GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(1.0), random_state=seed),
        x_train,
        y_train,
        tune=tune,
        param_grid=param_grid or {"kernel": [RBF(0.5), RBF(1.0), RBF(2.0)]},
        cv=cv,
        scoring="accuracy",
    )
    result = _classifier_result(
        model_name="gaussian_process_classifier",
        dataset=dataset,
        seed=seed,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        estimator=clf,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        extra={"kernel": str(clf.kernel)},
        tuning=tuning,
        fit_seconds=fit_seconds,
    )
    if save:
        stem = _classification_stem("gaussian_process_classifier", n_samples, noise, seed)
        save_json(result, results_path("classification_baselines", f"{stem}.json"))
    return result
