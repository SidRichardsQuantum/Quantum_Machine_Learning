from qml.benchmarks import (
    compare_classification_models,
    compare_regression_models,
)


def test_classification_benchmark_runs():
    result = compare_classification_models(
        models=["vqc", "qcnn", "logistic_regression"],
        seeds=[0],
        n_samples=20,
        model_kwargs={
            "vqc": {
                "n_layers": 1,
                "steps": 2,
            },
            "qcnn": {
                "steps": 2,
            },
        },
        save=False,
        dataset="circles",
    )
    assert "summary" in result
    assert result["dataset"] == "circles"
    assert {run["dataset"] for run in result["runs"]} == {"circles"}
    assert result["best_model"]["metric"] == "test_accuracy"
    assert result["best_model"]["higher_is_better"] is True
    assert "paired_vs_best_classical" in result
    assert "metadata" in result

    for run in result["runs"]:
        assert "generalization_gap" in run
        assert "timing" in run
        assert run["runtime_seconds"] >= 0.0

    for model_summary in result["summary"].values():
        assert "generalization_gap" in model_summary
        assert "runtime_seconds" in model_summary
        assert "ci95_low" in model_summary["test_accuracy"]


def test_regression_benchmark_runs():
    result = compare_regression_models(
        models=["vqr", "ridge_regression"],
        seeds=[0],
        n_samples=20,
        model_kwargs={
            "vqr": {
                "n_layers": 1,
                "steps": 2,
            },
        },
        save=False,
        dataset="sine",
    )
    assert "summary" in result
    assert result["dataset"] == "sine"
    assert {run["dataset"] for run in result["runs"]} == {"sine"}
    assert result["best_model"]["metric"] == "test_mse"
    assert result["best_model"]["higher_is_better"] is False
    assert "paired_vs_best_classical" in result
    assert "metadata" in result

    for run in result["runs"]:
        assert "generalization_gap" in run
        assert "timing" in run
        assert run["runtime_seconds"] >= 0.0

    for model_summary in result["summary"].values():
        assert "generalization_gap" in model_summary
        assert "runtime_seconds" in model_summary
        assert "ci95_low" in model_summary["test_mse"]


def test_benchmark_runs_new_classical_models_and_tuning():
    result = compare_classification_models(
        models=["random_forest_classifier", "knn_classifier"],
        seeds=[0],
        n_samples=28,
        dataset="linear",
        tune_classical=True,
        cv=2,
        model_kwargs={
            "random_forest_classifier": {
                "param_grid": {"n_estimators": [10], "max_depth": [None]},
            },
            "knn_classifier": {
                "param_grid": {"n_neighbors": [1, 3], "weights": ["uniform"]},
            },
        },
    )

    assert result["tune_classical"] is True
    assert result["paired_vs_best_classical"]["reference_model"] in result["models"]
    assert all(run["tuning"]["enabled"] is True for run in result["runs"])


def test_regression_benchmark_runs_new_classical_models_and_dataset():
    result = compare_regression_models(
        models=["kernel_ridge_regression", "svr_regression"],
        seeds=[0],
        n_samples=30,
        dataset="friedman",
        tune_classical=True,
        cv=2,
        model_kwargs={
            "kernel_ridge_regression": {
                "param_grid": {"alpha": [0.1], "kernel": ["rbf"], "gamma": [0.1]},
            },
            "svr_regression": {
                "param_grid": {"C": [1.0], "gamma": ["scale"], "epsilon": [0.1]},
            },
        },
    )

    assert result["dataset"] == "friedman"
    assert result["tune_classical"] is True
    assert all(run["tuning"]["enabled"] is True for run in result["runs"])


def test_classification_benchmark_runs_quantum_reservoir():
    result = compare_classification_models(
        models=["quantum_reservoir", "logistic_regression"],
        seeds=[0],
        n_samples=24,
        dataset="moons",
        model_kwargs={"quantum_reservoir": {"n_layers": 1}},
    )

    assert result["models"] == ["quantum_reservoir", "logistic_regression"]
    assert result["paired_vs_best_classical"]["reference_model"] == "logistic_regression"
    reservoir_run = next(run for run in result["runs"] if run["model"] == "quantum_reservoir")
    assert reservoir_run["timing"]["fit_seconds"] >= 0.0
    assert reservoir_run["timing"]["predict_seconds"] >= 0.0


def test_regression_benchmark_runs_new_quantum_regressors():
    result = compare_regression_models(
        models=[
            "quantum_kernel_regressor",
            "quantum_gaussian_process_regressor",
            "quantum_reservoir_regressor",
            "ridge_regression",
        ],
        seeds=[0],
        n_samples=18,
        dataset="sine",
        model_kwargs={
            "quantum_reservoir_regressor": {"n_layers": 1},
        },
    )

    assert result["models"] == [
        "quantum_kernel_regressor",
        "quantum_gaussian_process_regressor",
        "quantum_reservoir_regressor",
        "ridge_regression",
    ]
    assert result["paired_vs_best_classical"]["reference_model"] == "ridge_regression"
    for run in result["runs"]:
        assert "train_mse" in run
        assert "test_mse" in run
        assert "runtime_seconds" in run
        assert "total_seconds" in run["timing"]


def test_regression_benchmark_runs_trainable_quantum_kernel_regressor():
    result = compare_regression_models(
        models=["trainable_quantum_kernel_regressor", "ridge_regression"],
        seeds=[0],
        n_samples=12,
        dataset="sine",
        model_kwargs={
            "trainable_quantum_kernel_regressor": {
                "embedding": "angle",
                "embedding_layers": 1,
                "steps": 0,
            },
        },
    )

    run = next(
        run for run in result["runs"] if run["model"] == "trainable_quantum_kernel_regressor"
    )
    assert "final_loss" in run
    assert "train_mse" in run
    assert "test_mse" in run
