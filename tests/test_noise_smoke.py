from qml.benchmarks import compare_classification_models, compare_regression_models


def test_classification_benchmark_accepts_noise_aware_model_kwargs():
    result = compare_classification_models(
        models=["vqc", "kernel", "trainable_quantum_kernel"],
        seeds=[7],
        n_samples=24,
        noise=0.1,
        model_kwargs={
            "vqc": {
                "n_layers": 1,
                "steps": 2,
                "shots": 16,
            },
            "kernel": {
                "shots": 16,
            },
            "trainable_quantum_kernel": {
                "embedding": "angle",
                "steps": 0,
                "shots_train": 16,
                "shots_kernel": 16,
            },
        },
        save=False,
    )

    assert result["benchmark_type"] == "classification"
    assert result["models"] == [
        "vqc",
        "quantum_kernel",
        "trainable_quantum_kernel",
    ]

    assert "vqc" in result["summary"]
    assert "quantum_kernel" in result["summary"]
    assert "trainable_quantum_kernel" in result["summary"]

    assert len(result["runs"]) == 3

    for run in result["runs"]:
        assert "model" in run
        assert "seed" in run
        assert "train_accuracy" in run
        assert "test_accuracy" in run


def test_classification_benchmark_finite_shot_runs_are_deterministic_for_fixed_seed():
    kwargs = {
        "models": ["vqc", "kernel"],
        "seeds": [11],
        "n_samples": 24,
        "noise": 0.1,
        "model_kwargs": {
            "vqc": {
                "n_layers": 1,
                "steps": 2,
                "shots": 16,
            },
            "kernel": {
                "shots": 16,
            },
        },
        "save": False,
    }

    result_1 = compare_classification_models(**kwargs)
    result_2 = compare_classification_models(**kwargs)

    assert result_1["runs"] == result_2["runs"]
    assert result_1["summary"] == result_2["summary"]


def test_classification_benchmark_noise_aware_runs_for_fixed_seed():
    kwargs = {
        "models": ["vqc", "kernel"],
        "seeds": [11],
        "n_samples": 24,
        "noise": 0.1,
        "model_kwargs": {
            "vqc": {
                "n_layers": 1,
                "steps": 2,
                "shots": 16,
            },
            "kernel": {
                "shots": 16,
            },
        },
        "save": False,
    }

    result = compare_classification_models(**kwargs)

    assert result["benchmark_type"] == "classification"
    assert result["models"] == ["vqc", "quantum_kernel"]
    assert len(result["runs"]) == 2

    for run in result["runs"]:
        assert "model" in run
        assert "seed" in run
        assert "train_accuracy" in run
        assert "test_accuracy" in run


def test_regression_benchmark_accepts_noise_aware_model_kwargs():
    result = compare_regression_models(
        models=["vqr"],
        seeds=[7],
        n_samples=24,
        noise=0.1,
        model_kwargs={
            "vqr": {
                "n_layers": 1,
                "steps": 2,
                "shots": 16,
            },
        },
        save=False,
    )

    assert result["benchmark_type"] == "regression"
    assert result["models"] == ["vqr"]
    assert "vqr" in result["summary"]
    assert len(result["runs"]) == 1

    run = result["runs"][0]
    assert run["model"] == "vqr"
    assert run["seed"] == 7
    assert "train_mse" in run
    assert "test_mse" in run
    assert "train_mae" in run
    assert "test_mae" in run


def test_regression_benchmark_finite_shot_runs_are_deterministic_for_fixed_seed():
    kwargs = {
        "models": ["vqr"],
        "seeds": [11],
        "n_samples": 24,
        "noise": 0.1,
        "model_kwargs": {
            "vqr": {
                "n_layers": 1,
                "steps": 2,
                "shots": 16,
            },
        },
        "save": False,
    }

    result_1 = compare_regression_models(**kwargs)
    result_2 = compare_regression_models(**kwargs)

    assert result_1["runs"] == result_2["runs"]
    assert result_1["summary"] == result_2["summary"]
