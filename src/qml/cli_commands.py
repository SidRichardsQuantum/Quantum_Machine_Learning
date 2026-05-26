"""Command handlers for the qml command-line interface."""

from __future__ import annotations

import argparse


def _noise_model_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    from qml.noise import build_noise_model

    return build_noise_model(
        depolarizing=args.depolarizing,
        amplitude_damping=args.amplitude_damping,
        readout_error=args.readout_error,
    )


def _run_classification_benchmark_command(args: argparse.Namespace) -> int:
    from qml.benchmarks import compare_classification_models

    noise_model = _noise_model_from_args(args)
    result = compare_classification_models(
        models=args.models,
        seeds=args.seeds,
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        model_kwargs={
            "vqc": {"noise_model": noise_model},
            "qcnn": {"noise_model": noise_model},
            "quantum_kernel": {"noise_model": noise_model},
            "trainable_quantum_kernel": {"noise_model": noise_model},
            "quantum_reservoir": {"noise_model": noise_model},
        },
        save=args.save,
        tune_classical=args.tune_classical,
        cv=args.cv,
    )

    print("Benchmark type:", result["benchmark_type"])
    print("Models:", ", ".join(result["models"]))

    for model, metrics in result["summary"].items():
        train = metrics["train_accuracy"]
        test = metrics["test_accuracy"]

        print()
        print(model)
        print(f"  train mean: {train['mean']:.6f}")
        print(f"  train std : {train['std']:.6f}")
        print(f"  test mean : {test['mean']:.6f}")
        print(f"  test std  : {test['std']:.6f}")

    return 0


def _run_regression_benchmark_command(args: argparse.Namespace) -> int:
    from qml.benchmarks import compare_regression_models

    noise_model = _noise_model_from_args(args)
    result = compare_regression_models(
        models=args.models,
        seeds=args.seeds,
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        model_kwargs={
            "vqr": {"noise_model": noise_model},
            "quantum_kernel_regressor": {"noise_model": noise_model},
            "quantum_gaussian_process_regressor": {"noise_model": noise_model},
            "trainable_quantum_kernel_regressor": {"noise_model": noise_model},
            "quantum_reservoir_regressor": {"noise_model": noise_model},
        },
        save=args.save,
        tune_classical=args.tune_classical,
        cv=args.cv,
    )

    print("Benchmark type:", result["benchmark_type"])
    print("Models:", ", ".join(result["models"]))

    for model, metrics in result["summary"].items():
        train_mse = metrics["train_mse"]
        test_mse = metrics["test_mse"]

        print()
        print(model)
        print(f"  train MSE mean: {train_mse['mean']:.6f}")
        print(f"  train MSE std : {train_mse['std']:.6f}")
        print(f"  test MSE mean : {test_mse['mean']:.6f}")
        print(f"  test MSE std  : {test_mse['std']:.6f}")

    return 0


def _run_finite_shot_benchmark_command(args: argparse.Namespace) -> int:
    from qml.benchmarks import compare_classification_models, compare_regression_models
    from qml.io_utils import results_path, save_json

    print("Benchmark type: finite-shots")
    print(
        "Shot values:",
        ", ".join("analytic" if shots is None else str(shots) for shots in args.shots),
    )
    noise_model = _noise_model_from_args(args)
    results = []

    for shots in args.shots:
        label = "analytic" if shots is None else str(shots)
        classification = compare_classification_models(
            models=args.classification_models,
            seeds=args.seeds,
            n_samples=args.samples,
            noise=args.noise,
            test_size=args.test_size,
            dataset=args.classification_dataset,
            model_kwargs={
                "vqc": {
                    "n_layers": 1,
                    "steps": args.steps,
                    "shots": shots,
                    "noise_model": noise_model,
                },
                "qcnn": {"steps": args.steps, "shots": shots, "noise_model": noise_model},
                "quantum_kernel": {"shots": shots, "noise_model": noise_model},
                "trainable_quantum_kernel": {
                    "steps": max(1, args.steps // 2),
                    "embedding_layers": 1,
                    "shots_train": shots,
                    "shots_kernel": shots,
                    "noise_model": noise_model,
                },
                "quantum_reservoir": {"n_layers": 1, "shots": shots, "noise_model": noise_model},
            },
            save=False,
            tune_classical=args.tune_classical,
            cv=args.cv,
        )
        regression = compare_regression_models(
            models=args.regression_models,
            seeds=args.seeds,
            n_samples=args.samples,
            noise=args.noise,
            test_size=args.test_size,
            dataset=args.regression_dataset,
            model_kwargs={
                "vqr": {
                    "n_layers": 1,
                    "steps": args.steps,
                    "shots": shots,
                    "noise_model": noise_model,
                },
                "quantum_kernel_regressor": {"shots": shots, "noise_model": noise_model},
                "quantum_gaussian_process_regressor": {"shots": shots, "noise_model": noise_model},
                "quantum_reservoir_regressor": {
                    "n_layers": 1,
                    "shots": shots,
                    "noise_model": noise_model,
                },
                "trainable_quantum_kernel_regressor": {
                    "steps": max(1, args.steps // 2),
                    "embedding_layers": 1,
                    "shots_train": shots,
                    "shots_kernel": shots,
                    "noise_model": noise_model,
                },
            },
            save=False,
            tune_classical=args.tune_classical,
            cv=args.cv,
        )
        results.append(
            {
                "shots": label,
                "classification": classification,
                "regression": regression,
            }
        )

        print()
        print(f"Shots: {label}")
        print("  Classification")
        for model, metrics in classification["summary"].items():
            test = metrics["test_accuracy"]
            runtime = metrics["runtime_seconds"]
            print(
                f"    {model}: test={test['mean']:.6f} "
                f"ci95=[{test['ci95_low']:.6f}, {test['ci95_high']:.6f}] "
                f"runtime={runtime['mean']:.3f}s"
            )

        print("  Regression")
        for model, metrics in regression["summary"].items():
            test = metrics["test_mse"]
            runtime = metrics["runtime_seconds"]
            print(
                f"    {model}: test_mse={test['mean']:.6f} "
                f"ci95=[{test['ci95_low']:.6f}, {test['ci95_high']:.6f}] "
                f"runtime={runtime['mean']:.3f}s"
            )

    if args.save:
        save_json(
            {"benchmark_type": "finite-shots", "runs": results},
            results_path("benchmarks", "finite_shot_benchmark.json"),
        )

    return 0


def _run_vqc_command(args: argparse.Namespace) -> int:
    """
    Run the VQC workflow from parsed CLI arguments.
    """
    from qml.classifiers import run_vqc

    noise_model = _noise_model_from_args(args)
    result = run_vqc(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        n_layers=args.layers,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
        shots=args.shots,
        optimizer=args.optimizer,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        noise_model=noise_model,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_regression_command(args: argparse.Namespace) -> int:
    """
    Run the variational regression workflow from parsed CLI arguments.
    """
    from qml.regression import run_vqr

    noise_model = _noise_model_from_args(args)
    result = run_vqr(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        n_layers=args.layers,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
        shots=args.shots,
        optimizer=args.optimizer,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        noise_model=noise_model,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train MSE: {result['train_mse']:.6f}")
    print(f"Test MSE: {result['test_mse']:.6f}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_qcnn_command(args: argparse.Namespace) -> int:
    """
    Run the QCNN workflow from parsed CLI arguments.
    """
    from qml.qcnn import run_qcnn

    noise_model = _noise_model_from_args(args)
    result = run_qcnn(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
        shots=args.shots,
        optimizer=args.optimizer,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        noise_model=noise_model,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_trainable_kernel_command(args: argparse.Namespace) -> int:
    """
    Run the trainable quantum kernel workflow from parsed CLI arguments.
    """
    from qml.trainable_kernels import run_trainable_quantum_kernel_classifier

    result = run_trainable_quantum_kernel_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        embedding=args.embedding,
        embedding_layers=args.embedding_layers,
        steps=args.steps,
        step_size=args.step_size,
        reg_strength=args.reg_strength,
        svc_c=args.svc_c,
        plot=args.plot,
        save=args.save,
        shots_train=args.shots_train,
        shots_kernel=args.shots_kernel,
        optimizer=args.optimizer,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Embedding: {result['embedding']}")
    print(f"Embedding layers: {result['embedding_layers']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    print(f"Final alignment: {result['final_alignment']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_metric_learning_command(args: argparse.Namespace) -> int:
    """
    Run the quantum metric learning workflow from parsed CLI arguments.
    """
    from qml.metric_learning import run_quantum_metric_learner

    result = run_quantum_metric_learner(
        dataset=args.dataset,
        samples=args.samples,
        test_size=args.test_size,
        seed=args.seed,
        layers=args.layers,
        steps=args.steps,
        stepsize=args.step_size,
        margin=args.margin,
        pairs_per_step=args.pairs_per_step,
        log_every=args.log_every,
        scale_data=not args.no_scale_data,
        plot=args.plot,
        save=args.save,
    )

    print("Model: quantum_metric_learning")
    print(f"Dataset: {args.dataset}")
    print(f"Train accuracy: {result.train_accuracy:.6f}")
    print(f"Test accuracy: {result.test_accuracy:.6f}")
    print(f"Final loss: {result.loss_history[-1]:.6f}")
    return 0


def _run_autoencoder_command(args: argparse.Namespace) -> int:
    """
    Run the quantum autoencoder workflow from parsed CLI arguments.
    """
    from qml.autoencoder import run_quantum_autoencoder

    result = run_quantum_autoencoder(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        n_layers=args.layers,
        latent_qubits=args.latent_qubits,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
        family=args.family,
        optimizer=args.optimizer,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    print(f"Model: {result['model']}")
    print(f"Family: {result['family']}")
    print(f"Train compression fidelity: {result['train_compression_fidelity']:.6f}")
    print(f"Test compression fidelity: {result['test_compression_fidelity']:.6f}")
    print(f"Train reconstruction fidelity: {result['train_reconstruction_fidelity']:.6f}")
    print(f"Test reconstruction fidelity: {result['test_reconstruction_fidelity']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_kernel_command(args: argparse.Namespace) -> int:
    """
    Run the quantum kernel workflow from parsed CLI arguments.
    """
    from qml.kernel_methods import run_quantum_kernel_classifier

    noise_model = _noise_model_from_args(args)
    result = run_quantum_kernel_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        plot=args.plot,
        shots=args.shots,
        save=args.save,
        noise_model=noise_model,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    return 0


def _run_logistic_command(args: argparse.Namespace) -> int:
    """
    Run the logistic regression baseline from parsed CLI arguments.
    """
    from qml.classical_baselines import run_logistic_classifier

    result = run_logistic_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        plot=args.plot,
        save=args.save,
        max_iter=args.max_iter,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    return 0


def _run_svm_command(args: argparse.Namespace) -> int:
    """
    Run the SVM classifier baseline from parsed CLI arguments.
    """
    from qml.classical_baselines import run_svm_classifier

    try:
        gamma: str | float = float(args.gamma)
    except ValueError:
        gamma = args.gamma

    result = run_svm_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        plot=args.plot,
        save=args.save,
        kernel=args.kernel_name,
        c=args.c,
        gamma=gamma,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    return 0


def _run_mlp_classifier_command(args: argparse.Namespace) -> int:
    """
    Run the MLP classifier baseline from parsed CLI arguments.
    """
    from qml.classical_baselines import run_mlp_classifier

    result = run_mlp_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        plot=args.plot,
        save=args.save,
        hidden_layer_sizes=tuple(args.hidden_sizes),
        max_iter=args.max_iter,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train accuracy: {result['train_accuracy']:.6f}")
    print(f"Test accuracy: {result['test_accuracy']:.6f}")
    return 0


def _run_ridge_command(args: argparse.Namespace) -> int:
    """
    Run the ridge regression baseline from parsed CLI arguments.
    """
    from qml.classical_baselines import run_ridge_regression

    result = run_ridge_regression(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        plot=args.plot,
        save=args.save,
        alpha=args.alpha,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train MSE: {result['train_mse']:.6f}")
    print(f"Test MSE: {result['test_mse']:.6f}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
    return 0


def _run_mlp_regressor_command(args: argparse.Namespace) -> int:
    """
    Run the MLP regressor baseline from parsed CLI arguments.
    """
    from qml.classical_baselines import run_mlp_regressor

    result = run_mlp_regressor(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        plot=args.plot,
        save=args.save,
        hidden_layer_sizes=tuple(args.hidden_sizes),
        max_iter=args.max_iter,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train MSE: {result['train_mse']:.6f}")
    print(f"Test MSE: {result['test_mse']:.6f}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
    return 0
