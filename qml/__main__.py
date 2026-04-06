"""
qml.__main__

Command-line entrypoint for the qml package.
"""

from __future__ import annotations

import argparse

from qml.benchmarks import (
    compare_classification_models,
    compare_regression_models,
)
from qml.classical_baselines import (
    run_logistic_classifier,
    run_mlp_classifier,
    run_mlp_regressor,
    run_ridge_regression,
    run_svm_classifier,
)
from qml.classifiers import run_vqc
from qml.kernel_methods import run_quantum_kernel_classifier
from qml.regression import run_vqr
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier


def _run_classification_benchmark_command(args: argparse.Namespace) -> int:
    result = compare_classification_models(
        models=args.models,
        seeds=args.seeds,
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        save=args.save,
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
    result = compare_regression_models(
        models=args.models,
        seeds=args.seeds,
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        save=args.save,
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


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=200, help="Number of samples.")
    parser.add_argument("--noise", type=float, default=0.1, help="Dataset noise level.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction reserved for test data.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--plot", action="store_true", help="Display plots.")
    parser.add_argument("--save", action="store_true", help="Save results and figures.")


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=200, help="Number of samples.")
    parser.add_argument("--noise", type=float, default=0.1, help="Dataset noise level.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction reserved for test data.",
    )
    parser.add_argument("--save", action="store_true", help="Save benchmark results.")


def _add_shots_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Number of measurement shots (None = analytic mode).",
    )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m qml",
        description="Run quantum and classical machine learning workflows.",
    )

    subparsers = parser.add_subparsers(dest="command")

    vqc_parser = subparsers.add_parser(
        "vqc",
        help="Run a variational quantum classifier.",
    )
    _add_common_dataset_args(vqc_parser)
    vqc_parser.add_argument("--layers", type=int, default=2, help="Number of ansatz layers.")
    vqc_parser.add_argument("--steps", type=int, default=50, help="Number of optimizer steps.")
    vqc_parser.add_argument("--step-size", type=float, default=0.1, help="Optimizer step size.")
    _add_shots_arg(vqc_parser)
    vqc_parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs", "xor"],
        help="Classification dataset.",
    )

    kernel_parser = subparsers.add_parser(
        "kernel",
        help="Run a quantum kernel classifier.",
    )
    _add_common_dataset_args(kernel_parser)
    _add_shots_arg(kernel_parser)
    kernel_parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs", "xor"],
        help="Classification dataset.",
    )

    trainable_kernel_parser = subparsers.add_parser(
        "trainable-kernel",
        help="Run a trainable quantum kernel classifier.",
    )
    _add_common_dataset_args(trainable_kernel_parser)
    trainable_kernel_parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs", "xor"],
        help="Classification dataset.",
    )
    trainable_kernel_parser.add_argument(
        "--embedding",
        type=str,
        default="data_reupload",
        choices=["angle", "data_reupload"],
        help="Embedding used inside the trainable kernel feature map.",
    )
    trainable_kernel_parser.add_argument(
        "--embedding-layers",
        type=int,
        default=2,
        help="Number of trainable embedding layers.",
    )
    trainable_kernel_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of optimizer steps.",
    )
    trainable_kernel_parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Optimizer step size.",
    )
    trainable_kernel_parser.add_argument(
        "--reg-strength",
        type=float,
        default=1e-4,
        help="L2 regularisation strength for trainable kernel parameters.",
    )
    trainable_kernel_parser.add_argument(
        "--svc-c",
        type=float,
        default=1.0,
        help="SVM regularisation parameter for the learned precomputed kernel.",
    )
    trainable_kernel_parser.add_argument(
        "--shots-train",
        type=int,
        default=None,
        help="Shots used during kernel training (alignment optimisation).",
    )
    trainable_kernel_parser.add_argument(
        "--shots-kernel",
        type=int,
        default=None,
        help="Shots used when evaluating final kernel matrices.",
    )

    regression_parser = subparsers.add_parser(
        "regression",
        help="Run a variational quantum regressor.",
    )
    _add_common_dataset_args(regression_parser)
    regression_parser.add_argument(
        "--dataset",
        type=str,
        default="linear",
        choices=["linear", "sine", "polynomial"],
        help="Regression dataset.",
    )
    regression_parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of ansatz layers.",
    )
    regression_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of optimizer steps.",
    )
    regression_parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Optimizer step size.",
    )
    _add_shots_arg(regression_parser)

    logistic_parser = subparsers.add_parser(
        "logistic",
        help="Run a logistic regression classifier baseline.",
    )
    _add_common_dataset_args(logistic_parser)
    logistic_parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of optimizer iterations.",
    )

    svm_parser = subparsers.add_parser(
        "svm",
        help="Run a classical SVM classifier baseline.",
    )
    _add_common_dataset_args(svm_parser)
    svm_parser.add_argument(
        "--kernel-name",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="SVM kernel.",
    )
    svm_parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="SVM regularisation parameter.",
    )
    svm_parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient ('scale', 'auto', or numeric string).",
    )

    mlp_classifier_parser = subparsers.add_parser(
        "mlp-classifier",
        help="Run an MLP classifier baseline.",
    )
    _add_common_dataset_args(mlp_classifier_parser)
    mlp_classifier_parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[16, 16],
        help="Hidden layer sizes.",
    )
    mlp_classifier_parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of training iterations.",
    )

    ridge_parser = subparsers.add_parser(
        "ridge",
        help="Run a ridge regression baseline.",
    )
    _add_common_dataset_args(ridge_parser)
    ridge_parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularisation strength.",
    )

    mlp_regressor_parser = subparsers.add_parser(
        "mlp-regressor",
        help="Run an MLP regressor baseline.",
    )
    _add_common_dataset_args(mlp_regressor_parser)
    mlp_regressor_parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[32, 32],
        help="Hidden layer sizes.",
    )
    mlp_regressor_parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of training iterations.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run multi-seed benchmarks across models.",
    )

    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_type")

    classification_benchmark_parser = benchmark_subparsers.add_parser(
        "classification",
        help="Benchmark classification models.",
    )
    classification_benchmark_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to include.",
    )
    classification_benchmark_parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs", "xor"],
        help="Classification dataset.",
    )
    classification_benchmark_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[123],
        help="Random seeds.",
    )
    _add_common_benchmark_args(classification_benchmark_parser)

    regression_benchmark_parser = benchmark_subparsers.add_parser(
        "regression",
        help="Benchmark regression models.",
    )
    regression_benchmark_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to include.",
    )
    regression_benchmark_parser.add_argument(
        "--dataset",
        type=str,
        default="linear",
        choices=["linear", "sine", "polynomial"],
        help="Regression dataset.",
    )
    regression_benchmark_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[123],
        help="Random seeds.",
    )
    _add_common_benchmark_args(regression_benchmark_parser)

    return parser


def _run_vqc_command(args: argparse.Namespace) -> int:
    """
    Run the VQC workflow from parsed CLI arguments.
    """
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
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train MSE: {result['train_mse']:.6f}")
    print(f"Test MSE: {result['test_mse']:.6f}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
    print(f"Final loss: {result['final_loss']:.6f}")
    return 0


def _run_trainable_kernel_command(args: argparse.Namespace) -> int:
    """
    Run the trainable quantum kernel workflow from parsed CLI arguments.
    """
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


def _run_kernel_command(args: argparse.Namespace) -> int:
    """
    Run the quantum kernel workflow from parsed CLI arguments.
    """
    result = run_quantum_kernel_classifier(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        dataset=args.dataset,
        seed=args.seed,
        plot=args.plot,
        shots=args.shots,
        save=args.save,
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


def main() -> int:
    """
    Run the qml CLI.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "vqc":
        return _run_vqc_command(args)

    if args.command == "kernel":
        return _run_kernel_command(args)

    if args.command == "trainable-kernel":
        return _run_trainable_kernel_command(args)

    if args.command == "regression":
        return _run_regression_command(args)

    if args.command == "logistic":
        return _run_logistic_command(args)

    if args.command == "svm":
        return _run_svm_command(args)

    if args.command == "mlp-classifier":
        return _run_mlp_classifier_command(args)

    if args.command == "ridge":
        return _run_ridge_command(args)

    if args.command == "mlp-regressor":
        return _run_mlp_regressor_command(args)

    if args.command == "benchmark":
        if args.benchmark_type == "classification":
            return _run_classification_benchmark_command(args)

        if args.benchmark_type == "regression":
            return _run_regression_benchmark_command(args)

        print("Please specify 'classification' or 'regression'")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
