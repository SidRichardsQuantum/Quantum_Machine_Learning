"""
Command-line entrypoint for the qml package.
"""

from __future__ import annotations

import argparse

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

    kernel_parser = subparsers.add_parser(
        "kernel",
        help="Run a quantum kernel classifier.",
    )
    _add_common_dataset_args(kernel_parser)

    regression_parser = subparsers.add_parser(
        "regression",
        help="Run a variational quantum regressor.",
    )
    _add_common_dataset_args(regression_parser)
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

    return parser


def _run_vqc_command(args: argparse.Namespace) -> int:
    """
    Run the VQC workflow from parsed CLI arguments.
    """
    result = run_vqc(
        n_samples=args.samples,
        noise=args.noise,
        test_size=args.test_size,
        seed=args.seed,
        n_layers=args.layers,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
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
        seed=args.seed,
        n_layers=args.layers,
        steps=args.steps,
        step_size=args.step_size,
        plot=args.plot,
        save=args.save,
    )

    print(f"Model: {result['model']}")
    print(f"Dataset: {result['dataset']}")
    print(f"Train MSE: {result['train_mse']:.6f}")
    print(f"Test MSE: {result['test_mse']:.6f}")
    print(f"Train MAE: {result['train_mae']:.6f}")
    print(f"Test MAE: {result['test_mae']:.6f}")
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
        seed=args.seed,
        plot=args.plot,
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
