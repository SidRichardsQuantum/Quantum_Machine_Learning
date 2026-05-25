"""Argument parser construction for the qml command-line interface."""

from __future__ import annotations

import argparse


def _shot_value(value: str) -> int | None:
    if value.lower() in {"none", "analytic"}:
        return None
    return int(value)


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
    parser.add_argument(
        "--tune-classical",
        action="store_true",
        help="Tune classical baselines with small GridSearchCV defaults.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for tuned classical baselines.",
    )


def _add_shots_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Number of measurement shots (None = analytic mode).",
    )


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
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
        choices=["moons", "circles", "blobs", "xor", "linear", "breast_cancer", "wine"],
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
        choices=["moons", "circles", "blobs", "xor", "linear", "breast_cancer", "wine"],
        help="Classification dataset.",
    )

    qcnn_parser = subparsers.add_parser(
        "qcnn",
        help="Run a quantum convolutional neural network classifier.",
    )
    _add_common_dataset_args(qcnn_parser)
    qcnn_parser.add_argument("--steps", type=int, default=50, help="Number of optimizer steps.")
    qcnn_parser.add_argument("--step-size", type=float, default=0.1, help="Optimizer step size.")
    _add_shots_arg(qcnn_parser)
    qcnn_parser.add_argument(
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

    metric_learning_parser = subparsers.add_parser(
        "metric-learning",
        help="Run a quantum metric learning workflow.",
    )
    _add_common_dataset_args(metric_learning_parser)
    metric_learning_parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs"],
        help="Classification dataset.",
    )
    metric_learning_parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of trainable embedding layers.",
    )
    metric_learning_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of optimizer steps.",
    )
    metric_learning_parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Optimizer step size.",
    )
    metric_learning_parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Contrastive loss margin for negative pairs.",
    )
    metric_learning_parser.add_argument(
        "--pairs-per-step",
        type=int,
        default=32,
        help="Number of sampled training pairs per optimization step.",
    )
    metric_learning_parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print training progress every N steps.",
    )
    metric_learning_parser.add_argument(
        "--no-scale-data",
        action="store_true",
        help="Disable feature standardization before angle encoding.",
    )

    autoencoder_parser = subparsers.add_parser(
        "autoencoder",
        help="Run a quantum autoencoder workflow.",
    )
    _add_common_dataset_args(autoencoder_parser)
    autoencoder_parser.add_argument(
        "--family",
        type=str,
        default="correlated",
        choices=["correlated", "entangled", "hybrid"],
        help="Structured quantum state family.",
    )
    autoencoder_parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of autoencoder ansatz layers.",
    )
    autoencoder_parser.add_argument(
        "--latent-qubits",
        type=int,
        default=2,
        help="Number of latent qubits retained by the autoencoder.",
    )
    autoencoder_parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of optimizer steps.",
    )
    autoencoder_parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Optimizer step size.",
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
        choices=["linear", "sine", "polynomial", "friedman", "diabetes"],
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
        choices=["linear", "sine", "polynomial", "friedman", "diabetes"],
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

    finite_shot_benchmark_parser = benchmark_subparsers.add_parser(
        "finite-shots",
        help="Benchmark analytic versus finite-shot execution.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--classification-models",
        nargs="+",
        default=["vqc", "quantum_kernel", "quantum_reservoir", "svm_classifier"],
        help="Classification model names to include.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--regression-models",
        nargs="+",
        default=[
            "vqr",
            "quantum_kernel_regressor",
            "quantum_reservoir_regressor",
            "ridge_regression",
        ],
        help="Regression model names to include.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--classification-dataset",
        type=str,
        default="moons",
        choices=["moons", "circles", "blobs", "xor", "linear", "breast_cancer", "wine"],
        help="Classification dataset.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--regression-dataset",
        type=str,
        default="sine",
        choices=["linear", "sine", "polynomial", "friedman", "diabetes"],
        help="Regression dataset.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[123],
        help="Random seeds.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--shots",
        type=_shot_value,
        nargs="+",
        default=[None, 64, 128, 512],
        help="Shot counts to compare. Use 'analytic' or 'none' for analytic execution.",
    )
    finite_shot_benchmark_parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Small optimizer step count for trainable quantum models.",
    )
    _add_common_benchmark_args(finite_shot_benchmark_parser)

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)

    return parser
