"""Command-line interface dispatch for the qml package."""

from __future__ import annotations

from qml.cli_commands import (
    _run_autoencoder_command,
    _run_classification_benchmark_command,
    _run_kernel_command,
    _run_logistic_command,
    _run_metric_learning_command,
    _run_mlp_classifier_command,
    _run_mlp_regressor_command,
    _run_qcnn_command,
    _run_regression_benchmark_command,
    _run_regression_command,
    _run_ridge_command,
    _run_svm_command,
    _run_trainable_kernel_command,
    _run_vqc_command,
)
from qml.cli_parser import _build_parser


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """
    Run the qml CLI.
    """
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.command == "vqc":
        return _run_vqc_command(args)

    if args.command == "kernel":
        return _run_kernel_command(args)

    if args.command == "qcnn":
        return _run_qcnn_command(args)

    if args.command == "trainable-kernel":
        return _run_trainable_kernel_command(args)

    if args.command == "metric-learning":
        return _run_metric_learning_command(args)

    if args.command == "autoencoder":
        return _run_autoencoder_command(args)

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
