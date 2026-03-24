"""
Command-line entrypoint for the qml package.
"""

from __future__ import annotations

import argparse

from qml.classifiers import run_vqc
from qml.kernel_methods import run_quantum_kernel_classifier


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m qml",
        description="Run quantum machine learning workflows.",
    )

    subparsers = parser.add_subparsers(dest="command")

    vqc_parser = subparsers.add_parser(
        "vqc",
        help="Run a variational quantum classifier.",
    )
    vqc_parser.add_argument("--samples", type=int, default=200, help="Number of samples.")
    vqc_parser.add_argument("--noise", type=float, default=0.1, help="Dataset noise level.")
    vqc_parser.add_argument(
        "--test-size", type=float, default=0.25, help="Fraction reserved for test data."
    )
    vqc_parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    vqc_parser.add_argument("--layers", type=int, default=2, help="Number of ansatz layers.")
    vqc_parser.add_argument("--steps", type=int, default=50, help="Number of optimizer steps.")
    vqc_parser.add_argument("--step-size", type=float, default=0.1, help="Optimizer step size.")
    vqc_parser.add_argument("--plot", action="store_true", help="Display plots.")
    vqc_parser.add_argument("--save", action="store_true", help="Save results and figures.")

    kernel_parser = subparsers.add_parser(
        "kernel",
        help="Run a quantum kernel classifier.",
    )
    kernel_parser.add_argument("--samples", type=int, default=200, help="Number of samples.")
    kernel_parser.add_argument("--noise", type=float, default=0.1, help="Dataset noise level.")
    kernel_parser.add_argument(
        "--test-size", type=float, default=0.25, help="Fraction reserved for test data."
    )
    kernel_parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    kernel_parser.add_argument("--plot", action="store_true", help="Display plots.")
    kernel_parser.add_argument("--save", action="store_true", help="Save results and figures.")

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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
