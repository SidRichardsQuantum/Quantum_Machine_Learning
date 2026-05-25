import io
from contextlib import redirect_stderr, redirect_stdout

from qml.cli import main


def _run_cli_in_process(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        returncode = main(args)

    return returncode, stdout.getvalue(), stderr.getvalue()


def test_cli_vqc_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "vqc",
            "--samples",
            "8",
            "--steps",
            "1",
            "--layers",
            "1",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout


def test_cli_kernel_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "kernel",
            "--samples",
            "20",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout


def test_cli_qcnn_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "qcnn",
            "--samples",
            "20",
            "--steps",
            "2",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout
    assert "Final loss" in stdout


def test_cli_autoencoder_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "autoencoder",
            "--samples",
            "20",
            "--steps",
            "2",
            "--layers",
            "1",
        ],
    )

    assert returncode == 0
    assert "Train compression fidelity" in stdout
    assert "Test compression fidelity" in stdout
    assert "Final loss" in stdout


def test_cli_regression_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "regression",
            "--samples",
            "20",
            "--steps",
            "2",
            "--layers",
            "1",
        ],
    )

    assert returncode == 0
    assert "Train MSE" in stdout
    assert "Test MSE" in stdout
    assert "Final loss" in stdout


def test_cli_logistic_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "logistic",
            "--samples",
            "20",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout


def test_cli_ridge_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "ridge",
            "--samples",
            "20",
        ],
    )

    assert returncode == 0
    assert "Train MSE" in stdout
    assert "Test MSE" in stdout
    assert "Train MAE" in stdout
    assert "Test MAE" in stdout


def test_cli_trainable_kernel_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "trainable-kernel",
            "--samples",
            "8",
            "--steps",
            "0",
            "--embedding",
            "angle",
            "--embedding-layers",
            "1",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout
    assert "Final alignment" in stdout
    assert "Final loss" in stdout


def test_cli_metric_learning_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "metric-learning",
            "--samples",
            "20",
            "--steps",
            "1",
            "--layers",
            "1",
            "--pairs-per-step",
            "4",
            "--log-every",
            "0",
        ],
    )

    assert returncode == 0
    assert "Model: quantum_metric_learning" in stdout
    assert "Dataset: moons" in stdout
    assert "Train accuracy:" in stdout
    assert "Test accuracy:" in stdout
    assert "Final loss:" in stdout


def test_cli_without_command_prints_help():
    returncode, stdout, _ = _run_cli_in_process([])

    assert returncode == 1
    assert "Run quantum and classical machine learning workflows." in stdout


def test_cli_benchmark_requires_nested_command():
    returncode, stdout, _ = _run_cli_in_process(["benchmark"])

    assert returncode == 1
    assert "Please specify 'classification', 'regression', or 'finite-shots'" in stdout


def test_cli_finite_shot_benchmark_runs():
    returncode, stdout, _ = _run_cli_in_process(
        [
            "benchmark",
            "finite-shots",
            "--classification-models",
            "quantum_reservoir",
            "logistic_regression",
            "--regression-models",
            "quantum_reservoir_regressor",
            "ridge_regression",
            "--samples",
            "20",
            "--seeds",
            "0",
            "--shots",
            "analytic",
            "64",
        ],
    )

    assert returncode == 0
    assert "Benchmark type: finite-shots" in stdout
    assert "Shots: analytic" in stdout
    assert "Shots: 64" in stdout
    assert "quantum_reservoir" in stdout
    assert "quantum_reservoir_regressor" in stdout
