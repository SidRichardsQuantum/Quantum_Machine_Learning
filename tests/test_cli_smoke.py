import subprocess
import sys


def test_cli_vqc_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "vqc",
            "--samples",
            "40",
            "--steps",
            "5",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train accuracy" in result.stdout


def test_cli_kernel_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "kernel",
            "--samples",
            "40",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train accuracy" in result.stdout


def test_cli_regression_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "regression",
            "--samples",
            "40",
            "--steps",
            "5",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train MSE" in result.stdout
    assert "Test MSE" in result.stdout
    assert "Final loss" in result.stdout


def test_cli_logistic_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "logistic",
            "--samples",
            "40",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train accuracy" in result.stdout
    assert "Test accuracy" in result.stdout


def test_cli_ridge_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "ridge",
            "--samples",
            "40",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train MSE" in result.stdout
    assert "Test MSE" in result.stdout
    assert "Train MAE" in result.stdout
    assert "Test MAE" in result.stdout


def test_cli_trainable_kernel_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "trainable-kernel",
            "--samples",
            "40",
            "--steps",
            "3",
            "--embedding",
            "data_reupload",
            "--embedding-layers",
            "2",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train accuracy" in result.stdout
    assert "Test accuracy" in result.stdout
    assert "Final alignment" in result.stdout
    assert "Final loss" in result.stdout
