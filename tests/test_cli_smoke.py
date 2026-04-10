import io
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout

import pytest

from qml.__main__ import main


def _run_cli_in_process(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()

    monkeypatch.setattr(sys, "argv", ["python -m qml", *args])

    with redirect_stdout(stdout), redirect_stderr(stderr):
        returncode = main()

    return returncode, stdout.getvalue(), stderr.getvalue()


@pytest.mark.slow
def test_cli_vqc_runs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qml",
            "vqc",
            "--samples",
            "20",
            "--steps",
            "2",
            "--layers",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Train accuracy" in result.stdout
    assert "Test accuracy" in result.stdout


def test_cli_kernel_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
        [
            "kernel",
            "--samples",
            "20",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout


def test_cli_qcnn_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
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


def test_cli_autoencoder_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
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


def test_cli_regression_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
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


def test_cli_logistic_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
        [
            "logistic",
            "--samples",
            "20",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout


def test_cli_ridge_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
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


def test_cli_trainable_kernel_runs(monkeypatch: pytest.MonkeyPatch):
    returncode, stdout, _ = _run_cli_in_process(
        monkeypatch,
        [
            "trainable-kernel",
            "--samples",
            "20",
            "--steps",
            "1",
            "--embedding",
            "data_reupload",
            "--embedding-layers",
            "1",
        ],
    )

    assert returncode == 0
    assert "Train accuracy" in stdout
    assert "Test accuracy" in stdout
    assert "Final alignment" in stdout
    assert "Final loss" in stdout
