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
