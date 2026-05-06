from __future__ import annotations

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULT_ASSETS = ROOT / "docs/pages/assets/reference-results"

import pennylane as pennylane  # noqa: E402

from qml.autoencoder import run_quantum_autoencoder  # noqa: E402
from qml.classifiers import run_vqc  # noqa: E402
from qml.kernel_methods import run_quantum_kernel_classifier  # noqa: E402
from qml.metric_learning import run_quantum_metric_learner  # noqa: E402
from qml.qcnn import run_qcnn  # noqa: E402
from qml.regression import run_vqr  # noqa: E402
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier  # noqa: E402


def short_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version() -> str:
    try:
        return version("qml-pennylane")
    except PackageNotFoundError:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return str(data["project"]["version"])


def final_value(values: list[float] | tuple[float, ...]) -> float:
    return float(values[-1]) if values else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "analytic"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def row(
    model: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    elapsed: float,
    images: list[Path],
) -> dict[str, Any]:
    return {
        "model": model,
        "config": config,
        "metrics": metrics,
        "elapsed": elapsed,
        "images": images,
    }


def timed_run(label: str, fn, *args, **kwargs) -> tuple[Any, float]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    return result, perf_counter() - start


def run_images(run_dir: str) -> list[Path]:
    image_dir = RESULT_ASSETS / run_dir
    if not image_dir.exists():
        return []
    return sorted(path.relative_to(ROOT) for path in image_dir.glob("*.png"))


def output_dirs(run_dir: str) -> dict[str, Path]:
    return {
        "results_dir": RESULT_ASSETS / run_dir / "data",
        "images_dir": RESULT_ASSETS / run_dir,
    }


def run_reference_results() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if RESULT_ASSETS.exists():
        shutil.rmtree(RESULT_ASSETS)
    RESULT_ASSETS.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": "moons",
        "n_samples": 50,
        "noise": 0.1,
        "seed": 123,
        "n_layers": 1,
        "steps": 8,
        "shots": None,
    }
    result, elapsed = timed_run(
        "vqc",
        run_vqc,
        plot=False,
        save=True,
        **config,
        **output_dirs("vqc"),
    )
    runs.append(
        row(
            "Variational quantum classifier",
            config,
            {
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "final_loss": result["final_loss"],
            },
            elapsed,
            run_images("vqc"),
        )
    )

    config = {
        "dataset": "linear",
        "n_samples": 50,
        "noise": 0.1,
        "seed": 123,
        "n_layers": 1,
        "steps": 8,
        "shots": None,
    }
    result, elapsed = timed_run(
        "vqr",
        run_vqr,
        plot=False,
        save=True,
        **config,
        **output_dirs("vqr"),
    )
    runs.append(
        row(
            "Variational quantum regression",
            config,
            {
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "final_loss": result["final_loss"],
            },
            elapsed,
            run_images("vqr"),
        )
    )

    config = {
        "dataset": "moons",
        "n_samples": 40,
        "noise": 0.1,
        "seed": 123,
        "steps": 6,
        "shots": None,
    }
    result, elapsed = timed_run(
        "qcnn",
        run_qcnn,
        plot=False,
        save=True,
        **config,
        **output_dirs("qcnn"),
    )
    runs.append(
        row(
            "Quantum convolutional neural network",
            config,
            {
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "final_loss": result["final_loss"],
            },
            elapsed,
            run_images("qcnn"),
        )
    )

    config = {
        "family": "correlated",
        "n_samples": 32,
        "noise": 0.05,
        "seed": 123,
        "n_layers": 1,
        "latent_qubits": 2,
        "steps": 6,
    }
    result, elapsed = timed_run(
        "autoencoder",
        run_quantum_autoencoder,
        plot=False,
        save=True,
        **config,
        **output_dirs("autoencoder"),
    )
    runs.append(
        row(
            "Quantum autoencoder",
            config,
            {
                "test_compression_fidelity": result["test_compression_fidelity"],
                "test_reconstruction_fidelity": result["test_reconstruction_fidelity"],
                "final_loss": result["final_loss"],
            },
            elapsed,
            run_images("autoencoder"),
        )
    )

    config = {
        "dataset": "moons",
        "n_samples": 36,
        "noise": 0.1,
        "seed": 123,
        "shots": None,
    }
    result, elapsed = timed_run(
        "quantum_kernel",
        run_quantum_kernel_classifier,
        plot=False,
        save=True,
        **config,
        **output_dirs("quantum_kernel"),
    )
    runs.append(
        row(
            "Quantum kernel classifier",
            config,
            {
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
            },
            elapsed,
            run_images("quantum_kernel"),
        )
    )

    config = {
        "dataset": "moons",
        "n_samples": 20,
        "noise": 0.1,
        "seed": 123,
        "embedding_layers": 1,
        "steps": 2,
        "shots_train": None,
        "shots_kernel": None,
    }
    result, elapsed = timed_run(
        "trainable_kernel",
        run_trainable_quantum_kernel_classifier,
        plot=False,
        save=True,
        **config,
        **output_dirs("trainable_kernel"),
    )
    runs.append(
        row(
            "Trainable quantum kernel",
            config,
            {
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "final_alignment": result["final_alignment"],
                "final_loss": result["final_loss"],
            },
            elapsed,
            run_images("trainable_kernel"),
        )
    )

    config = {
        "dataset": "moons",
        "samples": 50,
        "seed": 42,
        "layers": 1,
        "steps": 8,
        "pairs_per_step": 16,
        "log_every": 0,
    }
    result, elapsed = timed_run(
        "metric_learning",
        run_quantum_metric_learner,
        plot=False,
        save=True,
        **config,
        **output_dirs("metric_learning"),
    )
    runs.append(
        row(
            "Quantum metric learning",
            config,
            {
                "train_accuracy": result.train_accuracy,
                "test_accuracy": result.test_accuracy,
                "final_loss": final_value(result.loss_history),
            },
            elapsed,
            run_images("metric_learning"),
        )
    )

    return runs


def format_config(config: dict[str, Any]) -> str:
    return ", ".join(f"`{key}={fmt(value)}`" for key, value in config.items())


def metrics_table(run: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in run["metrics"].items():
        lines.append(f"| `{key}` | {fmt(value)} |")
    lines.append(f"| `runtime_seconds` | {fmt(run['elapsed'], digits=2)} |")
    return "\n".join(lines)


def image_gallery(run: dict[str, Any]) -> str:
    if not run["images"]:
        return ""

    lines = ["", "Images:", ""]
    for image in run["images"]:
        title = image.stem.replace("_", " ")
        lines.append(f"![{title}]({image.as_posix()})")
    return "\n".join(lines)


def render_results(runs: list[dict[str, Any]]) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_rows = [
        "| Workflow | Primary metric | Value | Runtime |",
        "| --- | --- | ---: | ---: |",
    ]
    for run in runs:
        primary_name, primary_value = next(iter(run["metrics"].items()))
        summary_rows.append(
            f"| {run['model']} | `{primary_name}` | {fmt(primary_value)} | "
            f"{fmt(run['elapsed'], digits=2)} s |"
        )

    sections = []
    for run in runs:
        sections.append(f"""## {run["model"]}

Configuration:

{format_config(run["config"])}

{metrics_table(run)}
{image_gallery(run)}
""")

    return f"""# Results

These reference results are generated from the public package APIs used by the notebooks.
The notebooks remain thin clients; the API path is used here because it is deterministic,
CI-friendly, and avoids committing executed notebook outputs.

The configurations are intentionally small so the GitHub Pages workflow can refresh the
page quickly. They are reproducible smoke-scale examples, not quantum-advantage claims.

## Environment

- Generated: {generated_at}
- Git commit: `{short_commit()}`
- Python: `{platform.python_version()}`
- Package version: `{package_version()}`
- PennyLane: `{pennylane.__version__}`
- Matplotlib backend: `{os.environ.get("MPLBACKEND", "Agg")}`
- Default execution: analytic `default.qubit` unless a shot count is listed

## Summary

{chr(10).join(summary_rows)}

{chr(10).join(sections)}
## Reproduce

Regenerate this file from the repository root:

```bash
python docs/pages/generate_results.py
```

The GitHub Pages workflow also regenerates this file before building the web pages.
Generated images are written under `docs/pages/assets/reference-results/` and embedded above.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic QML reference results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "RESULTS.md",
        help="Markdown file to write.",
    )
    args = parser.parse_args()

    runs = run_reference_results()
    args.output.write_text(render_results(runs), encoding="utf-8")


if __name__ == "__main__":
    main()
