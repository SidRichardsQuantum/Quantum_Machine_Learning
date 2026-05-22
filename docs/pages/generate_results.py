from __future__ import annotations

import argparse
import base64
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import nbformat

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RESULT_ASSETS = ROOT / "docs/pages/assets/reference-results"
NOTEBOOK_ASSETS = ROOT / "docs/pages/assets/notebook-results"
NOTEBOOK_RESULTS = {
    "tutorial": {
        "title": "Tutorial Notebook Results",
        "description": (
            "Executed outputs from the tutorial notebooks in `notebooks/tutorials/`. "
            "These pages are generated from notebook outputs, including text tables and plots."
        ),
        "directory": ROOT / "notebooks/tutorials",
        "output": ROOT / "RESULTS_TUTORIALS.md",
    },
    "real_examples": {
        "title": "Real Example Notebook Results",
        "description": (
            "Executed outputs from the domain-oriented notebooks in `notebooks/real_examples/`. "
            "These examples use small reproducible physics, mathematics, or dynamical-system tasks."
        ),
        "directory": ROOT / "notebooks/real_examples",
        "output": ROOT / "RESULTS_REAL_EXAMPLES.md",
    },
}

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
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])

    try:
        return version("qml-pennylane")
    except PackageNotFoundError:
        return "unknown"


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


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


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


def notebook_title(path: Path, notebook) -> str:
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.startswith("# "):
                return line.removeprefix("# ").strip()
    return path.stem.replace("-", " ").title()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def is_relevant_stream(text: str) -> bool:
    if "+-" in text and "|" in text:
        return True
    keywords = (
        "Validation",
        "Dataset",
        "Results",
        "Summary",
        "Train accuracy",
        "Test accuracy",
        "Passed:",
        "Interpretation",
        "Sample ",
    )
    return any(keyword in text for keyword in keywords)


def notebook_stream_blocks(notebook) -> list[str]:
    blocks: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        parts = []
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
            elif output_type in {"execute_result", "display_data"}:
                data = output.get("data", {})
                text = data.get("text/plain", "")
                if isinstance(text, list):
                    text = "".join(text)
                if text.startswith("<Figure"):
                    continue
            else:
                continue
            text = strip_ansi(text).strip()
            if text and (is_relevant_stream(text) or len(text) <= 2000):
                parts.append(text)
        if parts:
            blocks.append("\n".join(parts))
    return blocks


def write_notebook_images(path: Path, notebook, group: str) -> list[Path]:
    output_dir = NOTEBOOK_ASSETS / group / path.stem
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    index = 1
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            encoded = data.get("image/png")
            if not encoded:
                continue
            if isinstance(encoded, list):
                encoded = "".join(encoded)
            image_path = output_dir / f"figure-{index:02d}.png"
            image_path.write_bytes(base64.b64decode(encoded))
            images.append(image_path.relative_to(ROOT))
            index += 1
    return images


def collect_notebook_result(path: Path, group: str) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    return {
        "title": notebook_title(path, notebook),
        "path": path.relative_to(ROOT),
        "streams": notebook_stream_blocks(notebook),
        "images": write_notebook_images(path, notebook, group),
    }


def execute_notebooks(paths: list[Path]) -> None:
    if not paths:
        return
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("JUPYTER_CONFIG_DIR", "/tmp/jupyter_config")
    env.setdefault("JUPYTER_DATA_DIR", "/tmp/jupyter_data")
    env.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/jupyter_runtime")
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--inplace",
        *(str(path.relative_to(ROOT)) for path in paths),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def collect_notebook_group(group: str, *, execute: bool) -> list[dict[str, Any]]:
    config = NOTEBOOK_RESULTS[group]
    paths = sorted(config["directory"].glob("*.ipynb"))
    if execute:
        execute_notebooks(paths)
    return [collect_notebook_result(path, group) for path in paths]


def render_notebook_results(group: str, results: list[dict[str, Any]]) -> str:
    config = NOTEBOOK_RESULTS[group]
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = [
        "| Notebook | Text result blocks | Plots |",
        "| --- | ---: | ---: |",
        *[
            f"| [{result['path'].as_posix()}](#{slugify(result['title'])}) | "
            f"{len(result['streams'])} | {len(result['images'])} |"
            for result in results
        ],
    ]

    sections = []
    for result in results:
        image_lines = []
        for image in result["images"]:
            title = image.stem.replace("-", " ")
            image_lines.append(f"![{title}]({image.as_posix()})")

        stream_lines = []
        for index, block in enumerate(result["streams"], start=1):
            stream_lines.append(f"Result block {index}:\n\n```text\n{block}\n```")

        sections.append(f"""## {result["title"]}

Notebook: `{result["path"].as_posix()}`

{chr(10).join(stream_lines) if stream_lines else "_No text result blocks were found._"}

{chr(10).join(image_lines) if image_lines else "_No plots were found._"}
""")

    return f"""# {config["title"]}

{config["description"]}

## Environment

- Generated: {generated_at}
- Git commit: `{short_commit()}`
- Python: `{platform.python_version()}`
- Package version: `{package_version()}`
- Matplotlib backend: `{os.environ.get("MPLBACKEND", "Agg")}`

## Summary

{chr(10).join(summary)}

{chr(10).join(sections)}
## Reproduce

Regenerate notebook result pages from existing executed notebook outputs:

```bash
python docs/pages/generate_results.py --skip-api-results
```

Execute notebooks first, then regenerate result pages:

```bash
python docs/pages/generate_results.py --skip-api-results --execute-notebooks
```
"""


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
Notebook-derived result pages are generated separately from executed notebook outputs:

- [Tutorial notebook results](results-tutorials.html)
- [Real example notebook results](results-real-examples.html)

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

Regenerate this file and notebook-result pages from the repository root:

```bash
python docs/pages/generate_results.py
```

The GitHub Pages workflow also regenerates this file before building the web pages.
Generated images are written under `docs/pages/assets/reference-results/` and embedded above.
"""


def write_notebook_result_pages(*, execute: bool) -> None:
    if NOTEBOOK_ASSETS.exists():
        shutil.rmtree(NOTEBOOK_ASSETS)
    NOTEBOOK_ASSETS.mkdir(parents=True, exist_ok=True)

    for group, config in NOTEBOOK_RESULTS.items():
        results = collect_notebook_group(group, execute=execute)
        config["output"].write_text(render_notebook_results(group, results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic QML reference results.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "RESULTS.md",
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--skip-notebook-results",
        action="store_true",
        help="Do not generate RESULTS_TUTORIALS.md or RESULTS_REAL_EXAMPLES.md.",
    )
    parser.add_argument(
        "--execute-notebooks",
        action="store_true",
        help="Execute notebooks before extracting notebook result pages.",
    )
    parser.add_argument(
        "--skip-api-results",
        action="store_true",
        help="Only generate notebook result pages.",
    )
    args = parser.parse_args()

    if not args.skip_api_results:
        runs = run_reference_results()
        args.output.write_text(render_results(runs), encoding="utf-8")

    if not args.skip_notebook_results:
        write_notebook_result_pages(execute=args.execute_notebooks)


if __name__ == "__main__":
    main()
