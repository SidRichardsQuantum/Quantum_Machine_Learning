"""
qml.io_utils
============

Saving, loading, and reproducibility helpers for qml experiments.

All output paths resolve relative to repository root:

results/<module>/
images/<module>/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------
# repo paths
# ---------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = _ROOT / "results"
IMAGES_DIR = _ROOT / "images"


def results_path(module: str, filename: str) -> Path:
    """
    Path for experiment outputs.

    Example
    -------
    results_path("vqc", "run.json")
    -> results/vqc/run.json
    """
    path = RESULTS_DIR / module / filename
    ensure_dir(path.parent)
    return path


def images_path(module: str, filename: str) -> Path:
    """
    Path for generated figures.

    Example
    -------
    images_path("kernel", "matrix.png")
    -> images/kernel/matrix.png
    """
    path = IMAGES_DIR / module / filename
    ensure_dir(path.parent)
    return path


# ---------------------------------------------------------------------
# json helpers
# ---------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """
    Convert NumPy objects into JSON-serializable Python objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    return obj


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary to JSON.

    Creates parent directories if required.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            _jsonable(data),
            f,
            indent=2,
            sort_keys=True,
        )


def load_json(path: str | Path) -> dict[str, Any]:
    """
    Load dictionary from JSON.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_result_dict(
    model: str,
    dataset: str,
    metrics: dict,
    config: dict,
):
    return {
        "model": model,
        "dataset": dataset,
        "metrics": metrics,
        "config": config,
    }


def ensure_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
