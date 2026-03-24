"""
qml.io_utils
============

Saving, loading, and reproducibility helpers for qml experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(obj: Any) -> Any:
    """
    Convert common NumPy objects into JSON-serializable Python objects.
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
    """Save a dictionary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a dictionary from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
