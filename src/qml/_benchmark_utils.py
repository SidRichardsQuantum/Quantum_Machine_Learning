"""
Shared helpers for benchmark aggregation and metadata.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from math import sqrt
from statistics import mean, pstdev
from typing import Any


def mean_std(values: list[float]) -> dict[str, float]:
    """
    Return mean and population standard deviation for a list of floats.
    """
    if not values:
        return {"mean": float("nan"), "std": float("nan")}

    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}

    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
    }


def summary_stats(values: list[float]) -> dict[str, float]:
    """
    Return mean, std, and a normal-approximation 95% confidence interval.
    """
    stats = mean_std(values)
    n = len(values)
    stats["n"] = n
    if n == 0:
        stats["ci95_low"] = float("nan")
        stats["ci95_high"] = float("nan")
        return stats

    half_width = 0.0 if n == 1 else 1.96 * stats["std"] / sqrt(n)
    stats["ci95_low"] = float(stats["mean"] - half_width)
    stats["ci95_high"] = float(stats["mean"] + half_width)
    return stats


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def benchmark_metadata() -> dict[str, Any]:
    """
    Return reproducibility metadata for saved or returned benchmark results.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "pennylane": package_version("pennylane"),
            "qml-pennylane": package_version("qml-pennylane"),
        },
    }


def timing_from_result(result: Any, runtime_seconds: float) -> dict[str, float]:
    if isinstance(result, dict) and isinstance(result.get("timing"), dict):
        timing = {key: float(value) for key, value in result["timing"].items()}
        timing.setdefault("total_seconds", float(runtime_seconds))
        return timing

    return {"total_seconds": float(runtime_seconds)}
