"""
qml
===

Quantum Machine Learning package built on PennyLane.

Provides reusable components for:
- Variational quantum classifiers
- Quantum kernel methods
- Hybrid quantum-classical training workflows
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qml-pennylane")
except PackageNotFoundError:
    # package is not installed (e.g. local development)
    __version__ = "0.0.0"


__all__ = [
    "data",
    "embeddings",
    "ansatz",
    "training",
    "losses",
    "metrics",
    "classifiers",
    "kernel_methods",
    "visualize",
    "io_utils",
]
