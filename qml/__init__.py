"""
qml
===

Quantum Machine Learning package built on PennyLane.

Provides reusable components for:
- Variational quantum classifiers (VQC)
- Variational quantum regression (VQR)
- Quantum kernel methods
- Classical baseline models
- Hybrid quantum–classical training workflows
"""

from importlib.metadata import PackageNotFoundError, version

from .classifiers import run_vqc
from .regression import run_vqr
from .kernel_methods import run_quantum_kernel_classifier

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
    "regression",
    "kernel_methods",
    "classical_baselines",
    "visualize",
    "io_utils",
    "run_vqc",
    "run_vqr",
    "run_quantum_kernel_classifier",
]
