"""
qml.__init__
============

Quantum Machine Learning package built on PennyLane.

Provides reusable components for:
- Variational quantum classifiers (VQC)
- Variational quantum regression (VQR)
- Quantum kernel methods
- Trainable quantum kernel learning
- Classical baseline models
- Hybrid quantum-classical training workflows
"""

from importlib.metadata import PackageNotFoundError, version

from .classifiers import run_vqc
from .kernel_methods import run_quantum_kernel_classifier
from .optimizers import get_optimizer, list_supported_optimizers
from .autoencoder import run_quantum_autoencoder
from .qcnn import run_qcnn
from .regression import run_vqr
from .trainable_kernels import run_trainable_quantum_kernel_classifier
from .metric_learning import run_quantum_metric_learner

try:
    __version__ = version("qml-pennylane")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = [
    "data",
    "embeddings",
    "ansatz",
    "training",
    "metric_learning",
    "autoencoder",
    "run_quantum_metric_learner",
    "losses",
    "metrics",
    "classifiers",
    "regression",
    "qcnn",
    "kernel_methods",
    "trainable_kernels",
    "classical_baselines",
    "visualize",
    "io_utils",
    "optimizers",
    "run_vqc",
    "run_vqr",
    "run_quantum_autoencoder",
    "run_qcnn",
    "run_quantum_kernel_classifier",
    "run_trainable_quantum_kernel_classifier",
    "get_optimizer",
    "list_supported_optimizers",
]
