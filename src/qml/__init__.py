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

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

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
    "estimators",
    "kernels",
    "preprocessing",
    "reporting",
    "visualize",
    "io_utils",
    "optimizers",
    "QuantumClassifier",
    "QuantumRegressor",
    "QuantumKernel",
    "QuantumKernelClassifier",
    "QuantumKernelRegressor",
    "kernel_target_alignment",
    "make_sequence_windows",
    "format_table",
    "print_table",
    "print_section",
    "run_vqc",
    "run_vqr",
    "run_quantum_autoencoder",
    "run_qcnn",
    "run_quantum_kernel_classifier",
    "run_trainable_quantum_kernel_classifier",
    "get_optimizer",
    "list_supported_optimizers",
]

_SUBMODULES = {
    "ansatz",
    "autoencoder",
    "classical_baselines",
    "classifiers",
    "data",
    "embeddings",
    "estimators",
    "io_utils",
    "kernel_methods",
    "kernels",
    "losses",
    "metric_learning",
    "metrics",
    "optimizers",
    "preprocessing",
    "qcnn",
    "regression",
    "reporting",
    "trainable_kernels",
    "training",
    "utils",
    "visualize",
}

_EXPORTS = {
    "QuantumClassifier": ("estimators", "QuantumClassifier"),
    "QuantumRegressor": ("estimators", "QuantumRegressor"),
    "QuantumKernel": ("kernels", "QuantumKernel"),
    "QuantumKernelClassifier": ("kernels", "QuantumKernelClassifier"),
    "QuantumKernelRegressor": ("kernels", "QuantumKernelRegressor"),
    "kernel_target_alignment": ("kernels", "kernel_target_alignment"),
    "make_sequence_windows": ("preprocessing", "make_sequence_windows"),
    "format_table": ("reporting", "format_table"),
    "print_table": ("reporting", "print_table"),
    "print_section": ("reporting", "print_section"),
    "run_vqc": ("classifiers", "run_vqc"),
    "run_vqr": ("regression", "run_vqr"),
    "run_quantum_autoencoder": ("autoencoder", "run_quantum_autoencoder"),
    "run_qcnn": ("qcnn", "run_qcnn"),
    "run_quantum_kernel_classifier": ("kernel_methods", "run_quantum_kernel_classifier"),
    "run_quantum_metric_learner": ("metric_learning", "run_quantum_metric_learner"),
    "run_trainable_quantum_kernel_classifier": (
        "trainable_kernels",
        "run_trainable_quantum_kernel_classifier",
    ),
    "get_optimizer": ("optimizers", "get_optimizer"),
    "list_supported_optimizers": ("optimizers", "list_supported_optimizers"),
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
