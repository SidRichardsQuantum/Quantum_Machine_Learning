"""
qml.kernel_methods
==================

Quantum kernel workflows and utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from sklearn.svm import SVC

from qml.data import make_moons_dataset
from qml.io_utils import save_json
from qml.metrics import accuracy_score


def _angle_feature_map(x, wires) -> None:
    """
    Simple 2-feature quantum feature map.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != len(wires):
        raise ValueError(f"Expected {len(wires)} features, got {x.shape[0]}.")

    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)

    if len(wires) > 1:
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def _compute_kernel_matrix(x_a, x_b, kernel_fn) -> np.ndarray:
    """
    Compute the kernel matrix K_ij = k(x_a[i], x_b[j]).
    """
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)

    kernel = np.empty((x_a.shape[0], x_b.shape[0]), dtype=float)
    for i, xa in enumerate(x_a):
        for j, xb in enumerate(x_b):
            kernel[i, j] = float(kernel_fn(xa, xb))
    return kernel


def run_quantum_kernel_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path = "results/kernel",
    images_dir: str | Path = "images/kernel",
) -> dict[str, Any]:
    """
    Run a minimal quantum kernel classifier on the two-moons dataset.
    """
    dataset = make_moons_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = dataset["x_train"]
    x_test = dataset["x_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]

    n_qubits = x_train.shape[1]
    wires = list(range(n_qubits))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        _angle_feature_map(x1, wires)
        qml.adjoint(_angle_feature_map)(x2, wires)
        return qml.probs(wires=wires)

    def kernel_fn(x1, x2) -> float:
        probs = kernel_circuit(x1, x2)
        return probs[0]

    kernel_matrix_train = _compute_kernel_matrix(x_train, x_train, kernel_fn)
    kernel_matrix_test = _compute_kernel_matrix(x_test, x_train, kernel_fn)

    clf = SVC(kernel="precomputed")
    clf.fit(kernel_matrix_train, y_train)

    y_train_pred = clf.predict(kernel_matrix_train)
    y_test_pred = clf.predict(kernel_matrix_test)

    result = {
        "model": "quantum_kernel_classifier",
        "dataset": "moons",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "kernel_matrix_train": kernel_matrix_train,
        "kernel_matrix_test": kernel_matrix_test,
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
    }

    stem = (
        f"moons_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}_seed{seed}"
    )

    if save:
        save_json(result, Path(results_dir) / f"{stem}.json")

    return result