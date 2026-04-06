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

from qml.data import make_classification_dataset
from qml.io_utils import images_path, results_path, save_json
from qml.io_utils import ensure_dir
from qml.metrics import accuracy_score
from qml.visualize import plot_dataset_2d, plot_kernel_matrix


def _angle_feature_map(x, wires) -> None:
    """
    Simple angle-based quantum feature map.

    Parameters
    ----------
    x
        Input feature vector.
    wires
        Circuit wires.
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
    Compute the kernel matrix K_ij = k(x_a^(i), x_b^(j)).
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
    shots: int | None = None,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    dataset: str = "moons",
) -> dict[str, Any]:
    """
    Run a minimal quantum kernel classifier on the two-moons dataset.

    Parameters
    ----------
    n_samples
        Number of dataset samples.
    noise
        Noise level used by ``make_moons``.
    test_size
        Fraction reserved for test data.
    seed
        Random seed.
    shots
        Number of measurement shots. If ``None``, uses analytic mode.
    plot
        Whether to display plots.
    save
        Whether to save results JSON and figures.
    """
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    n_qubits = x_train.shape[1]
    wires = list(range(n_qubits))

    dev = qml.device("default.qubit", wires=n_qubits, seed=seed)

    @qml.qnode(dev)
    def kernel_circuit_base(x1, x2):
        _angle_feature_map(x1, wires)
        qml.adjoint(_angle_feature_map)(x2, wires)
        return qml.probs(wires=wires)

    kernel_circuit = (
        qml.set_shots(kernel_circuit_base, shots) if shots is not None else kernel_circuit_base
    )

    def kernel_fn(x1, x2) -> float:
        probs = kernel_circuit(x1, x2)
        return float(probs[0])

    kernel_matrix_train = _compute_kernel_matrix(
        x_train,
        x_train,
        kernel_fn,
    )

    kernel_matrix_test = _compute_kernel_matrix(
        x_test,
        x_train,
        kernel_fn,
    )

    clf = SVC(kernel="precomputed")
    clf.fit(kernel_matrix_train, y_train)

    y_train_pred = clf.predict(kernel_matrix_train)
    y_test_pred = clf.predict(kernel_matrix_test)

    result = {
        "model": "quantum_kernel_classifier",
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "shots": shots,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "kernel_matrix_train": kernel_matrix_train,
        "kernel_matrix_test": kernel_matrix_test,
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
    }

    shots_tag = "analytic" if shots is None else f"shots{shots}"

    stem = (
        f"{dataset}_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}"
        f"_seed{seed}"
        f"_{shots_tag}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path

        return results_path(
            "kernel",
            filename,
        )

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
            return path

        return images_path(
            "kernel",
            filename,
        )

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="Quantum kernel training dataset",
            show=plot,
            save_path=_images_file(f"{stem}_dataset.png") if save else None,
        )

        plot_kernel_matrix(
            kernel_matrix_train,
            title="Quantum kernel matrix (train)",
            show=plot,
            save_path=_images_file(f"{stem}_kernel_train.png") if save else None,
        )

        plot_kernel_matrix(
            kernel_matrix_test,
            title="Quantum kernel matrix (test vs train)",
            show=plot,
            save_path=_images_file(f"{stem}_kernel_test.png") if save else None,
        )

    if save:
        save_json(
            result,
            _results_file(f"{stem}.json"),
        )

    return result
