"""
qml.trainable_kernels
=====================

Trainable quantum kernel workflows and utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC

from qml.data import make_classification_dataset
from qml.embeddings import embedding_parameter_shape, get_embedding
from qml.io_utils import images_path, results_path, save_json
from qml.metrics import accuracy_score
from qml.visualize import (
    plot_alignment_curve,
    plot_dataset_2d,
    plot_kernel_matrix,
    plot_loss_curve,
)


def _compute_kernel_matrix(x_a, x_b, kernel_fn) -> np.ndarray:
    """
    Compute the kernel matrix K_ij = k(x_a^(i), x_b^(j)).

    Parameters
    ----------
    x_a
        First input array of shape ``(n_a, n_features)``.
    x_b
        Second input array of shape ``(n_b, n_features)``.
    kernel_fn
        Callable returning a scalar kernel value.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape ``(n_a, n_b)``.
    """
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)

    kernel = np.empty((x_a.shape[0], x_b.shape[0]), dtype=float)
    for i, xa in enumerate(x_a):
        for j, xb in enumerate(x_b):
            kernel[i, j] = float(kernel_fn(xa, xb))
    return kernel


def _kernel_matrix_autodiff(x, kernel_fn):
    """
    Build a differentiable square kernel matrix for a single dataset.

    Parameters
    ----------
    x
        Input array of shape ``(n_samples, n_features)``.
    kernel_fn
        Callable returning a differentiable scalar kernel value.

    Returns
    -------
    tensor-like
        Differentiable kernel matrix.
    """
    rows = []
    for xa in x:
        row = [kernel_fn(xa, xb) for xb in x]
        rows.append(qml.math.stack(row))
    return qml.math.stack(rows)


def _kernel_target_alignment(kernel_matrix, y_pm) -> Any:
    """
    Compute normalized kernel-target alignment.

    Parameters
    ----------
    kernel_matrix
        Kernel matrix ``K``.
    y_pm
        Labels encoded as ``{-1, +1}``.

    Returns
    -------
    scalar
        Normalized alignment score.
    """
    target = qml.math.outer(y_pm, y_pm)

    numerator = qml.math.sum(kernel_matrix * target)
    kernel_norm = qml.math.sqrt(qml.math.sum(kernel_matrix * kernel_matrix) + 1e-12)
    target_norm = qml.math.sqrt(qml.math.sum(target * target) + 1e-12)

    return numerator / (kernel_norm * target_norm + 1e-12)


def run_trainable_quantum_kernel_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    embedding: str = "data_reupload",
    embedding_layers: int = 2,
    steps: int = 50,
    step_size: float = 0.1,
    reg_strength: float = 1e-4,
    svc_c: float = 1.0,
    shots_train: int | None = None,
    shots_kernel: int | None = None,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    dataset: str = "moons",
) -> dict[str, Any]:
    """
    Run a trainable quantum kernel classifier on the two-moons dataset.

    The kernel is defined through a parameterized quantum feature map
    ``U_phi(x; theta)`` and trained by maximizing kernel-target alignment
    on the training set before fitting a classical SVM on the learned
    precomputed kernel matrix.

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
    embedding
        Embedding name. Supported options depend on ``qml.embeddings``.
        Trainable kernel learning is most useful with ``"data_reupload"``.
    embedding_layers
        Number of trainable embedding layers for parameterized embeddings.
    steps
        Number of optimizer steps.
    step_size
        Optimizer step size.
    reg_strength
        L2 regularization strength applied to trainable embedding parameters.
    svc_c
        SVM regularization parameter used after kernel training.
    plot
        Whether to display plots.
    save
        Whether to save results JSON and figures.

    Returns
    -------
    dict[str, Any]
        Run summary including learned parameters, kernel matrices,
        alignment trace, predictions, and accuracies.
    """
    data = make_classification_dataset(
        dataset=dataset,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = np.asarray(data["x_train"], dtype=float)
    x_test = np.asarray(data["x_test"], dtype=float)
    y_train = np.asarray(data["y_train"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)

    n_qubits = x_train.shape[1]
    wires = list(range(n_qubits))

    embedding_fn = get_embedding(embedding)
    param_shape = embedding_parameter_shape(
        embedding,
        n_layers=embedding_layers,
        n_qubits=n_qubits,
    )

    dev_train = qml.device("default.qubit", wires=n_qubits, seed=seed)
    dev_kernel = qml.device("default.qubit", wires=n_qubits, seed=seed)

    def _apply_embedding(x, params) -> None:
        if param_shape:
            embedding_fn(x, params, wires)
        else:
            embedding_fn(x, wires)
            if len(wires) > 1:
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])

    @qml.qnode(dev_train, interface="autograd")
    def kernel_circuit_train_base(x1, x2, params):
        _apply_embedding(x1, params)
        qml.adjoint(_apply_embedding)(x2, params)
        return qml.probs(wires=wires)

    @qml.qnode(dev_kernel)
    def kernel_circuit_eval_base(x1, x2, params):
        _apply_embedding(x1, params)
        qml.adjoint(_apply_embedding)(x2, params)
        return qml.probs(wires=wires)

    kernel_circuit_train = (
        qml.set_shots(kernel_circuit_train_base, shots_train)
        if shots_train is not None
        else kernel_circuit_train_base
    )

    kernel_circuit_eval = (
        qml.set_shots(kernel_circuit_eval_base, shots_kernel)
        if shots_kernel is not None
        else kernel_circuit_eval_base
    )

    def kernel_value_autodiff(x1, x2, params):

        probs = kernel_circuit_train(x1, x2, params)

        return probs[0]

    def kernel_fn(x1, x2) -> float:

        probs = kernel_circuit_eval(
            np.asarray(x1, dtype=float),
            np.asarray(x2, dtype=float),
            trained_params,
        )

        return float(probs[0])

    x_train_q = pnp.array(x_train, requires_grad=False)
    y_train_pm = pnp.array(2 * y_train - 1, requires_grad=False)

    if param_shape:
        rng = np.random.default_rng(seed)
        initial_params = 0.01 * rng.standard_normal(param_shape)
        params = pnp.array(initial_params, requires_grad=True)
    else:
        params = pnp.array(0.0, requires_grad=False)

    optimizer = qml.GradientDescentOptimizer(stepsize=step_size)
    alignment_trace: list[float] = []
    loss_trace: list[float] = []

    def objective(trainable_params):
        kernel_matrix = _kernel_matrix_autodiff(
            x_train_q,
            lambda xa, xb: kernel_value_autodiff(xa, xb, trainable_params),
        )
        alignment = _kernel_target_alignment(kernel_matrix, y_train_pm)

        if param_shape:
            penalty = reg_strength * qml.math.mean(trainable_params * trainable_params)
        else:
            penalty = 0.0

        return -alignment + penalty

    if param_shape and steps > 0:
        for _ in range(steps):
            params = optimizer.step(objective, params)
            loss_value = float(objective(params))
            alignment_value = float(
                _kernel_target_alignment(
                    _kernel_matrix_autodiff(
                        x_train_q,
                        lambda xa, xb: kernel_value_autodiff(xa, xb, params),
                    ),
                    y_train_pm,
                )
            )
            loss_trace.append(loss_value)
            alignment_trace.append(alignment_value)
    else:
        alignment_value = float(
            _kernel_target_alignment(
                _kernel_matrix_autodiff(
                    x_train_q,
                    lambda xa, xb: kernel_value_autodiff(xa, xb, params),
                ),
                y_train_pm,
            )
        )
        loss_trace.append(float(objective(params)))
        alignment_trace.append(alignment_value)

    trained_params = np.asarray(params, dtype=float)

    kernel_matrix_train = _compute_kernel_matrix(x_train, x_train, kernel_fn)
    kernel_matrix_test = _compute_kernel_matrix(x_test, x_train, kernel_fn)

    clf = SVC(kernel="precomputed", C=svc_c)
    clf.fit(kernel_matrix_train, y_train)

    y_train_pred = clf.predict(kernel_matrix_train)
    y_test_pred = clf.predict(kernel_matrix_test)

    result = {
        "model": "trainable_quantum_kernel_classifier",
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "embedding": embedding,
        "embedding_layers": embedding_layers,
        "steps": steps,
        "step_size": step_size,
        "reg_strength": reg_strength,
        "svc_c": svc_c,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "alignment_trace": alignment_trace,
        "loss_trace": loss_trace,
        "final_alignment": alignment_trace[-1],
        "final_loss": loss_trace[-1],
        "trained_params": trained_params,
        "kernel_matrix_train": kernel_matrix_train,
        "kernel_matrix_test": kernel_matrix_test,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
        "shots_train": shots_train,
        "shots_kernel": shots_kernel,
    }

    train_tag = "analytic" if shots_train is None else f"train{shots_train}"
    kernel_tag = "analytic" if shots_kernel is None else f"kernel{shots_kernel}"

    stem = (
        f"{dataset}_trainable_kernel_"
        f"emb{embedding}_"
        f"layers{embedding_layers}_"
        f"samples{n_samples}_"
        f"noise{str(noise).replace('.', 'p')}_"
        f"seed{seed}_"
        f"{train_tag}_{kernel_tag}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return results_path("trainable_kernel", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return images_path("trainable_kernel", filename)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="Trainable quantum kernel training dataset",
            show=plot,
            save_path=_images_file(f"{stem}_dataset.png") if save else None,
        )

        plot_kernel_matrix(
            kernel_matrix_train,
            title="Trainable quantum kernel matrix (train)",
            show=plot,
            save_path=_images_file(f"{stem}_kernel_train.png") if save else None,
        )

        plot_kernel_matrix(
            kernel_matrix_test,
            title="Trainable quantum kernel matrix (test vs train)",
            show=plot,
            save_path=_images_file(f"{stem}_kernel_test.png") if save else None,
        )

        plot_alignment_curve(
            alignment_trace,
            title="Trainable quantum kernel alignment",
            show=plot,
            save_path=_images_file(f"{stem}_alignment.png") if save else None,
        )

        plot_loss_curve(
            loss_trace,
            title="Trainable quantum kernel loss",
            show=plot,
            save_path=_images_file(f"{stem}_loss.png") if save else None,
        )

    if save:
        save_json(result, _results_file(f"{stem}.json"))

    return result
