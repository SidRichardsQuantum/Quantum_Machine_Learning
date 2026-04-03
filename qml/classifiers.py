"""
qml.classifiers
===============

Classifier workflows for supervised quantum machine learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qml.ansatz import apply_hardware_efficient_ansatz, parameter_shape
from qml.data import make_moons_dataset
from qml.embeddings import apply_angle_embedding
from qml.io_utils import images_path, results_path, save_json
from qml.metrics import accuracy_score
from qml.visualize import (
    plot_dataset_2d,
    plot_decision_boundary,
    plot_loss_curve,
)


def _binary_cross_entropy(y_true, y_prob) -> float:
    """
    Compute binary cross-entropy for probabilities in [0, 1].
    """
    eps = 1e-8
    y_true = pnp.asarray(y_true, dtype=float)
    y_prob = pnp.clip(pnp.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    loss = -(y_true * pnp.log(y_prob) + (1.0 - y_true) * pnp.log(1.0 - y_prob))
    return float(pnp.mean(loss))


def run_vqc(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    n_layers: int = 2,
    steps: int = 50,
    step_size: float = 0.1,
    plot: bool = False,
    save: bool = False,
) -> dict[str, Any]:
    """
    Train a minimal variational quantum classifier on a two-moons dataset.

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
    n_layers
        Number of variational layers.
    steps
        Number of optimizer steps.
    step_size
        Optimizer step size.
    plot
        Whether to show generated plots.
    save
        Whether to save JSON results and plot outputs.

    Returns
    -------
    dict[str, Any]
        Run summary including fitted parameters, loss history, and accuracies.
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

    @qml.qnode(dev, interface="autograd")
    def circuit(x, params):
        apply_angle_embedding(x, wires=wires)
        apply_hardware_efficient_ansatz(params, wires=wires)
        return qml.expval(qml.PauliZ(wires[0]))

    def predict_proba_single(x, params):
        return 0.5 * (1.0 - circuit(x, params))

    def predict_proba_batch(x_data, params):
        return pnp.array([predict_proba_single(x, params) for x in x_data])

    def cost(params):
        probs = predict_proba_batch(x_train, params)
        return _binary_cross_entropy(y_train, probs)

    rng = np.random.default_rng(seed)
    init_params = 0.01 * rng.standard_normal(parameter_shape(n_layers=n_layers, n_qubits=n_qubits))
    params = pnp.array(init_params, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=step_size)
    loss_history: list[float] = []

    for _ in range(steps):
        params, current_loss = opt.step_and_cost(cost, params)
        loss_history.append(float(current_loss))

    train_probs = np.asarray(predict_proba_batch(x_train, params), dtype=float)
    test_probs = np.asarray(predict_proba_batch(x_test, params), dtype=float)

    y_train_pred = (train_probs >= 0.5).astype(int)
    y_test_pred = (test_probs >= 0.5).astype(int)

    result = {
        "model": "vqc",
        "dataset": "moons",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "steps": steps,
        "step_size": step_size,
        "loss_history": loss_history,
        "final_loss": _binary_cross_entropy(y_train, train_probs),
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "params": np.asarray(params, dtype=float),
        "y_train": np.asarray(y_train, dtype=int),
        "y_test": np.asarray(y_test, dtype=int),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_probabilities": train_probs,
        "test_probabilities": test_probs,
    }

    stem = (
        f"moons_layers{n_layers}_steps{steps}_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}_seed{seed}"
    )

    def predict_proba_grid(x_grid):
        return np.asarray([predict_proba_single(xi, params) for xi in x_grid], dtype=float)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            show=plot,
            save_path=images_path("vqc", f"{stem}_dataset.png") if save else None,
        )

        plot_loss_curve(
            loss_history,
            show=plot,
            save_path=images_path("vqc", f"{stem}_loss.png") if save else None,
        )

        plot_decision_boundary(
            predict_proba_grid,
            x_train,
            y_train,
            show=plot,
            save_path=images_path("vqc", f"{stem}_decision_boundary.png") if save else None,
        )

    if save:
        save_json(result, results_path("vqc", f"{stem}.json"))

    return result
