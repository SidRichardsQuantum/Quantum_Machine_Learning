"""
qml.regression
==============

Regression workflows for supervised quantum machine learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qml.ansatz import apply_hardware_efficient_ansatz, parameter_shape
from qml.data import make_regression_dataset
from qml.embeddings import apply_angle_embedding
from qml.io_utils import images_path, results_path, save_json
from qml.io_utils import ensure_dir
from qml.metrics import mean_absolute_error, mean_squared_error
from qml.visualize import (
    plot_dataset_2d,
    plot_loss_curve,
    plot_regression_predictions,
)
from qml.training import run_training_loop


def run_vqr(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    n_layers: int = 2,
    steps: int = 50,
    step_size: float = 0.1,
    shots: int | None = None,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    dataset: str = "linear",
) -> dict[str, Any]:
    """
    Train a minimal variational quantum regressor on a synthetic regression dataset.

    Parameters
    ----------
    n_samples
        Number of dataset samples.
    noise
        Noise level used by ``make_regression``.
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
    shots
        Number of measurement shots. If ``None``, uses analytic mode.
    plot
        Whether to display plots.
    save
        Whether to save results JSON and figures.

    Returns
    -------
    dict[str, Any]
        Run summary including fitted parameters, predictions, and regression metrics.
    """
    data = make_regression_dataset(
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

    @qml.qnode(dev, interface="autograd")
    def circuit_base(x, params):
        apply_angle_embedding(x, wires=wires)
        apply_hardware_efficient_ansatz(params, wires=wires)
        return qml.expval(qml.PauliZ(wires[0]))

    circuit = qml.set_shots(circuit_base, shots) if shots is not None else circuit_base

    def predict_single(x, params):
        return circuit(x, params)

    def predict_batch(x_data, params):
        return pnp.array([predict_single(x, params) for x in x_data])

    def cost(params):
        preds = predict_batch(x_train, params)
        targets = pnp.asarray(y_train, dtype=float)
        return pnp.mean((preds - targets) ** 2)

    rng = np.random.default_rng(seed)
    init_params = 0.01 * rng.standard_normal(parameter_shape(n_layers=n_layers, n_qubits=n_qubits))
    params = pnp.array(init_params, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=step_size)

    def step_fn(params):
        return opt.step_and_cost(cost, params)

    params, loss_history = run_training_loop(step_fn, params, steps)

    y_train_pred = np.asarray(predict_batch(x_train, params), dtype=float)
    y_test_pred = np.asarray(predict_batch(x_test, params), dtype=float)

    result = {
        "model": "vqr",
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "steps": steps,
        "step_size": step_size,
        "shots": shots,
        "loss_history": loss_history,
        "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "params": np.asarray(params, dtype=float),
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }

    shots_tag = "analytic" if shots is None else f"shots{shots}"
    stem = (
        f"{dataset}_layers{n_layers}_steps{steps}_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}_seed{seed}_{shots_tag}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path
        return results_path(f"{dataset}", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
            return path
        return images_path(f"{dataset}", filename)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title=f"{dataset} training dataset",
            show=plot,
            save_path=_images_file(f"{stem}_dataset.png") if save else None,
        )

        plot_loss_curve(
            loss_history,
            title="VQR training loss",
            show=plot,
            save_path=_images_file(f"{stem}_loss.png") if save else None,
        )

        plot_regression_predictions(
            y_test,
            y_test_pred,
            title="VQR test predictions",
            show=plot,
            save_path=_images_file(f"{stem}_predictions.png") if save else None,
        )

    if save:
        save_json(result, _results_file(f"{stem}.json"))

    return result
