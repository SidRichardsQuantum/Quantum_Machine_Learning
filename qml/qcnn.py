"""
qml.qcnn
========

Quantum convolutional neural network workflows for supervised classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qml.data import make_classification_dataset
from qml.io_utils import ensure_dir, images_path, results_path, save_json
from qml.metrics import accuracy_score
from qml.optimizers import get_optimizer
from qml.training import run_training_loop
from qml.visualize import plot_dataset_2d, plot_decision_boundary, plot_loss_curve


def _binary_cross_entropy_tensor(y_true, y_prob):
    """
    Compute binary cross-entropy for probabilities in [0, 1].
    """
    eps = 1e-8
    y_true = pnp.asarray(y_true, dtype=float)
    y_prob = pnp.clip(pnp.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    loss = -(y_true * pnp.log(y_prob) + (1.0 - y_true) * pnp.log(1.0 - y_prob))
    return pnp.mean(loss)


def _binary_cross_entropy(y_true, y_prob) -> float:
    """
    Compute binary cross-entropy and return a Python float.
    """
    return float(_binary_cross_entropy_tensor(y_true, y_prob))


def qcnn_parameter_shape() -> dict[str, tuple[int, ...]]:
    """
    Return the parameter shapes for the default QCNN classifier.
    """
    return {
        "embedding": (4, 3),
        "conv1": (2, 6),
        "conv2": (1, 6),
        "dense": (2,),
    }


def _apply_qcnn_embedding(x, params, wires) -> None:
    """
    Apply a small trainable data embedding across four qubits.
    """
    x = qml.math.ravel(qml.math.asarray(x))
    n_features = qml.math.shape(x)[0]

    for idx, wire in enumerate(wires):
        feature = x[idx % n_features]
        qml.RX(feature + params[idx, 0], wires=wire)
        qml.RY(feature + params[idx, 1], wires=wire)
        qml.RZ(params[idx, 2], wires=wire)


def _apply_convolution_block(params, wire_a: int, wire_b: int) -> None:
    """
    Apply a shared two-qubit convolution block.
    """
    qml.RY(params[0], wires=wire_a)
    qml.RZ(params[1], wires=wire_a)
    qml.RY(params[2], wires=wire_b)
    qml.RZ(params[3], wires=wire_b)
    qml.CNOT(wires=[wire_a, wire_b])
    qml.RY(params[4], wires=wire_b)
    qml.RZ(params[5], wires=wire_a)


def _apply_pooling_block(source: int, target: int) -> None:
    """
    Apply a lightweight pooling operation from source to target.
    """
    qml.CNOT(wires=[source, target])
    qml.PauliX(wires=source)


def _apply_qcnn(params) -> None:
    """
    Apply the default QCNN hierarchy on four qubits.
    """
    _apply_convolution_block(params["conv1"][0], 0, 1)
    _apply_convolution_block(params["conv1"][1], 2, 3)
    _apply_pooling_block(0, 1)
    _apply_pooling_block(2, 3)
    _apply_convolution_block(params["conv2"][0], 1, 3)
    qml.RY(params["dense"][0], wires=3)
    qml.RZ(params["dense"][1], wires=3)


def _flatten_params(params: dict[str, np.ndarray]) -> tuple[pnp.tensor, dict[str, tuple[int, ...]]]:
    """
    Flatten structured QCNN parameters into a single trainable vector.
    """
    flat_parts: list[np.ndarray] = []
    shapes: dict[str, tuple[int, ...]] = {}

    for name in ("embedding", "conv1", "conv2", "dense"):
        value = np.asarray(params[name], dtype=float)
        flat_parts.append(value.ravel())
        shapes[name] = value.shape

    flat = np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=float)
    return pnp.array(flat, requires_grad=True), shapes


def _unpack_params(flat_params, shapes: dict[str, tuple[int, ...]]) -> dict[str, pnp.tensor]:
    """
    Restore structured QCNN parameters from a flat vector.
    """
    offset = 0
    unpacked: dict[str, pnp.tensor] = {}

    for name in ("embedding", "conv1", "conv2", "dense"):
        shape = shapes[name]
        size = int(np.prod(shape))
        unpacked[name] = pnp.reshape(flat_params[offset : offset + size], shape)
        offset += size

    return unpacked


def run_qcnn(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    steps: int = 50,
    step_size: float = 0.1,
    optimizer: str = "adam",
    optimizer_kwargs: dict[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    shots: int | None = None,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    dataset: str = "moons",
) -> dict[str, Any]:
    """
    Train a small QCNN classifier on a synthetic binary classification dataset.
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

    n_qubits = 4
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits, seed=seed)

    param_shapes = qcnn_parameter_shape()
    rng = np.random.default_rng(seed)
    init_structured = {
        name: 0.01 * rng.standard_normal(shape) for name, shape in param_shapes.items()
    }
    params, shapes = _flatten_params(init_structured)

    @qml.qnode(dev, interface="autograd")
    def circuit_base(x, flat_params):
        structured = _unpack_params(flat_params, shapes)
        _apply_qcnn_embedding(x, structured["embedding"], wires)
        _apply_qcnn(structured)
        return qml.expval(qml.PauliZ(3))

    circuit = qml.set_shots(circuit_base, shots) if shots is not None else circuit_base

    def predict_proba_single(x, flat_params):
        return 0.5 * (1.0 - circuit(x, flat_params))

    def predict_proba_batch(x_data, flat_params):
        return pnp.array([predict_proba_single(x, flat_params) for x in x_data])

    def cost(flat_params):
        probs = predict_proba_batch(x_train, flat_params)
        return _binary_cross_entropy_tensor(y_train, probs)

    opt = get_optimizer(
        optimizer,
        stepsize=step_size,
        **(optimizer_kwargs or {}),
    )

    def step_fn(current_params):
        return opt.step_and_cost(cost, current_params)

    params, loss_history = run_training_loop(
        step_fn,
        params,
        steps,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )

    train_probs = np.asarray(predict_proba_batch(x_train, params), dtype=float)
    test_probs = np.asarray(predict_proba_batch(x_test, params), dtype=float)

    y_train_pred = (train_probs >= 0.5).astype(int)
    y_test_pred = (test_probs >= 0.5).astype(int)

    structured_params = _unpack_params(params, shapes)

    result = {
        "model": "qcnn",
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "steps": steps,
        "step_size": step_size,
        "optimizer": optimizer,
        "optimizer_kwargs": optimizer_kwargs or {},
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "shots": shots,
        "loss_history": loss_history,
        "final_loss": _binary_cross_entropy(y_train, train_probs),
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "params": np.asarray(params, dtype=float),
        "embedding_params": np.asarray(structured_params["embedding"], dtype=float),
        "conv1_params": np.asarray(structured_params["conv1"], dtype=float),
        "conv2_params": np.asarray(structured_params["conv2"], dtype=float),
        "dense_params": np.asarray(structured_params["dense"], dtype=float),
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_probabilities": train_probs,
        "test_probabilities": test_probs,
    }

    shots_tag = "analytic" if shots is None else f"shots{shots}"
    stem = (
        f"{dataset}_steps{steps}_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}_seed{seed}_{shots_tag}"
    )

    def predict_proba_grid(x_grid):
        return np.asarray([predict_proba_single(xi, params) for xi in x_grid], dtype=float)

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path
        return results_path("qcnn", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
            return path
        return images_path("qcnn", filename)

    if plot or save:
        plot_dataset_2d(
            x_train,
            y_train,
            title="QCNN training dataset",
            show=plot,
            save_path=_images_file(f"{stem}_dataset.png") if save else None,
        )

        plot_loss_curve(
            loss_history,
            title="QCNN training loss",
            show=plot,
            save_path=_images_file(f"{stem}_loss.png") if save else None,
        )

        plot_decision_boundary(
            predict_proba_grid,
            x_train,
            y_train,
            title="QCNN decision boundary",
            show=plot,
            save_path=_images_file(f"{stem}_decision_boundary.png") if save else None,
        )

    if save:
        save_json(result, _results_file(f"{stem}.json"))

    return result
