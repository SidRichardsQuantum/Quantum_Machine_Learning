"""
qml.autoencoder
===============

Quantum autoencoder workflows for compressing structured quantum state families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split

from qml.ansatz import apply_hardware_efficient_ansatz, parameter_shape
from qml.io_utils import ensure_dir, images_path, results_path, save_json
from qml.optimizers import get_optimizer
from qml.training import run_training_loop
from qml.visualize import plot_loss_curve


def _make_autoencoder_dataset(
    family: str,
    n_samples: int,
    noise: float,
    test_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Generate a low-dimensional family of angle parameters for compressible states.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-0.9 * np.pi, 0.9 * np.pi, size=(n_samples, 2))

    if family == "correlated":
        x = np.column_stack(
            [
                base[:, 0],
                0.7 * base[:, 0] + 0.3 * base[:, 1],
            ]
        )
    elif family == "entangled":
        x = np.column_stack(
            [
                base[:, 0] + 0.25 * np.sin(base[:, 1]),
                base[:, 1] - 0.25 * np.cos(base[:, 0]),
            ]
        )
    elif family == "hybrid":
        x = np.column_stack(
            [
                0.5 * (base[:, 0] + base[:, 1]),
                np.sin(base[:, 0]) + 0.25 * base[:, 1],
            ]
        )
    else:
        raise ValueError(
            f"Unsupported family '{family}'. Choose from: correlated, entangled, hybrid."
        )

    if noise > 0.0:
        x = x + rng.normal(0.0, noise, size=x.shape)

    x_train, x_test = train_test_split(
        np.asarray(x, dtype=float),
        test_size=test_size,
        random_state=seed,
    )

    return {
        "x_train": np.asarray(x_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
    }


def _prepare_state(x, wires, family: str) -> None:
    """
    Prepare a structured four-qubit state family from two latent angles.
    """
    phi = x[0]
    theta = x[1]

    if family == "correlated":
        qml.RY(phi, wires=wires[0])
        qml.RY(theta, wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.CNOT(wires=[wires[1], wires[3]])
        qml.RZ(0.5 * (phi + theta), wires=wires[2])
        qml.RX(0.5 * (phi - theta), wires=wires[3])
        return

    if family == "entangled":
        qml.RY(phi, wires=wires[0])
        qml.RY(theta, wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[2], wires[3]])
        qml.RZ(phi, wires=wires[2])
        qml.RY(theta, wires=wires[3])
        return

    if family == "hybrid":
        qml.Hadamard(wires=wires[0])
        qml.RY(phi, wires=wires[1])
        qml.RX(theta, wires=wires[2])
        qml.CNOT(wires=[wires[0], wires[2]])
        qml.CNOT(wires=[wires[1], wires[3]])
        qml.RY(phi - theta, wires=wires[3])
        return

    raise ValueError(f"Unsupported family '{family}'.")


def _state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """
    Compute fidelity between two pure states.
    """
    overlap = np.vdot(state_a, state_b)
    return float(np.abs(overlap) ** 2)


def run_quantum_autoencoder(
    n_samples: int = 200,
    noise: float = 0.05,
    test_size: float = 0.25,
    seed: int = 123,
    n_layers: int = 2,
    latent_qubits: int = 2,
    steps: int = 50,
    step_size: float = 0.1,
    optimizer: str = "adam",
    optimizer_kwargs: dict[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    family: str = "correlated",
) -> dict[str, Any]:
    """
    Train a quantum autoencoder on a structured family of four-qubit states.
    """
    n_qubits = 4
    if latent_qubits <= 0 or latent_qubits >= n_qubits:
        raise ValueError("latent_qubits must be between 1 and n_qubits - 1.")

    trash_qubits = n_qubits - latent_qubits
    wires = list(range(n_qubits))
    trash_wires = wires[-trash_qubits:]

    data = _make_autoencoder_dataset(
        family=family,
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
    x_train = np.asarray(data["x_train"], dtype=float)
    x_test = np.asarray(data["x_test"], dtype=float)

    param_shape = parameter_shape(n_layers=n_layers, n_qubits=n_qubits)
    rng = np.random.default_rng(seed)
    init_params = 0.01 * rng.standard_normal(param_shape)
    params = pnp.array(init_params, requires_grad=True)

    dev_train = qml.device("default.qubit", wires=n_qubits, seed=seed)
    dev_eval = qml.device("default.qubit", wires=n_qubits, seed=seed)

    @qml.qnode(dev_train, interface="autograd")
    def trash_probs(x, current_params):
        _prepare_state(x, wires=wires, family=family)
        apply_hardware_efficient_ansatz(current_params, wires=wires)
        return qml.probs(wires=trash_wires)

    @qml.qnode(dev_eval)
    def original_state(x):
        _prepare_state(x, wires=wires, family=family)
        return qml.state()

    @qml.qnode(dev_eval)
    def reconstructed_state(x, current_params):
        _prepare_state(x, wires=wires, family=family)
        apply_hardware_efficient_ansatz(current_params, wires=wires)
        qml.adjoint(apply_hardware_efficient_ansatz)(current_params, wires=wires)
        return qml.state()

    def compression_score_single(x, current_params):
        probs = trash_probs(x, current_params)
        return probs[0]

    def compression_score_batch(x_data, current_params):
        return pnp.array([compression_score_single(x, current_params) for x in x_data])

    def cost(current_params):
        scores = compression_score_batch(x_train, current_params)
        return 1.0 - pnp.mean(scores)

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

    train_scores = np.asarray(compression_score_batch(x_train, params), dtype=float)
    test_scores = np.asarray(compression_score_batch(x_test, params), dtype=float)

    train_reconstruction = np.asarray(
        [_state_fidelity(original_state(x), reconstructed_state(x, params)) for x in x_train],
        dtype=float,
    )
    test_reconstruction = np.asarray(
        [_state_fidelity(original_state(x), reconstructed_state(x, params)) for x in x_test],
        dtype=float,
    )

    result = {
        "model": "quantum_autoencoder",
        "family": family,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "n_qubits": n_qubits,
        "latent_qubits": latent_qubits,
        "trash_qubits": trash_qubits,
        "n_layers": n_layers,
        "steps": steps,
        "step_size": step_size,
        "optimizer": optimizer,
        "optimizer_kwargs": optimizer_kwargs or {},
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "loss_history": loss_history,
        "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
        "train_compression_fidelity": float(np.mean(train_scores)),
        "test_compression_fidelity": float(np.mean(test_scores)),
        "train_reconstruction_fidelity": float(np.mean(train_reconstruction)),
        "test_reconstruction_fidelity": float(np.mean(test_reconstruction)),
        "params": np.asarray(params, dtype=float),
        "x_train": x_train,
        "x_test": x_test,
        "train_compression_scores": train_scores,
        "test_compression_scores": test_scores,
        "train_reconstruction_scores": train_reconstruction,
        "test_reconstruction_scores": test_reconstruction,
    }

    stem = (
        f"{family}_layers{n_layers}_latent{latent_qubits}_steps{steps}_samples{n_samples}"
        f"_noise{str(noise).replace('.', 'p')}_seed{seed}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path
        return results_path("autoencoder", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
            return path
        return images_path("autoencoder", filename)

    if plot or save:
        plot_loss_curve(
            loss_history,
            title="Quantum autoencoder training loss",
            show=plot,
            save_path=_images_file(f"{stem}_loss.png") if save else None,
        )

    if save:
        save_json(result, _results_file(f"{stem}.json"))

    return result
