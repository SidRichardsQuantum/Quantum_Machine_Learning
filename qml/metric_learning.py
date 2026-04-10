"""
qml.metric_learning
===================

Quantum metric learning utilities built on PennyLane.

This module implements a simple supervised quantum metric learning workflow:
- a trainable quantum embedding circuit
- pairwise contrastive training on labelled data
- nearest-centroid classification in the learned feature space

The current implementation is intentionally lightweight and self-contained so it
fits naturally into the existing package structure and synthetic demo workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qml.io_utils import ensure_dir, images_path, results_path, save_json
from qml.visualize import plot_loss_curve, plot_metric_learning_embeddings

ArrayLike = np.ndarray


@dataclass
class QuantumMetricLearningConfig:
    """Configuration for quantum metric learning."""

    dataset: Literal["moons", "circles", "blobs"] = "moons"
    samples: int = 120
    test_size: float = 0.25
    seed: int = 42
    layers: int = 2
    steps: int = 100
    stepsize: float = 0.05
    margin: float = 0.5
    pairs_per_step: int = 32
    log_every: int = 10
    scale_data: bool = True
    plot: bool = False


@dataclass
class QuantumMetricLearningResult:
    """Outputs from quantum metric learning."""

    train_accuracy: float
    test_accuracy: float
    loss_history: list[float]
    params: ArrayLike
    train_embeddings: ArrayLike
    test_embeddings: ArrayLike
    train_centroids: dict[int, ArrayLike]
    y_train: ArrayLike
    y_test: ArrayLike


def _make_dataset(
    name: str,
    samples: int,
    seed: int,
) -> tuple[ArrayLike, ArrayLike]:
    """Generate a small synthetic classification dataset."""
    if name == "moons":
        x, y = make_moons(n_samples=samples, noise=0.15, random_state=seed)
    elif name == "circles":
        x, y = make_circles(
            n_samples=samples,
            noise=0.08,
            factor=0.45,
            random_state=seed,
        )
    elif name == "blobs":
        x, y = make_blobs(
            n_samples=samples,
            centers=2,
            n_features=2,
            cluster_std=1.25,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: moons, circles, blobs.")

    return x.astype(float), y.astype(int)


def _angle_scale(x: ArrayLike) -> ArrayLike:
    """Map features to a bounded angle range."""
    return np.pi * np.tanh(x)


def _pairwise_indices(
    y: ArrayLike,
    rng: np.random.Generator,
    num_pairs: int,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Sample labelled pairs.

    Returns arrays i_idx, j_idx, same_label where same_label is 1.0 if the pair
    belongs to the same class and 0.0 otherwise.
    """
    n = len(y)
    if n < 2:
        raise ValueError("At least two samples are required to form training pairs.")

    class_to_indices: dict[int, ArrayLike] = {
        int(label): np.where(y == label)[0] for label in np.unique(y)
    }

    i_list: list[int] = []
    j_list: list[int] = []
    s_list: list[float] = []

    available_same = [c for c, idx in class_to_indices.items() if len(idx) >= 2]
    available_diff = list(class_to_indices.keys())

    if not available_same:
        raise ValueError("Need at least one class with two or more samples.")

    for k in range(num_pairs):
        choose_same = (k % 2 == 0) or len(available_diff) < 2

        if choose_same:
            cls = int(rng.choice(available_same))
            idx = class_to_indices[cls]
            i, j = rng.choice(idx, size=2, replace=False)
            s = 1.0
        else:
            cls_i, cls_j = rng.choice(available_diff, size=2, replace=False)
            i = int(rng.choice(class_to_indices[int(cls_i)]))
            j = int(rng.choice(class_to_indices[int(cls_j)]))
            s = 0.0

        i_list.append(int(i))
        j_list.append(int(j))
        s_list.append(float(s))

    return (
        np.asarray(i_list, dtype=int),
        np.asarray(j_list, dtype=int),
        np.asarray(s_list, dtype=float),
    )


def _embedding_template(
    x: ArrayLike,
    params: ArrayLike,
    n_qubits: int,
) -> None:
    """Trainable data re-uploading style embedding."""
    x_angles = _angle_scale(x)

    for layer in range(params.shape[0]):
        for wire in range(n_qubits):
            feature = x_angles[wire % len(x_angles)]
            qml.RX(feature + params[layer, wire, 0], wires=wire)
            qml.RY(feature + params[layer, wire, 1], wires=wire)
            qml.RZ(params[layer, wire, 2], wires=wire)

        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])

        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])


def _build_embedding_qnode(n_qubits: int):
    """Create an embedding QNode returning Pauli-Z expectations."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def embedding_qnode(x: ArrayLike, params: ArrayLike) -> ArrayLike:
        _embedding_template(x, params, n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return embedding_qnode


def _contrastive_loss(
    embeds_i: ArrayLike,
    embeds_j: ArrayLike,
    same_label: ArrayLike,
    margin: float,
) -> pnp.tensor:
    """
    Contrastive loss over embedding vectors.

    Same-class pairs are pulled together.
    Different-class pairs are pushed apart beyond a margin.
    """
    dists = pnp.sqrt(pnp.sum((embeds_i - embeds_j) ** 2, axis=1) + 1e-8)
    positive_term = same_label * (dists**2)
    negative_term = (1.0 - same_label) * pnp.maximum(0.0, margin - dists) ** 2
    return pnp.mean(positive_term + negative_term)


def _compute_embeddings(
    x: ArrayLike,
    params: ArrayLike,
    embed_fn,
) -> ArrayLike:
    """Compute learned embeddings for a batch of samples."""
    embs = [np.asarray(embed_fn(sample, params), dtype=float) for sample in x]
    return np.asarray(embs, dtype=float)


def _fit_centroids(
    embeddings: ArrayLike,
    y: ArrayLike,
) -> dict[int, ArrayLike]:
    """Fit class centroids in learned embedding space."""
    centroids: dict[int, ArrayLike] = {}
    for cls in np.unique(y):
        cls = int(cls)
        centroids[cls] = embeddings[y == cls].mean(axis=0)
    return centroids


def _predict_nearest_centroid(
    embeddings: ArrayLike,
    centroids: dict[int, ArrayLike],
) -> ArrayLike:
    """Predict class labels by nearest centroid."""
    classes = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[c] for c in classes])
    dists = np.linalg.norm(
        embeddings[:, None, :] - centroid_matrix[None, :, :],
        axis=2,
    )
    preds = np.argmin(dists, axis=1)
    return np.asarray([classes[idx] for idx in preds], dtype=int)


def run_quantum_metric_learner(
    dataset: Literal["moons", "circles", "blobs"] = "moons",
    samples: int = 120,
    test_size: float = 0.25,
    seed: int = 42,
    layers: int = 2,
    steps: int = 100,
    stepsize: float = 0.05,
    margin: float = 0.5,
    pairs_per_step: int = 32,
    log_every: int = 10,
    scale_data: bool = True,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> QuantumMetricLearningResult:
    """
    Train a simple quantum metric learner and evaluate it.

    Parameters
    ----------
    dataset:
        Synthetic dataset name: "moons", "circles", or "blobs".
    samples:
        Total number of samples to generate.
    test_size:
        Fraction reserved for the test split.
    seed:
        Random seed for dataset generation, splitting, and pair sampling.
    layers:
        Number of trainable re-uploading layers in the embedding circuit.
    steps:
        Number of optimisation steps.
    stepsize:
        Optimiser learning rate.
    margin:
        Contrastive loss margin for negative pairs.
    pairs_per_step:
        Number of sampled training pairs per optimisation step.
    log_every:
        Print progress every `log_every` steps.
    scale_data:
        Standardise features before angle encoding.
    plot:
        Whether to plot the training loss curve.
    save:
        Whether to save JSON results and generated figures.
    results_dir:
        Optional override for result output directory.
    images_dir:
        Optional override for figure output directory.

    Returns
    -------
    QuantumMetricLearningResult
        Train/test scores, learned parameters, embeddings, and centroids.
    """
    config = QuantumMetricLearningConfig(
        dataset=dataset,
        samples=samples,
        test_size=test_size,
        seed=seed,
        layers=layers,
        steps=steps,
        stepsize=stepsize,
        margin=margin,
        pairs_per_step=pairs_per_step,
        log_every=log_every,
        scale_data=scale_data,
        plot=plot,
    )

    x, y = _make_dataset(config.dataset, config.samples, config.seed)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    if config.scale_data:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    n_qubits = max(2, x_train.shape[1])
    embed_qnode = _build_embedding_qnode(n_qubits)

    rng = np.random.default_rng(config.seed)
    params = pnp.array(
        0.01 * rng.standard_normal((config.layers, n_qubits, 3)),
        requires_grad=True,
    )

    opt = qml.AdamOptimizer(stepsize=config.stepsize)
    loss_history: list[float] = []

    def objective(current_params: ArrayLike) -> pnp.tensor:
        i_idx, j_idx, same_label = _pairwise_indices(
            y_train,
            rng=rng,
            num_pairs=config.pairs_per_step,
        )

        embeds_i = pnp.stack([embed_qnode(x_train[idx], current_params) for idx in i_idx])
        embeds_j = pnp.stack([embed_qnode(x_train[idx], current_params) for idx in j_idx])

        return _contrastive_loss(
            embeds_i=embeds_i,
            embeds_j=embeds_j,
            same_label=pnp.array(same_label),
            margin=config.margin,
        )

    for step in range(config.steps):
        params, loss = opt.step_and_cost(objective, params)
        loss_value = float(loss)
        loss_history.append(loss_value)

        if config.log_every > 0 and (
            step == 0 or (step + 1) % config.log_every == 0 or step == config.steps - 1
        ):
            print(f"[metric_learning] step={step + 1:04d} loss={loss_value:.6f}")

    train_embeddings = _compute_embeddings(x_train, params, embed_qnode)
    test_embeddings = _compute_embeddings(x_test, params, embed_qnode)

    train_centroids = _fit_centroids(train_embeddings, y_train)
    y_train_pred = _predict_nearest_centroid(train_embeddings, train_centroids)
    y_test_pred = _predict_nearest_centroid(test_embeddings, train_centroids)

    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))

    result = QuantumMetricLearningResult(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        loss_history=loss_history,
        params=np.asarray(params, dtype=float),
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_centroids=train_centroids,
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
    )

    stem = (
        f"{dataset}_layers{layers}_steps{steps}_samples{samples}"
        f"_margin{str(margin).replace('.', 'p')}_seed{seed}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path
        return results_path("metric_learning", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
            return path
        return images_path("metric_learning", filename)

    if config.plot or save:
        plot_loss_curve(
            loss_history,
            title="Quantum metric learning loss",
            show=config.plot,
            save_path=_images_file(f"{stem}_loss.png") if save else None,
        )
        if train_embeddings.shape[1] == 2:
            plot_metric_learning_embeddings(
                train_embeddings=train_embeddings,
                y_train=y_train,
                test_embeddings=test_embeddings,
                y_test=y_test,
                centroids=train_centroids,
                title="Quantum metric learning embeddings",
                show=config.plot,
                save_path=_images_file(f"{stem}_embeddings.png") if save else None,
            )

    if save:
        save_json(
            {
                "model": "quantum_metric_learning",
                "dataset": dataset,
                "config": asdict(config),
                "result": asdict(result),
            },
            _results_file(f"{stem}.json"),
        )

    return result
