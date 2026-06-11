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
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC

from qml.circuit_metadata import circuit_metadata, embedding_parameter_count
from qml.data import make_classification_dataset
from qml.data import make_regression_dataset
from qml.embeddings import embedding_parameter_shape, get_embedding
from qml.io_utils import ensure_dir, images_path, results_path, save_json
from qml.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from qml.noise import (
    apply_noise_channels,
    device_name_for_noise,
    noise_model_tag,
    noise_model_to_dict,
)
from qml.optimizers import get_optimizer
from qml.training import run_training_loop
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


def _regression_target_alignment(kernel_matrix, y) -> Any:
    """Compute normalized alignment against a continuous centered target kernel."""
    y = y - qml.math.mean(y)
    target = qml.math.outer(y, y)
    numerator = qml.math.sum(kernel_matrix * target)
    kernel_norm = qml.math.sqrt(qml.math.sum(kernel_matrix * kernel_matrix) + 1e-12)
    target_norm = qml.math.sqrt(qml.math.sum(target * target) + 1e-12)
    return numerator / (kernel_norm * target_norm + 1e-12)


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {x.shape}.")
    return x


class TrainableQuantumKernelRegressor:
    """
    Kernel-ridge regressor with a trainable quantum feature map.

    The feature-map parameters are optimized by maximizing normalized alignment
    between the quantum kernel matrix and the continuous target kernel ``y y^T``.
    """

    def __init__(
        self,
        *,
        embedding: str = "data_reupload",
        embedding_layers: int = 2,
        steps: int = 50,
        step_size: float = 0.1,
        optimizer: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        reg_strength: float = 1e-4,
        alpha: float = 1.0,
        shots_train: int | None = None,
        shots_kernel: int | None = None,
        noise_model: dict[str, float] | None = None,
        seed: int = 123,
    ) -> None:
        self.embedding = embedding
        self.embedding_layers = embedding_layers
        self.steps = steps
        self.step_size = step_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.reg_strength = reg_strength
        self.alpha = alpha
        self.shots_train = shots_train
        self.shots_kernel = shots_kernel
        self.noise_model = noise_model_to_dict(noise_model)
        self.seed = seed

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "embedding": self.embedding,
            "embedding_layers": self.embedding_layers,
            "steps": self.steps,
            "step_size": self.step_size,
            "optimizer": self.optimizer,
            "optimizer_kwargs": dict(self.optimizer_kwargs),
            "reg_strength": self.reg_strength,
            "alpha": self.alpha,
            "shots_train": self.shots_train,
            "shots_kernel": self.shots_kernel,
            "noise_model": self.noise_model,
            "seed": self.seed,
        }

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for TrainableQuantumKernelRegressor.")
            if key == "noise_model":
                value = noise_model_to_dict(value)
            setattr(self, key, value)
        return self

    def _build_kernel_functions(self, n_qubits: int):
        wires = list(range(n_qubits))
        embedding_name = self.embedding.strip().lower()
        embedding_fn = get_embedding(embedding_name)
        param_shape = embedding_parameter_shape(
            embedding_name,
            n_layers=self.embedding_layers,
            n_qubits=n_qubits,
        )
        is_trainable = bool(param_shape)
        if not is_trainable:
            if self.steps > 0:
                raise ValueError(
                    f"Embedding '{embedding_name}' is not trainable. Set steps=0 or use a "
                    "trainable embedding."
                )
            param_shape = (1,)

        dev_train = qml.device(
            device_name_for_noise(self.noise_model),
            wires=n_qubits,
            seed=self.seed,
        )
        dev_eval = qml.device(
            device_name_for_noise(self.noise_model),
            wires=n_qubits,
            seed=self.seed,
        )

        def apply_embedding(x, params) -> None:
            if is_trainable:
                embedding_fn(x, params, wires=wires)
            else:
                embedding_fn(x, wires=wires)

        @qml.qnode(dev_train, interface="autograd")
        def kernel_train_base(x1, x2, params):
            apply_embedding(x1, params)
            qml.adjoint(apply_embedding)(x2, params)
            apply_noise_channels(wires, self.noise_model, readout_wires=wires)
            return qml.probs(wires=wires)

        @qml.qnode(dev_eval)
        def kernel_eval_base(x1, x2, params):
            apply_embedding(x1, params)
            qml.adjoint(apply_embedding)(x2, params)
            apply_noise_channels(wires, self.noise_model, readout_wires=wires)
            return qml.probs(wires=wires)

        kernel_train = (
            qml.set_shots(kernel_train_base, self.shots_train)
            if self.shots_train is not None
            else kernel_train_base
        )
        kernel_eval = (
            qml.set_shots(kernel_eval_base, self.shots_kernel)
            if self.shots_kernel is not None
            else kernel_eval_base
        )

        return embedding_name, param_shape, is_trainable, kernel_train, kernel_eval

    def fit(self, x, y):
        x = _as_2d(x)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("y length must match the number of samples.")

        (
            self.embedding_name_,
            param_shape,
            is_trainable,
            kernel_train,
            kernel_eval,
        ) = self._build_kernel_functions(x.shape[1])

        def kernel_value_autodiff(x1, x2, params):
            return kernel_train(x1, x2, params)[0]

        x_q = pnp.array(x, requires_grad=False)
        y_q = pnp.array(y, requires_grad=False)
        rng = np.random.default_rng(self.seed)
        init_params = (
            0.01 * rng.standard_normal(param_shape) if is_trainable else np.zeros(param_shape)
        )
        params = pnp.array(init_params, requires_grad=is_trainable)
        opt = get_optimizer(self.optimizer, stepsize=self.step_size, **self.optimizer_kwargs)

        def objective(trainable_params):
            kernel_matrix = _kernel_matrix_autodiff(
                x_q,
                lambda xa, xb: kernel_value_autodiff(xa, xb, trainable_params),
            )
            alignment = _regression_target_alignment(kernel_matrix, y_q)
            penalty = self.reg_strength * qml.math.mean(trainable_params * trainable_params)
            return -alignment + penalty

        def step_fn(current_params):
            return opt.step_and_cost(objective, current_params)

        if self.steps > 0 and is_trainable:
            params, self.loss_trace_ = run_training_loop(step_fn, params, self.steps)
        else:
            self.loss_trace_ = [float(objective(params))]

        self.trained_params_ = np.asarray(params, dtype=float)

        def kernel_fn(x1, x2) -> float:
            return float(
                kernel_eval(
                    np.asarray(x1, dtype=float), np.asarray(x2, dtype=float), self.trained_params_
                )[0]
            )

        self.kernel_matrix_train_ = _compute_kernel_matrix(x, x, kernel_fn)
        self.alignment_ = float(_regression_target_alignment(self.kernel_matrix_train_, y))
        self.model_ = KernelRidge(alpha=self.alpha, kernel="precomputed")
        self.model_.fit(self.kernel_matrix_train_, y)
        self.x_train_ = x
        self.n_features_in_ = x.shape[1]
        self._kernel_fn_ = kernel_fn
        self.circuit_metadata_ = circuit_metadata(
            model="trainable_quantum_kernel_regressor",
            n_qubits=self.n_features_in_,
            n_layers=1,
            embedding=self.embedding_name_,
            embedding_layers=self.embedding_layers,
            ansatz=None,
            template="trainable_kernel",
            trainable_parameters=int(np.asarray(self.trained_params_).size if is_trainable else 0),
            extra={
                "shots_train": self.shots_train,
                "shots_kernel": self.shots_kernel,
                "noise_model": self.noise_model,
                "alignment": self.alignment_,
            },
        )
        return self

    def predict(self, x) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("TrainableQuantumKernelRegressor must be fitted before prediction.")
        x = _as_2d(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {x.shape[1]}.")
        k_test = _compute_kernel_matrix(x, self.x_train_, self._kernel_fn_)
        return np.asarray(self.model_.predict(k_test), dtype=float)

    def score(self, x, y) -> float:
        return -mean_squared_error(y, self.predict(x))

    def mean_absolute_error(self, x, y) -> float:
        return mean_absolute_error(y, self.predict(x))


def run_trainable_quantum_kernel_classifier(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    embedding: str = "data_reupload",
    embedding_layers: int = 2,
    steps: int = 50,
    optimizer: str = "adam",
    optimizer_kwargs: dict[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    step_size: float = 0.1,
    reg_strength: float = 1e-4,
    svc_c: float = 1.0,
    shots_train: int | None = None,
    shots_kernel: int | None = None,
    noise_model: dict[str, float] | None = None,
    plot: bool = False,
    save: bool = False,
    results_dir: str | Path | None = None,
    images_dir: str | Path | None = None,
    dataset: str = "moons",
) -> dict[str, Any]:
    """
    Run a trainable quantum kernel classifier.

    The kernel is defined through a parameterized quantum feature map
    ``U_phi(x; theta)`` and trained by maximizing kernel-target alignment
    on the training set before fitting a classical SVM on the learned
    precomputed kernel matrix.

    Parameters
    ----------
    n_samples
        Number of dataset samples.
    noise
        Noise level used by the dataset generator.
    test_size
        Fraction reserved for test data.
    seed
        Random seed.
    embedding
        Embedding name. Supported options depend on ``qml.embeddings``.
    embedding_layers
        Number of trainable embedding layers.
    steps
        Number of optimizer steps.
    optimizer
        Optimizer name understood by ``qml.optimizers.get_optimizer``.
    optimizer_kwargs
        Optional keyword arguments forwarded to the optimizer constructor.
    early_stopping_patience
        Number of consecutive non-improving steps allowed before stopping.
    early_stopping_min_delta
        Minimum loss decrease required to count as an improvement.
    step_size
        Optimizer step size.
    reg_strength
        L2 regularization strength applied to trainable embedding parameters.
    svc_c
        SVM regularization parameter used after kernel training.
    shots_train
        Shot count used during training-time kernel optimization. ``None``
        means analytic expectations.
    shots_kernel
        Shot count used for final kernel matrix evaluation. ``None`` means
        analytic expectations.
    plot
        Whether to display plots.
    save
        Whether to save results JSON and figures.
    results_dir
        Optional override for results output directory.
    images_dir
        Optional override for image output directory.
    dataset
        Classification dataset name.

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

    embedding_name = embedding.strip().lower()
    embedding_fn = get_embedding(embedding_name)
    param_shape = embedding_parameter_shape(
        embedding_name,
        n_layers=embedding_layers,
        n_qubits=n_qubits,
    )

    is_trainable = bool(param_shape)

    if not is_trainable:
        if steps > 0:
            raise ValueError(
                f"Embedding '{embedding_name}' is not trainable. "
                "Trainable quantum kernel learning requires an embedding with "
                "trainable parameters."
            )

        param_shape = (1,)

    noise_model = noise_model_to_dict(noise_model)
    dev_train = qml.device(device_name_for_noise(noise_model), wires=n_qubits, seed=seed)
    dev_kernel = qml.device(device_name_for_noise(noise_model), wires=n_qubits, seed=seed)

    def apply_embedding(x, params) -> None:
        if is_trainable:
            embedding_fn(x, params, wires=wires)
        else:
            embedding_fn(x, wires=wires)

    @qml.qnode(dev_train, interface="autograd")
    def kernel_circuit_train_base(x1, x2, params):
        apply_embedding(x1, params)
        qml.adjoint(apply_embedding)(x2, params)
        apply_noise_channels(wires, noise_model, readout_wires=wires)
        return qml.probs(wires=wires)

    @qml.qnode(dev_kernel)
    def kernel_circuit_eval_base(x1, x2, params):
        apply_embedding(x1, params)
        qml.adjoint(apply_embedding)(x2, params)
        apply_noise_channels(wires, noise_model, readout_wires=wires)
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

    x_train_q = pnp.array(x_train, requires_grad=False)
    y_train_pm = pnp.array(2 * y_train - 1, requires_grad=False)

    rng = np.random.default_rng(seed)
    init_params = 0.01 * rng.standard_normal(param_shape)
    params = pnp.array(init_params, requires_grad=True)

    opt = get_optimizer(
        optimizer,
        stepsize=step_size,
        **(optimizer_kwargs or {}),
    )

    def objective(trainable_params):
        kernel_matrix = _kernel_matrix_autodiff(
            x_train_q,
            lambda xa, xb: kernel_value_autodiff(xa, xb, trainable_params),
        )
        alignment = _kernel_target_alignment(kernel_matrix, y_train_pm)
        penalty = reg_strength * qml.math.mean(trainable_params * trainable_params)
        return -alignment + penalty

    def alignment_value(trainable_params) -> float:
        kernel_matrix = _kernel_matrix_autodiff(
            x_train_q,
            lambda xa, xb: kernel_value_autodiff(xa, xb, trainable_params),
        )
        return float(_kernel_target_alignment(kernel_matrix, y_train_pm))

    def step_fn(current_params):
        new_params, loss = opt.step_and_cost(objective, current_params)
        return new_params, loss

    if is_trainable:
        rng = np.random.default_rng(seed)
        init_params = 0.01 * rng.standard_normal(param_shape)
        params = pnp.array(init_params, requires_grad=True)
    else:
        params = pnp.array(np.zeros(param_shape), requires_grad=False)

    if steps > 0 and is_trainable:
        params, loss_trace = run_training_loop(
            step_fn,
            params,
            steps,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
    else:
        loss_trace = [float(objective(params))]

    alignment_trace = [alignment_value(params)]

    if loss_trace:
        trace_params = pnp.array(init_params, requires_grad=True)
        trace_opt = get_optimizer(
            optimizer,
            stepsize=step_size,
            **(optimizer_kwargs or {}),
        )

        def trace_step_fn(current_params):
            new_params, loss = trace_opt.step_and_cost(objective, current_params)
            return new_params, loss

        trace_params_current = trace_params
        alignment_trace = []
        for _ in range(len(loss_trace)):
            trace_params_current, _ = trace_step_fn(trace_params_current)
            alignment_trace.append(alignment_value(trace_params_current))

    trained_params = np.asarray(params, dtype=float)

    def kernel_fn(x1, x2) -> float:
        probs = kernel_circuit_eval(
            np.asarray(x1, dtype=float),
            np.asarray(x2, dtype=float),
            trained_params,
        )
        return float(probs[0])

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
        "embedding": embedding_name,
        "embedding_layers": embedding_layers,
        "circuit_metadata": circuit_metadata(
            model="trainable_quantum_kernel_classifier",
            n_qubits=n_qubits,
            n_layers=1,
            embedding=embedding_name,
            embedding_layers=embedding_layers,
            ansatz=None,
            template="trainable_kernel",
            trainable_parameters=embedding_parameter_count(
                embedding_name,
                n_layers=embedding_layers,
                n_qubits=n_qubits,
            ),
        ),
        "steps": steps,
        "optimizer": optimizer,
        "optimizer_kwargs": optimizer_kwargs or {},
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "step_size": step_size,
        "reg_strength": reg_strength,
        "svc_c": svc_c,
        "shots_train": shots_train,
        "shots_kernel": shots_kernel,
        "noise_model": noise_model,
        "loss_trace": loss_trace,
        "alignment_trace": alignment_trace,
        "final_loss": float(loss_trace[-1]) if loss_trace else float("nan"),
        "final_alignment": float(alignment_trace[-1]) if alignment_trace else float("nan"),
        "trained_params": trained_params,
        "kernel_matrix_train": kernel_matrix_train,
        "kernel_matrix_test": kernel_matrix_test,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": np.asarray(y_train_pred, dtype=int),
        "y_test_pred": np.asarray(y_test_pred, dtype=int),
    }

    train_tag = "analytic" if shots_train is None else f"train{shots_train}"
    kernel_tag = "analytic" if shots_kernel is None else f"kernel{shots_kernel}"

    stem = (
        f"{dataset}_trainable_kernel_"
        f"emb{embedding_name}_"
        f"layers{embedding_layers}_"
        f"steps{steps}_"
        f"samples{n_samples}_"
        f"noise{str(noise).replace('.', 'p')}_"
        f"seed{seed}_"
        f"{train_tag}_{kernel_tag}_"
        f"{noise_model_tag(noise_model)}"
    )

    def _results_file(filename: str) -> Path:
        if results_dir is not None:
            path = Path(results_dir) / filename
            ensure_dir(path.parent)
            return path
        return results_path("trainable_kernel", filename)

    def _images_file(filename: str) -> Path:
        if images_dir is not None:
            path = Path(images_dir) / filename
            ensure_dir(path.parent)
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


def run_trainable_quantum_kernel_regressor(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.25,
    seed: int = 123,
    dataset: str = "sine",
    embedding: str = "data_reupload",
    embedding_layers: int = 2,
    steps: int = 50,
    step_size: float = 0.1,
    optimizer: str = "adam",
    optimizer_kwargs: dict[str, Any] | None = None,
    reg_strength: float = 1e-4,
    alpha: float = 1.0,
    shots_train: int | None = None,
    shots_kernel: int | None = None,
    noise_model: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Run a trainable quantum kernel regressor on a named regression dataset.
    """
    data = make_regression_dataset(
        n_samples=n_samples,
        noise=noise,
        test_size=test_size,
        seed=seed,
        dataset=dataset,
    )
    model = TrainableQuantumKernelRegressor(
        embedding=embedding,
        embedding_layers=embedding_layers,
        steps=steps,
        step_size=step_size,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        reg_strength=reg_strength,
        alpha=alpha,
        shots_train=shots_train,
        shots_kernel=shots_kernel,
        noise_model=noise_model,
        seed=seed,
    )
    model.fit(data["x_train"], data["y_train"])
    y_train_pred = model.predict(data["x_train"])
    y_test_pred = model.predict(data["x_test"])
    n_qubits = int(np.asarray(data["x_train"], dtype=float).shape[1])
    return {
        "model": "trainable_quantum_kernel_regressor",
        "dataset": dataset,
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "embedding": model.embedding_name_,
        "embedding_layers": embedding_layers,
        "n_qubits": n_qubits,
        "circuit_metadata": circuit_metadata(
            model="trainable_quantum_kernel_regressor",
            n_qubits=n_qubits,
            n_layers=1,
            embedding=model.embedding_name_,
            embedding_layers=embedding_layers,
            ansatz=None,
            template="trainable_kernel",
            trainable_parameters=embedding_parameter_count(
                model.embedding_name_,
                n_layers=embedding_layers,
                n_qubits=n_qubits,
            ),
        ),
        "steps": steps,
        "shots_train": shots_train,
        "shots_kernel": shots_kernel,
        "noise_model": model.noise_model,
        "loss_trace": model.loss_trace_,
        "final_loss": float(model.loss_trace_[-1]),
        "final_alignment": model.alignment_,
        "trained_params": model.trained_params_,
        "kernel_matrix_train": model.kernel_matrix_train_,
        "train_mse": mean_squared_error(data["y_train"], y_train_pred),
        "test_mse": mean_squared_error(data["y_test"], y_test_pred),
        "train_mae": mean_absolute_error(data["y_train"], y_train_pred),
        "test_mae": mean_absolute_error(data["y_test"], y_test_pred),
        "x_train": data["x_train"],
        "x_test": data["x_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }
