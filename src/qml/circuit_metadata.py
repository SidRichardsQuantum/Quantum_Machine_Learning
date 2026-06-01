"""
qml.circuit_metadata
====================

Lightweight parameter-count and depth estimates for package circuit templates.
"""

from __future__ import annotations

from math import prod
from typing import Any

from qml.ansatz import parameter_shape, strongly_entangling_parameter_shape
from qml.embeddings import embedding_parameter_shape

__all__ = [
    "ansatz_parameter_count",
    "circuit_metadata",
    "embedding_parameter_count",
    "estimate_circuit_depth",
    "qcnn_parameter_count",
]


def _count_shape(shape: tuple[int, ...]) -> int:
    return int(prod(shape)) if shape else 0


def ansatz_parameter_count(
    n_layers: int,
    n_qubits: int,
    *,
    ansatz: str = "hardware_efficient",
) -> int:
    """Return the trainable parameter count for a supported ansatz template."""
    key = ansatz.strip().lower()
    if key in {"hardware_efficient", "hardware-efficient", "default"}:
        return _count_shape(parameter_shape(n_layers=n_layers, n_qubits=n_qubits))
    if key in {"strongly_entangling", "strongly-entangling", "strongly_entangling_layers"}:
        return _count_shape(
            strongly_entangling_parameter_shape(n_layers=n_layers, n_qubits=n_qubits)
        )
    raise ValueError(f"Unknown ansatz '{ansatz}'.")


def embedding_parameter_count(
    name: str,
    n_layers: int,
    n_qubits: int,
) -> int:
    """Return the trainable parameter count for a supported embedding."""
    return _count_shape(embedding_parameter_shape(name, n_layers=n_layers, n_qubits=n_qubits))


def qcnn_parameter_count() -> int:
    """Return the trainable parameter count for the default four-qubit QCNN."""
    shapes = {
        "embedding": (4, 3),
        "conv1": (2, 6),
        "pool1": (2, 2),
        "conv2": (1, 6),
        "pool2": (1, 2),
        "dense": (2,),
    }
    return sum(_count_shape(shape) for shape in shapes.values())


def _hardware_efficient_depth(n_layers: int, n_qubits: int) -> int:
    entangler_depth = 0
    if n_qubits > 1:
        entangler_depth += n_qubits - 1
    if n_qubits > 2:
        entangler_depth += 1
    return n_layers * (2 + entangler_depth)


def _embedding_depth(name: str, n_layers: int, n_qubits: int) -> int:
    key = name.strip().lower()
    if key in {
        "angle",
        "angle_embedding",
        "amplitude",
        "amplitude_embedding",
        "state_preparation",
        "reservoir_angle",
    }:
        return 1
    if key in {"zz", "zz_feature_map"}:
        return 1 + max(n_qubits - 1, 0) * 3
    if key in {"iqp", "iqp_feature_map"}:
        return 2 + (n_qubits if n_qubits > 1 else 0)
    if key in {"data_reupload", "data_reuploading", "data_reuploading_embedding"}:
        return n_layers * (2 + max(n_qubits - 1, 0))
    raise ValueError(f"Unknown embedding '{name}'.")


def estimate_circuit_depth(
    *,
    n_qubits: int,
    n_layers: int = 1,
    embedding: str = "angle",
    embedding_layers: int = 1,
    ansatz: str | None = "hardware_efficient",
    template: str = "variational",
    include_measurement: bool = True,
) -> int:
    """
    Return an approximate operation depth for package circuit templates.

    The estimate is intended for relative reporting in examples and benchmarks;
    it is not a device-compiled depth.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if embedding_layers <= 0:
        raise ValueError("embedding_layers must be positive.")

    key = template.strip().lower()
    measurement_depth = 1 if include_measurement else 0
    if key in {"variational", "vqc", "vqr", "autoencoder"}:
        depth = _embedding_depth(embedding, embedding_layers, n_qubits)
        if ansatz is not None:
            depth += _hardware_efficient_depth(n_layers, n_qubits)
        return int(depth + measurement_depth)
    if key in {"kernel", "quantum_kernel"}:
        depth = 2 * _embedding_depth(embedding, embedding_layers, n_qubits)
        return int(depth + measurement_depth)
    if key in {"trainable_kernel", "trainable_quantum_kernel"}:
        depth = 2 * _embedding_depth(embedding, embedding_layers, n_qubits)
        return int(depth + measurement_depth)
    if key == "qcnn":
        return int(23 + measurement_depth)
    raise ValueError(f"Unknown circuit template '{template}'.")


def circuit_metadata(
    *,
    model: str,
    n_qubits: int,
    n_layers: int = 1,
    embedding: str = "angle",
    embedding_layers: int = 1,
    ansatz: str | None = "hardware_efficient",
    template: str = "variational",
    trainable_parameters: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-friendly metadata dictionary for a package circuit workflow.
    """
    if trainable_parameters is None:
        if template.strip().lower() == "qcnn":
            trainable_parameters = qcnn_parameter_count()
        else:
            trainable_parameters = embedding_parameter_count(
                embedding,
                n_layers=embedding_layers,
                n_qubits=n_qubits,
            )
            if ansatz is not None:
                trainable_parameters += ansatz_parameter_count(
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    ansatz=ansatz,
                )

    metadata = {
        "model": model,
        "template": template,
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "embedding": embedding,
        "embedding_layers": int(embedding_layers),
        "ansatz": ansatz,
        "trainable_parameters": int(trainable_parameters),
        "estimated_depth": estimate_circuit_depth(
            n_qubits=n_qubits,
            n_layers=n_layers,
            embedding=embedding,
            embedding_layers=embedding_layers,
            ansatz=ansatz,
            template=template,
        ),
        "depth_is_estimate": True,
    }
    if extra:
        metadata.update(extra)
    return metadata
