from __future__ import annotations

import numpy as np
import pytest

from qml import QuantumClassifier, QuantumKernel, QuantumRegressor, make_sequence_windows
from qml.embeddings import embedding_parameter_shape
from qml.optimizers import get_optimizer


def test_quantum_regressor_rejects_mismatched_target_rows() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    y = np.asarray([0.0], dtype=float)

    with pytest.raises(ValueError, match="y must have shape"):
        QuantumRegressor(steps=0).fit(x, y)


def test_quantum_classifier_requires_at_least_two_classes() -> None:
    x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    y = np.asarray([1, 1])

    with pytest.raises(ValueError, match="At least two classes"):
        QuantumClassifier(steps=0).fit(x, y)


def test_quantum_kernel_rejects_feature_dimension_mismatch() -> None:
    kernel = QuantumKernel()

    with pytest.raises(ValueError, match="Feature dimensions must match"):
        kernel.evaluate([[0.0, 0.0]], [[0.0, 0.0, 0.0]])


def test_sequence_windows_rejects_too_short_series() -> None:
    with pytest.raises(ValueError, match="Sequence is too short"):
        make_sequence_windows([1.0, 2.0], window_size=2, horizon=2)


def test_unknown_embedding_and_optimizer_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown embedding"):
        embedding_parameter_shape("unknown", n_layers=1, n_qubits=2)

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        get_optimizer("unknown")
