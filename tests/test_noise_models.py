from __future__ import annotations

import numpy as np
import pytest

from qml import QuantumClassifier, QuantumKernel, QuantumRegressor, build_noise_model
from qml.noise import device_name_for_noise, noise_model_tag, normalize_noise_model
from qml.reservoir import QuantumReservoirFeatures


def test_noise_model_validation_and_helpers() -> None:
    assert normalize_noise_model(None) is None
    assert normalize_noise_model({"depolarizing": 0.0}) is None
    assert normalize_noise_model({"depolarizing_prob": 0.05}) == {"depolarizing": 0.05}
    assert build_noise_model(depolarizing=0.01, readout_error=0.02) == {
        "depolarizing": 0.01,
        "readout_error": 0.02,
    }
    assert device_name_for_noise({"readout_error": 0.1}) == "default.mixed"
    assert noise_model_tag({"amplitude_damping": 0.05}) == "amplitudedamping0p05"

    with pytest.raises(ValueError, match="Unknown noise channel"):
        normalize_noise_model({"thermal": 0.1})

    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_noise_model({"depolarizing": 1.1})


def test_noisy_quantum_kernel_smoke() -> None:
    x = np.asarray([[0.0, 0.0], [0.3, -0.2]], dtype=float)
    kernel = QuantumKernel(
        seed=0,
        noise_model={"depolarizing": 0.01, "readout_error": 0.02},
        cache=False,
    )

    matrix = kernel.evaluate(x)

    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()
    assert np.all((0.0 <= matrix) & (matrix <= 1.0))


def test_noisy_variational_estimators_smoke() -> None:
    x = np.asarray(
        [
            [-0.2, -0.1],
            [-0.1, -0.3],
            [0.2, 0.1],
            [0.3, 0.2],
        ],
        dtype=float,
    )
    y_class = np.asarray([0, 0, 1, 1])
    y_reg = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=float)
    noise_model = {
        "depolarizing": 0.01,
        "amplitude_damping": 0.01,
        "readout_error": 0.01,
    }

    clf = QuantumClassifier(n_layers=1, steps=1, seed=1, noise_model=noise_model)
    clf.fit(x, y_class)
    assert clf.predict_proba(x).shape == (4, 2)

    reg = QuantumRegressor(n_layers=1, steps=1, seed=2, noise_model=noise_model)
    reg.fit(x, y_reg)
    assert reg.predict(x).shape == y_reg.shape


def test_noisy_reservoir_features_smoke() -> None:
    x = np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=float)
    reservoir = QuantumReservoirFeatures(
        n_layers=1,
        seed=3,
        noise_model={"amplitude_damping": 0.01},
    )

    features = reservoir.fit_transform(x)

    assert features.shape == (2, 2)
    assert np.isfinite(features).all()
