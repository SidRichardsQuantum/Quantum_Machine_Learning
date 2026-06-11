import numpy as np
import pytest

from qml import (
    QuantumClassifier,
    QuantumGaussianProcessRegressor,
    QuantumKernel,
    QuantumKernelClassifier,
    QuantumKernelPCA,
    QuantumKernelRegressor,
    QuantumOneClassClassifier,
    QuantumRegressor,
    QuantumReservoirClassifier,
    QuantumReservoirFeatures,
    QuantumReservoirRegressor,
    TrainableQuantumKernelRegressor,
)

X_CLASS = np.asarray(
    [
        [-1.0, -1.0],
        [-0.8, -0.9],
        [0.8, 0.7],
        [1.0, 1.0],
    ],
    dtype=float,
)
Y_CLASS = np.asarray([0, 0, 1, 1])
Y_REG = np.asarray([-1.0, -0.8, 0.8, 1.0], dtype=float)


def test_variational_estimators_expose_consistent_fitted_attributes():
    clf = QuantumClassifier(n_layers=1, steps=1, seed=10)
    clf.fit(X_CLASS, Y_CLASS)

    assert clf.n_features_in_ == 2
    assert clf.classes_.tolist() == [0, 1]
    assert len(clf.trained_params_) == 1
    assert clf.loss_trace_ == clf.loss_history_
    assert clf.circuit_metadata_["model"] == "quantum_classifier"
    assert clf.predict(X_CLASS).shape == Y_CLASS.shape

    reg = QuantumRegressor(n_layers=1, steps=1, seed=11)
    reg.fit(X_CLASS, Y_REG)

    assert reg.n_features_in_ == 2
    assert reg.n_outputs_ == 1
    assert len(reg.trained_params_) == 1
    assert reg.loss_trace_ == reg.loss_history_
    assert reg.circuit_metadata_["model"] == "quantum_regressor"
    assert reg.predict(X_CLASS).shape == Y_REG.shape


def test_kernel_estimators_support_nested_params_and_fitted_state():
    clf = QuantumKernelClassifier(QuantumKernel(seed=12), c=0.5)
    assert clf.get_params()["kernel__shots"] is None
    clf.set_params(kernel__shots=64, kernel__cache=False, c=1.5)
    assert clf.kernel.shots == 64
    assert clf.kernel.cache is False
    assert clf.c == 1.5

    clf.fit(X_CLASS, Y_CLASS)
    assert clf.n_features_in_ == 2
    assert clf.classes_.tolist() == [0, 1]
    assert clf.kernel_matrix_train_.shape == (4, 4)
    assert clf.circuit_metadata_["shots"] == 64
    assert clf.predict(X_CLASS).shape == Y_CLASS.shape

    reg = QuantumKernelRegressor(QuantumKernel(seed=13), alpha=1e-3)
    reg.set_params(kernel__embedding="angle")
    reg.fit(X_CLASS, Y_REG)
    assert reg.n_features_in_ == 2
    assert reg.kernel_matrix_train_.shape == (4, 4)
    assert reg.circuit_metadata_["model"] == "quantum_kernel_regressor"
    assert reg.predict(X_CLASS).shape == Y_REG.shape


def test_advanced_kernel_estimators_expose_feature_counts_and_scoring_methods():
    kpca = QuantumKernelPCA(QuantumKernel(seed=14), n_components=2)
    z = kpca.fit_transform(X_CLASS)
    assert kpca.n_features_in_ == 2
    assert z.shape == (4, 2)
    kpca.set_params(kernel__cache=False)
    assert kpca.kernel.cache is False

    detector = QuantumOneClassClassifier(QuantumKernel(seed=15), nu=0.25)
    detector.fit(X_CLASS)
    assert detector.n_features_in_ == 2
    assert detector.score_samples(X_CLASS[:2]).shape == (2,)
    assert detector.circuit_metadata_["model"] == "quantum_one_class_classifier"

    gpr = QuantumGaussianProcessRegressor(QuantumKernel(seed=16), alpha=1e-4)
    gpr.fit(X_CLASS, Y_REG)
    assert gpr.n_features_in_ == 2
    assert gpr.kernel_matrix_train_.shape == (4, 4)
    mean, std = gpr.predict(X_CLASS[:2], return_std=True)
    assert mean.shape == (2,)
    assert std.shape == (2,)


def test_reservoir_estimators_support_nested_params_and_metadata():
    reg = QuantumReservoirRegressor(
        QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=17),
        alpha=1e-3,
    )
    reg.set_params(reservoir__shots=32, reservoir__input_scale=0.5)
    assert reg.reservoir.shots == 32
    assert reg.reservoir.input_scale == 0.5
    reg.fit(X_CLASS, Y_REG)
    assert reg.n_features_in_ == 2
    assert reg.feature_matrix_train_.shape == (4, 2)
    assert reg.circuit_metadata_["shots"] == 32

    clf = QuantumReservoirClassifier(
        QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=18),
        c=1.0,
    )
    clf.set_params(reservoir__noise_model={"depolarizing": 0.0})
    clf.fit(X_CLASS, Y_CLASS)
    assert clf.n_features_in_ == 2
    assert clf.classes_.tolist() == [0, 1]
    assert clf.predict_proba(X_CLASS).shape == (4, 2)
    assert clf.circuit_metadata_["model"] == "quantum_reservoir_classifier"


def test_trainable_kernel_regressor_metadata_and_feature_validation():
    reg = TrainableQuantumKernelRegressor(
        embedding_layers=1,
        steps=1,
        step_size=0.05,
        alpha=1e-3,
        seed=19,
    )
    reg.fit(X_CLASS, Y_REG)
    assert reg.n_features_in_ == 2
    assert reg.circuit_metadata_["model"] == "trainable_quantum_kernel_regressor"
    assert reg.circuit_metadata_["alignment"] == pytest.approx(reg.alignment_)

    with pytest.raises(ValueError, match="Expected 2 features"):
        reg.predict([[0.0, 0.0, 0.0]])


def test_unfitted_and_feature_mismatch_errors_are_clear():
    with pytest.raises(ValueError, match="fitted before prediction"):
        QuantumKernelRegressor().predict(X_CLASS)

    reg = QuantumKernelRegressor(QuantumKernel(seed=20), alpha=1e-3).fit(X_CLASS, Y_REG)
    with pytest.raises(ValueError, match="Expected 2 features"):
        reg.predict([[0.0, 0.0, 0.0]])


def test_seeded_estimators_are_deterministic_at_smoke_scale():
    reg_a = QuantumRegressor(n_layers=1, steps=1, seed=30).fit(X_CLASS, Y_REG)
    reg_b = QuantumRegressor(n_layers=1, steps=1, seed=30).fit(X_CLASS, Y_REG)
    assert np.allclose(reg_a.trained_params_[0], reg_b.trained_params_[0])
    assert np.allclose(reg_a.predict(X_CLASS), reg_b.predict(X_CLASS))

    kernel_a = QuantumKernel(seed=31).evaluate(X_CLASS)
    kernel_b = QuantumKernel(seed=31).evaluate(X_CLASS)
    assert np.allclose(kernel_a, kernel_b)

    reservoir_a = QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=32).fit(X_CLASS)
    reservoir_b = QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=32).fit(X_CLASS)
    assert np.allclose(reservoir_a.weights_, reservoir_b.weights_)
    assert np.allclose(reservoir_a.transform(X_CLASS), reservoir_b.transform(X_CLASS))

    trainable_a = TrainableQuantumKernelRegressor(
        embedding_layers=1,
        steps=1,
        step_size=0.05,
        alpha=1e-3,
        seed=33,
    ).fit(X_CLASS, Y_REG)
    trainable_b = TrainableQuantumKernelRegressor(
        embedding_layers=1,
        steps=1,
        step_size=0.05,
        alpha=1e-3,
        seed=33,
    ).fit(X_CLASS, Y_REG)
    assert np.allclose(trainable_a.trained_params_, trainable_b.trained_params_)
    assert np.allclose(trainable_a.predict(X_CLASS), trainable_b.predict(X_CLASS))
