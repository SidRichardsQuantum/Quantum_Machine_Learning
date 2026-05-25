import numpy as np

from qml import (
    QuantumGaussianProcessRegressor,
    QuantumKernel,
    QuantumKernelPCA,
    QuantumOneClassClassifier,
    QuantumReservoirClassifier,
    QuantumReservoirFeatures,
    QuantumReservoirRegressor,
    TrainableQuantumKernelRegressor,
    run_trainable_quantum_kernel_regressor,
)
from qml.ansatz import strongly_entangling_parameter_shape
from qml.embeddings import available_embeddings


def test_additional_embeddings_smoke():
    x = np.asarray([[0.1, 0.2], [0.3, -0.1]], dtype=float)

    assert {"angle", "amplitude", "zz", "iqp", "data_reupload"}.issubset(
        set(available_embeddings())
    )
    assert strongly_entangling_parameter_shape(2, 3) == (2, 3, 3)

    for embedding in ["amplitude", "zz", "iqp"]:
        kernel = QuantumKernel(embedding=embedding, seed=1)
        matrix = kernel.evaluate(x)
        assert matrix.shape == (2, 2)
        assert np.allclose(np.diag(matrix), 1.0)


def test_quantum_kernel_pca_smoke():
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.5, -0.3],
            [0.5, 0.4],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    kpca = QuantumKernelPCA(QuantumKernel(seed=2), n_components=2)
    z_train = kpca.fit_transform(x)
    z_test = kpca.transform(x[:2])

    assert z_train.shape == (4, 2)
    assert z_test.shape == (2, 2)
    assert np.isfinite(z_train).all()


def test_quantum_one_class_classifier_smoke():
    x_train = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]], dtype=float)
    x_test = np.asarray([[0.05, 0.05], [2.0, 2.0]], dtype=float)

    detector = QuantumOneClassClassifier(QuantumKernel(seed=3), nu=0.25)
    detector.fit(x_train)
    pred = detector.predict(x_test)
    score = detector.decision_function(x_test)

    assert pred.shape == (2,)
    assert set(pred).issubset({-1, 1})
    assert score.shape == (2,)


def test_quantum_gaussian_process_regressor_smoke():
    x = np.asarray([[-1.0, -1.0], [-0.3, -0.3], [0.3, 0.3], [1.0, 1.0]], dtype=float)
    y = np.asarray([-1.0, -0.2, 0.2, 1.0], dtype=float)

    reg = QuantumGaussianProcessRegressor(QuantumKernel(seed=4), alpha=1e-4)
    reg.fit(x, y)
    mean, std = reg.predict(x[:2], return_std=True)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.isfinite(mean).all()
    assert np.all(std >= 0.0)


def test_quantum_reservoir_regressor_and_classifier_smoke():
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.8, -0.7],
            [0.8, 0.7],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y_reg = np.asarray([-1.0, -0.8, 0.8, 1.0], dtype=float)
    y_cls = np.asarray([0, 0, 1, 1])

    features = QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=5)
    transformed = features.fit_transform(x)
    assert transformed.shape == (4, 2)

    reg = QuantumReservoirRegressor(
        QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=5), alpha=1e-3
    )
    reg.fit(x, y_reg)
    assert reg.predict(x).shape == y_reg.shape

    clf = QuantumReservoirClassifier(
        QuantumReservoirFeatures(n_qubits=2, n_layers=1, seed=6), c=1.0
    )
    clf.fit(x, y_cls)
    assert clf.predict(x).shape == y_cls.shape
    assert clf.predict_proba(x).shape == (4, 2)


def test_trainable_quantum_kernel_regressor_smoke():
    x = np.asarray(
        [
            [-1.0, -1.0],
            [-0.3, -0.3],
            [0.3, 0.3],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y = np.asarray([-0.9, -0.2, 0.2, 0.9], dtype=float)

    reg = TrainableQuantumKernelRegressor(
        embedding_layers=1,
        steps=1,
        step_size=0.05,
        alpha=1e-3,
        seed=7,
    )
    reg.fit(x, y)
    pred = reg.predict(x)

    assert pred.shape == y.shape
    assert len(reg.loss_trace_) == 1
    assert reg.trained_params_.shape == (1, 2, 3)
    assert np.isfinite(pred).all()


def test_trainable_quantum_kernel_regressor_runner_smoke():
    result = run_trainable_quantum_kernel_regressor(
        n_samples=8,
        noise=0.1,
        test_size=0.25,
        seed=8,
        dataset="sine",
        embedding_layers=1,
        steps=1,
        alpha=1e-3,
    )

    assert result["model"] == "trainable_quantum_kernel_regressor"
    assert result["kernel_matrix_train"].shape[0] == result["x_train"].shape[0]
    assert "test_mse" in result
