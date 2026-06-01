def test_imports_smoke():
    import qml
    import qml.autoencoder
    import qml.ansatz
    import qml.classifiers
    import qml.circuit_metadata
    import qml.data
    import qml.embeddings
    import qml.io_utils
    import qml.kernel_methods
    import qml.losses
    import qml.metrics
    import qml.model_selection
    import qml.qcnn
    import qml.reporting
    import qml.training
    import qml.utils
    import qml.visualize

    assert isinstance(qml.__version__, str)
    assert len(qml.__version__) > 0
