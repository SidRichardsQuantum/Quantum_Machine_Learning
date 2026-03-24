def test_imports_smoke():
    import qml
    import qml.ansatz
    import qml.classifiers
    import qml.data
    import qml.embeddings
    import qml.io_utils
    import qml.kernel_methods
    import qml.losses
    import qml.metrics
    import qml.training
    import qml.utils
    import qml.visualize

    assert qml.__version__
