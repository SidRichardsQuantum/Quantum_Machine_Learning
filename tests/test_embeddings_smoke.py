import math

from qml.classifiers import run_vqc
from qml.embeddings import available_embeddings, get_embedding


def test_available_embeddings():
    names = available_embeddings()

    assert isinstance(names, list)
    assert "angle" in names
    assert "data_reupload" in names


def test_get_embedding():
    fn = get_embedding("angle")

    assert callable(fn)


def test_run_vqc_data_reupload_smoke():
    result = run_vqc(
        n_samples=24,
        noise=0.1,
        test_size=0.25,
        seed=0,
        n_layers=1,
        embedding="data_reupload",
        embedding_layers=1,
        steps=2,
        step_size=0.1,
        plot=False,
        save=False,
    )

    assert result["model"] == "vqc"
    assert result["embedding"] == "data_reupload"

    assert math.isfinite(result["train_accuracy"])
    assert math.isfinite(result["test_accuracy"])
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0

    assert len(result["loss_history"]) > 0

    # embedding parameters should exist for data reupload
    assert result["embedding_params"] is not None
