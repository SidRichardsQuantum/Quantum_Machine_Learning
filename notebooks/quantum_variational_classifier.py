# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: '1.16.4'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quantum Variational Classifier
#
# Minimal package-client example for the PennyLane-based VQC workflow in `qml.classifiers`.

# %%
from qml.classifiers import run_vqc

# %% [markdown]
# ## Run a minimal VQC experiment
#
# This trains a small variational quantum classifier on the two-moons dataset.

# %%
result = run_vqc(
    n_samples=200,
    noise=0.1,
    test_size=0.25,
    seed=123,
    n_layers=2,
    steps=50,
    step_size=0.1,
    plot=True,
    save=False,
)

# %% [markdown]
# ## Summary metrics

# %%
print("Train accuracy:", result["train_accuracy"])
print("Test accuracy:", result["test_accuracy"])
print("Final loss:", result["final_loss"])

# %% [markdown]
# ## First few loss values

# %%
result["loss_history"][:10]

# %% [markdown]
# ## Returned result keys

# %%
sorted(result.keys())

# %% [markdown]
# ## Notes
#
# The notebook intentionally acts as a pure package client:
#
# - dataset generation lives in `qml.data`
# - embedding logic lives in `qml.embeddings`
# - ansatz logic lives in `qml.ansatz`
# - training/evaluation orchestration lives in `qml.classifiers`
#
# This keeps the notebook focused on usage rather than implementation.