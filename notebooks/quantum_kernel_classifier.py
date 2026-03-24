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
# # Quantum Kernel Classifier
#
# Minimal package-client example for the PennyLane-based quantum kernel workflow in `qml.kernel_methods`.

# %%
from qml.kernel_methods import run_quantum_kernel_classifier

# %% [markdown]
# ## Run a minimal quantum kernel experiment
#
# This fits a quantum kernel classifier on the two-moons dataset.

# %%
result = run_quantum_kernel_classifier(
    n_samples=200,
    noise=0.1,
    test_size=0.25,
    seed=123,
    plot=False,
    save=False,
)

# %% [markdown]
# ## Summary metrics

# %%
print("Train accuracy:", result["train_accuracy"])
print("Test accuracy:", result["test_accuracy"])

# %% [markdown]
# ## Kernel matrix shape

# %%
result["kernel_matrix_train"].shape

# %% [markdown]
# ## First few test predictions

# %%
result["y_test_pred"][:10]

# %% [markdown]
# ## Returned result keys

# %%
sorted(result.keys())