# Quantum Machine Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/SidRichardsQuantum/Quantum_Machine_Learning/actions/workflows/tests.yml/badge.svg)

Modular **PennyLane-based quantum machine learning library** implementing reusable workflows for:

• Variational quantum classification (VQC)  
• Variational quantum regression (VQR)  
• Quantum kernel methods  
• Trainable quantum kernels (kernel-target alignment)  
• Classical baseline models  
• Deterministic benchmark utilities  

The repository follows a **package-first design**:

• algorithms implemented in `qml/`  
• notebooks act as thin clients  
• experiments produce reproducible outputs  
• consistent plotting and result structures  
• deterministic execution via explicit seeds  

---

# Installation

Clone and install in editable mode:

```bash
pip install -e .
````

Install development tools:

```bash
pip install -e ".[dev]"
```

Requirements:

• Python ≥ 3.10
• PennyLane ≥ 0.34
• NumPy ≥ 1.24
• scikit-learn ≥ 1.3
• matplotlib ≥ 3.7

---

# Quick start

## Variational quantum classifier

```python
from qml.classifiers import run_vqc

result = run_vqc(
    n_samples=200,
    n_layers=2,
    steps=50,
    plot=True,
)
```

---

## Variational quantum regression

```python
from qml.regression import run_vqr

result = run_vqr(
    n_samples=200,
    n_layers=2,
    steps=50,
    plot=True,
)
```

---

## Quantum kernel classifier

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
    plot=True,
)
```

---

## Trainable quantum kernel (kernel-target alignment)

```python
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier

result = run_trainable_quantum_kernel_classifier(
    n_samples=200,
    steps=50,
    plot=True,
)
```

---

All workflows return structured dictionaries containing:

• training metrics
• predictions
• learned parameters
• configuration metadata

---

# Noise-aware execution (finite shots)

Quantum circuits can be evaluated either analytically or with finite sampling.

Finite-shot execution uses:

```python
qml.set_shots(qnode, shots)
```

Example:

```python
result = run_vqc(
    n_samples=200,
    n_layers=2,
    steps=50,
    shots=128,
)
```

Trainable kernel workflows support separate shot settings:

```python
result = run_trainable_quantum_kernel_classifier(
    n_samples=200,
    shots_train=64,
    shots_kernel=256,
)
```

All workflows remain deterministic when a fixed seed is provided.

---

# Benchmark framework

Benchmark utilities compare quantum and classical models across multiple seeds.

Example:

```python
from qml.benchmarks import compare_classification_models

result = compare_classification_models(
    models=[
        "vqc",
        "quantum_kernel",
        "trainable_quantum_kernel",
        "logistic_regression",
        "svm_classifier",
    ],
    seeds=[123, 456],
)
```

---

## Model-specific configuration

Benchmarks accept per-model kwargs:

```python
result = compare_classification_models(
    models=[
        "vqc",
        "quantum_kernel",
        "trainable_quantum_kernel",
    ],
    seeds=[123],
    model_kwargs={
        "vqc": {"shots": 128},

        "quantum_kernel": {"shots": 256},

        "trainable_quantum_kernel": {
            "shots_train": 64,
            "shots_kernel": 256,
        },
    },
)
```

Result structure remains consistent across models.

---

# Classical baselines

Included reference models:

• logistic regression
• ridge regression
• support vector machine
• multilayer perceptron

These provide performance context for quantum models.

---

# Command line interface

Run workflows directly:

```bash
python -m qml vqc --steps 50 --plot
python -m qml regression --steps 50 --plot
python -m qml kernel --plot
python -m qml trainable-kernel --steps 50 --plot
```

Run benchmarks:

```bash
python -m qml benchmark classification \
    --models vqc quantum_kernel svm_classifier logistic_regression \
    --seeds 123 456
```

```bash
python -m qml benchmark regression \
    --models vqr ridge_regression mlp_regressor \
    --seeds 123 456
```

CLI outputs include:

• training metrics
• test metrics
• final loss
• saved plots (optional)

---

# Documentation

Core documentation:

• **THEORY.md** — mathematical background
• **USAGE.md** — API examples

Algorithm notes:

• docs/qml/variational_quantum_classifier.md
• docs/qml/variational_regression.md
• docs/qml/quantum_kernels.md

Example notebooks:

• quantum_variational_classifier.ipynb
• quantum_regressor.ipynb
• quantum_kernel_classifier.ipynb
• classical_vs_quantum_classifier.ipynb

---

# Repository structure

```
qml/

    ansatz.py
        parameterised circuit templates

    embeddings.py
        feature encoding circuits

    classifiers.py
        variational quantum classification workflows

    regression.py
        variational quantum regression workflows

    kernel_methods.py
        quantum kernel workflows

    trainable_kernels.py
        kernel-target alignment optimisation

    classical_baselines.py
        logistic, ridge, svm, mlp

    benchmarks.py
        multi-seed benchmark utilities

    training.py
        hybrid optimisation loops

    metrics.py
        evaluation metrics

    losses.py
        objective functions

    data.py
        dataset generation utilities

    visualize.py
        plotting utilities

    io_utils.py
        reproducible saving utilities


notebooks/

    examples implemented as thin package clients


tests/

    smoke tests
    deterministic benchmarks


docs/

    theory notes and algorithm descriptions


results/

    saved experiment outputs (gitignored)


images/

    generated plots (gitignored)
```

---

# Design principles

## Package-first architecture

Core implementations live in:

```
qml.*
```

Notebooks import public APIs rather than defining circuits inline.

---

## Deterministic workflows

Reproducibility is prioritised:

• explicit random seeds
• deterministic dataset generation
• reproducible optimisation
• consistent JSON outputs
• deterministic finite-shot execution

---

## Minimal abstractions

Shared infrastructure intentionally remains lightweight:

• small set of embeddings
• hardware-efficient ansatz
• simple optimisation loops
• consistent plotting utilities

---

# Current algorithms

## Variational quantum classifier

Binary classification using:

• angle embedding
• hardware-efficient ansatz
• cross-entropy loss

---

## Variational quantum regression

Continuous prediction using:

• angle embedding
• expectation-value outputs
• mean squared error

---

## Quantum kernel classifier

Support vector machine using quantum feature maps:

$$
K(x_i, x_j)
===========

|\langle \phi(x_i) | \phi(x_j) \rangle|^2
$$

---

## Trainable quantum kernel

Kernel alignment objective:

$$
\max_\theta
;
\frac{
\langle K_\theta, Y \rangle_F
}{
|K_\theta|_F |Y|_F
}
$$

where:

• $K_\theta$ is the quantum kernel matrix
• $Y$ is the label similarity matrix

---

# Development workflow

Run tests:

```bash
pytest
```

Format code:

```bash
black .
ruff check .
```

Run module:

```bash
python -m qml
```

---

# Roadmap

Potential extensions:

• additional feature maps
• data re-uploading circuits
• quantum metric learning
• noise robustness studies
• additional benchmark datasets
• circuit architecture comparisons

---

## Author

Sid Richards

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see LICENSE
