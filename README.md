# Quantum Machine Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/SidRichardsQuantum/Quantum_Machine_Learning/actions/workflows/tests.yml/badge.svg)

Modular **PennyLane-based quantum machine learning suite** providing reusable implementations of:

- Variational quantum classifiers (VQC)
- Variational quantum regression (VQR)
- Quantum kernel methods
- Hybrid quantum–classical optimisation workflows

The repository follows a **package-first design**:

- algorithms implemented in `qml/`
- notebooks act as **thin clients**
- experiments produce **reproducible outputs**
- plots and results follow a **consistent structure**

---

# Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

Install development tools:

```bash
pip install -e ".[dev]"
```

Requirements:

- Python ≥ 3.10
- PennyLane ≥ 0.34
- NumPy ≥ 1.24
- scikit-learn ≥ 1.3
- matplotlib ≥ 3.7

---

# Quick start

### Variational quantum classifier

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

### Variational quantum regression

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

### Quantum kernel classifier

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
    plot=True,
)
```

---

All functions return structured dictionaries containing:

- training metrics
- predictions
- model parameters
- experiment configuration

---

# Command line interface

Run workflows directly:

```bash
python -m qml vqc --steps 50 --plot
python -m qml regression --steps 50 --plot
python -m qml kernel --plot
```

CLI outputs include metrics such as:

- training accuracy / MSE
- test accuracy / MSE
- final loss
- saved plots (optional)

---

# Documentation

Core documentation:

- **[THEORY.md](THEORY.md)** — mathematical background
- **[USAGE.md](USAGE.md)** — API examples and configuration

Algorithm notes:

- **[Variational quantum classifier](docs/qml/variational_quantum_classifier.md)**
- **[Variational quantum regression](docs/qml/variational_regression.md)**
- **[Quantum kernel methods](docs/qml/quantum_kernels.md)**

Example notebooks:

- `notebooks/quantum_variational_classifier.ipynb`
- `notebooks/quantum_regressor.ipynb`
- `notebooks/quantum_kernel_classifier.ipynb`
- `notebooks/classical_vs_quantum_classifier.ipynb`

---

# Repository structure

```
qml/
    data.py
        dataset generation and preprocessing

    embeddings.py
        feature encoding circuits

    ansatz.py
        parameterised circuit templates

    training.py
        hybrid optimisation loops

    losses.py
        objective functions

    metrics.py
        evaluation metrics

    classifiers.py
        variational quantum classification workflows

    regression.py
        variational quantum regression workflows

    kernel_methods.py
        quantum kernel workflows

    visualize.py
        plotting utilities

    io_utils.py
        reproducible saving/loading


notebooks/
    quantum_variational_classifier.ipynb
    quantum_regressor.ipynb
    quantum_kernel_classifier.ipynb
    classical_vs_quantum_classifier.ipynb


tests/
    smoke tests for CLI and core workflows


docs/
    theory notes and algorithm descriptions


results/
    saved experiment outputs (gitignored)


images/
    generated plots (gitignored)
```

---

# Design principles

### Package-first

Algorithms live in:

```
qml.*
```

Notebooks call stable public APIs rather than implementing circuits inline.

---

### Reproducibility

Experiments return structured dictionaries and optionally:

- save JSON outputs
- save figures
- use fixed random seeds
- produce consistent file naming

---

### Minimal abstractions

Shared infrastructure is intentionally lightweight:

- small set of embeddings
- hardware-efficient ansätze
- simple training loops
- consistent plotting utilities

---

# Example outputs

### Variational classifier

- dataset visualisation
- training loss curve
- decision boundary

---

### Variational regression

- dataset visualisation
- prediction curve
- training loss curve

---

### Quantum kernel classifier

- dataset visualisation
- kernel matrix heatmap
- classification accuracy

---

Outputs can be saved to:

```
results/
images/
```

---

# Current algorithms

## Variational Quantum Classifier

Binary classification using:

- angle embedding
- hardware-efficient ansatz
- Adam optimisation
- cross-entropy loss

---

## Variational Quantum Regression

Function approximation using:

- angle embedding
- hardware-efficient ansatz
- mean squared error loss

---

## Quantum Kernel Classifier

Support vector machine using a quantum feature map:

$$
K(x_i, x_j)
=
\left|
\langle \phi(x_i) \mid \phi(x_j) \rangle
\right|^2
$$

---

# Development workflow

Run tests:

```bash
pytest
```

Run module:

```bash
python -m qml
```

Format code:

```bash
black .
ruff check .
```

---

# Roadmap

Potential extensions:

- additional embeddings
- data re-uploading circuits
- kernel alignment methods
- noise studies
- trainable feature maps
- additional benchmark datasets
- comparison with classical baselines

---

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)
