# Quantum Machine Learning

Modular **PennyLane-based quantum machine learning suite** providing reusable implementations of:

- Variational quantum classifiers (VQC)
- Quantum kernel methods
- Hybrid quantum–classical training workflows

The repository follows a **package-first design**:

- core algorithms live in `qml/`
- notebooks act as **pure package clients**
- experiments produce **reproducible outputs**
- plots and results are generated consistently

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
- PennyLane
- NumPy
- scikit-learn
- matplotlib

---

# Quick start

Run a minimal variational quantum classifier:

```python
from qml.classifiers import run_vqc

result = run_vqc(
    n_samples=200,
    n_layers=2,
    steps=50,
    plot=True,
)
```

Run a minimal quantum kernel classifier:

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
)
```

Both functions return dictionaries containing:

- training metrics
- predictions
- model parameters
- experiment configuration

---

# Documentation

Core documentation:

- **[THEORY.md](THEORY.md)** — mathematical background
- **[USAGE.md](USAGE.md)** — package API usage

Algorithm notes:

- **[Variational quantum classifier](docs/qml/variational_classifier.md)**
- **[Quantum kernel methods](docs/qml/quantum_kernels.md)**

Example notebooks:

- `notebooks/quantum_variational_classifier.ipynb`
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
        variational quantum models

    kernel_methods.py
        quantum kernel workflows

    visualize.py
        plotting utilities

    io_utils.py
        reproducible saving/loading


notebooks/
    quantum_variational_classifier.ipynb
    quantum_kernel_classifier.ipynb
    classical_vs_quantum_classifier.ipynb


tests/
    smoke tests ensuring stable imports and execution


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

Algorithms are implemented in the package:

```
qml.*
```

Notebooks call public APIs rather than implementing circuits inline.

---

### Reproducibility

Experiments return structured dictionaries and can optionally:

- save JSON outputs
- save figures
- use fixed seeds

---

### Minimal abstractions

Shared infrastructure is intentionally lightweight:

- small set of embeddings
- small set of ansätze
- simple training loops
- consistent plotting

---

# Example outputs

Variational classifier:

- dataset visualization
- loss curve
- decision boundary

Quantum kernel classifier:

- dataset visualization
- kernel matrix heatmaps
- classification accuracy

Outputs can be saved to:

```
results/
images/
```

---

# Current algorithms

### Variational Quantum Classifier

Binary classification using:

- angle embedding
- hardware-efficient ansatz
- Adam optimisation

---

### Quantum Kernel Classifier

Support vector machine using a quantum feature map:

[
K(x_i, x_j) =
|\langle \phi(x_i) | \phi(x_j) \rangle|^2
]

---

# Development workflow

Run tests:

```bash
pytest
```

Run module directly:

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

Planned extensions:

- variational quantum regression
- additional feature maps
- data re-uploading architectures
- noise studies
- kernel visualisation utilities
- additional datasets

---

# License

MIT License
