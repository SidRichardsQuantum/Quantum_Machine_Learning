# Usage

This document describes how to use the public API provided by the `qml` package.

All notebooks in this repository act as **pure package clients** of these APIs.

---

# Installation

Install the package in editable mode:

```bash
pip install -e .
````

Install development tools:

```bash
pip install -e ".[dev]"
```

---

# Variational quantum classifier

Train a variational quantum classifier on a synthetic dataset:

```python
from qml.classifiers import run_vqc

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
```

---

## Parameters

| parameter   | description             | default |
| ----------- | ----------------------- | ------- |
| `n_samples` | dataset size            | 200     |
| `noise`     | dataset noise level     | 0.1     |
| `test_size` | test fraction           | 0.25    |
| `seed`      | random seed             | 123     |
| `n_layers`  | number of ansatz layers | 2       |
| `steps`     | optimisation steps      | 50      |
| `step_size` | Adam learning rate      | 0.1     |
| `plot`      | show plots              | False   |
| `save`      | save JSON and figures   | False   |

---

## Returned dictionary

The result contains:

```python
result.keys()
```

Typical fields:

```python
{
    "model",
    "dataset",
    "seed",
    "n_qubits",
    "n_layers",
    "steps",
    "step_size",
    "loss_history",
    "train_accuracy",
    "test_accuracy",
    "params",
    "y_train",
    "y_test",
    "y_train_pred",
    "y_test_pred",
    "train_probabilities",
    "test_probabilities",
}
```

---

## Generated plots

If `plot=True`, the following figures are displayed:

* dataset
* training loss
* decision boundary

If `save=True`, figures are saved to:

```
images/vqc/
```

and results are saved to:

```
results/vqc/
```

---

# Quantum kernel classifier

Train a support vector machine using a quantum kernel:

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
    noise=0.1,
    test_size=0.25,
    seed=123,
    plot=False,
    save=False,
)
```

---

## Parameters

| parameter   | description                   | default |
| ----------- | ----------------------------- | ------- |
| `n_samples` | dataset size                  | 200     |
| `noise`     | dataset noise level           | 0.1     |
| `test_size` | test fraction                 | 0.25    |
| `seed`      | random seed                   | 123     |
| `plot`      | show plots (future extension) | False   |
| `save`      | save results JSON             | False   |

---

## Returned dictionary

```python
{
    "model",
    "dataset",
    "seed",
    "n_qubits",
    "train_accuracy",
    "test_accuracy",
    "kernel_matrix_train",
    "kernel_matrix_test",
    "y_train",
    "y_test",
    "y_train_pred",
    "y_test_pred",
}
```

---

# Reproducibility

All workflows support deterministic runs using the `seed` parameter.

Outputs can optionally be saved:

```
results/
images/
```

Both directories are gitignored.

---

# Running from notebooks

Notebooks should import from the package:

```python
from qml.classifiers import run_vqc
```

rather than defining circuits inline.

This ensures:

* consistent behaviour
* reproducible outputs
* shared infrastructure
* minimal duplication

---

# Running tests

Execute:

```bash
pytest
```

---

# Running module

```bash
python -m qml
```

---

# Development workflow

Format code:

```bash
black .
ruff check .
```

---

# Next extensions

Future versions may add:

* regression models
* additional embeddings
* additional ansätze
* kernel visualisations
* noise-aware training
* CLI entrypoints
