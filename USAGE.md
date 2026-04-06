# Usage

This document describes how to use the public API provided by the `qml` package.

All notebooks in this repository act as **pure package clients** of these APIs.

---

# Installation

Install the package in editable mode:

```bash
pip install -e .
```

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

- dataset
- training loss
- decision boundary

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

Run a minimal quantum kernel classifier:

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
    plot=True,
)
```

---

## Parameters

| parameter     | description                      | default            |
| ------------- | -------------------------------- | ------------------ |
| `n_samples`   | dataset size                     | 200                |
| `noise`       | dataset noise level              | 0.1                |
| `test_size`   | test fraction                    | 0.25               |
| `seed`        | random seed                      | 123                |
| `plot`        | show dataset and kernel plots    | False              |
| `save`        | save JSON and figures            | False              |
| `results_dir` | directory for saved JSON outputs | `"results/kernel"` |
| `images_dir`  | directory for saved plots        | `"images/kernel"`  |

---

## Returned dictionary

```python
result.keys()
```

Typical fields:

```python
{
    "model",
    "dataset",
    "seed",
    "n_samples",
    "noise",
    "test_size",
    "n_qubits",

    "train_accuracy",
    "test_accuracy",

    "kernel_matrix_train",
    "kernel_matrix_test",

    "x_train",
    "x_test",

    "y_train",
    "y_test",

    "y_train_pred",
    "y_test_pred",
}
```

---

## Generated plots

If `plot=True`, the following figures are displayed:

- training dataset
- training kernel matrix
- test-vs-train kernel matrix

If `save=True`, figures are written to:

```
images/kernel/
```

and results are written to:

```
results/kernel/
```

Example filenames:

```
images/kernel/
    moons_samples200_noise0p1_seed123_dataset.png
    moons_samples200_noise0p1_seed123_kernel_train.png
    moons_samples200_noise0p1_seed123_kernel_test.png
```

```
results/kernel/
    moons_samples200_noise0p1_seed123.json
```

---

## Notes

The workflow separates:

- quantum feature map construction
- kernel matrix evaluation
- classical SVM optimisation

The quantum circuit is used only to compute kernel values:

$$
K(x_i, x_j)
=

|\langle \phi(x_i) | \phi(x_j) \rangle|^2
$$

Classification is performed using a classical support vector machine with a precomputed kernel.

---

# Benchmarking

Classification benchmark:

python -m qml benchmark classification \
    --models vqc quantum_kernel svm_classifier logistic_regression \
    --seeds 123 456 789


Regression benchmark:

python -m qml benchmark regression \
    --models vqr ridge_regression mlp_regressor \
    --seeds 123 456

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

- consistent behaviour
- reproducible outputs
- shared infrastructure
- minimal duplication

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

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)
