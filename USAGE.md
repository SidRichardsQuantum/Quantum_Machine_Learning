# Usage

This document describes how to use the public API provided by the `qml` package.

All notebooks in this repository act as **thin clients** of these APIs.

The design goal is:

• reusable workflows  
• deterministic experiments  
• consistent outputs  
• minimal configuration  

---

## Table of Contents

- [Installation](#installation)
- [Testing and CI](#testing-and-ci)

- [Variational quantum classifier (VQC)](#variational-quantum-classifier-vqc)

  - [Parameters](#parameters)
  - [Returned dictionary](#returned-dictionary)

- [Variational quantum regression (VQR)](#variational-quantum-regression-vqr)

  - [Parameters](#parameters-1)
  - [Returned metrics](#returned-metrics)

- [Quantum convolutional neural network (QCNN)](#quantum-convolutional-neural-network-qcnn)

  - [Parameters](#parameters-2)
  - [Returned dictionary](#returned-dictionary-1)

- [Quantum autoencoder](#quantum-autoencoder)

  - [Parameters](#parameters-3)
  - [Returned dictionary](#returned-dictionary-2)

- [Quantum kernel classifier](#quantum-kernel-classifier)

  - [Parameters](#parameters-4)
  - [Returned dictionary](#returned-dictionary-3)

- [Trainable quantum kernel classifier](#trainable-quantum-kernel-classifier)

  - [Parameters](#parameters-5)
  - [Returned metrics](#returned-metrics-1)

- [Quantum metric learning](#quantum-metric-learning)

  - [Parameters](#parameters-6)
  - [Returned object](#returned-object)

- [CLI usage](#cli-usage)
- [Noise-aware execution](#noise-aware-execution)
- [Classical baselines](#classical-baselines)
- [Benchmarking](#benchmarking)

  - [Classification benchmark](#classification-benchmark)
  - [Regression benchmark](#regression-benchmark)
  - [Model-specific kwargs](#model-specific-kwargs)

- [Command line interface](#command-line-interface)
- [CLI benchmarks](#cli-benchmarks)
- [Reproducibility](#reproducibility)
- [Running from notebooks](#running-from-notebooks)
- [Running tests](#running-tests)
- [Development workflow](#development-workflow)
- [Author](#author)
- [License](#license)

---

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Install development tools:

```bash
pip install -e ".[dev]"
```

---

## Testing and CI

Run the default fast local test pass:

```bash
pytest -m "not slow"
```

Run the slower end-to-end coverage as well:

```bash
pytest
```

Tests marked with `@pytest.mark.slow` are used for heavier CLI and determinism
checks. CI mirrors this split:

• fast tests run across the Python version matrix  
• the full suite runs on Python 3.12  

Linting remains separate:

```bash
ruff check .
```

---

## Variational quantum classifier (VQC)

Train a minimal variational quantum classifier on a synthetic dataset:

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

### Parameters

| parameter | description          | default |
| --------- | -------------------- | ------- |
| n_samples | dataset size         | 200     |
| noise     | dataset noise level  | 0.1     |
| test_size | test fraction        | 0.25    |
| seed      | random seed          | 123     |
| n_layers  | ansatz depth         | 2       |
| steps     | optimisation steps   | 50      |
| step_size | Adam learning rate   | 0.1     |
| shots     | finite-shot sampling | None    |
| plot      | show plots           | False   |
| save      | save JSON + plots    | False   |

---

### Returned dictionary

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

## Variational quantum regression (VQR)

Train a variational quantum regressor:

```python
from qml.regression import run_vqr

result = run_vqr(
    n_samples=200,
    seed=123,
    n_layers=2,
    steps=50,
    plot=True,
)
```

---

### Parameters

| parameter | description          | default |
| --------- | -------------------- | ------- |
| n_samples | dataset size         | 200     |
| noise     | dataset noise        | 0.1     |
| test_size | test fraction        | 0.25    |
| seed      | random seed          | 123     |
| n_layers  | ansatz depth         | 2       |
| steps     | optimisation steps   | 50      |
| step_size | Adam learning rate   | 0.1     |
| shots     | finite-shot sampling | None    |
| plot      | show plots           | False   |
| save      | save outputs         | False   |

---

### Returned metrics

```python
{
    "train_mse",
    "test_mse",

    "train_mae",
    "test_mae",

    "loss_history",
}
```

---

## Quantum convolutional neural network (QCNN)

Train a hierarchical quantum classifier on a synthetic dataset:

```python
from qml.qcnn import run_qcnn

result = run_qcnn(
    n_samples=200,
    noise=0.1,
    test_size=0.25,
    seed=123,
    steps=50,
    step_size=0.1,
    plot=True,
    save=False,
)
```

---

### Parameters

| parameter | description | default |
| --------- | ----------- | ------- |
| n_samples | dataset size | 200 |
| noise | dataset noise level | 0.1 |
| test_size | test fraction | 0.25 |
| seed | random seed | 123 |
| steps | optimisation steps | 50 |
| step_size | Adam learning rate | 0.1 |
| shots | finite-shot sampling | None |
| plot | show plots | False |
| save | save JSON + plots | False |

---

### Returned dictionary

Typical fields:

```python
{
    "model",
    "dataset",

    "seed",

    "n_qubits",

    "steps",
    "step_size",

    "loss_history",

    "train_accuracy",
    "test_accuracy",

    "params",
    "embedding_params",
    "conv1_params",
    "conv2_params",
    "dense_params",

    "y_train",
    "y_test",

    "y_train_pred",
    "y_test_pred",

    "train_probabilities",
    "test_probabilities",
}
```

---

## Quantum autoencoder

Train a quantum autoencoder on a structured family of four-qubit states:

```python
from qml.autoencoder import run_quantum_autoencoder

result = run_quantum_autoencoder(
    n_samples=200,
    noise=0.05,
    test_size=0.25,
    seed=123,
    n_layers=2,
    latent_qubits=2,
    steps=50,
    step_size=0.1,
    family="correlated",
    plot=True,
    save=False,
)
```

---

### Parameters

| parameter | description | default |
| --------- | ----------- | ------- |
| n_samples | dataset size | 200 |
| noise | family perturbation level | 0.05 |
| test_size | test fraction | 0.25 |
| seed | random seed | 123 |
| n_layers | autoencoder ansatz depth | 2 |
| latent_qubits | retained latent qubits | 2 |
| steps | optimisation steps | 50 |
| step_size | Adam learning rate | 0.1 |
| family | state family | "correlated" |
| plot | show plots | False |
| save | save JSON + plots | False |

---

### Returned dictionary

Typical fields:

```python
{
    "model",
    "family",

    "seed",

    "n_qubits",
    "latent_qubits",
    "trash_qubits",

    "n_layers",
    "steps",
    "step_size",

    "loss_history",

    "train_compression_fidelity",
    "test_compression_fidelity",

    "train_reconstruction_fidelity",
    "test_reconstruction_fidelity",

    "params",
}
```

---

## Quantum kernel classifier

Compute a quantum kernel matrix and train an SVM:

```python
from qml.kernel_methods import run_quantum_kernel_classifier

result = run_quantum_kernel_classifier(
    n_samples=200,
    seed=123,
    plot=True,
)
```

---

### Parameters

| parameter | description                   | default |
| --------- | ----------------------------- | ------- |
| n_samples | dataset size                  | 200     |
| noise     | dataset noise                 | 0.1     |
| test_size | test fraction                 | 0.25    |
| seed      | random seed                   | 123     |
| shots     | finite-shot kernel estimation | None    |
| plot      | show kernel plots             | False   |
| save      | save outputs                  | False   |

---

### Returned dictionary

```python
{
    "train_accuracy",
    "test_accuracy",

    "kernel_matrix_train",
    "kernel_matrix_test",

    "y_train_pred",
    "y_test_pred",
}
```

---

## Trainable quantum kernel classifier

Optimise feature-map parameters using kernel-target alignment.

```python
from qml.trainable_kernels import run_trainable_quantum_kernel_classifier

result = run_trainable_quantum_kernel_classifier(
    n_samples=200,
    steps=50,
    plot=True,
)
```

---

### Parameters

| parameter    | description                      | default |
| ------------ | -------------------------------- | ------- |
| embedding    | feature map type                 | "angle" |
| n_layers     | circuit depth                    | 2       |
| steps        | optimisation steps               | 50      |
| step_size    | learning rate                    | 0.1     |
| shots_train  | shots used during optimisation   | None    |
| shots_kernel | shots used for kernel evaluation | None    |

---

### Returned metrics

```python
{
    "train_accuracy",
    "test_accuracy",

    "final_alignment",

    "loss_history",

    "kernel_matrix_train",
}
```

---

## Quantum metric learning

Train a supervised quantum embedding model using contrastive loss.

```python
from qml.metric_learning import run_quantum_metric_learner

result = run_quantum_metric_learner(
    samples=200,
    test_size=0.25,
    seed=123,
    layers=2,
    steps=50,
    stepsize=0.05,
    margin=0.5,
    pairs_per_step=32,
    plot=True,
)
```

The model learns an embedding geometry such that:

• samples from the same class are mapped closer together  
• samples from different classes are separated by a margin  

Classification is performed using nearest-centroid prediction in the learned embedding space.

---

### Parameters

| parameter | description | default |
|----------|-------------|--------|
| dataset | dataset name ("moons", "circles", "blobs") | "moons" |
| samples | dataset size | 120 |
| test_size | test fraction | 0.25 |
| seed | random seed | 42 |
| layers | number of trainable embedding layers | 2 |
| steps | optimisation steps | 100 |
| stepsize | Adam learning rate | 0.05 |
| margin | contrastive separation margin | 0.5 |
| pairs_per_step | number of sampled training pairs per step | 32 |
| log_every | logging frequency | 10 |
| scale_data | standardise features before encoding | True |
| plot | display training loss | False |
| save | save JSON + plots | False |
| results_dir | override results output directory | None |
| images_dir | override images output directory | None |

---

### Returned object

Returns a dataclass:

```python
QuantumMetricLearningResult
```

Key attributes:

```python
result.train_accuracy
result.test_accuracy

result.loss_history

result.params

result.train_embeddings
result.test_embeddings

result.train_centroids
```

When `save=True`, the workflow writes JSON results and generated figures. By
default these are stored under:

```text
results/metric_learning/
images/metric_learning/
```

---

### CLI usage

```bash
python -m qml metric-learning \
    --samples 200 \
    --layers 2 \
    --steps 50 \
    --plot \
    --save
```

Optional arguments:

```bash
--margin 0.5
--pairs-per-step 32
--log-every 10
--no-scale-data
--save
```

---

## Noise-aware execution

Finite-shot simulation is supported across all quantum workflows.

Internally implemented via:

```python
qml.set_shots(qnode, shots)
```

Example:

```python
run_vqc(shots=128)

run_quantum_kernel_classifier(shots=256)

run_trainable_quantum_kernel_classifier(
    shots_train=64,
    shots_kernel=256,
)
```

When a seed is provided, runs remain deterministic.

---

## Classical baselines

Classical reference models:

```python
from qml.classical_baselines import (
    run_logistic_classifier,
    run_svm_classifier,
    run_mlp_classifier,
    run_ridge_regression,
    run_mlp_regressor,
)
```

Example:

```python
result = run_logistic_classifier(
    n_samples=200,
    seed=123,
)
```

---

## Benchmarking

Compare models across multiple seeds.

---

### Classification benchmark

```python
from qml.benchmarks import compare_classification_models

result = compare_classification_models(
    models=[
        "vqc",
        "qcnn",
        "quantum_kernel",
        "trainable_quantum_kernel",
        "logistic_regression",
        "svm_classifier",
        "mlp_classifier",
    ],

    seeds=[123, 456, 789],

    n_samples=200,
)
```

---

### Regression benchmark

```python
from qml.benchmarks import compare_regression_models

result = compare_regression_models(
    models=[
        "vqr",
        "ridge_regression",
        "mlp_regressor",
    ],

    seeds=[123, 456],
)
```

---

### Model-specific kwargs

Per-model configuration can be passed via:

```python
result = compare_classification_models(

    models=[
        "vqc",
        "qcnn",
        "quantum_kernel",
        "trainable_quantum_kernel",
    ],

    model_kwargs={

        "vqc": {
            "shots": 128,
            "n_layers": 2,
        },

        "quantum_kernel": {
            "shots": 256,
        },

        "trainable_quantum_kernel": {

            "shots_train": 64,
            "shots_kernel": 256,

            "steps": 25,
        },
    },
)
```

Benchmark results remain consistent in structure across models.

---

## Command line interface

Run workflows directly:

```bash
python -m qml vqc --steps 50 --plot

python -m qml qcnn --steps 50 --plot

python -m qml autoencoder --steps 50 --plot

python -m qml regression --steps 50 --plot

python -m qml kernel --plot

python -m qml trainable-kernel --steps 50 --plot

python -m qml metric-learning --steps 50 --plot
```

---

### CLI benchmarks

Classification:

```bash
python -m qml benchmark classification \
    --models vqc qcnn quantum_kernel logistic_regression svm_classifier \
    --seeds 123 456
```

Regression:

```bash
python -m qml benchmark regression \
    --models vqr ridge_regression mlp_regressor \
    --seeds 123 456
```

---

## Reproducibility

All workflows support deterministic execution via:

```
seed
```

Reproducibility applies to:

• dataset generation
• parameter initialisation
• optimisation trajectories
• finite-shot sampling

Outputs can optionally be saved:

```
results/
images/
```

These directories are gitignored.

---

## Running from notebooks

Notebooks should import from the package:

```python
from qml.classifiers import run_vqc
```

rather than defining circuits inline.

This ensures:

• consistent behaviour
• reproducible outputs
• shared infrastructure
• minimal duplication

---

## Running tests

Execute:

```bash
pytest
```

---

## Development workflow

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

## Author

Sid Richards

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see LICENSE
