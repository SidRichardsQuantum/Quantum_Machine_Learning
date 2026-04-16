# Benchmark Utilities

The `qml.benchmarks` module provides helpers for comparing quantum and classical models across multiple random seeds.

Benchmarking enables:

- reproducible evaluation of model performance
- comparison between quantum and classical approaches
- estimation of performance variability due to stochastic training effects
- consistent experiment logging

Both **classification** and **regression** workflows are supported.

---

## Overview

Benchmark functions run multiple training jobs using different random seeds and aggregate performance metrics.

Typical workflow:

1. choose models to compare
2. run multiple seeds
3. compute mean and standard deviation of metrics
4. optionally save results

Example metrics include:

- classification accuracy
- regression MSE / MAE
- final loss values
- variability across seeds

Results are returned as structured dictionaries and can optionally be saved to JSON.

---

## Classification Benchmarks

Compare multiple classifiers on the same dataset.

Supported models:

- `vqc`
- `qcnn`
- `quantum_kernel`
- `logistic_regression`
- `svm_classifier`
- `mlp_classifier`

Example:

```python
from qml.benchmarks import compare_classification_models

result = compare_classification_models(
    models=["vqc", "qcnn", "quantum_kernel", "svm_classifier"],
    seeds=[0, 1, 2, 3],
    n_samples=200,
    noise=0.1,
)
```

Returned structure:

```python
{
    "benchmark_type": "classification",
    "models": [...],
    "runs": [...],
    "summary": {
        "vqc": {
            "train_accuracy": {"mean": ..., "std": ...},
            "test_accuracy": {"mean": ..., "std": ...},
            "n_runs": 4
        }
    }
}
```

Each run record includes:

```python
{
    "model": "vqc",
    "seed": 0,
    "train_accuracy": ...,
    "test_accuracy": ...,
    "final_loss": ...
}
```

---

## Regression Benchmarks

Compare regression models on the same dataset.

Supported models:

- `vqr`
- `ridge_regression`
- `mlp_regressor`

Example:

```python
from qml.benchmarks import compare_regression_models

result = compare_regression_models(
    models=["vqr", "ridge_regression"],
    seeds=[0, 1, 2],
    n_samples=200,
    noise=0.1,
)
```

Returned structure:

```python
{
    "benchmark_type": "regression",
    "summary": {
        "vqr": {
            "train_mse": {"mean": ..., "std": ...},
            "test_mse": {"mean": ..., "std": ...},
            "train_mae": {"mean": ..., "std": ...},
            "test_mae": {"mean": ..., "std": ...},
            "n_runs": 3
        }
    }
}
```

Each run record includes:

```python
{
    "model": "vqr",
    "seed": 0,
    "train_mse": ...,
    "test_mse": ...,
    "train_mae": ...,
    "test_mae": ...,
    "final_loss": ...
}
```

---

## CLI Usage

Classification benchmark:

```bash
python -m qml benchmark classification \
    --models vqc qcnn quantum_kernel svm_classifier logistic_regression \
    --seeds 123 456 789
```

Regression benchmark:

```bash
python -m qml benchmark regression \
    --models vqr ridge_regression mlp_regressor \
    --seeds 123 456
```

Default settings:

- samples: 200
- noise: 0.1
- test split: 0.25
- seed: 123

---

## Saving Benchmark Results

Results can be saved to disk:

```python
compare_classification_models(
    seeds=[0, 1, 2],
    save=True,
)
```

Saved files are placed in:

```
results/benchmarks/
```

Example output file:

```
classification_benchmark.json
```

Saved JSON includes:

- individual run records
- aggregated metrics
- dataset configuration

This allows reproducibility and later analysis.

---

## Model Selection

Models are referenced by string identifiers.

Classification:

```
vqc
qcnn
quantum_kernel
logistic_regression
svm_classifier
mlp_classifier
```

Regression:

```
vqr
ridge_regression
mlp_regressor
```

Invalid model names raise an error.

Example:

```python
compare_classification_models(
    models=["vqc", "invalid_model"]
)
```

---

## Multi-seed Evaluation

Variational quantum models depend on:

- random parameter initialisation
- optimiser stochasticity
- dataset sampling variability

Performance should therefore be evaluated across multiple seeds.

Aggregate statistics:

$$
\mu = \frac{1}{N} \sum_{i=1}^N x_i
$$

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}
$$

These values are computed for each metric.

---

## Relationship to Other Modules

Benchmark utilities call the following workflows:

Classification:

- `qml.classifiers.run_vqc`
- `qml.qcnn.run_qcnn`
- `qml.kernel_methods.run_quantum_kernel_classifier`
- `qml.classical_baselines.run_logistic_classifier`
- `qml.classical_baselines.run_svm_classifier`
- `qml.classical_baselines.run_mlp_classifier`

Regression:

- `qml.regression.run_vqr`
- `qml.classical_baselines.run_ridge_regression`
- `qml.classical_baselines.run_mlp_regressor`

Datasets are generated using shared utilities from:

```
qml.data
qml.datasets
```

ensuring consistent experimental conditions across models.

---

## When to Use Benchmarks

Benchmarking is useful when:

- comparing quantum vs classical performance
- testing sensitivity to optimiser settings
- evaluating ansatz depth
- studying generalisation performance
- generating reproducible experiment summaries

Typical workflow:

1. explore behaviour in notebooks
2. run benchmark across seeds
3. analyse aggregated metrics
4. refine model configuration
