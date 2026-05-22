# Benchmark Utilities

The `qml.benchmarks` module provides helpers for comparing quantum and
classical models across multiple random seeds under the same dataset settings.

Benchmarking enables:

- reproducible evaluation of model performance
- comparison between quantum and classical approaches
- estimation of performance variability due to stochastic training effects
- runtime tracking for smoke-scale comparisons
- train/test gap tracking for basic overfitting checks
- consistent experiment logging

Both **classification** and **regression** workflows are supported.

---

## Overview

Benchmark functions run multiple training jobs using different random seeds and
aggregate performance metrics. They are intended for reproducible comparisons,
not claims of quantum advantage.

Typical workflow:

1. choose models to compare
2. run multiple seeds
3. compute mean and standard deviation of metrics, runtime, and train/test gap
4. optionally save results

Example metrics include:

- classification accuracy
- regression MSE / MAE
- final loss values when the model exposes them
- runtime in seconds
- train/test generalization gap
- variability across seeds

Results are returned as structured dictionaries and can optionally be saved to JSON.

---

## Classification Benchmarks

Compare multiple classifiers on the same dataset.

Supported models:

- `vqc`
- `qcnn`
- `quantum_kernel`
- `trainable_quantum_kernel`
- `quantum_metric_learning`
- `logistic_regression`
- `svm_classifier`
- `mlp_classifier`

Example:

```python
from qml.benchmarks import compare_classification_models

result = compare_classification_models(
    models=["vqc", "quantum_kernel", "svm_classifier", "logistic_regression"],
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
            "generalization_gap": {"mean": ..., "std": ...},
            "runtime_seconds": {"mean": ..., "std": ...},
            "n_runs": 4
        }
    },
    "best_model": {
        "model": "svm_classifier",
        "metric": "test_accuracy",
        "value": ...,
        "higher_is_better": True
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
    "generalization_gap": ...,
    "runtime_seconds": ...,
    "final_loss": ...
}
```

For classification, `generalization_gap` is computed as:

```text
train_accuracy - test_accuracy
```

A large positive value can indicate overfitting. A negative value can happen on
small splits and should be interpreted across multiple seeds rather than from a
single run.

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
            "generalization_gap": {"mean": ..., "std": ...},
            "runtime_seconds": {"mean": ..., "std": ...},
            "n_runs": 3
        }
    },
    "best_model": {
        "model": "ridge_regression",
        "metric": "test_mse",
        "value": ...,
        "higher_is_better": False
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
    "generalization_gap": ...,
    "runtime_seconds": ...,
    "final_loss": ...
}
```

For regression, `generalization_gap` is computed as:

```text
test_mse - train_mse
```

Positive values indicate worse test error than train error.

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

For release-quality comparisons, prefer explicit seed lists and include at
least one classical baseline in the model list. Small default runs are useful
for smoke checks, but they are not enough to evaluate model quality.

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
- runtime summaries
- train/test generalization-gap summaries
- best model according to the primary test metric
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
trainable_quantum_kernel
quantum_metric_learning
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

The `best_model` field is selected from the aggregated test metric:

- classification: highest mean `test_accuracy`
- regression: lowest mean `test_mse`

Use it as a convenience summary only. Always inspect the full run records,
standard deviations, and runtime before drawing conclusions.

---

## Relationship to Other Modules

Benchmark utilities call the following workflows:

Classification:

- `qml.classifiers.run_vqc`
- `qml.qcnn.run_qcnn`
- `qml.kernel_methods.run_quantum_kernel_classifier`
- `qml.trainable_kernels.run_trainable_quantum_kernel_classifier`
- `qml.metric_learning.run_quantum_metric_learner`
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
```

ensuring consistent experimental conditions across models.

Metric-learning benchmarks use the same classification dataset name, sample
count, split, and seed, but ignore the synthetic dataset `noise` parameter
because the metric-learning workflow does not expose that setting.

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

## Interpretation Checklist

Before publishing a benchmark table, record:

- model list and model-specific kwargs
- dataset name, sample count, split, noise level, and seed list
- analytic or finite-shot execution settings
- package version and Python/PennyLane versions
- classical baselines included in the comparison
- mean and standard deviation across seeds
- runtime and train/test gap

Benchmarks in this package are designed to make comparisons reproducible and
auditable. They do not establish quantum advantage by themselves.
