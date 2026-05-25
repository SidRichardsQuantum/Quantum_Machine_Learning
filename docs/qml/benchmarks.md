# Benchmark Utilities

The `qml.benchmarks` module compares quantum and classical models across
multiple random seeds under shared dataset settings. The helpers are intended
for reproducible model-quality comparisons, not claims of quantum advantage.

Benchmark outputs include:

- train/test metrics
- runtime totals
- fit/predict timing breakdowns when workflows expose them
- train/test generalization gaps
- mean, standard deviation, and 95% confidence intervals
- paired deltas against the best included classical baseline
- tuning metadata for tuned classical baselines
- environment metadata for reproducibility

Both classification and regression workflows are supported.

---

## Classification Benchmarks

Supported models:

- `vqc`
- `qcnn`
- `quantum_kernel`
- `trainable_quantum_kernel`
- `quantum_metric_learning`
- `quantum_reservoir`
- `logistic_regression`
- `svm_classifier`
- `mlp_classifier`
- `random_forest_classifier`
- `gradient_boosting_classifier`
- `knn_classifier`
- `gaussian_process_classifier`

Example:

```python
from qml.benchmarks import compare_classification_models

result = compare_classification_models(
    models=[
        "vqc",
        "quantum_kernel",
        "svm_classifier",
        "random_forest_classifier",
        "logistic_regression",
    ],
    seeds=[0, 1, 2, 3],
    n_samples=200,
    noise=0.1,
)
```

Each classification run record includes:

```python
{
    "model": "vqc",
    "seed": 0,
    "dataset": "moons",
    "train_accuracy": ...,
    "test_accuracy": ...,
    "generalization_gap": ...,
    "runtime_seconds": ...,
    "timing": {"total_seconds": ...},
    "final_loss": ...,
}
```

For classification, `generalization_gap` is:

```text
train_accuracy - test_accuracy
```

---

## Regression Benchmarks

Supported models:

- `vqr`
- `quantum_kernel_regressor`
- `trainable_quantum_kernel_regressor`
- `quantum_gaussian_process_regressor`
- `quantum_reservoir_regressor`
- `ridge_regression`
- `mlp_regressor`
- `kernel_ridge_regression`
- `svr_regression`
- `gaussian_process_regressor`
- `random_forest_regressor`
- `gradient_boosting_regressor`
- `knn_regressor`
- `lasso_regression`
- `elasticnet_regression`

Example:

```python
from qml.benchmarks import compare_regression_models

result = compare_regression_models(
    models=["vqr", "ridge_regression", "kernel_ridge_regression", "svr_regression"],
    seeds=[0, 1, 2],
    n_samples=200,
    noise=0.1,
)
```

For regression, `generalization_gap` is:

```text
test_mse - train_mse
```

---

## Result Structure

Returned benchmark dictionaries include:

```python
{
    "benchmark_type": "classification",
    "models": [...],
    "seeds": [...],
    "dataset": "moons",
    "runs": [...],
    "summary": {
        "svm_classifier": {
            "test_accuracy": {
                "mean": ...,
                "std": ...,
                "n": ...,
                "ci95_low": ...,
                "ci95_high": ...,
            },
            "runtime_seconds": {...},
            "generalization_gap": {...},
            "n_runs": 4,
        }
    },
    "best_model": {
        "model": "svm_classifier",
        "metric": "test_accuracy",
        "value": ...,
        "higher_is_better": True,
    },
    "paired_vs_best_classical": {...},
    "metadata": {...},
}
```

The `best_model` field is selected from the aggregate test metric:

- classification: highest mean `test_accuracy`
- regression: lowest mean `test_mse`

Use it as a convenience summary only. Always inspect run records, confidence
intervals, paired classical deltas, and runtime before drawing conclusions.

---

## Paired Classical Comparison

`paired_vs_best_classical` compares each selected model against the best
included classical baseline on matching seeds. It reports:

- reference classical model
- seed-wise mean delta and confidence interval
- win/loss/tie counts
- number of paired seeds

This is the preferred summary for evaluating whether a QML model improves over
classical references in the same benchmark call.

---

## Classical Hyperparameter Tuning

Classical baselines can be tuned with small `GridSearchCV` defaults:

```python
result = compare_classification_models(
    models=["quantum_kernel", "svm_classifier", "random_forest_classifier"],
    seeds=[0, 1, 2],
    tune_classical=True,
    cv=3,
)
```

Per-model grids can be overridden through `model_kwargs`:

```python
result = compare_regression_models(
    models=["vqr", "kernel_ridge_regression", "svr_regression"],
    tune_classical=True,
    model_kwargs={
        "kernel_ridge_regression": {
            "param_grid": {
                "alpha": [0.01, 0.1, 1.0],
                "kernel": ["rbf"],
                "gamma": [0.1, 1.0],
            }
        }
    },
)
```

Quantum model tuning is supplied explicitly through `model_kwargs`, for example
by sweeping layers, optimizer steps, step size, shots, or kernel settings across
separate benchmark calls.

---

## Datasets

Classification datasets:

```text
moons
circles
blobs
xor
linear
breast_cancer
wine
```

Regression datasets:

```text
linear
sine
polynomial
friedman
diabetes
```

---

## CLI Benchmark Presets

Run benchmark summaries directly from the package command:

```bash
qml-pennylane benchmark classification \
  --models vqc quantum_kernel quantum_reservoir svm_classifier \
  --seeds 0 1 2 \
  --samples 100
```

```bash
qml-pennylane benchmark regression \
  --models vqr quantum_kernel_regressor quantum_reservoir_regressor ridge_regression \
  --seeds 0 1 2 \
  --samples 100
```

Finite-shot sweeps compare analytic execution against selected shot counts:

```bash
qml-pennylane benchmark finite-shots \
  --shots analytic 64 128 512 \
  --seeds 0 1 2 \
  --samples 70
```

The real-data options are projected to two features so they remain compatible
with the compact quantum examples and visualizers. They are useful sanity
checks, not substitutes for domain-specific benchmark suites.

---

## CLI Usage

Classification benchmark:

```bash
python -m qml benchmark classification \
    --models vqc qcnn quantum_kernel svm_classifier random_forest_classifier \
    --seeds 123 456 789 \
    --tune-classical
```

Regression benchmark:

```bash
python -m qml benchmark regression \
    --models vqr ridge_regression kernel_ridge_regression svr_regression \
    --seeds 123 456 \
    --tune-classical
```

Default settings:

- samples: 200
- noise: 0.1
- test split: 0.25
- seed: 123
- classical tuning: disabled unless `--tune-classical` is provided

---

## Saving Results

Results can be saved to:

```text
results/benchmarks/
```

Saved JSON includes individual run records, aggregate metrics, timing summaries,
paired classical comparisons, tuning metadata, environment metadata, and the
dataset configuration.

---

## Interpretation Checklist

Before publishing a benchmark table, record:

- model list and model-specific kwargs
- dataset name, sample count, split, noise level, and seed list
- analytic or finite-shot execution settings
- package, Python, scikit-learn, and PennyLane versions
- classical baselines and tuning grids
- mean, standard deviation, and confidence intervals
- paired deltas against the best classical baseline
- runtime totals and fit/predict breakdowns
- train/test generalization gaps

Benchmarks in this package make comparisons reproducible and auditable. They do
not establish quantum advantage by themselves.
