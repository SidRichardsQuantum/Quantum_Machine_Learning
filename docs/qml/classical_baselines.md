# Classical Baselines

This module provides **classical reference models** for comparison with quantum machine learning workflows.

They serve two purposes:

1. establish performance baselines
2. provide sanity checks for datasets and training pipelines

All models follow the same interface pattern as the quantum workflows:

- consistent dataset generation
- structured result dictionaries
- optional plotting
- optional saving of results and figures

---

## Classification baselines

### Logistic regression

Linear classifier trained using maximum likelihood.

```python
from qml.classical_baselines import run_logistic_classifier

result = run_logistic_classifier(
    n_samples=200,
    noise=0.1,
    seed=123,
    plot=True,
)
```

Returns:

```
train_accuracy
test_accuracy
coefficients
intercept
predictions
probabilities
```

Logistic regression provides a simple reference for linear decision boundaries.

---

### Support vector machine (SVM)

Kernel-based classifier.

```python
from qml.classical_baselines import run_svm_classifier

result = run_svm_classifier(
    n_samples=200,
    noise=0.1,
    kernel="rbf",
)
```

Key hyperparameters:

```
kernel ∈ {linear, poly, rbf, sigmoid}
C
gamma
```

Useful for comparison with quantum kernel methods.

---

### Multi-layer perceptron (MLP) classifier

Feedforward neural network trained with backpropagation.

```python
from qml.classical_baselines import run_mlp_classifier

result = run_mlp_classifier(
    n_samples=200,
    hidden_layer_sizes=(16, 16),
)
```

Returns:

```
loss_curve
train_accuracy
test_accuracy
```

MLPs provide a flexible nonlinear baseline.

---

### Additional classifier baselines

The module also includes stronger nonlinear and nonparametric references:

```python
from qml.classical_baselines import (
    run_gaussian_process_classifier,
    run_gradient_boosting_classifier,
    run_knn_classifier,
    run_random_forest_classifier,
)
```

These baselines are useful when comparing QML classifiers against tuned
classical methods rather than only simple sanity checks.

---

## Regression baselines

### Ridge regression

Linear regression with L2 regularisation.

```python
from qml.classical_baselines import run_ridge_regression

result = run_ridge_regression(
    n_samples=200,
    noise=0.1,
    alpha=1.0,
)
```

Returns:

```
train_mse
test_mse
train_mae
test_mae
coefficients
intercept
```

Provides a simple baseline for variational quantum regression.

---

### Multi-layer perceptron (MLP) regressor

Nonlinear regression using a neural network.

```python
from qml.classical_baselines import run_mlp_regressor

result = run_mlp_regressor(
    n_samples=200,
    hidden_layer_sizes=(32, 32),
)
```

Returns:

```
loss_curve
train_mse
test_mse
train_mae
test_mae
```

MLP regression is a useful nonlinear comparison for VQR.

---

### Additional regression baselines

The module also includes:

```python
from qml.classical_baselines import (
    run_elasticnet_regression,
    run_gaussian_process_regressor,
    run_gradient_boosting_regressor,
    run_kernel_ridge_regression,
    run_knn_regressor,
    run_lasso_regression,
    run_random_forest_regressor,
    run_svr_regression,
)
```

`kernel_ridge_regression`, `svr_regression`, and
`gaussian_process_regressor` are especially relevant counterparts for quantum
kernel and quantum Gaussian-process workflows.

---

## Hyperparameter tuning and timing

Classical baseline runners accept optional tuning arguments:

```python
result = run_svm_classifier(
    tune=True,
    cv=3,
    param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
)
```

When tuning is enabled, results include:

```
tuning.enabled
tuning.cv
tuning.best_params
tuning.best_score
```

Classical result dictionaries also include a `timing` field:

```
timing.fit_seconds
timing.predict_seconds
timing.total_seconds
```

---

## Result structure

All baseline workflows return dictionaries containing:

```
model
dataset
seed
n_samples
noise
test_size

metrics
predictions
training data
test data
model parameters
```

Outputs can optionally be saved:

```
results/classification_baselines/
results/regression_baselines/

images/classification_baselines/
images/regression_baselines/
```

---

## CLI usage

Run baselines from the command line:

```bash
python -m qml logistic
python -m qml svm
python -m qml mlp-classifier

python -m qml ridge
python -m qml mlp-regressor
```

Example:

```bash
python -m qml ridge --samples 200 --noise 0.05 --alpha 0.5 --plot
```

---

## Relationship to quantum models

Classical baselines help contextualise quantum model performance:

| task                  | quantum model      | classical baseline       |
| --------------------- | ------------------ | ------------------------ |
| classification        | VQC                | logistic regression, MLP |
| kernel classification | quantum kernel SVM | classical SVM            |
| regression            | VQR                | ridge regression, MLP    |
| tree classification   | QML classifiers    | random forest, boosting  |
| kernel regression     | quantum kernels    | kernel ridge, SVR, GPR   |

Comparisons are demonstrated in the notebooks:

```
notebooks/tutorials/01-classical-vs-quantum-classifier.ipynb
notebooks/tutorials/02-classical-vs-quantum-regressor.ipynb
```

---

## Design notes

The implementation prioritises:

- minimal abstractions
- consistent interfaces
- reproducible experiments
- compatibility with sklearn
- clear comparison with quantum workflows

The goal is not to provide a full sklearn wrapper layer, but rather lightweight baselines that integrate naturally with the rest of the package.
