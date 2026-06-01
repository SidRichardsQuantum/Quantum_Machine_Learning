# Quantum Reservoir Models

Quantum reservoir models use a fixed quantum circuit as a nonlinear feature
map. The quantum circuit is not trained; a classical readout is fitted on the
measured expectation-value features.

This keeps training cheap and makes the estimators useful for small tabular
datasets, time-window features, and dynamical-system summaries.

## Feature Map

`QuantumReservoirFeatures` maps each input row to Pauli-Z expectations:

```python
from qml import QuantumReservoirFeatures

reservoir = QuantumReservoirFeatures(
    n_qubits=4,
    n_layers=2,
    seed=123,
)
features = reservoir.fit_transform(x_train)
```

Here `x_train` is the training feature matrix with one sample per row, and
`features` is the transformed matrix of reservoir expectation-value features.

The circuit:

1. repeats or truncates each input row to the reservoir qubit count
2. angle-encodes bounded features
3. applies fixed random rotations
4. applies nearest-neighbor entanglers
5. returns one expectation feature per qubit

Constructor parameters:

| Parameter | Meaning |
| --- | --- |
| `n_qubits` | Reservoir width. Defaults to input feature count. |
| `n_layers` | Number of fixed random reservoir layers. |
| `seed` | Random seed for fixed circuit weights. |
| `shots` | Optional finite-shot execution. `None` is analytic. |
| `input_scale` | Scale applied to bounded input angles. |
| `weight_scale` | Scale of fixed random circuit weights. |

## QuantumReservoirRegressor

`QuantumReservoirRegressor` fits a ridge readout on reservoir features.

```python
from qml import QuantumReservoirFeatures, QuantumReservoirRegressor

model = QuantumReservoirRegressor(
    QuantumReservoirFeatures(n_qubits=4, n_layers=2, seed=123),
    alpha=1e-3,
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
```

Here `y_train` is the continuous training-target vector, `x_test` is the test
feature matrix, and `pred` contains predicted continuous target values.

The regressor exposes `score(x, y)` as negative mean-squared error and
`mean_absolute_error(x, y)` as a convenience metric.

## QuantumReservoirClassifier

`QuantumReservoirClassifier` fits a logistic readout on reservoir features.

```python
from qml import QuantumReservoirClassifier, QuantumReservoirFeatures

model = QuantumReservoirClassifier(
    QuantumReservoirFeatures(n_qubits=4, n_layers=2, seed=123),
    c=1.0,
)
model.fit(x_train, y_train)
pred = model.predict(x_test)
proba = model.predict_proba(x_test)
```

Here `y_train` contains class labels, `pred` contains predicted class labels,
and `proba` contains predicted class probabilities.

The classifier supports binary and multiclass labels through scikit-learn's
logistic regression implementation.

## When To Use

Reservoir models are useful when:

- the input already contains short windows or summary features
- a nonlinear representation is needed
- full variational circuit training would be too expensive
- repeatable fixed-feature baselines are useful

They are general QML estimators. Domain-specific simulators and feature
extractors should stay in notebooks or user code.
