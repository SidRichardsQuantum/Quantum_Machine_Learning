# Embeddings and Ansatz Helpers

The package provides reusable circuit-building helpers for feature maps and
parameterized ansatz templates. These helpers are intentionally general and do
not encode any physics-specific assumptions.

## Available Embeddings

Use `qml.embeddings.available_embeddings()` to list canonical names:

```python
from qml.embeddings import available_embeddings

print(available_embeddings())
```

Current canonical names:

| Name | Trainable | Purpose |
| --- | --- | --- |
| `angle` | No | One feature per qubit using `qml.AngleEmbedding`. |
| `amplitude` | No | Amplitude encoding with validation, padding, and normalization. |
| `zz` | No | Second-order feature map with nearest-neighbor ZZ phases. |
| `iqp` | No | Compact IQP-style feature map with commuting phase structure. |
| `data_reupload` | Yes | Repeated feature encoding with trainable rotations. |

## Angle Embedding

```python
from qml.embeddings import apply_angle_embedding

apply_angle_embedding(x, wires=[0, 1, 2])
```

The input length must match the number of wires.

Here `x` is the input feature vector and `wires` is the ordered list of qubits
receiving those features.

## Amplitude Embedding

```python
from qml.embeddings import apply_amplitude_embedding

apply_amplitude_embedding(x, wires=[0, 1])
```

The helper accepts up to `2 ** n_wires` features, pads shorter vectors with
zeros, and normalizes the state. Longer vectors raise `ValueError` rather than
silently truncating information.

Here `n_wires` is the number of qubits in `wires`, so the amplitude state has
dimension `2 ** n_wires`.

## ZZ Feature Map

```python
from qml.embeddings import apply_zz_feature_map

apply_zz_feature_map(x, wires=[0, 1, 2])
```

This map applies single-qubit phase rotations and nearest-neighbor pairwise
products through CNOT-RZ-CNOT blocks.

Here each component of `x` supplies a feature angle, and adjacent entries define
the pairwise products used in the ZZ interaction terms.

## IQP Feature Map

```python
from qml.embeddings import apply_iqp_feature_map

apply_iqp_feature_map(x, wires=[0, 1, 2])
```

The IQP-style map uses Hadamards, single-qubit phase rotations, and ring ZZ
interactions.

Here "ring" means the pairwise interaction pattern wraps from the final wire
back to the first wire.

## Data Re-uploading Embedding

```python
from qml.embeddings import (
    apply_data_reuploading_embedding,
    embedding_parameter_shape,
)

shape = embedding_parameter_shape("data_reupload", n_layers=2, n_qubits=3)
apply_data_reuploading_embedding(x, weights, wires=[0, 1, 2])
```

Expected weight shape is `(n_layers, n_qubits, 3)`.

Here `n_layers` is the number of repeated encoding layers, `n_qubits` is the
number of wires, and the final dimension stores three trainable rotation values
per qubit.

## Ansatz Helpers

The default hardware-efficient ansatz uses trainable `RY` and `RZ` rotations
with a CNOT entangling chain:

```python
from qml.ansatz import apply_hardware_efficient_ansatz, parameter_shape

params = rng.normal(size=parameter_shape(n_layers=2, n_qubits=3))
apply_hardware_efficient_ansatz(params, wires=[0, 1, 2])
```

Here `params` is the trainable parameter tensor, `n_layers` is the number of
ansatz layers, and `n_qubits` is the number of circuit wires.

For stronger entangling templates:

```python
from qml.ansatz import (
    apply_strongly_entangling_ansatz,
    strongly_entangling_parameter_shape,
)

params = rng.normal(size=strongly_entangling_parameter_shape(2, 3))
apply_strongly_entangling_ansatz(params, wires=[0, 1, 2])
```

These helpers are used by package estimators and can also be used directly in
custom PennyLane QNodes.
