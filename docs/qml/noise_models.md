# Noise Models

The package supports small, explicit noise-model dictionaries for circuit-backed
workflows. Noise is opt-in: when `noise_model=None` or all probabilities are
zero, workflows keep using noiseless analytic or finite-shot execution.

## Supported Channels

Use `qml.noise.build_noise_model(...)` or pass a dictionary with these keys:

| Key | Circuit effect |
| --- | --- |
| `depolarizing` | Applies `qml.DepolarizingChannel` to every circuit wire. |
| `amplitude_damping` | Applies `qml.AmplitudeDamping` to every circuit wire. |
| `readout_error` | Applies a `qml.BitFlip` channel immediately before measurement. |

All values are probabilities in `[0, 1]`. Nonzero noise models run on
PennyLane `default.mixed`; noiseless workflows continue to use `default.qubit`.

## Example

```python
from qml import QuantumKernel, build_noise_model, run_vqc

noise_model = build_noise_model(
    depolarizing=0.01,
    amplitude_damping=0.0,
    readout_error=0.02,
)

result = run_vqc(
    n_samples=80,
    n_layers=1,
    steps=10,
    shots=256,
    noise_model=noise_model,
)

kernel = QuantumKernel(shots=256, noise_model=noise_model)
```

## CLI

Circuit-backed commands and benchmark commands accept:

```bash
qml-pennylane vqc --shots 256 --depolarizing 0.01 --readout-error 0.02

qml-pennylane benchmark finite-shots \
  --shots analytic 128 512 \
  --depolarizing 0.005 \
  --readout-error 0.02
```

## Interpretation

Noisy results should be reported next to the matching noiseless analytic or
finite-shot result using the same train/test split and seeds. Treat these runs
as robustness checks, not hardware-validated performance claims.
