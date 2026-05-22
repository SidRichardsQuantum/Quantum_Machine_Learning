# Notebooks

These notebooks are thin clients around the reusable APIs in `src/qml/`. They are
intended to be runnable from the repository root, `notebooks/`, or their own
subdirectories.

## Tutorial notebooks

These are tutorial-style walkthroughs of the package algorithms and reusable
implementations. They live in `notebooks/tutorials/` so they are separate from
domain real examples.

| Notebook | Purpose |
| --- | --- |
| `tutorials/01-classical-vs-quantum-classifier.ipynb` | Compare classical and quantum classifiers on synthetic classification data. |
| `tutorials/02-classical-vs-quantum-regressor.ipynb` | Compare classical and quantum regressors on synthetic regression data. |
| `tutorials/03-variational-quantum-classifier.ipynb` | Train a variational quantum classifier. |
| `tutorials/04-variational-quantum-regressor.ipynb` | Train a variational quantum regressor. |
| `tutorials/05-quantum-kernel-classifier.ipynb` | Train and evaluate a quantum kernel classifier. |
| `tutorials/06-quantum-kernel-estimators.ipynb` | Use dataset-agnostic quantum kernel classifier and regressor estimators. |
| `tutorials/07-variational-quantum-estimators.ipynb` | Use sklearn-like variational quantum classifier and regressor estimators. |
| `tutorials/08-sequence-window-quantum-forecasting.ipynb` | Build sequence windows and train a quantum one-step forecaster. |
| `tutorials/09-quantum-metric-learning.ipynb` | Learn a quantum embedding geometry for classification. |
| `tutorials/10-quantum-convolutional-neural-network.ipynb` | Train a compact quantum convolutional neural network. |
| `tutorials/11-quantum-autoencoder.ipynb` | Compress and reconstruct structured quantum states. |

## Real example notebooks

These notebooks use small reproducible physics or maths simulators and then
solve the resulting supervised-learning task with reusable package components.
Each notebook ends with a validation block that prints dataset metadata,
metrics, sample predictions, and a `passed` flag.
They are correctness and usage examples with classical sanity baselines; they
do not claim quantum advantage.

Shared text output uses `qml.reporting.print_section`, which renders compact
human-readable tables.

| Notebook | Problem | Package-backed approach |
| --- | --- | --- |
| `real_examples/01-rabi-oscillation-parameter-inference.ipynb` | Recover the Rabi frequency of a driven two-level system from sparse population measurements. | Package angle embedding, hardware-efficient ansatz, optimizer, training loop, and regression metrics. |
| `real_examples/02-ising-correlation-temperature-classifier.ipynb` | Classify low- and high-temperature Ising model samples from Monte Carlo correlation features. | Angle embedding, PennyLane quantum fidelity kernel, package accuracy metric. |
| `real_examples/03-lorenz-regime-classifier.ipynb` | Classify Lorenz-system parameter regimes from short trajectory statistics. | Angle embedding, PennyLane quantum fidelity kernel, package accuracy metric. |
| `real_examples/04-condensed-matter-tfim-phase-classifier.ipynb` | Classify finite-size transverse-field Ising model samples as ferromagnetic or paramagnetic from correlation features. | Angle embedding, PennyLane quantum fidelity kernel, package accuracy metric. |
| `real_examples/05-pendulum-trajectory-surrogate.ipynb` | Learn a small-angle pendulum trajectory surrogate from initial state and time. | Package angle embedding, hardware-efficient ansatz, optimizer, training loop, and regression metrics. |
| `real_examples/06-damped-oscillator-parameter-inference.ipynb` | Recover a damped oscillator damping coefficient from sparse displacement measurements. | Package angle embedding, hardware-efficient ansatz, optimizer, training loop, and regression metrics. |

## Application notebook ideas

These are good candidates for future notebooks because each maps to a concrete
physics, mathematics, or engineering workflow and can be benchmarked against a
classical baseline.

| Area | Example problem | Candidate QML approach | Useful output |
| --- | --- | --- | --- |
| Condensed matter | Classify phases of the transverse-field Ising model from spin correlation features. | Quantum kernel classifier or VQC. | Phase-boundary accuracy and robustness to noisy measurements. |
| Dynamical systems | Learn a surrogate for harmonic oscillator, pendulum, or Duffing oscillator trajectories. | Variational quantum regressor. | Forecast error versus ridge/MLP baselines. |
| Signal processing | Classify real sensor signals or spectra after dimensionality reduction. | Quantum metric learning or quantum kernel classifier. | Accuracy, confusion matrix, embedding visualization. |
| Inverse problems | Infer model parameters from simulated measurements. | VQR or metric learning. | Parameter recovery error under measurement noise. |

For real-world usefulness, each application notebook should include:

- a clearly named domain problem
- a small reproducible dataset or simulator
- a classical baseline
- deterministic seeds
- finite-shot or noise-aware execution when relevant
- metrics that matter in the domain, not only generic loss curves
