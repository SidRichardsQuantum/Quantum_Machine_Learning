# Results

These reference results are generated from the public package APIs used by the notebooks.
Notebook-derived result pages are generated separately from executed notebook outputs:

- [Tutorial notebook results](results-tutorials.html)
- [Real example notebook results](results-real-examples.html)
- [Benchmark notebook results](results-benchmarks.html)

The configurations are intentionally small enough for the result-refresh workflow to
regenerate in CI. They are reproducible smoke-scale examples, not quantum-advantage
claims.

## Environment

<<<<<<< HEAD
- Generated: 2026-06-01 08:51:28 UTC
- Git commit: `508c84e`
- Python: `3.12.13`
- Package version: `0.2.9`
=======
- Generated: 2026-06-01 09:47:12 UTC
- Git commit: `508c84e`
- Python: `3.12.1`
- Package version: `0.2.10`
>>>>>>> 0f4b077 (Release v0.2.10)
- PennyLane: `0.45.0`
- Matplotlib backend: `Agg`
- Default execution: analytic `default.qubit` unless a shot count is listed

## Summary

| Workflow | Primary metric | Value | Runtime |
| --- | --- | ---: | ---: |
<<<<<<< HEAD
| Variational quantum classifier | `train_accuracy` | 0.4595 | 15.81 s |
| Variational quantum regression | `train_mse` | 0.9098 | 1.95 s |
| Quantum convolutional neural network | `train_accuracy` | 0.8333 | 79.08 s |
| Quantum autoencoder | `test_compression_fidelity` | 0.7014 | 1.79 s |
| Quantum kernel classifier | `train_accuracy` | 0.8519 | 1.32 s |
| Trainable quantum kernel | `train_accuracy` | 0.7333 | 14.95 s |
| Trainable quantum kernel regressor | `train_mse` | 0.0125 | 6.46 s |
| Quantum metric learning | `train_accuracy` | 0.5946 | 2.22 s |
=======
| Variational quantum classifier | `train_accuracy` | 0.4595 | 29.49 s |
| Variational quantum regression | `train_mse` | 0.9098 | 4.13 s |
| Quantum convolutional neural network | `train_accuracy` | 0.8333 | 159.74 s |
| Quantum autoencoder | `test_compression_fidelity` | 0.7014 | 3.89 s |
| Quantum kernel classifier | `train_accuracy` | 0.8519 | 2.24 s |
| Trainable quantum kernel | `train_accuracy` | 0.7333 | 27.68 s |
| Trainable quantum kernel regressor | `train_mse` | 0.0125 | 12.39 s |
| Quantum metric learning | `train_accuracy` | 0.5946 | 4.76 s |
>>>>>>> 0f4b077 (Release v0.2.10)

## Variational quantum classifier

Configuration:

`dataset=moons`, `n_samples=50`, `noise=0.1000`, `seed=123`, `n_layers=1`, `steps=8`, `shots=analytic`

| Metric | Value |
| --- | ---: |
| `train_accuracy` | 0.4595 |
| `test_accuracy` | 0.6154 |
| `final_loss` | 1.4790 |
<<<<<<< HEAD
| `runtime_seconds` | 15.81 |
=======
| `runtime_seconds` | 29.49 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![moons embedangle layers1 steps8 samples50 noise0p1 seed123 analytic noiseless dataset](../pages/assets/reference-results/vqc/moons_embedangle_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_dataset.png)
![moons embedangle layers1 steps8 samples50 noise0p1 seed123 analytic noiseless decision boundary](../pages/assets/reference-results/vqc/moons_embedangle_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_decision_boundary.png)
![moons embedangle layers1 steps8 samples50 noise0p1 seed123 analytic noiseless loss](../pages/assets/reference-results/vqc/moons_embedangle_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_loss.png)

## Variational quantum regression

Configuration:

`dataset=linear`, `n_samples=50`, `noise=0.1000`, `seed=123`, `n_layers=1`, `steps=8`, `shots=analytic`

| Metric | Value |
| --- | ---: |
| `train_mse` | 0.9098 |
| `test_mse` | 0.3316 |
| `final_loss` | 0.9841 |
<<<<<<< HEAD
| `runtime_seconds` | 1.95 |
=======
| `runtime_seconds` | 4.13 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![linear layers1 steps8 samples50 noise0p1 seed123 analytic noiseless dataset](../pages/assets/reference-results/vqr/linear_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_dataset.png)
![linear layers1 steps8 samples50 noise0p1 seed123 analytic noiseless loss](../pages/assets/reference-results/vqr/linear_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_loss.png)
![linear layers1 steps8 samples50 noise0p1 seed123 analytic noiseless predictions](../pages/assets/reference-results/vqr/linear_layers1_steps8_samples50_noise0p1_seed123_analytic_noiseless_predictions.png)

## Quantum convolutional neural network

Configuration:

`dataset=moons`, `n_samples=40`, `noise=0.1000`, `seed=123`, `steps=6`, `shots=analytic`

| Metric | Value |
| --- | ---: |
| `train_accuracy` | 0.8333 |
| `test_accuracy` | 0.9000 |
| `final_loss` | 0.4556 |
<<<<<<< HEAD
| `runtime_seconds` | 79.08 |
=======
| `runtime_seconds` | 159.74 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![moons steps6 samples40 noise0p1 seed123 analytic noiseless dataset](../pages/assets/reference-results/qcnn/moons_steps6_samples40_noise0p1_seed123_analytic_noiseless_dataset.png)
![moons steps6 samples40 noise0p1 seed123 analytic noiseless decision boundary](../pages/assets/reference-results/qcnn/moons_steps6_samples40_noise0p1_seed123_analytic_noiseless_decision_boundary.png)
![moons steps6 samples40 noise0p1 seed123 analytic noiseless loss](../pages/assets/reference-results/qcnn/moons_steps6_samples40_noise0p1_seed123_analytic_noiseless_loss.png)

## Quantum autoencoder

Configuration:

`family=correlated`, `n_samples=32`, `noise=0.0500`, `seed=123`, `n_layers=1`, `latent_qubits=2`, `steps=6`

| Metric | Value |
| --- | ---: |
| `test_compression_fidelity` | 0.7014 |
| `test_reconstruction_fidelity` | 0.7014 |
| `final_loss` | 0.3676 |
<<<<<<< HEAD
| `runtime_seconds` | 1.79 |
=======
| `runtime_seconds` | 3.89 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![correlated layers1 latent2 steps6 samples32 noise0p05 seed123 loss](../pages/assets/reference-results/autoencoder/correlated_layers1_latent2_steps6_samples32_noise0p05_seed123_loss.png)

## Quantum kernel classifier

Configuration:

`dataset=moons`, `n_samples=36`, `noise=0.1000`, `seed=123`, `shots=analytic`

| Metric | Value |
| --- | ---: |
| `train_accuracy` | 0.8519 |
| `test_accuracy` | 0.8889 |
<<<<<<< HEAD
| `runtime_seconds` | 1.32 |
=======
| `runtime_seconds` | 2.24 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![moons samples36 noise0p1 seed123 analytic noiseless dataset](../pages/assets/reference-results/quantum_kernel/moons_samples36_noise0p1_seed123_analytic_noiseless_dataset.png)
![moons samples36 noise0p1 seed123 analytic noiseless kernel test](../pages/assets/reference-results/quantum_kernel/moons_samples36_noise0p1_seed123_analytic_noiseless_kernel_test.png)
![moons samples36 noise0p1 seed123 analytic noiseless kernel train](../pages/assets/reference-results/quantum_kernel/moons_samples36_noise0p1_seed123_analytic_noiseless_kernel_train.png)

## Trainable quantum kernel

Configuration:

`dataset=moons`, `n_samples=20`, `noise=0.1000`, `seed=123`, `embedding_layers=1`, `steps=2`, `shots_train=analytic`, `shots_kernel=analytic`

| Metric | Value |
| --- | ---: |
| `train_accuracy` | 0.7333 |
| `test_accuracy` | 0.8000 |
| `final_alignment` | 0.1755 |
| `final_loss` | -0.1755 |
<<<<<<< HEAD
| `runtime_seconds` | 14.95 |
=======
| `runtime_seconds` | 27.68 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![moons trainable kernel embdata reupload layers1 steps2 samples20 noise0p1 seed123 analytic analytic noiseless alignment](../pages/assets/reference-results/trainable_kernel/moons_trainable_kernel_embdata_reupload_layers1_steps2_samples20_noise0p1_seed123_analytic_analytic_noiseless_alignment.png)
![moons trainable kernel embdata reupload layers1 steps2 samples20 noise0p1 seed123 analytic analytic noiseless dataset](../pages/assets/reference-results/trainable_kernel/moons_trainable_kernel_embdata_reupload_layers1_steps2_samples20_noise0p1_seed123_analytic_analytic_noiseless_dataset.png)
![moons trainable kernel embdata reupload layers1 steps2 samples20 noise0p1 seed123 analytic analytic noiseless kernel test](../pages/assets/reference-results/trainable_kernel/moons_trainable_kernel_embdata_reupload_layers1_steps2_samples20_noise0p1_seed123_analytic_analytic_noiseless_kernel_test.png)
![moons trainable kernel embdata reupload layers1 steps2 samples20 noise0p1 seed123 analytic analytic noiseless kernel train](../pages/assets/reference-results/trainable_kernel/moons_trainable_kernel_embdata_reupload_layers1_steps2_samples20_noise0p1_seed123_analytic_analytic_noiseless_kernel_train.png)
![moons trainable kernel embdata reupload layers1 steps2 samples20 noise0p1 seed123 analytic analytic noiseless loss](../pages/assets/reference-results/trainable_kernel/moons_trainable_kernel_embdata_reupload_layers1_steps2_samples20_noise0p1_seed123_analytic_analytic_noiseless_loss.png)

## Trainable quantum kernel regressor

Configuration:

`dataset=sine`, `n_samples=20`, `noise=0.1000`, `seed=123`, `embedding_layers=1`, `steps=2`, `shots_train=analytic`, `shots_kernel=analytic`, `alpha=0.0010`

| Metric | Value |
| --- | ---: |
| `train_mse` | 0.0125 |
| `test_mse` | 0.4359 |
| `final_alignment` | 0.4287 |
| `final_loss` | -0.4287 |
<<<<<<< HEAD
| `runtime_seconds` | 6.46 |
=======
| `runtime_seconds` | 12.39 |
>>>>>>> 0f4b077 (Release v0.2.10)


## Quantum metric learning

Configuration:

`dataset=moons`, `samples=50`, `seed=42`, `layers=1`, `steps=8`, `pairs_per_step=16`, `log_every=0`

| Metric | Value |
| --- | ---: |
| `train_accuracy` | 0.5946 |
| `test_accuracy` | 0.6923 |
| `final_loss` | 0.1152 |
<<<<<<< HEAD
| `runtime_seconds` | 2.22 |
=======
| `runtime_seconds` | 4.76 |
>>>>>>> 0f4b077 (Release v0.2.10)

Images:

![moons layers1 steps8 samples50 margin0p5 seed42 embeddings](../pages/assets/reference-results/metric_learning/moons_layers1_steps8_samples50_margin0p5_seed42_embeddings.png)
![moons layers1 steps8 samples50 margin0p5 seed42 loss](../pages/assets/reference-results/metric_learning/moons_layers1_steps8_samples50_margin0p5_seed42_loss.png)

## Reproduce

Regenerate this file and notebook-result pages from the repository root:

```bash
python docs/pages/generate_results.py
```

The Refresh results workflow regenerates this file before the Pages workflow publishes
the committed result artifacts.
Generated images are written under `docs/pages/assets/reference-results/` and embedded above.
