# Real Example Notebook Results

Executed outputs from the domain-oriented notebooks in `notebooks/real_examples/`. These examples use small reproducible physics, mathematics, or dynamical-system tasks.

## Environment

- Generated: 2026-05-25 03:43:26 UTC
- Git commit: `9672523`
- Python: `3.12.1`
- Package version: `0.2.4`
- Matplotlib backend: `Agg`

## Summary

| Notebook | Text result blocks | Plots |
| --- | ---: | ---: |
| [notebooks/real_examples/01-rabi-oscillation-parameter-inference.ipynb](#quantum-dynamics-rabi-oscillation-parameter-inference) | 3 | 1 |
| [notebooks/real_examples/02-ising-correlation-temperature-classifier.ipynb](#statistical-physics-ising-temperature-classification) | 3 | 1 |
| [notebooks/real_examples/03-lorenz-regime-classifier.ipynb](#nonlinear-dynamics-lorenz-regime-classification) | 3 | 1 |
| [notebooks/real_examples/04-condensed-matter-tfim-phase-classifier.ipynb](#condensed-matter-tfim-phase-classification) | 3 | 1 |
| [notebooks/real_examples/05-pendulum-trajectory-surrogate.ipynb](#dynamical-systems-pendulum-trajectory-surrogate) | 3 | 1 |
| [notebooks/real_examples/06-damped-oscillator-parameter-inference.ipynb](#inverse-problems-damped-oscillator-parameter-inference) | 3 | 1 |
| [notebooks/real_examples/07-tfim-hamiltonian-parameter-inference.ipynb](#condensed-matter-tfim-hamiltonian-parameter-inference) | 3 | 1 |
| [notebooks/real_examples/08-quantum-kernel-phase-discovery.ipynb](#condensed-matter-quantum-kernel-phase-discovery) | 3 | 1 |
| [notebooks/real_examples/09-potential-energy-curve-interpolation.ipynb](#molecular-physics-potential-energy-curve-interpolation) | 3 | 1 |
| [notebooks/real_examples/10-lorenz-quantum-reservoir-regime-classifier.ipynb](#dynamical-systems-lorenz-quantum-reservoir-regime-classification) | 3 | 1 |
| [notebooks/real_examples/11-noisy-oscillator-quantum-reservoir-inference.ipynb](#dynamical-systems-noisy-oscillator-quantum-reservoir-inference) | 3 | 1 |

## Quantum Dynamics: Rabi Oscillation Parameter Inference

Notebook: `notebooks/real_examples/01-rabi-oscillation-parameter-inference.ipynb`

Result block 1:

```text
Dataset
+-------------------+---------------------+
| Metric            | Value               |
+-------------------+---------------------+
| Samples           | 58                  |
| feature_shape     | [58, 3]             |
| Measurement times | [0.45, 0.95, 1.55]  |
| Omega range       | [0.721799, 2.38716] |
+-------------------+---------------------+
```
Result block 2:

```text
Results
+-------------------+------------+
| Metric            | Value      |
+-------------------+------------+
| Quantum omega MAE | 0.0393237  |
| Quantum omega MSE | 0.00319466 |
| Ridge omega MAE   | 0.0273694  |
| Initial loss      | 0.580552   |
| Final loss        | 0.00697781 |
| Loss steps        | 72         |
+-------------------+------------+
```
Result block 3:

```text
Validation
Dataset
+---------------+-----------------------------------------+
| Metric        | Value                                   |
+---------------+-----------------------------------------+
| Problem       | rabi_frequency_inference                |
| Features      | [P_e(t=0.45), P_e(t=0.95), P_e(t=1.55)] |
| Target        | rabi_frequency_omega                    |
| Train samples | 40                                      |
| Test samples  | 18                                      |
+---------------+-----------------------------------------+

Results
+-------------------+------------+
| Metric            | Value      |
+-------------------+------------+
| Quantum omega MAE | 0.0393237  |
| Quantum omega MSE | 0.00319466 |
| Ridge omega MAE   | 0.0273694  |
| Initial loss      | 0.580552   |
| Final loss        | 0.00697781 |
| Loss steps        | 72         |
+-------------------+------------+

Sample recoveries
1. Actual Omega: 1.999511, Quantum Omega: 2.027799, Ridge Omega: 1.981000
2. Actual Omega: 2.296350, Quantum Omega: 2.222178, Ridge Omega: 2.256309
3. Actual Omega: 1.474035, Quantum Omega: 1.486740, Ridge Omega: 1.458339
4. Actual Omega: 1.918194, Quantum Omega: 1.940107, Ridge Omega: 1.933709
5. Actual Omega: 1.762004, Quantum Omega: 1.732852, Ridge Omega: 1.756742
6. Actual Omega: 1.762079, Quantum Omega: 1.713610, Ridge Omega: 1.772137

Interpretation
The quantum inverse model trained successfully and recovers Rabi frequency within the validation tolerance.
The baseline is included as a sanity check; this validates package usage rather than quantum advantage.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/01-rabi-oscillation-parameter-inference/figure-01.png)

## Statistical Physics: Ising Temperature Classification

Notebook: `notebooks/real_examples/02-ising-correlation-temperature-classifier.ipynb`

Result block 1:

```text
Dataset
+-------------------+--------------+
| Metric            | Value        |
+-------------------+--------------+
| Samples           | 48           |
| feature_shape     | [48, 3]      |
| Labels            | [24, 24]     |
| Temperature range | [1.35, 3.45] |
+-------------------+--------------+
```
Result block 2:

```text
Results
+-------------------------+------------------+
| Metric                  | Value            |
+-------------------------+------------------+
| Quantum kernel accuracy | 0.933333         |
| Logistic accuracy       | 0.933333         |
| Confusion matrix        | [[8, 0], [1, 6]] |
| Kernel train shape      | [33, 33]         |
| Kernel diagonal minimum | 1                |
| Kernel symmetry error   | 6.66134e-16      |
+-------------------------+------------------+
```
Result block 3:

```text
Validation
Dataset
+-------------------+--------------------------------------------------------------------------------------------------------+
| Metric            | Value                                                                                                  |
+-------------------+--------------------------------------------------------------------------------------------------------+
| Problem           | ising_temperature_regime_classification                                                                |
| Lattice size      | 8                                                                                                      |
| Samples           | 48                                                                                                     |
| Feature names     | [absolute_magnetization, nearest_neighbor_correlation, energy_density]                                 |
| Test temperatures | [1.959, 1.685, 3.137, 2.05, 2.589, 3.333, 1.654, 3.254, 3.215, 1.533, 2.98, 1.837, 1.411, 3.45, 1.776] |
| Test labels       | [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]                                                          |
| Test predictions  | [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]                                                          |
+-------------------+--------------------------------------------------------------------------------------------------------+

Results
+-------------------------+------------------+
| Metric                  | Value            |
+-------------------------+------------------+
| Quantum kernel accuracy | 0.933333         |
| Logistic accuracy       | 0.933333         |
| Confusion matrix        | [[8, 0], [1, 6]] |
| Kernel train shape      | [33, 33]         |
| Kernel diagonal minimum | 1                |
| Kernel symmetry error   | 6.66134e-16      |
+-------------------------+------------------+

Interpretation
The Ising features separate low- and high-temperature regimes for this small Monte Carlo dataset.
The quantum kernel and logistic baseline are sanity-checked side by side; this validates workflow correctness, not quantum advantage.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/02-ising-correlation-temperature-classifier/figure-01.png)

## Nonlinear Dynamics: Lorenz Regime Classification

Notebook: `notebooks/real_examples/03-lorenz-regime-classifier.ipynb`

Result block 1:

```text
Dataset
+---------------+----------+
| Metric        | Value    |
+---------------+----------+
| Samples       | 44       |
| feature_shape | [44, 3]  |
| Labels        | [22, 22] |
| Rho range     | [16, 38] |
+---------------+----------+
```
Result block 2:

```text
Results
+-------------------------+------------------+
| Metric                  | Value            |
+-------------------------+------------------+
| Quantum kernel accuracy | 1                |
| Logistic accuracy       | 1                |
| Confusion matrix        | [[7, 0], [0, 7]] |
| Kernel train shape      | [30, 30]         |
| Kernel diagonal minimum | 1                |
| Kernel symmetry error   | 6.66134e-16      |
+-------------------------+------------------+
```
Result block 3:

```text
Validation
Dataset
+------------------+---------------------------------------------------------------------------------------------------------+
| Metric           | Value                                                                                                   |
+------------------+---------------------------------------------------------------------------------------------------------+
| Problem          | lorenz_parameter_regime_classification                                                                  |
| Samples          | 44                                                                                                      |
| Feature names    | [mean_tail_z, std_tail_x, mean_abs_dx_tail]                                                             |
| Test rho values  | [16.571, 16, 19.429, 35.619, 38, 36.095, 16.286, 21.714, 37.048, 33.238, 28.952, 34.19, 16.857, 20.571] |
| Test labels      | [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]                                                              |
| Test predictions | [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]                                                              |
+------------------+---------------------------------------------------------------------------------------------------------+

Results
+-------------------------+------------------+
| Metric                  | Value            |
+-------------------------+------------------+
| Quantum kernel accuracy | 1                |
| Logistic accuracy       | 1                |
| Confusion matrix        | [[7, 0], [0, 7]] |
| Kernel train shape      | [30, 30]         |
| Kernel diagonal minimum | 1                |
| Kernel symmetry error   | 6.66134e-16      |
+-------------------------+------------------+

Interpretation
The Lorenz summary features distinguish the two chosen parameter regimes for this reproducible simulator.
The quantum kernel and logistic baseline are both sanity checks; this validates package usage rather than quantum advantage.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/03-lorenz-regime-classifier/figure-01.png)

## Condensed Matter: TFIM Phase Classification

Notebook: `notebooks/real_examples/04-condensed-matter-tfim-phase-classifier.ipynb`

Result block 1:

```text
Dataset
+---------------+----------+
| Metric        | Value    |
+---------------+----------+
| Samples       | 48       |
| feature_shape | [48, 2]  |
| Labels        | [24, 24] |
+---------------+----------+
```
Result block 2:

```text
Results
+-------------------------+-------------+
| Metric                  | Value       |
+-------------------------+-------------+
| Quantum kernel accuracy | 1           |
| Logistic accuracy       | 1           |
| Kernel train shape      | [33, 33]    |
| Kernel diagonal minimum | 1           |
| Kernel symmetry error   | 4.44089e-16 |
+-------------------------+-------------+
```
Result block 3:

```text
Validation
Dataset
+------------------+----------------------------------------------------------------------------------------------------------+
| Metric           | Value                                                                                                    |
+------------------+----------------------------------------------------------------------------------------------------------+
| Problem          | finite_size_tfim_phase_classification                                                                    |
| N spins          | 4                                                                                                        |
| Samples          | 48                                                                                                       |
| Feature names    | [nearest_neighbor_zz_correlation, transverse_x_magnetization]                                            |
| Test fields      | [0.473, 1.463, 1.686, 0.697, 0.952, 0.346, 1.207, 0.378, 1.176, 1.527, 1.303, 0.505, 1.08, 1.016, 0.856] |
| Test labels      | [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]                                                            |
| Test predictions | [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]                                                            |
+------------------+----------------------------------------------------------------------------------------------------------+

Results
+-------------------------+-------------+
| Metric                  | Value       |
+-------------------------+-------------+
| Quantum kernel accuracy | 1           |
| Logistic accuracy       | 1           |
| Kernel train shape      | [33, 33]    |
| Kernel diagonal minimum | 1           |
| Kernel symmetry error   | 4.44089e-16 |
+-------------------------+-------------+

Interpretation
The TFIM feature set is cleanly separable in this finite-size example.
Both the quantum kernel model and the logistic baseline solve the held-out split perfectly, so this validates the workflow rather than demonstrating quantum advantage.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/04-condensed-matter-tfim-phase-classifier/figure-01.png)

## Dynamical Systems: Pendulum Trajectory Surrogate

Notebook: `notebooks/real_examples/05-pendulum-trajectory-surrogate.ipynb`

Result block 1:

```text
Dataset
+---------------+----------------------+
| Metric        | Value                |
+---------------+----------------------+
| Samples       | 54                   |
| feature_shape | [54, 3]              |
| Target range  | [-0.57019, 0.562638] |
+---------------+----------------------+
```
Result block 2:

```text
Results
+------------------+-----------+
| Metric           | Value     |
+------------------+-----------+
| Quantum test MAE | 0.241666  |
| Quantum test MSE | 0.0977921 |
| Ridge test MAE   | 0.261172  |
| Initial loss     | 0.557283  |
| Final loss       | 0.0745353 |
| Loss steps       | 75        |
+------------------+-----------+
```
Result block 3:

```text
Validation
Dataset
+---------------+-------------------------------------------+
| Metric        | Value                                     |
+---------------+-------------------------------------------+
| Problem       | small_angle_pendulum_trajectory_surrogate |
| Features      | [theta0, omega0, time]                    |
| Target        | theta_at_time                             |
| Train samples | 37                                        |
| Test samples  | 17                                        |
+---------------+-------------------------------------------+

Results
+------------------+-----------+
| Metric           | Value     |
+------------------+-----------+
| Quantum test MAE | 0.241666  |
| Quantum test MSE | 0.0977921 |
| Ridge test MAE   | 0.261172  |
| Initial loss     | 0.557283  |
| Final loss       | 0.0745353 |
| Loss steps       | 75        |
+------------------+-----------+

Sample predictions
1. Actual: -0.340563, Quantum: -0.148739, Ridge: -0.148951
2. Actual: 0.295556, Quantum: 0.136990, Ridge: -0.018507
3. Actual: -0.075147, Quantum: -0.212564, Ridge: -0.145427
4. Actual: -0.244626, Quantum: 0.147142, Ridge: 0.100026
5. Actual: 0.392050, Quantum: 0.164307, Ridge: 0.110848
6. Actual: 0.036099, Quantum: 0.153176, Ridge: 0.111547

Interpretation
The quantum regressor trained successfully and reduced the optimization loss.
The held-out error is reasonable for a compact illustrative surrogate, but individual predictions can still be visibly imperfect.
The ridge baseline is included as a sanity benchmark; no quantum advantage is claimed.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/05-pendulum-trajectory-surrogate/figure-01.png)

## Inverse Problems: Damped Oscillator Parameter Inference

Notebook: `notebooks/real_examples/06-damped-oscillator-parameter-inference.ipynb`

Result block 1:

```text
Dataset
+---------------+-----------------------+
| Metric        | Value                 |
+---------------+-----------------------+
| Samples       | 60                    |
| feature_shape | [60, 2]               |
| Gamma range   | [0.0702778, 0.574796] |
+---------------+-----------------------+
```
Result block 2:

```text
Results
+-------------------+-----------+
| Metric            | Value     |
+-------------------+-----------+
| Quantum gamma MAE | 0.0562714 |
| Ridge gamma MAE   | 0.0177185 |
| Initial loss      | 0.829008  |
| Final loss        | 0.0328188 |
| Loss steps        | 85        |
+-------------------+-----------+
```
Result block 3:

```text
Validation
Dataset
+---------------+-------------------------------------+
| Metric        | Value                               |
+---------------+-------------------------------------+
| Problem       | damped_oscillator_damping_inference |
| Features      | [x(t=0.7), x(t=1.4)]                |
| Target        | damping_gamma                       |
| Train samples | 42                                  |
| Test samples  | 18                                  |
+---------------+-------------------------------------+

Results
+-------------------+-----------+
| Metric            | Value     |
+-------------------+-----------+
| Quantum gamma MAE | 0.0562714 |
| Ridge gamma MAE   | 0.0177185 |
| Initial loss      | 0.829008  |
| Final loss        | 0.0328188 |
| Loss steps        | 85        |
+-------------------+-----------+

Sample recoveries
1. Actual gamma: 0.405297, Quantum gamma: 0.502452, Ridge gamma: 0.415919
2. Actual gamma: 0.299665, Quantum gamma: 0.396080, Ridge gamma: 0.318860
3. Actual gamma: 0.085770, Quantum gamma: 0.130895, Ridge gamma: 0.094191
4. Actual gamma: 0.442487, Quantum gamma: 0.448133, Ridge gamma: 0.447158
5. Actual gamma: 0.142181, Quantum gamma: 0.133497, Ridge gamma: 0.155010
6. Actual gamma: 0.477434, Quantum gamma: 0.514308, Ridge gamma: 0.466378

Interpretation
The quantum inverse model trained successfully and recovers damping coefficients within the validation tolerance.
The ridge baseline is stronger on this simple inverse problem, so this validates package usage rather than quantum advantage.

Passed: True
```

![figure 01](docs/pages/assets/notebook-results/real_examples/06-damped-oscillator-parameter-inference/figure-01.png)

## Condensed Matter: TFIM Hamiltonian Parameter Inference

Notebook: `notebooks/real_examples/07-tfim-hamiltonian-parameter-inference.ipynb`

Result block 1:

```text
Dataset
+----------+---------------------------------+
| Metric   | Value                           |
+----------+---------------------------------+
| Problem  | TFIM transverse-field inference |
| Samples  | 24                              |
| Features | [<Z>, <X>, <ZZ>, E0/N]          |
| Target   | transverse field h              |
+----------+---------------------------------+
```
Result block 2:

```text
Results
+-------------------------+-----------+
| Metric                  | Value     |
+-------------------------+-----------+
| Trainable kernel h MAE  | 0.0483953 |
| Quantum GPR h MAE       | 0.147426  |
| Ridge h MAE             | 0.0103647 |
| Kernel-target alignment | 0.466965  |
+-------------------------+-----------+
```
Result block 3:

```text
Validation
Dataset
+---------------+---------------------------------+
| Metric        | Value                           |
+---------------+---------------------------------+
| problem       | tfim_transverse_field_inference |
| n_train       | 16                              |
| n_test        | 8                               |
| feature_count | 4                               |
+---------------+---------------------------------+

Results
+-----------------------------+-----------+
| Metric                      | Value     |
+-----------------------------+-----------+
| trainable_kernel_h_mae      | 0.0483953 |
| quantum_gpr_h_mae           | 0.147426  |
| ridge_h_mae                 | 0.0103647 |
| trainable_kernel_alignment  | 0.466965  |
| trainable_kernel_loss_final | -0.466965 |
+-----------------------------+-----------+

Sample predictions
+----------+--------------------+---------------+
| actual_h | trainable_kernel_h | quantum_gpr_h |
+----------+--------------------+---------------+
| 1.19783  | 1.17612            | 1.14429       |
| 0.480435 | 0.488462           | 0.481546      |
| 1.4587   | 1.45176            | 1.44352       |
| 1.85     | 1.80081            | 1.61504       |
+----------+--------------------+---------------+

Passed
+--------+-------+
| Metric | Value |
+--------+-------+
| passed | True  |
+--------+-------+
```

![figure 01](docs/pages/assets/notebook-results/real_examples/07-tfim-hamiltonian-parameter-inference/figure-01.png)

## Condensed Matter: Quantum Kernel Phase Discovery

Notebook: `notebooks/real_examples/08-quantum-kernel-phase-discovery.ipynb`

Result block 1:

```text
Dataset
+----------+-----------------------------------------+
| Metric   | Value                                   |
+----------+-----------------------------------------+
| Problem  | TFIM phase discovery/classification     |
| Samples  | 36                                      |
| Classes  | {0: 'ferromagnetic', 1: 'paramagnetic'} |
| Features | [<Z>, <X>, <ZZ>, E0/N]                  |
+----------+-----------------------------------------+
```
Result block 2:

```text
Results
+--------------------------+--------------------+
| Metric                   | Value              |
+--------------------------+--------------------+
| Quantum kernel accuracy  | 1                  |
| One-class phase accuracy | 0.727273           |
| Logistic accuracy        | 1                  |
| Kernel PCA eigenvalues   | [4.69121, 3.03512] |
+--------------------------+--------------------+
```
Result block 3:

```text
Validation
Dataset
+---------------+----------------------+
| Metric        | Value                |
+---------------+----------------------+
| problem       | tfim_phase_discovery |
| n_train       | 25                   |
| n_test        | 11                   |
| feature_count | 4                    |
+---------------+----------------------+

Results
+--------------------------+--------------------+
| Metric                   | Value              |
+--------------------------+--------------------+
| quantum_kernel_accuracy  | 1                  |
| one_class_phase_accuracy | 0.727273           |
| logistic_accuracy        | 1                  |
| kpca_eigenvalues         | [4.69121, 3.03512] |
+--------------------------+--------------------+

Sample predictions
+----------+--------+--------+-----------+
| h        | actual | kernel | one_class |
+----------+--------+--------+-----------+
| 0.709562 | 0      | 0      | 0         |
| 0.525813 | 0      | 0      | 0         |
| 0.387803 | 0      | 0      | 1         |
| 0.707049 | 0      | 0      | 0         |
| 1.64825  | 1      | 1      | 1         |
+----------+--------+--------+-----------+

Passed
+--------+-------+
| Metric | Value |
+--------+-------+
| passed | True  |
+--------+-------+
```

![figure 01](docs/pages/assets/notebook-results/real_examples/08-quantum-kernel-phase-discovery/figure-01.png)

## Molecular Physics: Potential Energy Curve Interpolation

Notebook: `notebooks/real_examples/09-potential-energy-curve-interpolation.ipynb`

Result block 1:

```text
Dataset
+-----------------+-------------------------------+
| Metric          | Value                         |
+-----------------+-------------------------------+
| Problem         | Morse potential interpolation |
| Training points | 18                            |
| Held-out points | 24                            |
| Features        | [bond length r, r^2]          |
| Target          | potential energy              |
+-----------------+-------------------------------+
```
Result block 2:

```text
Results
+---------------------------------+-----------+
| Metric                          | Value     |
+---------------------------------+-----------+
| Quantum GPR energy MAE          | 0.0221625 |
| Quantum kernel ridge energy MAE | 0.073289  |
| Ridge energy MAE                | 0.572759  |
+---------------------------------+-----------+
```
Result block 3:

```text
Validation
Dataset
+---------+-------------------------------+
| Metric  | Value                         |
+---------+-------------------------------+
| problem | morse_potential_interpolation |
| n_train | 18                            |
| n_test  | 24                            |
+---------+-------------------------------+

Results
+---------------------------------+-------------+
| Metric                          | Value       |
+---------------------------------+-------------+
| quantum_gpr_energy_mae          | 0.0221625   |
| quantum_kernel_ridge_energy_mae | 0.073289    |
| ridge_energy_mae                | 0.572759    |
| quantum_gpr_energy_mse          | 0.000778001 |
+---------------------------------+-------------+

Sample predictions
+----------+---------------+--------------------+
| r        | actual_energy | quantum_gpr_energy |
+----------+---------------+--------------------+
| 0.79878  | -0.0330634    | 0.0471411          |
| 0.896341 | -1.82574      | -1.78565           |
| 0.945122 | -2.50641      | -2.50454           |
| 0.993902 | -3.06713      | -3.0818            |
| 1.09146  | -3.88831      | -3.88523           |
+----------+---------------+--------------------+

Passed
+--------+-------+
| Metric | Value |
+--------+-------+
| passed | True  |
+--------+-------+
```

![figure 01](docs/pages/assets/notebook-results/real_examples/09-potential-energy-curve-interpolation/figure-01.png)

## Dynamical Systems: Lorenz Quantum Reservoir Regime Classification

Notebook: `notebooks/real_examples/10-lorenz-quantum-reservoir-regime-classifier.ipynb`

Result block 1:

```text
Dataset
+----------+----------------------------------------+
| Metric   | Value                                  |
+----------+----------------------------------------+
| Problem  | Lorenz regime classification           |
| Samples  | 40                                     |
| Classes  | {0: 'settled/periodic', 1: 'chaotic'}  |
| Features | [std_x, std_y, std_z, mean_step_speed] |
+----------+----------------------------------------+
```
Result block 2:

```text
Results
+----------------------------+----------+
| Metric                     | Value    |
+----------------------------+----------+
| Quantum reservoir accuracy | 0.916667 |
| Quantum kernel accuracy    | 1        |
| Logistic accuracy          | 1        |
+----------------------------+----------+
```
Result block 3:

```text
Validation
Dataset
+---------------+---------------------------------------------+
| Metric        | Value                                       |
+---------------+---------------------------------------------+
| problem       | lorenz_regime_classification_with_reservoir |
| n_train       | 28                                          |
| n_test        | 12                                          |
| feature_count | 4                                           |
+---------------+---------------------------------------------+

Results
+----------------------------+----------+
| Metric                     | Value    |
+----------------------------+----------+
| quantum_reservoir_accuracy | 0.916667 |
| quantum_kernel_accuracy    | 1        |
| logistic_accuracy          | 1        |
+----------------------------+----------+

Sample predictions
+---------+--------+-----------+--------+
| rho     | actual | reservoir | kernel |
+---------+--------+-----------+--------+
| 31.3047 | 1      | 1         | 1      |
| 26.8206 | 1      | 1         | 1      |
| 16.8081 | 0      | 0         | 0      |
| 30.7191 | 1      | 0         | 1      |
| 19.0278 | 0      | 0         | 0      |
+---------+--------+-----------+--------+

Passed
+--------+-------+
| Metric | Value |
+--------+-------+
| passed | True  |
+--------+-------+
```

![figure 01](docs/pages/assets/notebook-results/real_examples/10-lorenz-quantum-reservoir-regime-classifier/figure-01.png)

## Dynamical Systems: Noisy Oscillator Quantum Reservoir Inference

Notebook: `notebooks/real_examples/11-noisy-oscillator-quantum-reservoir-inference.ipynb`

Result block 1:

```text
Dataset
+----------+------------------------------------------+
| Metric   | Value                                    |
+----------+------------------------------------------+
| Problem  | damped oscillator damping inference      |
| Samples  | 40                                       |
| Features | [x(t=0.2), x(t=0.7), x(t=1.2), x(t=1.7)] |
| Target   | damping coefficient gamma                |
+----------+------------------------------------------+
```
Result block 2:

```text
Results
+-----------------------------+-----------+
| Metric                      | Value     |
+-----------------------------+-----------+
| Quantum reservoir gamma MAE | 0.112894  |
| Quantum GPR gamma MAE       | 0.114993  |
| Ridge gamma MAE             | 0.0389513 |
+-----------------------------+-----------+
```
Result block 3:

```text
Validation
Dataset
+---------------+---------------------------------------+
| Metric        | Value                                 |
+---------------+---------------------------------------+
| problem       | damped_oscillator_reservoir_inference |
| n_train       | 28                                    |
| n_test        | 12                                    |
| feature_count | 4                                     |
+---------------+---------------------------------------+

Results
+-----------------------------+-----------+
| Metric                      | Value     |
+-----------------------------+-----------+
| quantum_reservoir_gamma_mae | 0.112894  |
| quantum_gpr_gamma_mae       | 0.114993  |
| ridge_gamma_mae             | 0.0389513 |
+-----------------------------+-----------+

Sample predictions
+--------------+-----------------+-----------+
| actual_gamma | reservoir_gamma | gpr_gamma |
+--------------+-----------------+-----------+
| 0.087205     | 0.224281        | 0.207131  |
| 0.312743     | 0.397964        | 0.493448  |
| 0.289192     | 0.236855        | 0.338626  |
| 0.102039     | 0.269038        | 0.186091  |
| 0.266321     | 0.282579        | 0.560718  |
+--------------+-----------------+-----------+

Passed
+--------+-------+
| Metric | Value |
+--------+-------+
| passed | True  |
+--------+-------+
```

![figure 01](docs/pages/assets/notebook-results/real_examples/11-noisy-oscillator-quantum-reservoir-inference/figure-01.png)

## Reproduce

Regenerate notebook result pages from existing executed notebook outputs:

```bash
python docs/pages/generate_results.py --skip-api-results
```

Execute notebooks first, then regenerate result pages:

```bash
python docs/pages/generate_results.py --skip-api-results --execute-notebooks
```
