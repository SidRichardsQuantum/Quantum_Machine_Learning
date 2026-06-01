# Benchmark Notebook Results

Executed outputs from benchmark notebooks in `notebooks/benchmarks/`. These compare QML workflows with classical baselines using deterministic seeds, confidence intervals, paired deltas, runtime summaries, and finite-shot sweeps.

## Environment

<<<<<<< HEAD
- Generated: 2026-06-01 09:15:14 UTC
- Git commit: `508c84e`
- Python: `3.12.13`
- Package version: `0.2.9`
=======
- Generated: 2026-06-01 09:47:13 UTC
- Git commit: `508c84e`
- Python: `3.12.1`
- Package version: `0.2.10`
>>>>>>> 0f4b077 (Release v0.2.10)
- Matplotlib backend: `Agg`

## Summary

| Notebook | Text result blocks | Plots |
| --- | ---: | ---: |
<<<<<<< HEAD
| [notebooks/benchmarks/01-classification-model-benchmark.ipynb](#classification-model-benchmark) | 2 | 0 |
| [notebooks/benchmarks/02-regression-model-benchmark.ipynb](#regression-model-benchmark) | 2 | 0 |
| [notebooks/benchmarks/03-quantum-kernel-family-benchmark.ipynb](#quantum-kernel-family-benchmark) | 2 | 0 |
| [notebooks/benchmarks/04-variational-model-capacity-benchmark.ipynb](#variational-model-capacity-benchmark) | 2 | 0 |
| [notebooks/benchmarks/05-finite-shot-benchmark.ipynb](#finite-shot-benchmark) | 2 | 0 |
| [notebooks/benchmarks/06-real-data-small-sample-benchmark.ipynb](#real-data-small-sample-benchmark) | 2 | 0 |
=======
| [notebooks/benchmarks/01-classification-model-benchmark.ipynb](#classification-model-benchmark) | 2 | 3 |
| [notebooks/benchmarks/02-regression-model-benchmark.ipynb](#regression-model-benchmark) | 2 | 3 |
| [notebooks/benchmarks/03-quantum-kernel-family-benchmark.ipynb](#quantum-kernel-family-benchmark) | 2 | 2 |
| [notebooks/benchmarks/04-variational-model-capacity-benchmark.ipynb](#variational-model-capacity-benchmark) | 2 | 2 |
| [notebooks/benchmarks/05-finite-shot-benchmark.ipynb](#finite-shot-benchmark) | 2 | 2 |
| [notebooks/benchmarks/06-real-data-small-sample-benchmark.ipynb](#real-data-small-sample-benchmark) | 2 | 2 |
| [notebooks/benchmarks/07-noise-model-benchmark.ipynb](#noise-model-benchmark) | 3 | 2 |
>>>>>>> 0f4b077 (Release v0.2.10)

## Classification Model Benchmark

Notebook: `notebooks/benchmarks/01-classification-model-benchmark.ipynb`

Result block 1:

```text
Classification summary
+--------------------------+---------------+----------+-----------+----------+------------+--------+-------+------+
| model                    | test_accuracy | ci95_low | ci95_high | gap      | runtime_s  | params | depth | runs |
+--------------------------+---------------+----------+-----------+----------+------------+--------+-------+------+
| qcnn                     | 0.75          | 0.75     | 0.75      | 0.125    | 4.3048     | 38     | 24    | 1    |
| quantum_kernel           | 0.75          | 0.75     | 0.75      | 0.166667 | 0.582978   | 0      | 3     | 1    |
| logistic_regression      | 0.75          | 0.75     | 0.75      | 0.166667 | 0.00431984 |        |       | 1    |
| svm_classifier           | 0.75          | 0.75     | 0.75      | 0.166667 | 0.00377182 |        |       | 1    |
| random_forest_classifier | 0.75          | 0.75     | 0.75      | 0.25     | 0.035071   |        |       | 1    |
| vqc                      | 0.625         | 0.625    | 0.625     | 0        | 0.60458    | 4      | 5     | 1    |
| quantum_reservoir        | 0.5           | 0.5      | 0.5       | 0.125    | 0.0850361  | 0      | 2     | 1    |
+--------------------------+---------------+----------+-----------+----------+------------+--------+-------+------+

Paired deltas vs logistic_regression
+--------------------------+------------+------+--------+------+-------+
| model                    | mean_delta | wins | losses | ties | pairs |
+--------------------------+------------+------+--------+------+-------+
| vqc                      | -0.125     | 0    | 1      | 0    | 1     |
| qcnn                     | 0          | 0    | 0      | 1    | 1     |
| quantum_kernel           | 0          | 0    | 0      | 1    | 1     |
| quantum_reservoir        | -0.25      | 0    | 1      | 0    | 1     |
| logistic_regression      | 0          | 0    | 0      | 1    | 1     |
| svm_classifier           | 0          | 0    | 0      | 1    | 1     |
| random_forest_classifier | 0          | 0    | 0      | 1    | 1     |
+--------------------------+------------+------+--------+------+-------+

Best model
+------------------+---------------+
| Metric           | Value         |
+------------------+---------------+
| model            | qcnn          |
| metric           | test_accuracy |
| value            | 0.75          |
| higher_is_better | True          |
+------------------+---------------+
```
Result block 2:

```text
Classification dataset sweep
+---------+-------------------+---------------+----------+-----------+------------+------------+--------+-------+------+
| dataset | model             | test_accuracy | ci95_low | ci95_high | gap        | runtime_s  | params | depth | runs |
+---------+-------------------+---------------+----------+-----------+------------+------------+--------+-------+------+
| moons   | quantum_kernel    | 0.75          | 0.75     | 0.75      | 0.166667   | 0.582349   | 0      | 3     | 1    |
| moons   | svm_classifier    | 0.75          | 0.75     | 0.75      | 0.166667   | 0.00402999 |        |       | 1    |
| moons   | vqc               | 0.625         | 0.625    | 0.625     | 0          | 0.483978   | 4      | 5     | 1    |
| moons   | quantum_reservoir | 0.5           | 0.5      | 0.5       | 0.125      | 0.0848197  | 0      | 2     | 1    |
| circles | quantum_kernel    | 0.875         | 0.875    | 0.875     | 0.125      | 0.588742   | 0      | 3     | 1    |
| circles | svm_classifier    | 0.875         | 0.875    | 0.875     | 0.125      | 0.00421187 |        |       | 1    |
| circles | vqc               | 0.5           | 0.5      | 0.5       | -0.208333  | 0.585971   | 4      | 5     | 1    |
| circles | quantum_reservoir | 0.5           | 0.5      | 0.5       | 0.291667   | 0.0871217  | 0      | 2     | 1    |
| wine    | quantum_kernel    | 1             | 1        | 1         | -0.0416667 | 0.585737   | 0      | 3     | 1    |
| wine    | svm_classifier    | 1             | 1        | 1         | -0.0416667 | 0.00502436 |        |       | 1    |
| wine    | quantum_reservoir | 0.625         | 0.625    | 0.625     | 0.0833333  | 0.085927   | 0      | 2     | 1    |
| wine    | vqc               | 0.375         | 0.375    | 0.375     | 0.125      | 0.585558   | 4      | 5     | 1    |
+---------+-------------------+---------------+----------+-----------+------------+------------+--------+-------+------+
```

_No plots were found._

## Regression Model Benchmark

Notebook: `notebooks/benchmarks/02-regression-model-benchmark.ipynb`

Result block 1:

```text
Regression summary by MSE
+------------------------------------+-----------+-----------+-----------+------------+------------+--------+-------+------+
| model                              | test_mse  | ci95_low  | ci95_high | gap        | runtime_s  | params | depth | runs |
+------------------------------------+-----------+-----------+-----------+------------+------------+--------+-------+------+
| svr_regression                     | 0.0801907 | 0.0801907 | 0.0801907 | -0.0136897 | 0.00233544 |        |       | 1    |
| vqr                                | 0.122342  | 0.122342  | 0.122342  | -0.773642  | 0.674183   | 4      | 5     | 1    |
| kernel_ridge_regression            | 0.197459  | 0.197459  | 0.197459  | 0.0338305  | 0.0028616  |        |       | 1    |
| quantum_gaussian_process_regressor | 0.373078  | 0.373078  | 0.373078  | 0.11375    | 0.587449   | 0      | 3     | 1    |
| quantum_kernel_regressor           | 0.419824  | 0.419824  | 0.419824  | 0.0973739  | 0.59671    | 0      | 3     | 1    |
| ridge_regression                   | 0.622718  | 0.622718  | 0.622718  | 0.121189   | 0.00239304 |        |       | 1    |
| quantum_reservoir_regressor        | 0.986287  | 0.986287  | 0.986287  | 0.0713367  | 0.0758315  | 0      | 2     | 1    |
+------------------------------------+-----------+-----------+-----------+------------+------------+--------+-------+------+

Regression summary by MAE
+------------------------------------+----------+----------+-----------+------------+------------+--------+-------+------+
| model                              | test_mae | ci95_low | ci95_high | gap        | runtime_s  | params | depth | runs |
+------------------------------------+----------+----------+-----------+------------+------------+--------+-------+------+
| svr_regression                     | 0.230482 | 0.230482 | 0.230482  | -0.0136897 | 0.00233544 |        |       | 1    |
| vqr                                | 0.300124 | 0.300124 | 0.300124  | -0.773642  | 0.674183   | 4      | 5     | 1    |
| kernel_ridge_regression            | 0.408842 | 0.408842 | 0.408842  | 0.0338305  | 0.0028616  |        |       | 1    |
| quantum_gaussian_process_regressor | 0.562695 | 0.562695 | 0.562695  | 0.11375    | 0.587449   | 0      | 3     | 1    |
| quantum_kernel_regressor           | 0.60704  | 0.60704  | 0.60704   | 0.0973739  | 0.59671    | 0      | 3     | 1    |
| ridge_regression                   | 0.705256 | 0.705256 | 0.705256  | 0.121189   | 0.00239304 |        |       | 1    |
| quantum_reservoir_regressor        | 0.837164 | 0.837164 | 0.837164  | 0.0713367  | 0.0758315  | 0      | 2     | 1    |
+------------------------------------+----------+----------+-----------+------------+------------+--------+-------+------+

Paired MSE deltas vs svr_regression
+------------------------------------+------------+------+--------+------+-------+
| model                              | mean_delta | wins | losses | ties | pairs |
+------------------------------------+------------+------+--------+------+-------+
| vqr                                | 0.0421516  | 0    | 1      | 0    | 1     |
| quantum_kernel_regressor           | 0.339634   | 0    | 1      | 0    | 1     |
| quantum_gaussian_process_regressor | 0.292887   | 0    | 1      | 0    | 1     |
| quantum_reservoir_regressor        | 0.906096   | 0    | 1      | 0    | 1     |
| ridge_regression                   | 0.542528   | 0    | 1      | 0    | 1     |
| kernel_ridge_regression            | 0.117269   | 0    | 1      | 0    | 1     |
| svr_regression                     | 0          | 0    | 0      | 1    | 1     |
+------------------------------------+------------+------+--------+------+-------+

Best model
+------------------+----------------+
| Metric           | Value          |
+------------------+----------------+
| model            | svr_regression |
| metric           | test_mse       |
| value            | 0.0801907      |
| higher_is_better | False          |
+------------------+----------------+
```
Result block 2:

```text
Regression dataset sweep
+----------+-----------------------------+------------+------------+------------+------------+------------+--------+-------+------+
| dataset  | model                       | test_mse   | ci95_low   | ci95_high  | gap        | runtime_s  | params | depth | runs |
+----------+-----------------------------+------------+------------+------------+------------+------------+--------+-------+------+
| linear   | ridge_regression            | 0.00322048 | 0.00322048 | 0.00322048 | 0.00141887 | 0.00290031 |        |       | 1    |
| linear   | quantum_kernel_regressor    | 0.168163   | 0.168163   | 0.168163   | -0.0173032 | 0.597858   | 0      | 3     | 1    |
| linear   | vqr                         | 1.12147    | 1.12147    | 1.12147    | 0.0589046  | 0.775226   | 4      | 5     | 1    |
| linear   | quantum_reservoir_regressor | 1.29671    | 1.29671    | 1.29671    | 0.325749   | 0.0764549  | 0      | 2     | 1    |
| sine     | vqr                         | 0.122342   | 0.122342   | 0.122342   | -0.773642  | 0.646735   | 4      | 5     | 1    |
| sine     | quantum_kernel_regressor    | 0.419824   | 0.419824   | 0.419824   | 0.0973739  | 0.593465   | 0      | 3     | 1    |
| sine     | ridge_regression            | 0.622718   | 0.622718   | 0.622718   | 0.121189   | 0.00249402 |        |       | 1    |
| sine     | quantum_reservoir_regressor | 0.986287   | 0.986287   | 0.986287   | 0.0713367  | 0.0771625  | 0      | 2     | 1    |
| diabetes | quantum_kernel_regressor    | 0.667531   | 0.667531   | 0.667531   | 0.0185848  | 0.53703    | 0      | 3     | 1    |
| diabetes | ridge_regression            | 0.677572   | 0.677572   | 0.677572   | -0.145698  | 0.00408987 |        |       | 1    |
| diabetes | vqr                         | 0.816185   | 0.816185   | 0.816185   | -0.726353  | 0.653704   | 4      | 5     | 1    |
| diabetes | quantum_reservoir_regressor | 0.886238   | 0.886238   | 0.886238   | 0.0638437  | 0.0776479  | 0      | 2     | 1    |
+----------+-----------------------------+------------+------------+------------+------------+------------+--------+-------+------+
```

_No plots were found._

## Quantum Kernel Family Benchmark

Notebook: `notebooks/benchmarks/03-quantum-kernel-family-benchmark.ipynb`

Result block 1:

```text
Kernel-family benchmark summary
+---------+--------------------------+---------------+----------+-----------+----------+------------+
| dataset | model                    | test_accuracy | ci95_low | ci95_high | gap      | runtime_s  |
+---------+--------------------------+---------------+----------+-----------+----------+------------+
| moons   | quantum_kernel           | 0.75          | 0.75     | 0.75      | 0.166667 | 0.598445   |
| moons   | trainable_quantum_kernel | 0.75          | 0.75     | 0.75      | 0.166667 | 39.0974    |
| moons   | svm_classifier           | 0.75          | 0.75     | 0.75      | 0.166667 | 0.00334335 |
| moons   | knn_classifier           | 0.75          | 0.75     | 0.75      | 0.208333 | 0.00556761 |
| circles | quantum_kernel           | 0.875         | 0.875    | 0.875     | 0.125    | 0.576662   |
| circles | trainable_quantum_kernel | 0.875         | 0.875    | 0.875     | 0.125    | 38.0612    |
| circles | svm_classifier           | 0.875         | 0.875    | 0.875     | 0.125    | 0.00329018 |
| circles | knn_classifier           | 0.5           | 0.5      | 0.5       | 0.166667 | 0.00516108 |
+---------+--------------------------+---------------+----------+-----------+----------+------------+

Paired deltas vs best classical kernel-style baseline
+---------+----------------+--------------------------+------------+------+--------+-------+
| dataset | reference      | model                    | mean_delta | wins | losses | pairs |
+---------+----------------+--------------------------+------------+------+--------+-------+
| moons   | svm_classifier | quantum_kernel           | 0          | 0    | 0      | 1     |
| moons   | svm_classifier | trainable_quantum_kernel | 0          | 0    | 0      | 1     |
| moons   | svm_classifier | svm_classifier           | 0          | 0    | 0      | 1     |
| moons   | svm_classifier | knn_classifier           | 0          | 0    | 0      | 1     |
| circles | svm_classifier | quantum_kernel           | 0          | 0    | 0      | 1     |
| circles | svm_classifier | trainable_quantum_kernel | 0          | 0    | 0      | 1     |
| circles | svm_classifier | svm_classifier           | 0          | 0    | 0      | 1     |
| circles | svm_classifier | knn_classifier           | -0.375     | 0    | 1      | 1     |
+---------+----------------+--------------------------+------------+------+--------+-------+
```
Result block 2:

```text
Trainable quantum kernel diagnostics: moons
+---------+------+---------------+-----------------+-----------+
| dataset | seed | test_accuracy | final_alignment | runtime_s |
+---------+------+---------------+-----------------+-----------+
| moons   | 0    | 0.75          | 0.32394         | 39.0974   |
+---------+------+---------------+-----------------+-----------+

Trainable quantum kernel diagnostics: circles
+---------+------+---------------+-----------------+-----------+
| dataset | seed | test_accuracy | final_alignment | runtime_s |
+---------+------+---------------+-----------------+-----------+
| circles | 0    | 0.875         | 0.133107        | 38.0612   |
+---------+------+---------------+-----------------+-----------+
```

_No plots were found._

## Variational Model Capacity Benchmark

Notebook: `notebooks/benchmarks/04-variational-model-capacity-benchmark.ipynb`

Result block 1:

```text
Classification capacity summary
+---------------------+-----------------------------+---------------+------------+------+
| model               | capacity                    | test_accuracy | runtime_s  | runs |
+---------------------+-----------------------------+---------------+------------+------+
| qcnn                | {'steps': 8}                | 0.875         | 7.81388    | 1    |
| logistic_regression | reference                   | 0.75          | 0.00609831 | 1    |
| qcnn                | {'steps': 4}                | 0.625         | 4.03315    | 1    |
| vqc                 | {'n_layers': 1, 'steps': 4} | 0.375         | 0.576707   | 1    |
| vqc                 | {'n_layers': 2, 'steps': 4} | 0.375         | 0.930775   | 1    |
+---------------------+-----------------------------+---------------+------------+------+
```
Result block 2:

```text
Regression capacity summary
+------------------+-----------------------------+----------+----------+------------+------+
| model            | capacity                    | test_mse | test_mae | runtime_s  | runs |
+------------------+-----------------------------+----------+----------+------------+------+
| vqr              | {'n_layers': 1, 'steps': 5} | 0.131641 | 0.307639 | 0.678531   | 1    |
| vqr              | {'n_layers': 2, 'steps': 5} | 0.161685 | 0.370644 | 1.11291    | 1    |
| ridge_regression | reference                   | 0.634695 | 0.70884  | 0.00287814 | 1    |
+------------------+-----------------------------+----------+----------+------------+------+
```

_No plots were found._

## Finite-Shot Benchmark

Notebook: `notebooks/benchmarks/05-finite-shot-benchmark.ipynb`

Result block 1:

```text
Classification finite-shot summary
+----------+-------------------+---------------+----------+-----------+--------------------+------------+----------------------------+
| shots    | model             | test_accuracy | ci95_low | ci95_high | generalization_gap | runtime_s  | accuracy_delta_vs_analytic |
+----------+-------------------+---------------+----------+-----------+--------------------+------------+----------------------------+
| analytic | vqc               | 0.166667      | 0.166667 | 0.166667  | 0.222222           | 0.479709   | 0                          |
| analytic | qcnn              | 0.833333      | 0.833333 | 0.833333  | -0.111111          | 2.9871     | 0                          |
| analytic | quantum_kernel    | 0.833333      | 0.833333 | 0.833333  | -0.0555556         | 0.333284   | 0                          |
| analytic | quantum_reservoir | 0.333333      | 0.333333 | 0.333333  | 0.444444           | 0.0650134  | 0                          |
| analytic | svm_classifier    | 1             | 1        | 1         | -0.166667          | 0.00404593 | 0                          |
| 64       | vqc               | 0.5           | 0.5      | 0.5       | 0.166667           | 0.971879   | 0.333333                   |
| 64       | qcnn              | 0.833333      | 0.833333 | 0.833333  | -0.277778          | 32.0347    | 0                          |
| 64       | quantum_kernel    | 0.833333      | 0.833333 | 0.833333  | -0.0555556         | 0.49959    | 0                          |
| 64       | quantum_reservoir | 0.333333      | 0.333333 | 0.333333  | 0.388889           | 0.100106   | 0                          |
| 64       | svm_classifier    | 1             | 1        | 1         | -0.166667          | 0.00388598 | 0                          |
| 128      | vqc               | 0.5           | 0.5      | 0.5       | 0.166667           | 0.87107    | 0.333333                   |
| 128      | qcnn              | 0.833333      | 0.833333 | 0.833333  | -0.277778          | 31.9389    | 0                          |
| 128      | quantum_kernel    | 0.833333      | 0.833333 | 0.833333  | -0.0555556         | 0.501992   | 0                          |
| 128      | quantum_reservoir | 0.333333      | 0.333333 | 0.333333  | 0.444444           | 0.0991378  | 0                          |
| 128      | svm_classifier    | 1             | 1        | 1         | -0.166667          | 0.00386756 | 0                          |
| 512      | vqc               | 0.5           | 0.5      | 0.5       | 0.222222           | 0.965432   | 0.333333                   |
| 512      | qcnn              | 1             | 1        | 1         | -0.277778          | 32.0585    | 0.166667                   |
| 512      | quantum_kernel    | 0.833333      | 0.833333 | 0.833333  | -0.0555556         | 0.506092   | 0                          |
| 512      | quantum_reservoir | 0.333333      | 0.333333 | 0.333333  | 0.388889           | 0.100098   | 0                          |
| 512      | svm_classifier    | 1             | 1        | 1         | -0.166667          | 0.00384376 | 0                          |
+----------+-------------------+---------------+----------+-----------+--------------------+------------+----------------------------+
```
Result block 2:

```text
Regression finite-shot summary
+----------+------------------------------------+----------+----------+-----------+--------------------+------------+-----------------------+
| shots    | model                              | test_mse | ci95_low | ci95_high | generalization_gap | runtime_s  | mse_delta_vs_analytic |
+----------+------------------------------------+----------+----------+-----------+--------------------+------------+-----------------------+
| analytic | vqr                                | 0.665687 | 0.665687 | 0.665687  | -0.0945696         | 0.519257   | 0                     |
| analytic | quantum_kernel_regressor           | 0.618937 | 0.618937 | 0.618937  | 0.39149            | 0.33243    | 0                     |
| analytic | quantum_gaussian_process_regressor | 1.13144  | 1.13144  | 1.13144   | 1.06286            | 0.330008   | 0                     |
| analytic | quantum_reservoir_regressor        | 0.982325 | 0.982325 | 0.982325  | 0.100131           | 0.0565304  | 0                     |
| analytic | ridge_regression                   | 0.65928  | 0.65928  | 0.65928   | 0.345539           | 0.00236014 | 0                     |
| 64       | vqr                                | 0.603867 | 0.603867 | 0.603867  | -0.122759          | 1.0474     | -0.0618199            |
| 64       | quantum_kernel_regressor           | 0.58976  | 0.58976  | 0.58976   | 0.368003           | 0.499589   | -0.0291769            |
| 64       | quantum_gaussian_process_regressor | 0.78001  | 0.78001  | 0.78001   | 0.646353           | 0.499195   | -0.35143              |
| 64       | quantum_reservoir_regressor        | 1.04479  | 1.04479  | 1.04479   | 0.177314           | 0.0899226  | 0.0624619             |
| 64       | ridge_regression                   | 0.65928  | 0.65928  | 0.65928   | 0.345539           | 0.0024138  | 0                     |
| 128      | vqr                                | 0.664742 | 0.664742 | 0.664742  | -0.0626584         | 1.13819    | -0.000945064          |
| 128      | quantum_kernel_regressor           | 0.577133 | 0.577133 | 0.577133  | 0.379557           | 0.499953   | -0.0418044            |
| 128      | quantum_gaussian_process_regressor | 0.861045 | 0.861045 | 0.861045  | 0.816126           | 0.498282   | -0.270395             |
| 128      | quantum_reservoir_regressor        | 0.998827 | 0.998827 | 0.998827  | 0.119465           | 0.0891002  | 0.016502              |
| 128      | ridge_regression                   | 0.65928  | 0.65928  | 0.65928   | 0.345539           | 0.00243169 | 0                     |
| 512      | vqr                                | 0.66036  | 0.66036  | 0.66036   | -0.0898559         | 1.06532    | -0.00532629           |
| 512      | quantum_kernel_regressor           | 0.575466 | 0.575466 | 0.575466  | 0.374851           | 0.510124   | -0.0434711            |
| 512      | quantum_gaussian_process_regressor | 0.83301  | 0.83301  | 0.83301   | 0.771806           | 0.504893   | -0.29843              |
| 512      | quantum_reservoir_regressor        | 0.945361 | 0.945361 | 0.945361  | 0.062365           | 0.0901844  | -0.0369647            |
| 512      | ridge_regression                   | 0.65928  | 0.65928  | 0.65928   | 0.345539           | 0.00234506 | 0                     |
+----------+------------------------------------+----------+----------+-----------+--------------------+------------+-----------------------+
```

_No plots were found._

## Real-Data Small-Sample Benchmark

Notebook: `notebooks/benchmarks/06-real-data-small-sample-benchmark.ipynb`

Result block 1:

```text
Real-data classification summary
+---------------+---------------------+---------------+----------+-----------+------------+------------+
| dataset       | model               | test_accuracy | ci95_low | ci95_high | gap        | runtime_s  |
+---------------+---------------------+---------------+----------+-----------+------------+------------+
| breast_cancer | logistic_regression | 0.8           | 0.8      | 0.8       | 0.0666667  | 0.0106015  |
| breast_cancer | quantum_kernel      | 0.666667      | 0.666667 | 0.666667  | 0.2        | 2.0256     |
| breast_cancer | svm_classifier      | 0.666667      | 0.666667 | 0.666667  | 0.2        | 0.0101666  |
| breast_cancer | vqc                 | 0.533333      | 0.533333 | 0.533333  | -0.0666667 | 1.05909    |
| breast_cancer | quantum_reservoir   | 0.466667      | 0.466667 | 0.466667  | 0.177778   | 0.159356   |
| wine          | quantum_kernel      | 1             | 1        | 1         | -0.0888889 | 2.05211    |
| wine          | svm_classifier      | 1             | 1        | 1         | -0.0888889 | 0.00502391 |
| wine          | logistic_regression | 0.8           | 0.8      | 0.8       | 0.0888889  | 0.0055552  |
| wine          | vqc                 | 0.533333      | 0.533333 | 0.533333  | 0.111111   | 1.07763    |
| wine          | quantum_reservoir   | 0.533333      | 0.533333 | 0.533333  | 0.0222222  | 0.155162   |
+---------------+---------------------+---------------+----------+-----------+------------+------------+
```
Result block 2:

```text
Real-data regression summary
+----------+-----------------------------+----------+----------+----------+-----------+------------+
| dataset  | model                       | test_mse | test_mae | ci95_low | ci95_high | runtime_s  |
+----------+-----------------------------+----------+----------+----------+-----------+------------+
| diabetes | quantum_kernel_regressor    | 0.949192 | 0.85899  | 0.949192 | 0.949192  | 1.28635    |
| diabetes | ridge_regression            | 0.992731 | 0.913113 | 0.992731 | 0.992731  | 0.00419083 |
| diabetes | svr_regression              | 1.09356  | 0.963873 | 1.09356  | 1.09356   | 0.00397112 |
| diabetes | quantum_reservoir_regressor | 1.24796  | 1.03509  | 1.24796  | 1.24796   | 0.138367   |
| diabetes | vqr                         | 1.62663  | 1.01501  | 1.62663  | 1.62663   | 1.2307     |
+----------+-----------------------------+----------+----------+----------+-----------+------------+
```

_No plots were found._

## Noise-Model Benchmark

Notebook: `notebooks/benchmarks/07-noise-model-benchmark.ipynb`

Result block 1:

```text
Classification noise-model summary
+------------------------+--------------------------------------------------------+----------------+---------------+----------+-----------+--------------------+------------+-----------------------------+
| noise_model            | noise_tag                                              | model          | test_accuracy | ci95_low | ci95_high | generalization_gap | runtime_s  | accuracy_delta_vs_noiseless |
+------------------------+--------------------------------------------------------+----------------+---------------+----------+-----------+--------------------+------------+-----------------------------+
| noiseless              | noiseless                                              | vqc            | 0.6           | 0.6      | 0.6       | 0                  | 0.854189   | 0                           |
| noiseless              | noiseless                                              | quantum_kernel | 0.6           | 0.6      | 0.6       | 0.266667           | 0.702677   | 0                           |
| noiseless              | noiseless                                              | svm_classifier | 0.8           | 0.8      | 0.8       | 0.0666667          | 0.0130219  | 0                           |
| depolarizing_0.02      | depolarizing0p02                                       | vqc            | 0.6           | 0.6      | 0.6       | 0                  | 1.5845     | 0                           |
| depolarizing_0.02      | depolarizing0p02                                       | quantum_kernel | 0.6           | 0.6      | 0.6       | 0.266667           | 1.13225    | 0                           |
| depolarizing_0.02      | depolarizing0p02                                       | svm_classifier | 0.8           | 0.8      | 0.8       | 0.0666667          | 0.00870595 | 0                           |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | vqc            | 0.6           | 0.6      | 0.6       | 0                  | 1.27449    | 0                           |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | quantum_kernel | 0.6           | 0.6      | 0.6       | 0.266667           | 1.25527    | 0                           |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | svm_classifier | 0.8           | 0.8      | 0.8       | 0.0666667          | 0.00360445 | 0                           |
| readout_error_0.03     | readouterror0p03                                       | vqc            | 0.6           | 0.6      | 0.6       | 0                  | 0.596281   | 0                           |
| readout_error_0.03     | readouterror0p03                                       | quantum_kernel | 0.6           | 0.6      | 0.6       | 0.266667           | 0.840027   | 0                           |
| readout_error_0.03     | readouterror0p03                                       | svm_classifier | 0.8           | 0.8      | 0.8       | 0.0666667          | 0.00448502 | 0                           |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | vqc            | 0.6           | 0.6      | 0.6       | 0                  | 0.815886   | 0                           |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | quantum_kernel | 0.6           | 0.6      | 0.6       | 0.266667           | 0.923056   | 0                           |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | svm_classifier | 0.8           | 0.8      | 0.8       | 0.0666667          | 0.00458748 | 0                           |
+------------------------+--------------------------------------------------------+----------------+---------------+----------+-----------+--------------------+------------+-----------------------------+
```
Result block 2:

```text
Regression noise-model summary
+------------------------+--------------------------------------------------------+--------------------------+----------+----------+-----------+--------------------+------------+------------------------+
| noise_model            | noise_tag                                              | model                    | test_mse | ci95_low | ci95_high | generalization_gap | runtime_s  | mse_delta_vs_noiseless |
+------------------------+--------------------------------------------------------+--------------------------+----------+----------+-----------+--------------------+------------+------------------------+
| noiseless              | noiseless                                              | vqr                      | 5.08599  | 5.08599  | 5.08599   | 4.35799            | 0.567244   | 0                      |
| noiseless              | noiseless                                              | quantum_kernel_regressor | 2.49846  | 2.49846  | 2.49846   | 2.225              | 0.391775   | 0                      |
| noiseless              | noiseless                                              | ridge_regression         | 2.10303  | 2.10303  | 2.10303   | 1.81081            | 0.00449462 | 0                      |
| depolarizing_0.02      | depolarizing0p02                                       | vqr                      | 5.02772  | 5.02772  | 5.02772   | 4.30715            | 1.00622    | -0.0582679             |
| depolarizing_0.02      | depolarizing0p02                                       | quantum_kernel_regressor | 2.50201  | 2.50201  | 2.50201   | 2.22621            | 0.841362   | 0.00355135             |
| depolarizing_0.02      | depolarizing0p02                                       | ridge_regression         | 2.10303  | 2.10303  | 2.10303   | 1.81081            | 0.00269852 | 0                      |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | vqr                      | 5.11607  | 5.11607  | 5.11607   | 4.37205            | 0.784504   | 0.0300832              |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | quantum_kernel_regressor | 2.495    | 2.495    | 2.495     | 2.22125            | 0.600997   | -0.00346088            |
| amplitude_damping_0.02 | amplitudedamping0p02                                   | ridge_regression         | 2.10303  | 2.10303  | 2.10303   | 1.81081            | 0.004028   | 0                      |
| readout_error_0.03     | readouterror0p03                                       | vqr                      | 4.95601  | 4.95601  | 4.95601   | 4.24358            | 0.863779   | -0.129981              |
| readout_error_0.03     | readouterror0p03                                       | quantum_kernel_regressor | 2.50679  | 2.50679  | 2.50679   | 2.22785            | 0.734005   | 0.008329               |
| readout_error_0.03     | readouterror0p03                                       | ridge_regression         | 2.10303  | 2.10303  | 2.10303   | 1.81081            | 0.00424434 | 0                      |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | vqr                      | 4.98595  | 4.98595  | 4.98595   | 4.26417            | 0.889175   | -0.100039              |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | quantum_kernel_regressor | 2.50417  | 2.50417  | 2.50417   | 2.22575            | 1.13066    | 0.00571259             |
| combined_low           | depolarizing0p01_amplitudedamping0p01_readouterror0p02 | ridge_regression         | 2.10303  | 2.10303  | 2.10303   | 1.81081            | 0.00422156 | 0                      |
+------------------------+--------------------------------------------------------+--------------------------+----------+----------+-----------+--------------------+------------+------------------------+
```
Result block 3:

```text
Noise benchmark validation
+---------------------+-----------------+--------------+--------+
| classification_rows | regression_rows | noise_models | passed |
+---------------------+-----------------+--------------+--------+
| 15                  | 15              | 5            | True   |
+---------------------+-----------------+--------------+--------+
```

![figure 01](../pages/assets/notebook-results/benchmarks/07-noise-model-benchmark/figure-01.png)
![figure 02](../pages/assets/notebook-results/benchmarks/07-noise-model-benchmark/figure-02.png)

## Reproduce

Regenerate notebook result pages from existing executed notebook outputs:

```bash
python docs/pages/generate_results.py --skip-api-results
```

Execute notebooks first, then regenerate result pages:

```bash
python docs/pages/generate_results.py --skip-api-results --execute-notebooks
```
