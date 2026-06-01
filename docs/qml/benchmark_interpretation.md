# Benchmark Interpretation Guide

Benchmarks in this project are reproducible comparison templates. They are
designed to make model behavior visible under shared settings, not to establish
general quantum advantage.

Use the benchmark pages to answer narrow questions:

- Which included model works best for this small dataset and configuration?
- Does a QML model improve over the strongest included classical reference?
- How much quality is traded for runtime or finite-shot execution?
- Are results stable across seeds, datasets, and model settings?

## Required Context

Interpret every benchmark with these fields visible:

- dataset name and sample count
- model list
- seed list
- train and test metrics
- confidence intervals or multi-seed summaries
- generalization gaps
- runtime summaries
- circuit metadata when available, especially trainable-parameter count and
  estimated template depth
- paired deltas against the best included classical baseline
- shot count or analytic execution mode
- tuning metadata for classical references

Single-seed benchmark outputs are smoke checks. Treat them as examples of the
reporting format, not as evidence that one model is generally better.

## Primary Metrics

Classification benchmarks rank models by mean `test_accuracy`.

Regression benchmarks rank models by mean `test_mse`, with lower values better.
Secondary metrics such as MAE are useful for checking whether conclusions are
consistent across loss functions.

The `best_model` field is a convenience summary. It should not be the only
result used when deciding whether a model is useful.

## Confidence Intervals

The benchmark summaries report 95 percent confidence intervals when multiple
seed results are available. Wide intervals mean the run is not precise enough to
support a strong comparison.

Use intervals this way:

- overlapping intervals suggest the benchmark is inconclusive
- narrow intervals make seed-to-seed behavior easier to compare
- single-run intervals are placeholders around one observed value
- more seeds are preferred before making release notes or documentation claims

For small notebook defaults, the interval is mostly a reminder to rerun with
more seeds before drawing stronger conclusions.

## Paired Classical Deltas

`paired_vs_best_classical` compares each model against the best included
classical baseline on matching seeds. This is the most useful summary for QML
comparisons because it keeps dataset splits aligned.

For classification, positive deltas mean better test accuracy than the classical
reference.

For regression, negative deltas mean lower test error than the classical
reference.

Read the paired table together with win/loss/tie counts. A small average delta
with mixed wins and losses should be treated as inconclusive.

## Generalization Gaps

Generalization gaps indicate how differently a model behaves on train and test
splits.

For classification:

```text
train_accuracy - test_accuracy
```

For regression:

```text
test_mse - train_mse
```

Large positive gaps can indicate overfitting. Negative gaps can happen on small
or noisy splits and should be checked across more seeds.

## Runtime Tradeoffs

QML models often have higher runtime at these small sample sizes because the
notebooks favor transparent circuit evaluation over optimized deployment.

Runtime should be interpreted with quality:

- a slower model with no paired metric gain is not a useful default
- a slower model with consistent paired gains may justify deeper investigation
- analytic simulators and finite-shot runs should not be compared as if they
  had the same execution cost
- runtime scaling benchmarks are needed before making claims about larger data

Circuit metadata adds a second complexity signal. Parameter counts and estimated
depths are package-template summaries, not hardware-compiled resource estimates,
but they help distinguish a small fixed-feature circuit from a deeper trainable
workflow when runtimes or metrics are similar.

The smoke-scale notebooks are intentionally small enough for documentation
generation. They are not tuned for maximum model quality.

## Finite-Shot Results

Finite-shot benchmarks compare analytic execution with sampled execution at
fixed shot counts. Expect additional variance as shots decrease.

Useful signals include:

- metric degradation from analytic to low-shot settings
- whether higher shot counts recover analytic behavior
- whether rankings change under finite-shot execution
- whether runtime increases enough to change the practical recommendation

Report finite-shot results as robustness checks, not as hardware performance
claims.

## Tuning and Fairness

Classical baselines may be tuned with small grid-search defaults. Quantum model
settings are supplied explicitly through benchmark arguments and `model_kwargs`.

A comparison is more meaningful when:

- all models use the same train/test splits
- classical baselines include at least one strong nonlinear reference
- model settings are stated
- seed counts are high enough for stable summaries
- runtime is reported with the primary metric

If only the default notebook settings were used, describe the result as a
smoke-scale comparison.

## Release Language

Use careful wording in release notes and docs:

- "matched the best included classical baseline on this smoke-scale run"
- "improved the paired test metric in this benchmark configuration"
- "degraded under low-shot execution"
- "requires more seeds or larger sweeps before stronger claims"

Avoid wording such as:

- "quantum advantage"
- "outperforms classical machine learning" without scope
- "state of the art"
- "proves superiority"

The benchmark suite is most useful as a reproducible harness for inspecting
model behavior under controlled settings.
