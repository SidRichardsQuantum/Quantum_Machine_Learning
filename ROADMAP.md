# Roadmap

This project should grow as a package-first quantum machine learning library:
general QML implementations live in `src/qml/`, while notebooks demonstrate,
validate, and benchmark those implementations. Domain-specific physics and
mathematics should appear in notebooks as data generation, task framing, and
interpretation, not as hidden assumptions inside reusable package algorithms.

## Guiding Principles

- General implementations first: every major QML algorithm should be usable on
  user-supplied arrays or named package datasets without depending on one
  specific physics problem.
- Thin notebooks: tutorial, benchmark, and real-example notebooks should call
  reusable package APIs rather than reimplementing algorithms inline.
- Truthful scope: results should be framed as reproducible demonstrations or
  empirical comparisons, not quantum-advantage claims.
- Classical baselines are mandatory for benchmarks and strongly preferred for
  real examples.
- Plots and tables should be compact, relevant, and reproducible. Every
  benchmark and real-example notebook should produce succinct metric tables and
  diagnostic plots.
- Finite-shot or noise-aware execution should be included where it materially
  affects the algorithm or task interpretation.

## Notebook Contract

### Tutorial notebooks

Location: `notebooks/tutorials/`

Tutorial notebooks are educational walkthroughs for package algorithms. Their
numerical results are less important than clarity, correctness, and API
coverage.

Each major QML implementation should have at least one tutorial notebook that:

- explains the public API by example
- uses small deterministic settings
- shows the expected inputs and returned result structure
- includes one or two simple plots when useful
- avoids large benchmark claims
- links naturally to the relevant docs page

### Real-example notebooks

Location: `notebooks/real_examples/`

Real-example notebooks tailor general package algorithms to concrete physics,
mathematics, or engineering problems. The simulator, dataset construction, and
domain metrics may live in the notebook, but the QML model should come from
`src/qml/`.

Each real-example notebook should include:

- a clearly named domain problem
- a small reproducible simulator or dataset
- deterministic seeds
- a package-backed QML approach
- at least one relevant classical sanity baseline
- metrics that mean something for the domain
- succinct tables and plots
- a final validation block with a `passed` flag

### Benchmark notebooks

Location: `notebooks/benchmarks/`

Benchmark notebooks compare package QML implementations against relevant
classical implementations under shared data splits, seeds, metrics, and runtime
reporting.

Each benchmark notebook should include:

- QML and classical model lists
- deterministic seed lists
- confidence intervals or multi-seed summaries where feasible
- train/test metrics and generalization gaps
- runtime summaries
- paired deltas against the best included classical baseline when applicable
- succinct plots, such as metric bars with intervals, runtime/quality scatter
  plots, finite-shot degradation plots, or dataset sweep charts
- explicit wording that small notebook defaults are templates, not definitive
  performance claims

## Implementation Tracks

### Core QML algorithms

Keep these implementations general and estimator-like:

- Variational quantum classifier and regressor
- Quantum kernel classifier and regressor
- Trainable quantum kernel classifier and regressor
- Quantum Gaussian-process regression
- Quantum kernel PCA and one-class detection
- Quantum reservoir classifier and regressor
- Quantum metric learning
- Quantum convolutional neural network
- Quantum autoencoder
- Cross-validation and model-selection helpers for estimator-style APIs

Potential additions:

- Multiclass variational classifier with one-vs-rest and native multiclass
  readout options
- Multi-output regression examples and tests beyond the existing estimator
  surface
- Quantum support vector regression wrapper around fidelity kernels
- Additional trainable-kernel objectives, such as centered alignment,
  regularized alignment, and task-weighted alignment
- Noise model utilities for depolarizing, amplitude damping, readout error,
  and shot-noise comparisons
- Additional model-selection scoring options and result-format integrations

### Embeddings and ansatz library

The package should expose reusable, documented building blocks:

- Angle, amplitude, ZZ, IQP, and data-reuploading embeddings
- Hardware-efficient and strongly entangling ansatz helpers
- Shape validation and parameter initialization helpers
- Circuit metadata helpers for wires, layers, parameter counts, and depth

Potential additions:

- Basis, phase, and displacement-inspired embeddings
- Problem-independent feature scaling utilities for quantum embeddings
- Optional circuit drawing utilities for tutorials and docs

### Training and optimization

Training should stay reusable across algorithms:

- shared optimizer selection
- early stopping
- deterministic initialization
- loss history and timing metadata
- clear validation errors

Potential additions:

- mini-batch training for variational models
- learning-rate schedules
- callback hooks for logging and custom stopping rules
- gradient diagnostics for barren-plateau-style examples
- repeated-restart utilities for nonconvex training

## Real-Example Expansion Ideas

Good candidates should be small, reproducible, and meaningful with two to four
features after preprocessing.

- Quantum dynamics: Rabi, Ramsey, Landau-Zener, noisy oscillator, spin-chain
  parameter inference
- Condensed matter: TFIM/XXZ phase classification, finite-size phase-boundary
  regression, correlation-feature anomaly detection
- Differential equations: heat, wave, diffusion-reaction, Burgers, and Poisson
  inverse problems with compact feature summaries
- Dynamical systems: Lorenz, Duffing, forced pendulum, logistic map, regime
  classification, and short-horizon surrogate modeling
- Spectroscopy and signals: peak classification, frequency inference,
  low-dimensional sensor summaries, and anomaly detection
- Molecular and materials surrogates: potential-energy interpolation,
  small-geometry property regression, and kernel uncertainty estimates
- Numerical linear algebra and approximation: function interpolation,
  manifold classification, and low-sample kernel regression

## Benchmark Expansion Ideas

Benchmark notebooks should answer one narrow question each.

- Classification model benchmark across VQC, QCNN, quantum kernels, trainable
  kernels, metric learning, reservoirs, and classical baselines
- Regression model benchmark across VQR, quantum kernel regression, trainable
  kernel regression, quantum GPR, reservoirs, and classical baselines
- Kernel-family benchmark comparing quantum kernels with RBF, polynomial,
  Gaussian-process, kNN, and kernel-ridge references
- Finite-shot benchmark comparing analytic, 64, 128, 512, and 1024 shots
- Noise-model benchmark for selected algorithms and realistic error channels
- Capacity benchmark for layers, qubits/features, trainable parameters, and
  optimizer steps
- Small-sample benchmark on real tabular datasets
- Runtime scaling benchmark over samples, features, qubits, and shot counts

## Documentation and Results

Keep generated outputs discoverable:

- `docs/results/api-reference.md`: smoke-scale API reference outputs
- `docs/results/tutorials.md`: extracted tutorial notebook outputs
- `docs/results/real-examples.md`: extracted real-example outputs
- `docs/results/benchmarks.md`: extracted benchmark outputs

Potential additions:

- A notebook authoring guide with required sections, validation blocks, and
  plotting expectations
- Expanded API examples for user-supplied arrays, model selection, and circuit
  metadata reporting

## Quality Gates

Before adding or advertising a new algorithm:

- implementation lives in `src/qml/`
- public API has validation errors for invalid shapes and options
- unit or smoke tests cover fit/predict/result behavior
- implementation contract is documented
- circuit-backed workflow results include `circuit_metadata` where applicable
- tutorial notebook exists or is planned in the same milestone
- benchmark coverage exists or is explicitly deferred
- at least one real-example candidate is identified if the method is useful for
  domain workflows

Before adding a new real-example notebook:

- QML model is imported from the package
- domain simulator is deterministic and small
- classical baseline is included
- final metrics are domain-relevant
- outputs include compact tables and plots
- notebook executes from repository root, `notebooks/`, and its own directory

Before adding a new benchmark notebook:

- compared models are relevant to the task
- model settings are stated
- defaults are small enough for docs generation
- stronger-run guidance is included in text or config comments
- results include tables and plots
- runtime and generalization gaps are shown

## Near-Term Milestones

### Release hardening

- Stabilize benchmark coverage for all currently advertised major algorithms.
- Keep benchmark notebooks smoke-scale but useful.
- Ensure generated benchmark plots appear on the Pages site.
- Finalize `CHANGELOG.md` and package metadata before tagging.

### Generalization release

- Improve estimator consistency across classifiers, regressors, kernels,
  reservoirs, and trainable kernels.
- Extend cross-validation/model-selection helpers with additional scorers and
  notebook/reporting integrations as real usage patterns emerge.
- Expand implementation contracts for noise and finite-shot behavior.

### Domain examples release

- Add a curated set of real-example notebooks across condensed matter,
  dynamics, inverse problems, and signal/spectral tasks.
- Prefer fewer high-quality notebooks over many shallow examples.
- Ensure every real example has a classical baseline and validation block.

### Robustness and scaling release

- Add runtime scaling benchmarks.
- Add noise-channel benchmarks.
- Use circuit metadata in benchmark summaries and runtime/quality/depth plots.
- Improve CI/docs controls for long-running notebooks.
