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
- Estimator APIs are the primary user surface: algorithms should expose
  predictable package objects for user arrays before relying on convenience
  `run_*` workflows or notebooks.
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
- Algorithm truth should be testable: implementation contracts need behavioral
  tests that prove the advertised circuit, kernel, objective, readout, or
  reconstruction metric is what the package actually executes.

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
- a package-backed QML estimator or workflow, with domain logic kept in the
  notebook
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

Estimator-style implementations should be preferred over notebook-specific
workflow additions. Convenience `run_*` functions may remain useful for demos,
CLIs, and generated results, but they should wrap the same reusable algorithms
that package users can call directly on their own arrays.

Potential additions:

- Multiclass variational classifier with one-vs-rest and native multiclass
  readout options
- Multi-output regression examples and tests beyond the existing estimator
  surface
- Quantum support vector regression wrapper around fidelity kernels
- Additional trainable-kernel objectives, such as centered alignment,
  regularized alignment, and task-weighted alignment
- Additional model-selection result-format integrations as real usage patterns
  emerge

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

- learning-rate schedules
- callback hooks for logging and custom stopping rules
- gradient diagnostics for barren-plateau-style examples
- repeated-restart utilities for nonconvex training

### Estimator consistency and model selection

The estimator layer is the main near-term package hardening target. Quantum
estimators should be boring to use from Python: instantiate, `fit`, `predict`
or `transform`, `score` where meaningful, inspect fitted attributes, and pass
through lightweight model-selection helpers.

Required consistency targets:

- `fit`, `predict`, `transform`, `predict_proba`, `decision_function`, and
  `score` should raise clear fitted-state and shape errors.
- All fitted array estimators should record `n_features_in_`; classifiers
  should record `classes_`.
- Constructor arguments should round-trip through `get_params` and
  `set_params`.
- Composed estimators should support useful nested parameter updates, such as
  `kernel__shots`, `kernel__embedding`, `reservoir__n_layers`, and
  `reservoir__noise_model`, where compatible with the local design.
- Fitted attributes should use consistent trailing-underscore names for learned
  parameters, traces, fitted classical models, kernels, reservoir features,
  training data references, and circuit metadata.
- Score semantics should be documented and stable: classifiers return accuracy,
  regressors return negative mean squared error unless a deliberate alternative
  is documented.
- Finite-shot and `noise_model` support should be accepted, rejected, or
  documented explicitly for each circuit-backed estimator.
- Smoke-scale deterministic behavior should be tested for seeded trainable and
  random-feature estimators.

The tracking checklist lives in `docs/qml/estimator_consistency.md`. Treat it as
a release blocker for the next generalization release, not as optional cleanup.

### Algorithm truth checks

Implementation contracts should be backed by tests that verify algorithmic
claims, not just importability or notebook execution.

Required checks include:

- Fidelity kernels are symmetric with unit diagonals in analytic mode and
  positive semidefinite up to numerical tolerance on smoke datasets.
- Quantum kernel classifiers, regressors, PCA, one-class detection, and
  Gaussian-process regression consume precomputed quantum kernel matrices rather
  than silently replacing them with classical kernels.
- Trainable quantum kernels optimize the documented alignment objective and
  expose the learned parameters and objective trace.
- Variational classifiers and regressors train the stated loss against
  circuit-generated predictions, not cached labels or shortcut metrics.
- QCNN pooling reports and follows the documented active-wire reduction.
- Autoencoder reconstruction fidelity is computed after compression and
  decoding, not by applying an encoder immediately followed by its inverse.
- Finite-shot and noise-aware paths change the circuit execution path and are
  reflected in result metadata where applicable.

## Real-Example Expansion Ideas

Good candidates should be small, reproducible, and meaningful with two to four
features after preprocessing. Prefer a small curated set of high-quality
examples over many shallow notebooks. Add new domain examples only when they
exercise stable package APIs rather than filling gaps with notebook-local QML
implementations.

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

- Expanded API examples for user-supplied arrays, model selection, and circuit
  metadata reporting
- A user-facing estimator guide centered on `QuantumClassifier`,
  `QuantumRegressor`, quantum kernel estimators, trainable-kernel estimators,
  and reservoir estimators.
- Explicit examples for custom arrays, nested model-selection parameters,
  finite-shot/noise settings, fitted attributes, and result inspection.

## Quality Gates

Before adding or advertising a new algorithm:

- implementation lives in `src/qml/`
- public API has validation errors for invalid shapes and options
- unit or smoke tests cover fit/predict/result behavior
- behavioral tests verify the advertised circuit, kernel, objective, readout, or
  reconstruction metric
- implementation contract is documented
- circuit-backed workflow results include `circuit_metadata` where applicable
- estimator-style APIs are available or the absence of an estimator surface is
  explicitly justified
- estimator objects round-trip constructor parameters through `get_params` and
  `set_params`
- tutorial notebook exists or is planned in the same milestone
- benchmark coverage exists or is explicitly deferred
- at least one real-example candidate is identified if the method is useful for
  domain workflows

Before adding a new real-example notebook:

- QML model is imported from the package
- package estimator or workflow APIs are used directly; notebook-local QML
  model implementations are avoided
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

### Completed: 0.2.11 release hardening and scaling

The 0.2.x release-hardening and robustness/scaling work has landed:

- Benchmark coverage now spans the currently advertised major algorithm
  families with smoke-scale notebooks and generated result pages.
- Finite-shot, noise-model, capacity, kernel-family, runtime-scaling, and
  classification/regression benchmark notebooks are present.
- Runtime-scaling diagnostics are exposed through
  `qml.benchmarks.benchmark_runtime_scaling(...)` and the
  `qml-pennylane benchmark runtime-scaling` CLI preset.
- Circuit metadata is included in circuit-backed workflow results and benchmark
  summaries where applicable.
- Generated API, tutorial, real-example, and benchmark outputs are published
  under `docs/results/` and committed Pages assets.
- `CHANGELOG.md`, package metadata, and the `v0.2.11` tag reflect the latest
  release.

### Completed: 0.2.12 generalization release

The v0.2.12 release prioritized the package API surface over adding more
notebooks. The result is a more consistent estimator layer across variational,
kernel, trainable-kernel, and reservoir models. The first audit is tracked in
`docs/qml/estimator_consistency.md`.

- Audited estimator classes for consistent `fit`, `predict`, `score`,
  `get_params`, and `set_params` behavior.
- Treated estimator consistency as the release blocker before expanding the
  notebook catalog.
- Standardized fitted attributes, including learned parameters, training traces,
  fitted-model references, class labels, feature counts, and circuit metadata.
- Added useful nested parameter support for composed estimators, including
  kernel and reservoir parameters used by model-selection workflows.
- Tightened validation errors for input shapes, target types, unsupported options,
  shot counts, and noise-model compatibility.
- Added behavioral tests for deterministic seeds, parameter round-tripping,
  prediction shapes, scoring semantics, and fitted-state errors.
- Added algorithm truth tests for kernels, trainable-kernel alignment,
  variational losses, QCNN pooling, autoencoder reconstruction, and finite-shot
  or noise-aware execution paths.
- Documented estimator consistency guarantees in the API reference and
  implementation contracts.
- Extended model-selection/reporting integrations only where they clarify actual
  estimator workflows.
- Expanded implementation contracts for finite-shot and noise behavior across
  estimator-style APIs, not just workflow functions.

Potential follow-up scope, after the consistency pass:

- Multiclass variational classifier support through one-vs-rest or native
  multiclass readouts.
- Multi-output regression examples and tests beyond the existing estimator
  surface.
- Additional trainable-kernel objectives such as centered alignment,
  regularized alignment, and task-weighted alignment.

### Next: Domain examples release

- Add a curated set of four to six real-example notebooks across condensed
  matter, dynamics, inverse problems, and signal/spectral tasks.
- Prefer fewer high-quality notebooks over many shallow examples.
- Ensure every real example has a classical baseline and validation block.
- Keep notebooks near-pure package clients: simulators, feature extraction,
  metrics, and interpretation may live in notebooks, but QML estimators should
  come from `src/qml/`.
- Do not add domain examples to compensate for missing package capabilities;
  move reusable QML logic into the package first.
