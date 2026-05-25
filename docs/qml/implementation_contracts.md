# Implementation Contracts

This package treats an algorithm implementation as true when the code exposes
the model family, builds the stated quantum or classical computation, optimizes
the stated objective when training is required, and reports metrics that are
computed from the trained model rather than from a shortcut.

## Quantum models

- Variational quantum classifier: a PennyLane QNode applies a named feature
  embedding, a trainable hardware-efficient ansatz, and a Pauli-Z readout. It
  trains with binary cross-entropy on the training split.
- Variational quantum regressor: a PennyLane QNode applies angle embedding, a
  trainable hardware-efficient ansatz, and a Pauli-Z readout. It trains with
  mean squared error on standardized regression targets.
- Quantum kernel classifier/regressor: a fidelity kernel is evaluated as the
  probability of returning to `|0...0>` after `U(x)` followed by `U(y)^dagger`.
  Classical SVM or kernel-ridge models consume the precomputed kernel matrix.
- Trainable quantum kernel classifier: a parameterized quantum feature map is
  trained by maximizing normalized kernel-target alignment before fitting an
  SVM on the learned fidelity kernel.
- Trainable quantum kernel regressor: a parameterized quantum feature map is
  trained by maximizing normalized alignment with a continuous target kernel
  before fitting kernel-ridge regression on the learned fidelity kernel.
- Quantum reservoir estimators: a fixed random quantum circuit maps inputs to
  expectation-value features, then a classical ridge or logistic readout is
  fitted on those features.
- Quantum kernel PCA, one-class detection, and Gaussian process regression:
  general classical kernel algorithms consume quantum fidelity kernel matrices
  without embedding any domain-specific physics assumptions.
- Quantum convolutional neural network: a four-qubit QCNN with trainable
  convolution blocks, trainable pooling blocks, active-wire reduction from
  four to two to one wire, and a final Pauli-Z classifier readout.
- Quantum autoencoder: a four-qubit encoder is trained to concentrate state
  information into latent qubits by maximizing trash-zero probability.
  Reconstruction is evaluated by postselecting the encoded state onto the
  trash-zero subspace, renormalizing, and decoding with the adjoint of the
  learned encoder.
- Quantum metric learner: a trainable quantum embedding is optimized with a
  pairwise contrastive loss, then evaluated by nearest-centroid classification
  in the learned expectation-value embedding space.

## Classical models and benchmarks

- Classical baselines use scikit-learn estimators directly: logistic
  regression, SVM, ridge regression, MLP classifier, and MLP regressor.
- Benchmark helpers run the selected quantum and classical models over the
  same named dataset, sample count, noise level, test fraction, and seed list.
  They aggregate train/test metrics without changing the underlying model
  implementations.

## Required behavioral checks

- Trainable models must return learned parameters and a loss/alignment trace
  when an optimization objective is present.
- Fidelity kernels in analytic mode must be symmetric with diagonal entries
  near one and nonnegative eigenvalues up to numerical tolerance.
- Autoencoder reconstruction fidelity must be computed after compression loss,
  not by applying an encoder immediately followed by its inverse.
- QCNN results must expose the active-wire reduction stages used by pooling.
