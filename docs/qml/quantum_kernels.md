# Quantum Kernel Methods

This note describes the quantum kernel classifier implemented in `qml.kernel_methods`.

The model is a **hybrid quantum–classical classifier**:

- classical data is encoded into a quantum state
- pairwise quantum state overlaps define a kernel
- the kernel matrix is passed to a classical support vector machine (SVM)
- prediction is performed classically using the quantum-computed kernel

Unlike the variational quantum classifier, there are **no trainable quantum circuit parameters** in the current implementation.

---

# Data

We consider a binary classification dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
$$

where:

- $N$ is the number of samples
- $x_i \in \mathbb{R}^d$ is the feature vector for sample $i$
- $y_i \in \{0,1\}$ is the binary label for sample $i$
- $d$ is the feature dimension

In the current implementation:

- $d = 2$
- the dataset is the two-moons dataset
- features are standardised before entering the quantum feature map

Let:

$$
x = (x_1, x_2)
$$

denote one standardised input sample.

---

# Quantum feature map

A quantum kernel method begins by encoding each classical input $x$ into a quantum state:

$$
|\phi(x)\rangle = U(x)\,|0\rangle^{\otimes n}
$$

where:

- $n$ is the number of qubits
- $U(x)$ is the feature-map unitary
- $|0\rangle^{\otimes n}$ is the initial computational basis state

In the current implementation:

- $n = d$
- each feature is encoded by a single-qubit $R_Y$ rotation
- a short entangling chain is applied afterward

## Feature map definition

For an input vector $x \in \mathbb{R}^n$, the feature map is:

$$
U(x)
=
U_{\mathrm{ent}}
\left(
\prod_{j=1}^{n} R_Y(x_j)
\right)
$$

where:

- $x_j$ is the $j$th feature
- $R_Y(x_j)$ is a rotation about the $Y$ axis on qubit $j$
- $U_{\mathrm{ent}}$ is the entangling unitary

The entangling unitary is a nearest-neighbour CNOT chain:

$$
U_{\mathrm{ent}}
=
\prod_{j=1}^{n-1} \mathrm{CNOT}_{j,j+1}
$$

So the encoded quantum state is:

$$
|\phi(x)\rangle = U(x)\,|0\rangle^{\otimes n}
$$

---

# Quantum kernel

The kernel between two samples $x_i$ and $x_j$ is defined by the squared state overlap:

$$
K(x_i, x_j)
=
|\langle \phi(x_i) \mid \phi(x_j) \rangle|^2
$$

where:

- $K(x_i, x_j)$ is the kernel value between inputs $x_i$ and $x_j$
- $|\phi(x_i)\rangle$ and $|\phi(x_j)\rangle$ are the encoded quantum states

This kernel is:

- symmetric
- non-negative
- bounded in $[0,1]$

It acts as a similarity measure in the Hilbert space induced by the feature map.

---

# Circuit evaluation of the kernel

The overlap can be computed using a single quantum circuit.

Starting from $|0\rangle^{\otimes n}$, apply:

1. $U(x_i)$
2. $U^\dagger(x_j)$

The resulting state is:

$$
|\psi(x_i, x_j)\rangle
=
U^\dagger(x_j)\,U(x_i)\,|0\rangle^{\otimes n}
$$

The kernel is then the probability of returning to the all-zero state:

$$
K(x_i, x_j)
=
|\langle 0^{\otimes n} \mid \psi(x_i, x_j)\rangle|^2
$$

equivalently,

$$
K(x_i, x_j)
=
|\langle 0^{\otimes n} \mid U^\dagger(x_j)\,U(x_i) \mid 0^{\otimes n}\rangle|^2
$$

In the implementation, the circuit returns the full computational-basis probability vector, and the kernel value is the probability of the zero state:

$$
K(x_i, x_j) = p_{0\cdots0}
$$

where $p_{0\cdots0}$ is the measured probability of bitstring $00\cdots0$.

---

# Kernel matrix

For a training set:

$$
\{x_i\}_{i=1}^{N_{\mathrm{train}}}
$$

the training kernel matrix is:

$$
K^{\mathrm{train}} \in \mathbb{R}^{N_{\mathrm{train}} \times N_{\mathrm{train}}}
$$

with entries:

$$
K^{\mathrm{train}}_{ij} = K(x_i, x_j)
$$

where:

- $N_{\mathrm{train}}$ is the number of training samples
- $K^{\mathrm{train}}_{ij}$ is the kernel between training samples $x_i$ and $x_j$

For test samples:

$$
\{x'_a\}_{a=1}^{N_{\mathrm{test}}}
$$

the test kernel matrix is:

$$
K^{\mathrm{test}} \in \mathbb{R}^{N_{\mathrm{test}} \times N_{\mathrm{train}}}
$$

with entries:

$$
K^{\mathrm{test}}_{aj} = K(x'_a, x_j)
$$

where:

- $N_{\mathrm{test}}$ is the number of test samples
- $x'_a$ is the $a$th test input
- $x_j$ is the $j$th training input

So:

- the training kernel matrix compares training samples to training samples
- the test kernel matrix compares test samples to training samples

---

# Classical support vector machine

Once the kernel matrix is computed, classification is performed by a classical SVM using the precomputed kernel.

The classifier decision function takes the form:

$$
f(x)
=
\sum_{i=1}^{N_{\mathrm{train}}} \alpha_i\,K(x_i, x) + b
$$

where:

- $\alpha_i$ are learned dual coefficients
- $b$ is the bias term
- $x_i$ are support vectors from the training set
- $K(x_i, x)$ is the kernel between training point $x_i$ and input $x$

The predicted class is determined by the sign of the decision function in the SVM formulation. In the binary setting used here, the classifier outputs labels in $\{0,1\}$ after fitting on the training labels.

The quantum computer is used only for kernel evaluation. The optimisation of the SVM is entirely classical.

---

# Why kernels are useful

Kernel methods are attractive because they separate two tasks:

1. define a feature space
2. fit a classifier in that feature space

The quantum circuit defines the feature space implicitly through the map:

$$
x \mapsto |\phi(x)\rangle
$$

The classifier then works with inner products in that space, rather than explicit coordinates.

This means the model can represent nonlinear decision boundaries in the original input space even though the SVM optimisation remains classical.

---

# Accuracy

After fitting the SVM, predictions are made on both train and test sets.

For any evaluation set of size $M$, let:

- $y_i$ be the true label of sample $i$
- $\hat{y}_i$ be the predicted label of sample $i$

Accuracy is:

$$
\mathrm{Accuracy}
=
\frac{1}{M}
\sum_{i=1}^{M}
\mathbf{1}\{y_i = \hat{y}_i\}
$$

where:

- $M$ is the number of evaluated samples
- $\mathbf{1}\{\cdot\}$ is the indicator function

The implementation reports:

- training accuracy
- test accuracy

---

# Computational structure

Let:

- $N_{\mathrm{train}}$ be the number of training samples
- $N_{\mathrm{test}}$ be the number of test samples

Then the number of kernel evaluations is:

- $N_{\mathrm{train}}^2$ for the training kernel matrix
- $N_{\mathrm{test}}N_{\mathrm{train}}$ for the test kernel matrix

This means the current implementation is simple and clear, but not yet optimised for large datasets.

It is appropriate for small educational and research-scale examples.

---

# Current implementation choices

The current quantum kernel classifier is intentionally minimal.

## Included

- binary classification
- two-dimensional input
- angle-based feature map
- nearest-neighbour entangling structure
- exact statevector simulation
- precomputed-kernel SVM
- train/test accuracy
- returned train and test kernel matrices

## Not yet included

- trainable feature maps
- alternative kernels
- multiclass extensions
- shot-based kernel estimation
- kernel centering or alignment analysis
- kernel matrix visualisation
- classical baseline comparisons inside the same workflow
- larger dataset support

---

# Relation to the code

The implemented workflow is organised as follows:

- `qml.data` prepares the dataset
- `qml.kernel_methods` defines the feature map, kernel circuit, kernel matrices, and SVM workflow
- `qml.metrics` computes accuracy
- `qml.io_utils` optionally saves results

So the notebook remains a package client, while the full kernel logic lives in the package.

---

# Summary

The implemented quantum kernel classifier is defined by:

1. a feature map $U(x)$
2. encoded states $|\phi(x)\rangle = U(x)|0\rangle^{\otimes n}$
3. a kernel
   $$
   K(x_i,x_j)=|\langle \phi(x_i)\mid\phi(x_j)\rangle|^2
   $$
4. a training kernel matrix and test kernel matrix
5. a classical SVM trained on the precomputed kernel

Formally:

$$
|\phi(x)\rangle = U(x)\,|0\rangle^{\otimes n}
$$

$$
K(x_i, x_j)
=
|\langle \phi(x_i) \mid \phi(x_j) \rangle|^2
$$

$$
K^{\mathrm{train}}_{ij} = K(x_i, x_j)
$$

$$
K^{\mathrm{test}}_{aj} = K(x'_a, x_j)
$$

$$
f(x)
=
\sum_{i=1}^{N_{\mathrm{train}}} \alpha_i\,K(x_i, x) + b
$$

This is the core quantum kernel workflow used in the repository.