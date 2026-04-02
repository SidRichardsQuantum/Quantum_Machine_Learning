# Variational Quantum Regression

This note describes the variational quantum regressor (VQR) implemented in `qml.regression`.

The model is a **hybrid quantum–classical regressor**:

- a classical feature vector is encoded into a quantum circuit
- a parameterised ansatz is applied
- an observable is measured
- the measured scalar is used as the prediction
- parameters are trained by minimising a regression loss

---

# Data

We consider a regression dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
$$

where:

- $N$ is the number of samples
- $x_i \in \mathbb{R}^d$ is the feature vector for sample $i$
- $y_i \in \mathbb{R}$ is the target value for sample $i$
- $d$ is the feature dimension

In the current implementation:

- $d = 2$
- the dataset is a synthetic regression dataset generated with `sklearn.datasets.make_regression`
- input features are standardised
- targets are standardised

Let:

$$
x = (x_1, x_2)
$$

denote one standardised input sample.

---

# Quantum state preparation

The input is encoded into a quantum state using an angle embedding.

Let:

- $n$ be the number of qubits
- $n = d$ in the current implementation
- $|0\rangle^{\otimes n}$ be the initial computational basis state

The feature map is:

$$
|\phi(x)\rangle = U_{\text{enc}}(x)\,|0\rangle^{\otimes n}
$$

where $U_{\text{enc}}(x)$ is the encoding unitary.

## Angle embedding

For an input vector $x \in \mathbb{R}^n$, the encoding applies one $R_Y$ rotation per qubit:

$$
U_{\text{enc}}(x)
=
\prod_{j=1}^{n} R_Y(x_j)
$$

where:

- $x_j$ is feature $j$
- $R_Y(\alpha)$ is a single-qubit rotation by angle $\alpha$ about the $Y$ axis

The matrix form is:

$$
R_Y(\alpha)
=
\begin{pmatrix}
\cos(\alpha/2) & -\sin(\alpha/2) \\
\sin(\alpha/2) & \cos(\alpha/2)
\end{pmatrix}
$$

---

# Variational ansatz

After encoding, a trainable circuit is applied.

Let:

$$
\theta
$$

denote the full set of trainable parameters.

The ansatz unitary is:

$$
U_{\text{ans}}(\theta)
$$

and the full circuit state is:

$$
|\psi(x,\theta)\rangle
=
U_{\text{ans}}(\theta)\,U_{\text{enc}}(x)\,|0\rangle^{\otimes n}
$$

## Layered hardware-efficient ansatz

The implemented ansatz uses:

- one layer index $\ell = 1,\dots,L$
- one qubit index $j = 1,\dots,n$

where:

- $L$ is the number of variational layers
- $n$ is the number of qubits

Each layer applies:

1. $R_Y$ on each qubit
2. $R_Z$ on each qubit
3. a chain of CNOT gates for entanglement

The parameter tensor is:

$$
\theta \in \mathbb{R}^{L \times n \times 2}
$$

where:

- $\theta_{\ell,j,1}$ is the $R_Y$ angle for layer $\ell$, qubit $j$
- $\theta_{\ell,j,2}$ is the $R_Z$ angle for layer $\ell$, qubit $j$

One layer has the form:

$$
U_{\text{layer}}^{(\ell)}
=
U_{\text{ent}}
\left(
\prod_{j=1}^{n}
R_Z(\theta_{\ell,j,2})\,
R_Y(\theta_{\ell,j,1})
\right)
$$

where $U_{\text{ent}}$ is the entangling unitary.

For the chain entangler:

$$
U_{\text{ent}}
=
\prod_{j=1}^{n-1} \mathrm{CNOT}_{j,j+1}
$$

Thus the full ansatz is:

$$
U_{\text{ans}}(\theta)
=
\prod_{\ell=1}^{L}
U_{\text{layer}}^{(\ell)}
$$

---

# Measurement and model output

The circuit measures the expectation value of the Pauli-$Z$ observable on the first qubit.

Let:

$$
M = Z_1
$$

where $Z_1$ is Pauli $Z$ acting on qubit 1.

The model prediction is:

$$
\hat{y}(x,\theta)
=
\langle \psi(x,\theta) | M | \psi(x,\theta) \rangle
$$

where:

- $\hat{y}(x,\theta)$ is the predicted target value
- $\hat{y}(x,\theta) \in [-1,1]$

Because the targets are standardised, this bounded output is sufficient for the current minimal implementation.

---

# Loss function

Training uses mean squared error.

For a batch of $N$ training samples, let:

- $y_i \in \mathbb{R}$ be the true target of sample $i$
- $\hat{y}_i = \hat{y}(x_i,\theta)$ be the predicted target of sample $i$

The loss is:

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)^2
$$

where:

- $\mathcal{L}(\theta)$ is the training objective
- $N$ is the number of training samples

---

# Optimisation

The parameters $\theta$ are trained using a classical optimiser.

The current implementation uses Adam with step size:

$$
\eta > 0
$$

where $\eta$ is the learning rate.

Training proceeds for a fixed number of steps:

$$
T
$$

where $T$ is the total number of optimisation iterations.

At each step:

1. evaluate the circuit on the training set
2. compute predictions $\hat{y}_i$
3. compute loss $\mathcal{L}(\theta)$
4. compute gradients with respect to $\theta$
5. update $\theta$ using Adam

The loss history is recorded as:

$$
\{\mathcal{L}^{(t)}\}_{t=1}^{T}
$$

where $\mathcal{L}^{(t)}$ is the loss after optimisation step $t$.

---

# Regression metrics

After training, predictions are formed on both train and test sets.

For any evaluation set of size $M$, let:

- $y_i$ be the true target of sample $i$
- $\hat{y}_i$ be the predicted target of sample $i$

## Mean squared error

$$
\mathrm{MSE}
=
\frac{1}{M}
\sum_{i=1}^{M}
(\hat{y}_i - y_i)^2
$$

where:

- $\mathrm{MSE}$ is the mean squared error
- $M$ is the number of evaluated samples

## Mean absolute error

$$
\mathrm{MAE}
=
\frac{1}{M}
\sum_{i=1}^{M}
|\hat{y}_i - y_i|
$$

where:

- $\mathrm{MAE}$ is the mean absolute error

The implementation reports:

- training MSE
- test MSE
- training MAE
- test MAE

---

# Parameter count

The ansatz parameter tensor has shape:

$$
(L, n, 2)
$$

so the total number of trainable parameters is:

$$
P = 2Ln
$$

where:

- $P$ is the total number of trainable parameters
- $L$ is the number of layers
- $n$ is the number of qubits

For the current minimal model:

- $n = 2$
- typical choice: $L = 2$

so:

$$
P = 2 \cdot 2 \cdot 2 = 8
$$

---

# Current implementation choices

The current VQR is intentionally minimal.

## Included

- regression with scalar targets
- two-dimensional input
- angle embedding
- hardware-efficient ansatz
- Pauli-$Z$ measurement on the first qubit
- mean squared error training
- Adam optimisation
- MSE and MAE reporting
- prediction visualisation

## Not yet included

- multi-output regression
- minibatching
- alternative observables
- alternative ansätze
- alternative embeddings
- rescaling from bounded outputs to unbounded targets
- shot noise / hardware execution
- uncertainty estimates

---

# Relation to the code

The implemented workflow is organised as follows:

- `qml.data` prepares the dataset
- `qml.embeddings` applies the feature map
- `qml.ansatz` applies the trainable circuit
- `qml.regression.run_vqr` performs training and evaluation
- `qml.visualize` creates plots

So the notebook remains a package client, while the full VQR logic lives in the package.

---

# Summary

The implemented VQR is a regressor defined by:

1. a feature map $U_{\text{enc}}(x)$
2. a trainable ansatz $U_{\text{ans}}(\theta)$
3. an observable $M = Z_1$
4. a scalar prediction from the expectation value
5. a mean squared error training objective

Formally:

$$
|\psi(x,\theta)\rangle
=
U_{\text{ans}}(\theta)\,U_{\text{enc}}(x)\,|0\rangle^{\otimes n}
$$

$$
\hat{y}(x,\theta)
=
\langle \psi(x,\theta) | Z_1 | \psi(x,\theta) \rangle
$$

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)^2
$$

This is the core variational regression workflow used in the repository.