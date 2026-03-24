# Variational Quantum Classifier

This note describes the variational quantum classifier (VQC) implemented in `qml.classifiers`.

The model is a **hybrid quantum–classical binary classifier**:

- a classical feature vector is encoded into a quantum circuit
- a parameterised ansatz is applied
- an observable is measured
- the resulting scalar is converted into a class probability
- parameters are trained by minimising a classical loss

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
- features are standardised before entering the circuit

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

So the encoded state depends directly on the input features.

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

The raw model output is:

$$
z(x,\theta)
=
\langle \psi(x,\theta) | M | \psi(x,\theta) \rangle
$$

with:

$$
z(x,\theta) \in [-1,1]
$$

This expectation value is then mapped to a probability:

$$
p_\theta(y=1 \mid x)
=
\frac{1 - z(x,\theta)}{2}
$$

and therefore:

$$
p_\theta(y=0 \mid x)
=
1 - p_\theta(y=1 \mid x)
$$

where:

- $p_\theta(y=1 \mid x)$ is the predicted probability of class 1
- $p_\theta(y=0 \mid x)$ is the predicted probability of class 0

## Decision rule

The predicted class is:

$$
\hat{y}(x)
=
\begin{cases}
1, & p_\theta(y=1 \mid x) \ge 0.5 \\
0, & p_\theta(y=1 \mid x) < 0.5
\end{cases}
$$

where $\hat{y}(x)$ is the predicted label for input $x$.

---

# Loss function

Training uses binary cross-entropy.

For a batch of $N$ training samples, let:

- $y_i \in \{0,1\}$ be the true label of sample $i$
- $p_i = p_\theta(y=1 \mid x_i)$ be the predicted probability of class 1 for sample $i$

The loss is:

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right]
$$

where:

- $\mathcal{L}(\theta)$ is the training objective
- $N$ is the number of training samples

In implementation, probabilities are clipped slightly away from 0 and 1 for numerical stability.

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
2. compute probabilities $p_i$
3. compute loss $\mathcal{L}(\theta)$
4. compute gradients with respect to $\theta$
5. update $\theta$ using Adam

The loss history is recorded as:

$$
\{\mathcal{L}^{(t)}\}_{t=1}^{T}
$$

where $\mathcal{L}^{(t)}$ is the loss after optimisation step $t$.

---

# Accuracy

After training, predictions are formed on both train and test sets.

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

The current VQC is intentionally minimal.

## Included

- binary classification
- two-dimensional input
- angle embedding
- hardware-efficient ansatz
- Pauli-$Z$ measurement on the first qubit
- binary cross-entropy
- Adam optimisation
- train/test accuracy
- loss curve and decision-boundary visualisation

## Not yet included

- multiclass classification
- minibatching
- alternative observables
- alternative ansätze
- alternative embeddings
- shot noise / hardware execution
- regularisation terms
- calibration or uncertainty estimates

---

# Interpretation

The model learns a map:

$$
x \mapsto p_\theta(y=1 \mid x)
$$

where the nonlinearity comes from:

- quantum state preparation
- entangling operations
- nonlinear dependence of expectation values on the parameters and inputs

In practice, the VQC behaves like a compact hybrid classifier whose expressive power depends on:

- number of qubits $n$
- number of layers $L$
- embedding choice
- entanglement structure
- optimiser settings

---

# Relation to the code

The implemented workflow is organised as follows:

- `qml.data` prepares the dataset
- `qml.embeddings` applies the feature map
- `qml.ansatz` applies the trainable circuit
- `qml.classifiers.run_vqc` performs training and evaluation
- `qml.visualize` creates plots

So the notebook remains a package client, while the full VQC logic lives in the package.

---

# Summary

The implemented VQC is a binary classifier defined by:

1. a feature map $U_{\text{enc}}(x)$
2. a trainable ansatz $U_{\text{ans}}(\theta)$
3. an observable $M = Z_1$
4. a probability map from expectation values to class probabilities
5. a binary cross-entropy training objective

Formally:

$$
|\psi(x,\theta)\rangle
=
U_{\text{ans}}(\theta)\,U_{\text{enc}}(x)\,|0\rangle^{\otimes n}
$$

$$
z(x,\theta)
=
\langle \psi(x,\theta) | Z_1 | \psi(x,\theta) \rangle
$$

$$
p_\theta(y=1 \mid x)
=
\frac{1 - z(x,\theta)}{2}
$$

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right]
$$

This is the core VQC workflow used in the repository.