# Theory

This repository implements core methods in **quantum machine learning (QML)** using parameterised quantum circuits and quantum feature maps.

The focus is on **supervised learning** tasks, where a hybrid quantum–classical model learns from labelled data.

---

# Hybrid quantum–classical learning

Most QML models take the form:

$$
f_\theta(x)
$$

where:

- $x$ is a classical input vector
- $\theta$ are trainable circuit parameters
- $f_\theta$ is computed using a quantum circuit

A typical workflow:

1. encode classical data into a quantum state
2. apply a parameterised circuit
3. measure an observable
4. compute a classical loss
5. update parameters using classical optimisation

---

# Data encoding (feature maps)

Classical data must be embedded into quantum states.

Given a feature vector:

$$
x \in \mathbb{R}^d
$$

we construct a quantum state:

$$
|\phi(x)\rangle
$$

using a parameterised unitary:

$$
|\phi(x)\rangle = U(x)|0\rangle
$$

## Angle embedding

One simple encoding is angle embedding:

$$
U(x) =
\prod_i R_Y(x_i)
$$

where:

$$
R_Y(\alpha) =
\begin{pmatrix}
\cos(\alpha/2) & -\sin(\alpha/2) \\
\sin(\alpha/2) & \cos(\alpha/2)
\end{pmatrix}
$$

Angle embedding maps features directly to rotation angles.

---

# Variational quantum circuits

Variational models use a parameterised unitary:

$$
U(\theta)
$$

composed of single-qubit rotations and entangling gates.

The model output is an expectation value:

$$
f_\theta(x) =
\langle 0 |
U^\dagger(x)
U^\dagger(\theta)
M
U(\theta)
U(x)
|0\rangle
$$

where:

- $M$ is an observable (e.g. Pauli $Z$)
- $U(x)$ encodes data
- $U(\theta)$ contains trainable parameters

---

# Hardware-efficient ansatz

A simple ansatz uses layers of single-qubit rotations followed by entanglement:

$$
U(\theta) =
\prod_{\ell=1}^{L}
\left(
\prod_{i=1}^{n}
R_Y(\theta_{\ell,i,1})
R_Z(\theta_{\ell,i,2})
\right)
\cdot
U_{\text{ent}}
$$

where:

$$
U_{\text{ent}}
=
\prod_i \text{CNOT}_{i,i+1}
$$

Properties:

- shallow circuits
- hardware compatible
- expressive but trainable
- commonly used baseline architecture

---

# Variational quantum classifier (VQC)

The classifier predicts probabilities using expectation values.

For a single observable:

$$
z(x,\theta) =
\langle Z_0 \rangle
$$

we map:

$$
p(y=1|x,\theta) =
\frac{1 - z(x,\theta)}{2}
$$

Binary prediction:

$$
\hat{y} =
\begin{cases}
1 & p \ge 0.5 \\
0 & p < 0.5
\end{cases}
$$

---

# Variational quantum regression

A variational quantum regressor uses the same basic hybrid structure as a variational classifier, but predicts a continuous target rather than a class probability.

For an input $x$, define:

$$
\hat{y}(x,\theta)
=
\langle \psi(x,\theta) | M | \psi(x,\theta) \rangle
$$

where:

- $\hat{y}(x,\theta) \in \mathbb{R}$ is the predicted target
- $\theta$ are trainable circuit parameters
- $M$ is a measured observable, such as Pauli $Z$

In the current implementation, the observable is:

$$
M = Z_1
$$

so the prediction is:

$$
\hat{y}(x,\theta)
=
\langle Z_1 \rangle
$$

Since Pauli-$Z$ expectation values lie in $[-1,1]$, the current workflow standardises the target values before training.

## Regression loss

Training uses mean squared error:

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)^2
$$

where:

- $N$ is the number of training samples
- $y_i$ is the true target for sample $i$
- $\hat{y}_i$ is the predicted target for sample $i$

Evaluation also reports mean absolute error:

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{y}_i - y_i|
$$

So the variational regressor shares the same quantum building blocks as the classifier:

- data encoding
- parameterised ansatz
- expectation-value measurement
- classical optimisation

but differs in how the measured scalar is interpreted and how the loss is defined.

---

# Loss functions

Binary classification uses cross-entropy:

$$
\mathcal{L}(\theta) =
-\frac{1}{N}
\sum_i
\left[
y_i \log p_i +
(1-y_i)\log(1-p_i)
\right]
$$

Optimisation is performed using gradient-based methods such as Adam.

Gradients are computed using automatic differentiation and the parameter-shift rule.

---

# Quantum kernel methods

Kernel methods avoid explicit parameter training.

Instead, we compute a similarity measure:

$$
K(x_i,x_j)
=
|\langle \phi(x_i) | \phi(x_j) \rangle|^2
$$

where:

$$
|\phi(x)\rangle = U(x)|0\rangle
$$

This defines a kernel matrix:

$$
K_{ij} = K(x_i,x_j)
$$

which is used by classical algorithms such as support vector machines.

---

# Kernel evaluation using quantum circuits

The overlap can be computed using:

$$
K(x_i,x_j)
=
|\langle 0|
U^\dagger(x_j)
U(x_i)
|0\rangle|^2
$$

In practice:

1. apply feature map $U(x_i)$
2. apply inverse feature map $U^\dagger(x_j)$
3. measure probability of the zero state

---

# Support vector machines with quantum kernels

Given a kernel matrix:

$$
K_{ij}
$$

the classifier takes the form:

$$
f(x)
=
\sum_i
\alpha_i
K(x_i,x)
+
b
$$

where:

- $\alpha_i$ are learned weights
- $b$ is a bias term

Training solves a convex optimisation problem.

The quantum computer provides the kernel evaluations.

---

# Model capacity

Expressivity depends on:

- embedding choice
- circuit depth
- entanglement structure
- number of qubits

Tradeoffs:

- deeper circuits increase expressivity
- deeper circuits increase noise sensitivity
- more qubits increase dimensional capacity

---

# General workflow

For both VQC and kernel methods:

1. prepare dataset
2. encode data
3. evaluate quantum model
4. compute classical objective
5. update classical parameters or classifier

---

# Future extensions

Possible extensions include:

- data re-uploading models
- quantum convolutional feature maps
- variational regression
- noise-aware training
- trainability analysis
- expressivity studies
- classical baselines for comparison

---

# References

Schuld, M., Sinayskiy, I., & Petruccione, F. (2015)  
An introduction to quantum machine learning.

Havlíček et al. (2019)  
Supervised learning with quantum-enhanced feature spaces.

Farhi & Neven (2018)  
Classification with quantum neural networks.

Mitarai et al. (2018)  
Quantum circuit learning.