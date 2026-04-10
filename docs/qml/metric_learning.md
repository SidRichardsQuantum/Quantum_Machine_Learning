# Quantum Metric Learning

Quantum metric learning trains a parameterised quantum embedding such that distances between samples reflect label similarity.

Instead of directly predicting class labels, the model learns a representation in which:

• samples from the same class are close  
• samples from different classes are separated  

Classification can then be performed using simple classical methods such as nearest centroid or k-nearest neighbours.

This separates:

representation learning (quantum)
classification rule (classical)

---

# Overview

Given input features:

$$
x \in \mathbb{R}^d
$$

a quantum circuit defines an embedding:

$$
z(x, \theta)
\in \mathbb{R}^k
$$

where:

• $k$ is the number of qubits  
• $\theta$ are trainable circuit parameters  

The embedding is constructed using expectation values of Pauli observables:

$$
z(x, \theta)
=
\left(
\langle Z_0 \rangle,
\langle Z_1 \rangle,
\dots,
\langle Z_{k-1} \rangle
\right)
$$

Distances in this embedding space are used to measure similarity between inputs.

---

# Model structure

The quantum embedding uses a parameterised circuit:

$$
|\phi(x,\theta)\rangle
=
U(x,\theta)|0\rangle
$$

where:

• $U(x,\theta)$ contains both data encoding and trainable parameters
• entangling gates allow correlations between features

The embedding vector is obtained from expectation values:

$$
z_i(x,\theta)
=
\langle \phi(x,\theta) | Z_i | \phi(x,\theta) \rangle
$$

giving an embedding dimension equal to the number of qubits.

---

# Data re-uploading embedding

To increase expressivity without increasing qubit count, features may be encoded multiple times:

$$
U(x,\theta)
=
\prod_{\ell=1}^{L}
U_{ent}
U_{enc}(x,\theta_\ell)
$$

Example encoding layer:

$$
U_{enc}(x,\theta)
=
\prod_i
R_X(x_i + \theta_{i1})
R_Y(x_i + \theta_{i2})
R_Z(\theta_{i3})
$$

Repeated encoding allows the circuit to learn nonlinear transformations of the input space.

---

# Contrastive training objective

Training uses pairs of labelled samples.

Given two inputs:

$$
x_i, x_j
$$

define embedding distance:

$$
d_{ij}
=
\|z(x_i,\theta) - z(x_j,\theta)\|_2
$$

Define similarity indicator:

$$
y_{ij}
=
\begin{cases}
1 & y_i = y_j \\
0 & y_i \ne y_j
\end{cases}
$$

Contrastive loss:

$$
\mathcal{L}(\theta)
=
y_{ij} d_{ij}^2
+
(1 - y_{ij})
\max(0, m - d_{ij})^2
$$

where:

• $m$ is a margin hyperparameter  
• $d_{ij}$ is Euclidean distance between embeddings  

This objective:

• pulls same-class samples together  
• pushes different-class samples apart  

---

# Training workflow

Typical training loop:

1. sample labelled pairs from the dataset
2. compute quantum embeddings
3. compute pairwise distances
4. evaluate contrastive loss
5. update parameters using gradient-based optimisation

Optimisation is performed using classical optimisers such as Adam.

Gradients are computed using automatic differentiation of expectation values.

---

# Classification in embedding space

After training, embeddings can be used for classical classification.

A simple approach uses nearest centroid prediction.

Compute centroid for each class:

$$
c_k
=
\frac{1}{N_k}
\sum_{i : y_i = k}
z(x_i,\theta)
$$

Prediction rule:

$$
\hat{y}
=
\arg\min_k
\|z(x,\theta) - c_k\|_2
$$

Other possible classifiers include:

• k-nearest neighbours
• logistic regression
• support vector machines

---

# Relationship to other QML methods

## Variational classifiers

Variational classifiers optimise prediction error directly:

$$
f_\theta(x)
\rightarrow y
$$

Metric learning instead optimises representation geometry.

---

## Quantum kernel methods

Kernel methods compute similarity:

$$
K(x_i,x_j)
=
|\langle \phi(x_i) | \phi(x_j) \rangle|^2
$$

Metric learning uses Euclidean distance in embedding space:

$$
d(x_i,x_j)
=
\|z(x_i,\theta) - z(x_j,\theta)\|
$$

Both approaches use quantum circuits as feature maps.

---

## Trainable quantum kernels

Trainable kernels optimise similarity structure via kernel alignment.

Metric learning directly shapes the embedding geometry.

---

# Model capacity

Expressivity depends on:

• number of qubits
• circuit depth
• entanglement structure
• number of re-uploading layers

Increasing depth allows more complex similarity structure but may increase optimisation difficulty.

---

# Example usage

```python
from qml.metric_learning import run_quantum_metric_learner

result = run_quantum_metric_learner(
    dataset="moons",
    samples=200,
    layers=2,
    steps=50,
)
```

Outputs include:

• learned embedding parameters
• embedding vectors
• centroid positions
• training and test accuracy
• optimisation loss history

The public API returns a `QuantumMetricLearningResult` dataclass rather than a
plain dictionary, so values are accessed via attributes such as
`result.train_accuracy` and `result.loss_history`.

When `save=True`, the workflow also writes JSON results and generated figures to:

• `results/metric_learning/`
• `images/metric_learning/`

---

# When to use quantum metric learning

Metric learning is useful when:

• classification boundaries are complex
• similarity structure is more important than direct prediction
• small datasets require expressive embeddings
• classical classifiers benefit from learned representations

---

# References

Hadsell et al. (2006)
Dimensionality reduction by learning an invariant mapping.

Mitarai et al. (2018)
Quantum circuit learning.

Schuld & Killoran (2019)
Quantum machine learning in feature Hilbert spaces.

Cristianini et al. (2002)
Kernel-target alignment.
