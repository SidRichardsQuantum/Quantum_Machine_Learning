# Theory

This repository implements core methods in **quantum machine learning (QML)** using parameterised quantum circuits and quantum feature maps.

The focus is on **supervised learning** using hybrid quantum–classical models.

Workflows include:

• variational quantum classifiers  
• variational quantum regressors  
• quantum kernel methods  
• trainable quantum kernels  
• quantum metric learning  

All models rely on parameterised quantum circuits evaluated within classical optimisation loops.

---

# Hybrid quantum–classical learning

Most QML models take the form:

$$
f_\theta(x)
$$

where:

• $x \in \mathbb{R}^d$ is a classical input vector  
• $\theta$ are trainable circuit parameters  
• $f_\theta$ is computed using a quantum circuit  

Typical workflow:

1. encode classical data into a quantum state
2. apply a parameterised circuit
3. measure an observable
4. compute a classical loss
5. update parameters using classical optimisation

Optimisation is performed using gradient-based methods such as Adam.

Gradients are computed using automatic differentiation and the parameter-shift rule.

---

# Data encoding (feature maps)

Classical data must be embedded into quantum states.

Given a feature vector:

$$
x \in \mathbb{R}^d
$$

we prepare a quantum state:

$$
|\phi(x)\rangle
=
U(x)|0\rangle
$$

where:

$$
U(x)
$$

is a parameterised unitary.

---

## Angle embedding

A simple encoding uses single-qubit rotations:

$$
U(x)
=
\prod_{i=1}^d
R_Y(x_i)
$$

where:

$$
R_Y(\alpha)
=
\begin{pmatrix}
\cos(\alpha/2)
&
-\sin(\alpha/2)
\\
\sin(\alpha/2)
&
\cos(\alpha/2)
\end{pmatrix}
$$

Angle embedding maps classical features directly to rotation angles.

---

# Variational quantum circuits

Variational models use parameterised circuits:

$$
U(\theta)
$$

composed of single-qubit rotations and entangling gates.

The model output is an expectation value:

$$
f_\theta(x)
=
\langle 0 |
U^\dagger(x)
U^\dagger(\theta)
M
U(\theta)
U(x)
|0\rangle
$$

where:

• $M$ is an observable  
• $U(x)$ encodes data  
• $U(\theta)$ contains trainable parameters  

---

# Hardware-efficient ansatz

A common ansatz uses repeated layers:

$$
U(\theta)
=
\prod_{\ell=1}^{L}
\left(
\prod_{i=1}^{n}
R_Y(\theta_{\ell,i,1})
R_Z(\theta_{\ell,i,2})
\right)
U_{ent}
$$

with entanglement:

$$
U_{ent}
=
\prod_{i=1}^{n-1}
\text{CNOT}_{i,i+1}
$$

Properties:

• shallow circuit depth  
• hardware compatible  
• expressive but trainable  
• widely used baseline  

---

# Expectation values

Variational models produce scalar outputs via expectation values:

$$
f_\theta(x)
=
\langle \psi(x,\theta) | M | \psi(x,\theta) \rangle
$$

with:

$$
|\psi(x,\theta)\rangle
=
U(\theta)U(x)|0\rangle
$$

Typical observable:

$$
M = Z_i
$$

giving outputs in:

$$
[-1,1]
$$

---

# Finite-shot estimation (noise-aware execution)

Expectation values may be computed either analytically or via sampling.

Given $S$ measurement shots:

$$
\hat{f}_\theta(x)
=
\frac{1}{S}
\sum_{s=1}^{S}
m_s
$$

where:

$$
m_s \in \{-1,1\}
$$

Finite-shot evaluation introduces sampling variance:

$$
\mathrm{Var}[\hat{f}_\theta(x)]
=
\frac{\sigma^2}{S}
$$

As:

$$
S \rightarrow \infty
$$

the estimate converges to the analytic expectation value.

Finite-shot sampling simulates noise effects present on real hardware.

---

# Variational quantum classifier (VQC)

Binary classification uses expectation values mapped to probabilities.

Define:

$$
z(x,\theta)
=
\langle Z_0 \rangle
$$

Probability of class 1:

$$
p(y=1|x,\theta)
=
\frac{1 - z(x,\theta)}{2}
$$

Prediction rule:

$$
\hat{y}
=
\begin{cases}
1 & p \ge 0.5 \\
0 & p < 0.5
\end{cases}
$$

---

## Classification loss

Binary cross-entropy:

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^N
\left[
y_i \log p_i
+
(1-y_i)\log(1-p_i)
\right]
$$

Optimisation adjusts parameters $\theta$ to minimise classification error.

---

# Variational quantum regression (VQR)

Regression uses expectation values as continuous predictions.

Prediction:

$$
\hat{y}(x,\theta)
=
\langle Z_1 \rangle
$$

Targets are typically standardised:

$$
y \rightarrow \tilde{y}
$$

to match the observable output range.

---

## Regression loss

Mean squared error:

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat{y}_i - y_i)^2
$$

Evaluation metrics include:

Mean absolute error:

$$
\mathrm{MAE}
=
\frac{1}{N}
\sum_i
|\hat{y}_i - y_i|
$$

Regression uses the same quantum architecture as classification but a different loss.

---

# Quantum kernel methods

Kernel methods avoid explicit parameter optimisation.

Instead, similarity between inputs is computed:

$$
K(x_i,x_j)
=
|\langle \phi(x_i) | \phi(x_j) \rangle|^2
$$

where:

$$
|\phi(x)\rangle
=
U(x)|0\rangle
$$

---

# Kernel evaluation using quantum circuits

Kernel values are computed using:

$$
K(x_i,x_j)
=
|\langle 0|
U^\dagger(x_j)
U(x_i)
|0\rangle|^2
$$

Procedure:

1. apply feature map $U(x_i)$
2. apply inverse feature map $U^\dagger(x_j)$
3. measure probability of the zero state

Construct kernel matrix:

$$
K_{ij}
=
K(x_i,x_j)
$$

---

# Support vector machines with quantum kernels

Given kernel matrix:

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

• $\alpha_i$ are learned coefficients  
• $b$ is a bias term  

Training solves a convex optimisation problem.

The quantum computer supplies kernel values.

---

# Trainable quantum kernels

Instead of fixing the feature map, parameters $\theta$ can be introduced:

$$
|\phi(x,\theta)\rangle
=
U(x,\theta)|0\rangle
$$

giving kernel:

$$
K_\theta(x_i,x_j)
=
|\langle \phi(x_i,\theta) | \phi(x_j,\theta) \rangle|^2
$$

---

## Kernel-target alignment

Trainable kernels optimise similarity between:

• kernel matrix $K_\theta$  
• label similarity matrix $Y$

Label similarity:

$$
Y_{ij}
=
y_i y_j
$$

Alignment objective:

$$
A(\theta)
=
\frac{
\langle K_\theta, Y \rangle_F
}{
\|K_\theta\|_F
\|Y\|_F
}
$$

where Frobenius inner product:

$$
\langle A,B \rangle_F
=
\sum_{ij} A_{ij}B_{ij}
$$

Optimisation objective:

$$
\max_\theta A(\theta)
$$

Alignment encourages kernel similarity to reflect class structure.

---

# Quantum metric learning

Quantum metric learning aims to learn an embedding geometry in which distances between samples reflect label similarity.

Instead of directly predicting labels, the model learns a parameterised quantum embedding:

$$
|\phi(x,\theta)\rangle
=
U(x,\theta)|0\rangle
$$

The quantum circuit defines a feature map:

$$
z(x,\theta)
=
\langle Z_i \rangle
$$

where expectation values of Pauli observables form an embedding vector:

$$
z(x,\theta)
\in \mathbb{R}^k
$$

for a $k$-qubit circuit.

---

## Distance-based supervision

Given two samples:

$$
x_i, x_j
$$

define embedding distance:

$$
d_{ij}
=
\|z(x_i,\theta) - z(x_j,\theta)\|_2
$$

Training encourages:

• small distances for same-class pairs  
• large distances for different-class pairs  

---

## Contrastive loss

Define label similarity indicator:

$$
y_{ij}
=
\begin{cases}
1 & y_i = y_j \\
0 & y_i \ne y_j
\end{cases}
$$

Contrastive objective:

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
• $d_{ij}$ is Euclidean distance in embedding space  

The margin encourages separation between classes.

---

## Data re-uploading embeddings

Expressive embeddings may be constructed using repeated feature encoding layers:

$$
U(x,\theta)
=
\prod_{\ell=1}^L
U_{ent}
U_{enc}(x,\theta_\ell)
$$

where:

$$
U_{enc}(x,\theta)
=
\prod_i
R_X(x_i + \theta_{i1})
R_Y(x_i + \theta_{i2})
R_Z(\theta_{i3})
$$

Repeated encoding increases expressivity without increasing qubit count.

---

## Classification in embedding space

After optimisation, predictions may be performed using classical methods.

One simple approach uses nearest centroid classification.

Compute class centroids:

$$
c_k
=
\frac{1}{N_k}
\sum_{i : y_i = k}
z(x_i,\theta)
$$

Prediction:

$$
\hat{y}
=
\arg\min_k
\|z(x,\theta) - c_k\|_2
$$

Metric learning therefore separates:

• representation learning (quantum)
• classification rule (classical)

---

## Relationship to kernel methods

Metric learning and kernel methods both rely on quantum feature maps.

Kernel methods compute similarity:

$$
K(x_i,x_j)
=
|\langle \phi(x_i)|\phi(x_j)\rangle|^2
$$

Metric learning instead optimises parameters such that Euclidean distances in embedding space reflect label similarity.

Both approaches use quantum circuits to construct feature representations.

---

## Relationship to variational models

Variational classifiers directly optimise prediction error.

Metric learning optimises geometry of the feature space.

Advantages:

• decouples representation learning from classifier choice  
• allows classical classifiers to operate on quantum features  
• supports few-shot learning scenarios  
• provides interpretable embedding structure  

---

## Model capacity considerations

Embedding expressivity depends on:

• number of qubits  
• circuit depth  
• entanglement structure  
• number of re-uploading layers  

As circuit depth increases, the embedding may represent more complex similarity structure.

---

# Relationship between models

Variational models:

learn parameters inside quantum circuits.

Kernel models:

use quantum circuits to compute similarity measures.

Trainable kernels:

learn parameters inside the feature map rather than classifier weights.

---

# Model capacity

Expressivity depends on:

• embedding structure  
• circuit depth  
• entanglement pattern  
• number of qubits  

Tradeoffs:

• deeper circuits increase expressivity  
• deeper circuits increase noise sensitivity  
• more qubits increase dimensional capacity  

---

# General workflow

Common structure across models:

1. prepare dataset
2. encode data into quantum state
3. evaluate circuit
4. compute classical objective
5. update parameters or classifier

---

# Noise considerations

Finite-shot sampling introduces:

• variance in expectation values  
• stochastic gradients  
• sensitivity to circuit depth  

Noise-aware evaluation allows study of:

• robustness of variational models  
• stability of kernel matrices  
• sensitivity of optimisation  

Finite-shot execution approximates behaviour of real quantum hardware.

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

Cristianini et al. (2002)  
On kernel-target alignment.

---

## License

MIT License — see LICENSE
