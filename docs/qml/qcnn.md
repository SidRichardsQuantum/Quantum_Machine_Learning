# Quantum Convolutional Neural Networks

This note describes the QCNN classifier implemented in `qml.qcnn`.

The current implementation is intentionally compact and package-oriented:

• four-qubit trainable input embedding  
• shared local convolution-style blocks  
• pooling-style entangling reductions  
• final single-qubit readout for binary classification  

---

# Overview

A quantum convolutional neural network mirrors the high-level idea of a
classical CNN: local processing is applied first, and information is then
compressed into a smaller effective representation before the final readout.

In this repository, the QCNN classifier acts on a fixed four-qubit register and
is trained directly on synthetic binary classification datasets.

---

# Model structure

Let the embedded input state be

$$
|\phi(x)\rangle.
$$

The QCNN applies a hierarchical circuit of the form

$$
|\psi(x,\theta)\rangle
=
U_{\mathrm{dense}}(\theta_d)
U_{\mathrm{conv},2}(\theta_2)
U_{\mathrm{pool}}
U_{\mathrm{conv},1}(\theta_1)
U_{\mathrm{emb}}(x,\theta_e)
|0\rangle^{\otimes 4}.
$$

The components are:

• `U_emb`: trainable feature embedding using data-dependent single-qubit rotations  
• `U_conv,1`: first-stage shared two-qubit convolution blocks on neighbouring pairs  
• `U_pool`: pooling-style entangling reductions  
• `U_conv,2`: second-stage convolution on the reduced representation  
• `U_dense`: final single-qubit rotations before readout  

---

# Readout and loss

The model measures a Pauli-$Z$ expectation on the readout qubit:

$$
s(x,\theta) = \langle Z_3 \rangle.
$$

This expectation is mapped to a binary probability:

$$
p(y=1 \mid x,\theta)
=
\frac{1 - s(x,\theta)}{2}.
$$

Training minimizes binary cross-entropy over the training set.

---

# Example usage

```python
from qml.qcnn import run_qcnn

result = run_qcnn(
    dataset="moons",
    n_samples=200,
    steps=50,
    step_size=0.1,
)
```

Outputs include:

• flattened trainable parameters  
• structured QCNN parameter blocks  
• training and test accuracy  
• optimisation loss history  
• predicted probabilities on train and test splits  

When `save=True`, the workflow writes JSON results and generated figures to:

• `results/qcnn/`
• `images/qcnn/`

---

# Relationship to Other Models

Compared with the existing VQC workflow:

• VQC uses a repeated global ansatz template  
• QCNN uses an explicitly hierarchical convolution/pooling structure  

Compared with quantum kernels:

• QCNN learns a direct discriminative classifier  
• kernel workflows learn similarity structure and rely on a classical SVM readout  

---

# When to Use QCNN

QCNN is useful when:

• you want a more structured architecture than a flat variational circuit  
• local hierarchical feature extraction is a better inductive bias  
• you want a classifier that differs materially from the existing VQC workflow  

---

# References

Cong et al. (2019)
Quantum convolutional neural networks.
