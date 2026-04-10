# Quantum Autoencoder

This note describes the quantum autoencoder workflow implemented in `qml.autoencoder`.

The current implementation is intentionally compact and package-oriented:

• structured four-qubit input state families  
• a trainable encoder/decoder ansatz  
• latent and trash subsystem separation  
• compression and reconstruction fidelity reporting  

---

# Overview

A quantum autoencoder learns a unitary compression map that preserves the
informative degrees of freedom of a quantum state in a smaller latent subspace.

Rather than predicting labels directly, it learns a transformation that moves
discardable information into a trash subsystem.

---

# Model structure

Let the input state be

$$
|\psi(x)\rangle.
$$

The encoder applies a trainable unitary

$$
|\phi(x,\theta)\rangle
=
U(\theta)|\psi(x)\rangle.
$$

If compression succeeds, the state factorizes approximately as

$$
|\phi(x,\theta)\rangle
\approx
|\tilde{\psi}(x)\rangle_{\mathrm{latent}}
\otimes
|0\rangle_{\mathrm{trash}}.
$$

The implementation retains a configurable number of latent qubits and measures
how often the trash subsystem lands in the all-zero basis state.

---

# Training objective

The training signal is the probability of measuring the trash subsystem in
$|0\rangle^{\otimes k}$.

If

$$
p_{\mathrm{trash}}(0 \cdots 0 \mid x,\theta)
$$

denotes that probability, the loss is

$$
\mathcal{L}(\theta)
=
1 - \mathbb{E}_x \left[p_{\mathrm{trash}}(0 \cdots 0 \mid x,\theta)\right].
$$

Minimizing this loss encourages the encoder to compress the structured state
family into the latent subsystem.

---

# Reconstruction fidelity

To assess whether useful information is preserved, the workflow also computes a
reconstruction fidelity by applying the decoder

$$
U(\theta)^\dagger
$$

after the encoder and comparing the resulting state to the original state.

This yields two complementary metrics:

• compression fidelity on the trash subsystem  
• reconstruction fidelity on the full state  

---

# Example usage

```python
from qml.autoencoder import run_quantum_autoencoder

result = run_quantum_autoencoder(
    family="correlated",
    n_samples=200,
    n_layers=2,
    latent_qubits=2,
    steps=50,
)
```

Outputs include:

• train/test compression fidelity  
• train/test reconstruction fidelity  
• learned ansatz parameters  
• loss history  

When `save=True`, the workflow writes JSON results and generated figures to:

• `results/autoencoder/`
• `images/autoencoder/`

---

# State families

The current implementation provides several synthetic state families:

• `correlated`
• `entangled`
• `hybrid`

These are designed to provide structured low-dimensional families that are
meaningful compression targets for a small autoencoder.
