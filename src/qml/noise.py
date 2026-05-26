"""
qml.noise
=========

Small reusable helpers for opt-in noisy circuit simulations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pennylane as qml

NoiseModel = Mapping[str, float] | None

_ALIASES = {
    "depolarizing": "depolarizing",
    "depolarizing_prob": "depolarizing",
    "amplitude_damping": "amplitude_damping",
    "amplitude_damping_prob": "amplitude_damping",
    "readout_error": "readout_error",
    "readout_error_prob": "readout_error",
}


def normalize_noise_model(noise_model: NoiseModel) -> dict[str, float] | None:
    """
    Validate and canonicalize a simple channel-probability noise model.

    Supported keys are ``depolarizing``, ``amplitude_damping``, and
    ``readout_error``. Values must be probabilities in ``[0, 1]``. Empty models
    and all-zero models return ``None`` so noiseless callers keep their existing
    simulator path.
    """
    if noise_model is None:
        return None

    if not isinstance(noise_model, Mapping):
        raise TypeError("noise_model must be a mapping or None.")

    normalized: dict[str, float] = {}
    for key, value in noise_model.items():
        canonical = _ALIASES.get(str(key).strip().lower())
        if canonical is None:
            supported = ", ".join(sorted(set(_ALIASES.values())))
            raise ValueError(f"Unknown noise channel {key!r}. Supported channels: {supported}.")

        probability = float(value)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"{canonical} probability must be between 0 and 1.")

        if probability > 0.0:
            normalized[canonical] = probability

    return normalized or None


def noise_model_to_dict(noise_model: NoiseModel) -> dict[str, float] | None:
    """Return a JSON-serializable canonical noise-model dictionary."""
    normalized = normalize_noise_model(noise_model)
    return None if normalized is None else dict(normalized)


def device_name_for_noise(noise_model: NoiseModel) -> str:
    """Return the PennyLane device name required by the optional noise model."""
    return "default.mixed" if normalize_noise_model(noise_model) else "default.qubit"


def apply_noise_channels(
    wires: Sequence[int],
    noise_model: NoiseModel,
    *,
    readout_wires: Sequence[int] | None = None,
) -> None:
    """
    Apply supported noise channels to a circuit.

    Depolarizing and amplitude-damping channels are applied to every listed
    wire. Readout error is approximated by a bit-flip channel immediately before
    measurement on ``readout_wires``.
    """
    normalized = normalize_noise_model(noise_model)
    if normalized is None:
        return

    wires = list(wires)
    for wire in wires:
        if "depolarizing" in normalized:
            qml.DepolarizingChannel(normalized["depolarizing"], wires=wire)
        if "amplitude_damping" in normalized:
            qml.AmplitudeDamping(normalized["amplitude_damping"], wires=wire)

    for wire in list(wires if readout_wires is None else readout_wires):
        if "readout_error" in normalized:
            qml.BitFlip(normalized["readout_error"], wires=wire)


def noise_model_tag(noise_model: NoiseModel) -> str:
    """Return a compact filename-safe tag for a noise model."""
    normalized = normalize_noise_model(noise_model)
    if normalized is None:
        return "noiseless"

    parts: list[str] = []
    for key in ("depolarizing", "amplitude_damping", "readout_error"):
        if key in normalized:
            value = str(normalized[key]).replace(".", "p")
            parts.append(f"{key.replace('_', '')}{value}")
    return "_".join(parts)


def noise_model_cache_key(noise_model: NoiseModel) -> tuple[tuple[str, float], ...] | None:
    """Return a stable hashable cache key for a noise model."""
    normalized = normalize_noise_model(noise_model)
    if normalized is None:
        return None
    return tuple(sorted(normalized.items()))


def build_noise_model(
    *,
    depolarizing: float = 0.0,
    amplitude_damping: float = 0.0,
    readout_error: float = 0.0,
) -> dict[str, float] | None:
    """Build a canonical noise model from explicit channel probabilities."""
    return noise_model_to_dict(
        {
            "depolarizing": depolarizing,
            "amplitude_damping": amplitude_damping,
            "readout_error": readout_error,
        }
    )


def has_noise(noise_model: Any) -> bool:
    """Return whether a noise model enables at least one nonzero channel."""
    return normalize_noise_model(noise_model) is not None
