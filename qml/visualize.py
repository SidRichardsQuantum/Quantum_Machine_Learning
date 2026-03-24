"""
qml.visualize
=============

Plotting utilities for qml experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_current_figure(path: str | Path, dpi: int = 200) -> None:
    """Save the current Matplotlib figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_loss_curve(
    loss_history: list[float] | np.ndarray,
    title: str = "Training loss",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot training loss versus optimization step.
    """
    loss_history = np.asarray(loss_history, dtype=float)

    plt.figure()
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_classification_data(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Dataset",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a 2D binary classification dataset.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    plt.figure()
    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(x[mask, 0], x[mask, 1], label=f"class {cls}")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()

    if save_path is not None:
        save_current_figure(save_path)

    if show:
        plt.show()
    else:
        plt.close()