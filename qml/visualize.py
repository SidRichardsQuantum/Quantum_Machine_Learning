"""
qml.visualize
=============

Plotting utilities for qml experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(
    loss_history,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    loss_history = np.asarray(loss_history, dtype=float)

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")

    save_path = _prepare_path(save_path)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_2d(
    x,
    y,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    plt.figure()

    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(
            x[mask, 0],
            x[mask, 1],
            label=f"class {cls}",
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Dataset")
    plt.legend()

    save_path = _prepare_path(save_path)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_decision_boundary(
    predict_proba_fn,
    x,
    y,
    *,
    grid_points: int = 80,
    show: bool = True,
    save_path: str | Path | None = None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    pad = 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min - pad, x_max + pad, grid_points),
        np.linspace(y_min - pad, y_max + pad, grid_points),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = predict_proba_fn(grid)
    probs = np.asarray(probs).reshape(xx.shape)

    plt.figure()

    plt.contourf(
        xx,
        yy,
        probs,
        levels=20,
        alpha=0.5,
    )

    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(
            x[mask, 0],
            x[mask, 1],
            label=f"class {cls}",
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision boundary")
    plt.legend()

    save_path = _prepare_path(save_path)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()