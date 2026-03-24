"""
qml.visualize
=============

Plotting utilities for qml experiments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path | None) -> Path | None:
    """
    Prepare an output path for saving a figure.

    Parameters
    ----------
    path
        Target file path.

    Returns
    -------
    Path | None
        Normalized path, or ``None`` if no path was provided.
    """
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_figure(
    *,
    show: bool,
    save_path: str | Path | None,
    dpi: int = 200,
) -> None:
    """
    Save and/or show the current Matplotlib figure.
    """
    save_path = _prepare_path(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_curve(
    loss_history,
    *,
    title: str = "Training loss",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot training loss versus optimization step.

    Parameters
    ----------
    loss_history
        Sequence of loss values.
    title
        Plot title.
    show
        Whether to display the figure.
    save_path
        Optional figure output path.
    """
    loss_history = np.asarray(loss_history, dtype=float)

    plt.figure()
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)

    _finalize_figure(show=show, save_path=save_path)


def plot_dataset_2d(
    x,
    y,
    *,
    title: str = "Dataset",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a 2D labeled dataset.

    Parameters
    ----------
    x
        Feature matrix of shape ``(n_samples, 2)``.
    y
        Label vector.
    title
        Plot title.
    show
        Whether to display the figure.
    save_path
        Optional figure output path.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape (n_samples, 2), got {x.shape}.")

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
    plt.title(title)
    plt.legend()

    _finalize_figure(show=show, save_path=save_path)


def plot_decision_boundary(
    predict_proba_fn,
    x,
    y,
    *,
    grid_points: int = 80,
    title: str = "Decision boundary",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a 2D decision surface together with labeled data.

    Parameters
    ----------
    predict_proba_fn
        Callable mapping an array of shape ``(n_grid, 2)`` to class-1 probabilities.
    x
        Feature matrix of shape ``(n_samples, 2)``.
    y
        Label vector.
    grid_points
        Number of grid points per axis.
    title
        Plot title.
    show
        Whether to display the figure.
    save_path
        Optional figure output path.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape (n_samples, 2), got {x.shape}.")

    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    pad = 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min - pad, x_max + pad, grid_points),
        np.linspace(y_min - pad, y_max + pad, grid_points),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = np.asarray(predict_proba_fn(grid), dtype=float).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, probs, levels=20, alpha=0.5)

    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(
            x[mask, 0],
            x[mask, 1],
            label=f"class {cls}",
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()

    _finalize_figure(show=show, save_path=save_path)


def plot_kernel_matrix(
    kernel_matrix,
    *,
    title: str = "Kernel matrix",
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a kernel matrix as a heatmap.

    Parameters
    ----------
    kernel_matrix
        Array of shape ``(n_rows, n_cols)``.
    title
        Plot title.
    show
        Whether to display the figure.
    save_path
        Optional figure output path.
    """
    kernel_matrix = np.asarray(kernel_matrix, dtype=float)

    if kernel_matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D kernel matrix, got array with shape {kernel_matrix.shape}."
        )

    plt.figure()
    plt.imshow(kernel_matrix, aspect="auto", origin="lower")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.title(title)
    plt.colorbar(label="Kernel value")

    _finalize_figure(show=show, save_path=save_path)