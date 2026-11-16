"""Figures for the case study: confusion matrices and sample embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_confusion(cm: pd.DataFrame, path: str | Path, title: str = "",
                   normalize: bool = True) -> Path:
    """Confusion matrix; with ``normalize`` each row shows the fraction of the true class."""
    values = cm.to_numpy(dtype=float)
    shown = values / values.sum(axis=1, keepdims=True).clip(min=1) if normalize else values
    size = 1.2 + 0.7 * len(cm)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            colour = "white" if shown[i, j] > 0.6 * np.nanmax(shown) else "black"
            ax.text(j, i, f"{int(values[i, j])}", ha="center", va="center", color=colour,
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of true class" if normalize else "Samples")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_embedding(coords: np.ndarray, groups: pd.Series, path: str | Path,
                   title: str = "", xlabel: str = "UMAP 1", ylabel: str = "UMAP 2") -> Path:
    """Scatter plot of a 2-D embedding, coloured by group."""
    groups = groups.astype(str)
    names = sorted(groups.unique())
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    markers = "osD^vP*Xh<>"
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for k, name in enumerate(names):
        mask = (groups == name).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.85, label=name,
                   color=cmap(k % cmap.N), marker=markers[k % len(markers)],
                   edgecolors="white", linewidths=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
