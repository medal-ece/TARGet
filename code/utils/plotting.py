# code/utils/plotting.py
"""Plotting helpers used during training and diagnostics."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import ALIAS  # same alias map used in metrics


def history_plot(history, metric_list, save_path, save_title, img_format='png'):
    """Plot selected training and validation metrics from a Keras history object."""
    def _canon(name):
        n = name.lower()
        return ALIAS.get(n, n)
    
    for m in metric_list:
        key = _canon(m)
        if key not in history.history:
            # also try validation name
            if f"val_{key}" not in history.history:
                print(f"Warn: key: '{m}' not found!")
                continue
        if key in history.history:
            plt.plot(history.history[key])
        if f"val_{key}" in history.history:
            plt.plot(history.history[f"val_{key}"])
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.title(f"{save_title}_{key}")
        plt.ylabel(key.capitalize())
        # y_lim = 0.3
        # plt.ylim([0, y_lim])
        plt.xlabel('Epoch')
        plt.grid(True)
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        plt.tight_layout()
        save_dir = os.path.join(save_path, "hist_plots")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"history_{save_title}_{m}.{img_format}"), dpi=200, bbox_inches="tight")
        plt.close()


def PCA_plot(y, path, file_name):
    """Plot cumulative explained variance from a PCA run."""
    save_path = os.path.join(path, f"{file_name}.png")
    # Plot explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(y), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs Number of PCA Components")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # optional, closes the figure so it doesn’t stay in memory


def scatter_plotter(x, y, save_path: str, name: str, color_values=None, cmap: str = "hsv"):
    """Save a Cartesian scatter plot, optionally colored by point density or counts."""
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color="gray", linewidth=0.1)
    plt.axvline(0, color="gray", linewidth=0.1)
    sc = plt.scatter(x, y, c=color_values, cmap=cmap, s=12, alpha=0.6, linewidths=0, edgecolors="none", rasterized=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{name} Scatter Plot")
    plt.axis("equal")
    plt.grid(True)

    if color_values is not None:
        plt.colorbar(sc, label="Repetitions per point")

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{name}_scatterplot.png"))
    plt.close()
