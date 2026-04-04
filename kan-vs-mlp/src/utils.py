"""Shared utility helpers for experiments, training, and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MLP_COLOR = "#2196F3"
KAN_COLOR = "#FF9800"
BSPLINE_COLOR = "#4CAF50"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed used for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the preferred compute device for the current machine.

    Returns:
        A CUDA device when available, otherwise a CPU device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def _apply_plot_style(ax: plt.Axes) -> None:
    """Apply the shared plotting style to an axis.

    Args:
        ax: Matplotlib axis to style.
    """
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _finalize_figure(fig: plt.Figure, save_path: str | None = None) -> None:
    """Save and close a figure.

    Args:
        fig: Matplotlib figure to finalize.
        save_path: Optional output path for saving the figure.
    """
    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_training_curves(
    history: dict,
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot training and validation curves for an experiment.

    Args:
        history: Dictionary containing ``train_loss``, ``val_loss``, and
            optionally ``val_metric`` as lists of per-epoch values.
        title: Title displayed above the figure.
        save_path: Optional path to save the figure as a PNG.

    Raises:
        KeyError: If required history keys are missing.
        ValueError: If the provided loss curves are empty.
    """
    required_keys = {"train_loss", "val_loss"}
    missing_keys = required_keys.difference(history)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing required history keys: {missing}")

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    if not train_loss or not val_loss:
        raise ValueError("train_loss and val_loss must contain at least one epoch.")

    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle(title, fontsize=14)

    loss_ax, metric_ax = axes
    _apply_plot_style(loss_ax)
    loss_ax.plot(epochs, train_loss, color=MLP_COLOR, linewidth=2, label="Train Loss")
    loss_ax.plot(epochs, val_loss, color=KAN_COLOR, linewidth=2, label="Val Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Curves")
    loss_ax.legend()

    _apply_plot_style(metric_ax)
    if "val_metric" in history and history["val_metric"]:
        metric_ax.plot(
            range(1, len(history["val_metric"]) + 1),
            history["val_metric"],
            color=BSPLINE_COLOR,
            linewidth=2,
            label="Validation Metric",
        )
        metric_ax.set_ylabel("Metric")
        metric_ax.legend()
    else:
        metric_ax.text(
            0.5,
            0.5,
            "No validation metric provided",
            ha="center",
            va="center",
            fontsize=11,
            transform=metric_ax.transAxes,
        )
        metric_ax.set_ylabel("Metric")

    metric_ax.set_xlabel("Epoch")
    metric_ax.set_title("Validation Metric")

    _finalize_figure(fig, save_path)


def plot_bar_comparison(
    labels: list[str],
    values: list[float],
    errors: list[float] | None = None,
    ylabel: str = "",
    title: str = "",
    colors: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a comparison bar chart with optional error bars.

    Args:
        labels: Tick labels for the bars.
        values: Values associated with each label.
        errors: Optional error bars for each bar.
        ylabel: Label for the y-axis.
        title: Chart title.
        colors: Optional list of bar colors.
        save_path: Optional path to save the figure as a PNG.

    Raises:
        ValueError: If labels and values have mismatched lengths.
    """
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length.")

    if errors is not None and len(errors) != len(values):
        raise ValueError("errors must have the same length as values.")

    bar_colors = colors if colors is not None else [MLP_COLOR] * len(values)
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    _apply_plot_style(ax)
    ax.bar(
        positions,
        values,
        yerr=errors,
        color=bar_colors,
        alpha=0.9,
        capsize=4 if errors is not None else 0,
        edgecolor="none",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _finalize_figure(fig, save_path)


def save_results_csv(results: dict, filepath: str) -> None:
    """Save experiment results to a CSV file.

    Args:
        results: Dictionary whose keys are column names and values are column data.
        filepath: Destination CSV path.
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def load_results_csv(filepath: str) -> pd.DataFrame:
    """Load experiment results from a CSV file.

    Args:
        filepath: Source CSV path.

    Returns:
        A pandas DataFrame containing the saved results.
    """
    return pd.read_csv(filepath)


def format_param_count(n: int) -> str:
    """Format a parameter count into a human-readable string.

    Args:
        n: Raw parameter count.

    Returns:
        A compact string such as ``15.2K`` or ``1.31M``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
