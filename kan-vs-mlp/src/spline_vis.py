"""
Spline Visualization Utilities for KAN Models

Provides tools for visualizing and analyzing the learned activation
functions in trained KAN (Kolmogorov-Arnold Network) models. These
visualizations are a key interpretability advantage of KANs over MLPs.

Two approaches are implemented:
1. Functional probe: evaluate the KAN layer on swept inputs while holding
   other features at a reference value, capturing the effective univariate
   response of each input feature.
2. Basis reconstruction: directly reconstruct the B-spline curves from
   the learned coefficients using scipy.interpolate.BSpline.

Both approaches should produce very similar curves for the first layer
of a KAN. The functional probe is more general (works for any layer),
while basis reconstruction is more precise (no interference from other
features at the reference value).
"""

from __future__ import annotations

import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.stats import spearmanr

from src.kan_layer import KANLinear

KAN_COLOR = "#FF9800"
MLP_COLOR = "#2196F3"
GREY_COLOR = "#757575"


def probe_kan_layer_responses(
    layer: KANLinear,
    input_range: tuple[float, float] = (-3.0, 3.0),
    num_points: int = 200,
    reference_value: float = 0.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Probe the effective response of each input feature in a KAN layer.

    Each feature is swept across a 1D input range while the remaining features
    are fixed at a reference value. The layer output is recorded for every
    sweep point and aggregated across output neurons.

    Args:
        layer: Trained ``KANLinear`` layer to probe.
        input_range: Inclusive lower and upper sweep bounds.
        num_points: Number of points in the sweep grid.
        reference_value: Value used for non-swept features.
        device: Execution device, typically ``"cpu"`` or ``"cuda"``.

    Returns:
        Dictionary containing the sweep inputs, aggregated responses, and
        full per-output responses.
    """
    if num_points <= 1:
        raise ValueError("num_points must be greater than 1.")

    layer_device = next(layer.parameters()).device
    requested_device = torch.device(device)
    was_training = layer.training

    layer = layer.to(requested_device)
    layer.eval()

    input_values = torch.linspace(
        input_range[0],
        input_range[1],
        steps=num_points,
        device=requested_device,
        dtype=torch.float32,
    )

    aggregated_responses: list[np.ndarray] = []
    full_responses: list[np.ndarray] = []

    with torch.no_grad():
        for feature_idx in range(layer.in_features):
            probe_inputs = torch.full(
                (num_points, layer.in_features),
                fill_value=reference_value,
                device=requested_device,
                dtype=torch.float32,
            )
            probe_inputs[:, feature_idx] = input_values

            outputs = layer(probe_inputs)
            aggregated = outputs.mean(dim=1)

            aggregated_responses.append(aggregated.detach().cpu().numpy())
            full_responses.append(outputs.detach().cpu().numpy())

    layer = layer.to(layer_device)
    layer.train(was_training)

    return {
        "input_values": input_values.detach().cpu().numpy(),
        "aggregated_responses": np.stack(aggregated_responses, axis=0),
        "full_responses": np.stack(full_responses, axis=0),
        "in_features": layer.in_features,
        "out_features": layer.out_features,
    }


def reconstruct_bsplines(layer: KANLinear) -> dict[str, Any]:
    """Reconstruct the learned B-spline functions from layer coefficients.

    Args:
        layer: Trained ``KANLinear`` layer.

    Returns:
        Dictionary containing reconstructed spline objects and sampled curves.
    """
    grid = layer.grid.detach().cpu().numpy()
    if hasattr(layer, "scaled_spline_weight"):
        coefficients_tensor = layer.scaled_spline_weight.detach().cpu()
    else:
        coefficients_tensor = layer.spline_weight.detach().cpu()
    coefficients = coefficients_tensor.numpy()

    spline_functions: list[list[dict[str, Any] | None]] = []
    spline_order: int | None = None

    for input_idx in range(layer.in_features):
        feature_splines: list[dict[str, Any] | None] = []
        knots = grid[input_idx].copy()

        if not np.all(np.diff(knots) >= 0):
            print(f"Warning: grid for feature {input_idx} is not sorted; sorting knots.")
            knots = np.sort(knots)

        for output_idx in range(layer.out_features):
            coeffs = coefficients[output_idx, input_idx].copy()
            n_knots = len(knots)
            n_coeffs = len(coeffs)
            degree = n_knots - n_coeffs - 1

            if spline_order is None:
                spline_order = degree

            if n_knots != n_coeffs + degree + 1:
                print(
                    "Warning: unexpected spline dimensions for "
                    f"feature {input_idx}, output {output_idx}: "
                    f"n_knots={n_knots}, n_coeffs={n_coeffs}, degree={degree}"
                )

            try:
                spline = BSpline(knots, coeffs, degree)
                x_values = np.linspace(knots[degree], knots[-degree - 1], 200)
                y_values = spline(x_values)
                feature_splines.append(
                    {
                        "bspline": spline,
                        "x_values": x_values,
                        "y_values": y_values,
                    }
                )
            except Exception as exc:
                print(
                    f"Warning: failed to reconstruct spline for feature {input_idx}, "
                    f"output {output_idx}: {exc}"
                )
                feature_splines.append(None)

        spline_functions.append(feature_splines)

    return {
        "spline_functions": spline_functions,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "spline_order": spline_order if spline_order is not None else 0,
    }


def compute_feature_nonlinearity(responses: dict[str, Any]) -> np.ndarray:
    """Compute a nonlinearity score for each probed input feature.

    Args:
        responses: Dictionary returned by ``probe_kan_layer_responses``.

    Returns:
        Array of nonlinearity scores with shape ``(in_features,)``.
    """
    input_values = np.asarray(responses["input_values"], dtype=np.float64)
    aggregated_responses = np.asarray(responses["aggregated_responses"], dtype=np.float64)

    scores = np.zeros(aggregated_responses.shape[0], dtype=np.float64)
    for feature_idx, curve in enumerate(aggregated_responses):
        linear_coeffs = np.polyfit(input_values, curve, deg=1)
        linear_fit = np.polyval(linear_coeffs, input_values)
        residual = curve - linear_fit
        scores[feature_idx] = float(np.std(residual))

    return scores


def plot_kan_splines(
    responses: dict[str, Any],
    feature_names: list[str] | None = None,
    title: str = "Learned KAN Activation Functions",
    save_path: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot probed KAN response curves for each input feature.

    Args:
        responses: Output dictionary from ``probe_kan_layer_responses``.
        feature_names: Optional human-readable feature names.
        title: Overall figure title.
        save_path: Optional path for saving the figure.
        figsize: Optional figure size override.
    """
    input_values = np.asarray(responses["input_values"])
    aggregated_responses = np.asarray(responses["aggregated_responses"])
    n_features = aggregated_responses.shape[0]

    if feature_names is None:
        feature_names = [f"Feature {idx}" for idx in range(n_features)]

    scores = compute_feature_nonlinearity(responses)
    n_cols = min(4, n_features)
    n_rows = math.ceil(n_features / n_cols)
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for feature_idx in range(n_features):
        ax = flat_axes[feature_idx]
        curve = aggregated_responses[feature_idx]
        linear_coeffs = np.polyfit(input_values, curve, deg=1)
        linear_fit = np.polyval(linear_coeffs, input_values)

        ax.plot(input_values, curve, color=KAN_COLOR, linewidth=2)
        ax.plot(input_values, linear_fit, color=GREY_COLOR, linestyle="--", linewidth=1.5)
        ax.set_title(feature_names[feature_idx])
        ax.set_xlabel("Input value")
        ax.set_ylabel("Activation output")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03,
            0.95,
            f"NL: {scores[feature_idx]:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for ax in flat_axes[n_features:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_feature_importance_comparison(
    kan_nonlinearity: np.ndarray,
    shap_importance: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Importance: KAN Nonlinearity vs SHAP",
    save_path: str | None = None,
) -> None:
    """Plot side-by-side feature-importance rankings for KAN and SHAP.

    Args:
        kan_nonlinearity: KAN nonlinearity scores.
        shap_importance: Mean absolute SHAP values.
        feature_names: Human-readable feature names.
        title: Overall figure title.
        save_path: Optional path for saving the figure.
    """
    kan_nonlinearity = np.asarray(kan_nonlinearity, dtype=np.float64)
    shap_importance = np.asarray(shap_importance, dtype=np.float64)

    kan_order = np.argsort(kan_nonlinearity)[::-1]
    shap_order = np.argsort(shap_importance)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(
        np.arange(len(feature_names)),
        kan_nonlinearity[kan_order],
        color=KAN_COLOR,
        alpha=0.9,
    )
    axes[0].set_yticks(np.arange(len(feature_names)))
    axes[0].set_yticklabels([feature_names[idx] for idx in kan_order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Nonlinearity Score (std of residual from linear fit)")
    axes[0].set_ylabel("Feature")
    axes[0].set_title("KAN Nonlinearity")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(
        np.arange(len(feature_names)),
        shap_importance[shap_order],
        color=MLP_COLOR,
        alpha=0.9,
    )
    axes[1].set_yticks(np.arange(len(feature_names)))
    axes[1].set_yticklabels([feature_names[idx] for idx in shap_order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean |SHAP Value|")
    axes[1].set_ylabel("Feature")
    axes[1].set_title("SHAP Importance")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_spline_vs_shap_scatter(
    kan_nonlinearity: np.ndarray,
    shap_importance: np.ndarray,
    feature_names: list[str],
    title: str = "KAN Nonlinearity vs SHAP Importance",
    save_path: str | None = None,
) -> None:
    """Plot the relationship between KAN nonlinearity and SHAP importance.

    Args:
        kan_nonlinearity: KAN nonlinearity scores.
        shap_importance: Mean absolute SHAP values.
        feature_names: Human-readable feature names.
        title: Plot title.
        save_path: Optional path for saving the figure.
    """
    kan_nonlinearity = np.asarray(kan_nonlinearity, dtype=np.float64)
    shap_importance = np.asarray(shap_importance, dtype=np.float64)

    rho, p_value = spearmanr(kan_nonlinearity, shap_importance)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(kan_nonlinearity, shap_importance, color=KAN_COLOR, s=70, alpha=0.9)

    for idx, feature_name in enumerate(feature_names):
        ax.annotate(
            feature_name,
            (kan_nonlinearity[idx], shap_importance[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    if len(kan_nonlinearity) >= 2:
        coeffs = np.polyfit(kan_nonlinearity, shap_importance, deg=1)
        x_line = np.linspace(kan_nonlinearity.min(), kan_nonlinearity.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, linestyle="--", color=GREY_COLOR, linewidth=1.5)

    ax.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:.3f} (p = {p_value:.3f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set_xlabel("KAN Nonlinearity Score")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _run_smoke_test() -> None:
    """Run smoke tests for the spline visualization utilities."""
    torch.manual_seed(42)
    np.random.seed(42)

    layer = KANLinear(8, 4, grid_size=5, spline_order=3)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

    for _ in range(50):
        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        loss = ((layer(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    responses = probe_kan_layer_responses(layer)
    assert responses["input_values"].shape == (200,)
    assert responses["aggregated_responses"].shape == (8, 200)
    assert responses["full_responses"].shape == (8, 200, 4)
    print("Probe responses computed successfully")

    reconstruction = reconstruct_bsplines(layer)
    assert len(reconstruction["spline_functions"]) == 8
    assert len(reconstruction["spline_functions"][0]) == 4

    first_valid = None
    for feature_group in reconstruction["spline_functions"]:
        for spline_info in feature_group:
            if spline_info is not None:
                first_valid = spline_info
                break
        if first_valid is not None:
            break

    if first_valid is not None:
        print(
            "First spline y-range:",
            float(np.min(first_valid["y_values"])),
            float(np.max(first_valid["y_values"])),
        )
    print("BSpline reconstruction completed successfully")

    scores = compute_feature_nonlinearity(responses)
    assert scores.shape == (8,)
    assert np.all(scores >= 0.0)
    feature_names = [f"F{idx}" for idx in range(8)]
    print("Nonlinearity scores:")
    for name, score in zip(feature_names, scores, strict=True):
        print(f"  {name}: {score:.4f}")
    print("Nonlinearity computation passed")

    spline_plot_path = os.path.join("results", "exp3", "test_splines.png")
    plot_kan_splines(
        responses,
        feature_names=feature_names,
        save_path=spline_plot_path,
    )
    assert os.path.exists(spline_plot_path)
    print("Spline plot saved successfully")

    shap_importance = np.random.rand(8) * 0.5
    comparison_plot_path = os.path.join("results", "exp3", "test_comparison.png")
    plot_feature_importance_comparison(
        kan_nonlinearity=scores,
        shap_importance=shap_importance,
        feature_names=feature_names,
        save_path=comparison_plot_path,
    )
    assert os.path.exists(comparison_plot_path)
    print("Comparison plot saved successfully")

    scatter_plot_path = os.path.join("results", "exp3", "test_scatter.png")
    plot_spline_vs_shap_scatter(
        kan_nonlinearity=scores,
        shap_importance=shap_importance,
        feature_names=feature_names,
        save_path=scatter_plot_path,
    )
    assert os.path.exists(scatter_plot_path)
    print("Scatter plot saved successfully")

    print("All spline_vis.py smoke tests passed!")


if __name__ == "__main__":
    _run_smoke_test()
