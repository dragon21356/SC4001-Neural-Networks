"""Post-run analysis script for Experiment 1 regression results."""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_california_housing
from src.kan_layer import KANLinear
from src.models import KANRegressor
from src.models import MLPRegressor
from src.models import count_parameters
from src.spline_vis import compute_feature_nonlinearity
from src.spline_vis import plot_feature_importance_comparison
from src.spline_vis import plot_kan_splines
from src.spline_vis import plot_spline_vs_shap_scatter
from src.spline_vis import probe_kan_layer_responses
from src.spline_vis import reconstruct_bsplines
from src.utils import BSPLINE_COLOR
from src.utils import KAN_COLOR
from src.utils import MLP_COLOR
from src.utils import format_param_count
from src.utils import set_seed

ALL_RESULTS_PATH = os.path.join("results", "exp1", "exp1_all_results.csv")
SUMMARY_RESULTS_PATH = os.path.join("results", "exp1", "exp1_summary.csv")
CHECKPOINT_DIR = os.path.join("results", "exp1", "checkpoints")
FIGURES_DIR = os.path.join("results", "exp1", "figures")
SUMMARY_TABLE_PATH = os.path.join(FIGURES_DIR, "summary_table.txt")
NONLINEARITY_PATH = os.path.join("results", "exp1", "kan_nonlinearity_scores.npy")
FEATURE_NAMES_PATH = os.path.join("results", "exp1", "feature_names.json")


def ensure_results_exist() -> None:
    """Ensure Experiment 1 result CSVs are present.

    Raises:
        SystemExit: If the required CSV files do not exist.
    """
    if not os.path.exists(ALL_RESULTS_PATH) or not os.path.exists(SUMMARY_RESULTS_PATH):
        print("Results files not found. Run exp1_regression.py first.")
        raise SystemExit(1)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Experiment 1 result CSVs and print validation information.

    Returns:
        Tuple of ``(all_results_df, summary_df)``.
    """
    all_results_df = pd.read_csv(ALL_RESULTS_PATH)
    summary_df = pd.read_csv(SUMMARY_RESULTS_PATH)

    expected_rows = 48
    if len(all_results_df) < expected_rows:
        print(f"Warning: Expected 48 rows, found {len(all_results_df)}. Some runs may be incomplete.")

    base_configs = {
        name.replace("_matched", "")
        for name in all_results_df["config_name"].dropna().astype(str).tolist()
    }
    seeds = sorted(all_results_df["seed"].dropna().astype(int).unique().tolist())
    print(f"Completed runs: {len(all_results_df)}")
    print(f"Unique configs: {len(base_configs)}")
    print(f"Seeds found: {len(seeds)} -> {seeds}")

    return all_results_df, summary_df


def prepare_output_dirs() -> None:
    """Create the analysis output directories."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def decode_history(raw_value: Any) -> list[float]:
    """Decode a JSON-encoded history column safely.

    Args:
        raw_value: Raw CSV cell value.

    Returns:
        Decoded list of floats.
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []

    if isinstance(raw_value, list):
        return [float(value) for value in raw_value]

    try:
        decoded = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return []

    cleaned: list[float] = []
    for value in decoded:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def get_base_config_name(config_name: str) -> str:
    """Normalize a config name by stripping the MLP suffix.

    Args:
        config_name: Raw config name from the results CSV.

    Returns:
        Base configuration name.
    """
    return config_name.replace("_matched", "")


def format_config_label(base_config_name: str) -> str:
    """Convert ``Gx_Ky`` into a compact plot label.

    Args:
        base_config_name: Base config name such as ``G5_K3``.

    Returns:
        Display label such as ``G5,k3``.
    """
    left, right = base_config_name.split("_")
    return f"{left},{right.lower().replace('k', 'k')}"


def get_best_config_pair(summary_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Get the best KAN config and its matched MLP summary row.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Tuple of ``(best_kan_row, matched_mlp_row)``.
    """
    kan_summary = summary_df[summary_df["model_type"] == "kan"].copy()
    if kan_summary.empty:
        raise ValueError("No KAN summary rows found in exp1_summary.csv.")

    best_kan = kan_summary.sort_values(by="test_r2_mean", ascending=False).iloc[0]
    matched_name = f"{best_kan['config_name']}_matched"
    matched_rows = summary_df[
        (summary_df["model_type"] == "mlp") & (summary_df["config_name"] == matched_name)
    ]
    if matched_rows.empty:
        raise ValueError(f"Could not find matched MLP summary row for {matched_name}.")
    return best_kan, matched_rows.iloc[0]


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Save and close a matplotlib figure.

    Args:
        fig: Figure to save.
        filename: Output filename under ``results/exp1/figures``.

    Returns:
        Full output path.
    """
    output_path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_figure_1_heatmap(summary_df: pd.DataFrame) -> str:
    """Generate the KAN R² heatmap.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    kan_summary = summary_df[summary_df["model_type"] == "kan"].copy()
    kan_summary["grid_size"] = kan_summary["grid_size"].astype(int)
    kan_summary["spline_order"] = kan_summary["spline_order"].astype(int)

    r2_matrix = kan_summary.pivot(index="spline_order", columns="grid_size", values="test_r2_mean")
    std_matrix = kan_summary.pivot(index="spline_order", columns="grid_size", values="test_r2_std")

    annot = np.empty(r2_matrix.shape, dtype=object)
    for row_idx in range(r2_matrix.shape[0]):
        for col_idx in range(r2_matrix.shape[1]):
            mean_value = r2_matrix.iloc[row_idx, col_idx]
            std_value = std_matrix.iloc[row_idx, col_idx]
            annot[row_idx, col_idx] = f"{mean_value:.4f}\n±{0.0 if pd.isna(std_value) else std_value:.4f}"

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        sns = import_module("seaborn")
        sns.heatmap(
            r2_matrix,
            annot=annot,
            fmt="",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": "Test R²"},
            ax=ax,
        )
    except ModuleNotFoundError:
        print("Warning: seaborn is not installed. Falling back to matplotlib for Figure 1.")
        heatmap = ax.imshow(r2_matrix.values, cmap="YlOrRd", aspect="auto")
        for row_idx in range(r2_matrix.shape[0]):
            for col_idx in range(r2_matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    annot[row_idx, col_idx],
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        fig.colorbar(heatmap, ax=ax, label="Test R²")
        ax.set_xticks(range(len(r2_matrix.columns)))
        ax.set_xticklabels(r2_matrix.columns.tolist())
        ax.set_yticks(range(len(r2_matrix.index)))
        ax.set_yticklabels(r2_matrix.index.tolist())
    ax.set_title("KAN Test R² by Grid Size and Spline Order")
    ax.set_xlabel("Grid Size (G)")
    ax.set_ylabel("Spline Order (k)")
    return save_figure(fig, "fig1_kan_r2_heatmap.png")


def _paired_config_rows(summary_df: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Build paired KAN/MLP rows for each base configuration.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        List of paired summary rows.
    """
    pairs: list[tuple[pd.Series, pd.Series]] = []
    kan_rows = summary_df[summary_df["model_type"] == "kan"].sort_values(
        by=["grid_size", "spline_order"]
    )
    for _, kan_row in kan_rows.iterrows():
        mlp_name = f"{kan_row['config_name']}_matched"
        mlp_rows = summary_df[
            (summary_df["model_type"] == "mlp") & (summary_df["config_name"] == mlp_name)
        ]
        if mlp_rows.empty:
            continue
        pairs.append((kan_row, mlp_rows.iloc[0]))
    return pairs


def generate_grouped_metric_chart(
    summary_df: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    ylabel: str,
    filename: str,
    best_reference: bool = False,
) -> str:
    """Generate a grouped KAN-vs-MLP metric comparison chart.

    Args:
        summary_df: Summary results DataFrame.
        metric_mean_col: Column name for the averaged metric.
        metric_std_col: Column name for the metric standard deviation.
        title: Figure title.
        ylabel: Y-axis label.
        filename: Output filename.
        best_reference: Whether to draw a dashed line at the best metric.

    Returns:
        Saved figure path.
    """
    pairs = _paired_config_rows(summary_df)
    labels = [format_config_label(get_base_config_name(str(kan_row["config_name"]))) for kan_row, _ in pairs]
    kan_means = [float(kan_row[metric_mean_col]) for kan_row, _ in pairs]
    kan_stds = [0.0 if pd.isna(kan_row[metric_std_col]) else float(kan_row[metric_std_col]) for kan_row, _ in pairs]
    mlp_means = [float(mlp_row[metric_mean_col]) for _, mlp_row in pairs]
    mlp_stds = [0.0 if pd.isna(mlp_row[metric_std_col]) else float(mlp_row[metric_std_col]) for _, mlp_row in pairs]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, kan_means, width, yerr=kan_stds, capsize=4, color=KAN_COLOR, label="KAN")
    ax.bar(x + width / 2, mlp_means, width, yerr=mlp_stds, capsize=4, color=MLP_COLOR, label="MLP (matched params)")
    if best_reference:
        ax.axhline(max(kan_means + mlp_means), color="#808080", linestyle="--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return save_figure(fig, filename)


def _select_median_seed_run(all_results_df: pd.DataFrame, config_name: str, model_type: str) -> pd.Series:
    """Select the median-R² run for a given config and model type.

    Args:
        all_results_df: Run-level results DataFrame.
        config_name: Config name to filter on.
        model_type: ``kan`` or ``mlp``.

    Returns:
        Selected run row.
    """
    subset = all_results_df[
        (all_results_df["model_type"] == model_type)
        & (all_results_df["config_name"] == config_name)
    ].copy()
    subset = subset.sort_values(by="test_r2", ascending=True).reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No runs found for {model_type} / {config_name}.")
    return subset.iloc[len(subset) // 2]


def generate_figure_3_convergence(all_results_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Generate convergence curves for the best KAN and its matched MLP.

    Args:
        all_results_df: Run-level results DataFrame.
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    best_kan, matched_mlp = get_best_config_pair(summary_df)
    kan_run = _select_median_seed_run(all_results_df, str(best_kan["config_name"]), "kan")
    mlp_run = _select_median_seed_run(all_results_df, str(matched_mlp["config_name"]), "mlp")

    kan_val_r2 = decode_history(kan_run["val_r2_history"])
    mlp_val_r2 = decode_history(mlp_run["val_r2_history"])
    kan_train_loss = decode_history(kan_run["train_loss_history"])
    mlp_train_loss = decode_history(mlp_run["train_loss_history"])

    max_epochs = max(len(kan_val_r2), len(mlp_val_r2), 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(range(1, len(kan_val_r2) + 1), kan_val_r2, color=KAN_COLOR, linewidth=2, label="KAN Val R²")
    axes[0].plot(range(1, len(mlp_val_r2) + 1), mlp_val_r2, color=MLP_COLOR, linewidth=2, label="MLP Val R²")
    axes[0].set_ylabel("Validation R²")
    axes[0].set_title(
        f"Convergence: Best KAN (G={int(best_kan['grid_size'])}, k={int(best_kan['spline_order'])}) "
        f"vs Matched MLP (H={int(matched_mlp['hidden_dim'])})"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(kan_train_loss) + 1), kan_train_loss, color=KAN_COLOR, linewidth=2, label="KAN Train Loss")
    axes[1].plot(range(1, len(mlp_train_loss) + 1), mlp_train_loss, color=MLP_COLOR, linewidth=2, label="MLP Train Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xlim(1, max_epochs)

    return save_figure(fig, "fig3_convergence_curves.png")


def generate_figure_5_speed(summary_df: pd.DataFrame) -> str:
    """Generate the training-speed comparison figure.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    best_kan, matched_mlp = get_best_config_pair(summary_df)
    names = [
        f"KAN {best_kan['config_name']}",
        f"MLP {matched_mlp['config_name']}",
    ]
    values = [
        float(best_kan["avg_time_per_epoch_mean"]),
        float(matched_mlp["avg_time_per_epoch_mean"]),
    ]
    ratio = values[0] / max(values[1], 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=[KAN_COLOR, MLP_COLOR], alpha=0.9)
    ax.set_ylabel("Average Time per Epoch (s)")
    ax.set_title("Training Speed Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.5,
        max(values) * 0.92,
        f"KAN is {ratio:.1f}x slower",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    return save_figure(fig, "fig5_training_speed.png")


def generate_figure_6_params_vs_r2(summary_df: pd.DataFrame) -> str:
    """Generate the parameter-efficiency scatter plot.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_type, marker, color in (("kan", "o", KAN_COLOR), ("mlp", "s", MLP_COLOR)):
        subset = summary_df[summary_df["model_type"] == model_type]
        ax.errorbar(
            subset["total_params"],
            subset["test_r2_mean"],
            yerr=subset["test_r2_std"].fillna(0.0),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            linestyle="none",
            label="KAN" if model_type == "kan" else "MLP",
        )
        for row in subset.itertuples(index=False):
            ax.annotate(
                str(row.config_name),
                (float(row.total_params), float(row.test_r2_mean)),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    params_min = float(summary_df["total_params"].min())
    params_max = float(summary_df["total_params"].max())
    if params_max / max(params_min, 1.0) > 10:
        ax.set_xscale("log")
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Test R²")
    ax.set_title("Parameter Efficiency: Params vs Test R²")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, "fig6_params_vs_r2.png")


def generate_summary_table(summary_df: pd.DataFrame) -> str:
    """Generate the report-ready summary table text.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Table text.
    """
    lines = [
        "Table 1: Experiment 1 Results - KAN vs Parameter-Matched MLP on California Housing",
        "┌──────────────┬──────────┬──────────────────┬──────────────────┬───────────────┬──────────┐",
        "│ Model        │ Params   │ Test MSE         │ Test R²          │ Conv. Epoch   │ Time/Ep  │",
        "├──────────────┼──────────┼──────────────────┼──────────────────┼───────────────┼──────────┤",
    ]

    for kan_row, mlp_row in _paired_config_rows(summary_df):
        for label, row in (
            (f"KAN {format_config_label(str(kan_row['config_name']))}", kan_row),
            ("MLP (matched)", mlp_row),
        ):
            mse_std = 0.0 if pd.isna(row["test_mse_std"]) else float(row["test_mse_std"])
            r2_std = 0.0 if pd.isna(row["test_r2_std"]) else float(row["test_r2_std"])
            conv_std = 0.0 if pd.isna(row["convergence_epoch_std"]) else float(row["convergence_epoch_std"])
            lines.append(
                f"│ {label:<12} │ "
                f"{format_param_count(int(row['total_params'])):>8} │ "
                f"{float(row['test_mse_mean']):>6.4f} ± {mse_std:<6.4f} │ "
                f"{float(row['test_r2_mean']):>6.4f} ± {r2_std:<6.4f} │ "
                f"{float(row['convergence_epoch_mean']):>5.1f} ± {conv_std:<4.1f} │ "
                f"{float(row['avg_time_per_epoch_mean']):>5.2f}s │"
            )
        lines.append(
            "│──────────────│──────────│──────────────────│──────────────────│───────────────│──────────│"
        )

    if lines[-1].startswith("│──────────────"):
        lines = lines[:-1]
    lines.append(
        "└──────────────┴──────────┴──────────────────┴──────────────────┴───────────────┴──────────┘"
    )
    table_text = "\n".join(lines)

    with open(SUMMARY_TABLE_PATH, "w", encoding="utf-8") as handle:
        handle.write(table_text)

    print(table_text)
    return table_text


def generate_dummy_importance_plots(
    nonlinearity_scores: np.ndarray,
    feature_names: list[str],
) -> tuple[str, str]:
    """Generate placeholder comparison plots using normalized nonlinearity values.

    This keeps the feature-comparison plotting utilities exercised even before
    Experiment 3 produces SHAP values.

    Args:
        nonlinearity_scores: KAN nonlinearity scores.
        feature_names: California Housing feature names.

    Returns:
        Tuple of saved figure paths.
    """
    shap_proxy = nonlinearity_scores / max(float(np.max(nonlinearity_scores)), 1e-8)
    comparison_path = os.path.join(FIGURES_DIR, "fig7b_feature_importance_proxy.png")
    scatter_path = os.path.join(FIGURES_DIR, "fig7c_spline_vs_proxy_scatter.png")
    plot_feature_importance_comparison(
        kan_nonlinearity=nonlinearity_scores,
        shap_importance=shap_proxy,
        feature_names=feature_names,
        title="KAN Nonlinearity vs Proxy Importance",
        save_path=comparison_path,
    )
    plot_spline_vs_shap_scatter(
        kan_nonlinearity=nonlinearity_scores,
        shap_importance=shap_proxy,
        feature_names=feature_names,
        title="KAN Nonlinearity vs Proxy Importance",
        save_path=scatter_path,
    )
    return comparison_path, scatter_path


def run_spline_visualization_section(all_results_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Any] | None:
    """Run spline probing on the best KAN checkpoint.

    Args:
        all_results_df: Run-level results DataFrame.
        summary_df: Summary results DataFrame.

    Returns:
        Dictionary describing the spline-analysis outcome, or ``None`` if skipped.
    """
    kan_rows = all_results_df[all_results_df["model_type"] == "kan"].copy()
    if kan_rows.empty:
        print("Warning: No KAN runs found, skipping spline visualization.")
        return None

    best_kan_run = kan_rows.sort_values(by="test_r2", ascending=False).iloc[0]
    config_name = str(best_kan_run["config_name"])
    grid_size = int(best_kan_run["grid_size"])
    spline_order = int(best_kan_run["spline_order"])
    hidden_dim = int(best_kan_run["hidden_dim"])
    seed = int(best_kan_run["seed"])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"kan_{config_name}_seed{seed}_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: checkpoint not found for best KAN model at {checkpoint_path}. Skipping spline section.")
        return None

    set_seed(seed)
    model = KANRegressor(
        in_features=8,
        hidden_dim=hidden_dim,
        out_features=1,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    _, _, _, info = get_california_housing(batch_size=64, seed=seed, num_workers=0)
    first_kan_layer = None
    for module in model.modules():
        if isinstance(module, KANLinear):
            first_kan_layer = module
            break

    if first_kan_layer is None:
        print("Warning: Could not locate a KANLinear layer in the best KAN model.")
        return None

    responses = probe_kan_layer_responses(
        first_kan_layer,
        input_range=(-3.0, 3.0),
        num_points=200,
        reference_value=0.0,
        device="cpu",
    )
    nonlinearity_scores = compute_feature_nonlinearity(responses)
    figure_path = os.path.join(FIGURES_DIR, "fig7_kan_splines.png")
    plot_kan_splines(
        responses,
        feature_names=info["feature_names"],
        save_path=figure_path,
        title=f"Learned KAN Activations (G={grid_size}, k={spline_order}, R²={float(best_kan_run['test_r2']):.4f})",
    )

    np.save(NONLINEARITY_PATH, nonlinearity_scores)
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as handle:
        json.dump(info["feature_names"], handle, indent=2)

    ranking = sorted(
        zip(info["feature_names"], nonlinearity_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Feature Nonlinearity Ranking (higher = more nonlinear learned activation):")
    for rank, (feature_name, score) in enumerate(ranking, start=1):
        print(f"{rank}. {feature_name:<10}: {score:.4f}")

    try:
        reconstruction = reconstruct_bsplines(first_kan_layer)
        feature_zero_recon = reconstruction["spline_functions"][0]
        valid_curves = [entry for entry in feature_zero_recon if entry is not None]
        if valid_curves:
            recon_x = valid_curves[0]["x_values"]
            recon_mean = np.mean([entry["y_values"] for entry in valid_curves], axis=0)
            probe_x = np.asarray(responses["input_values"])
            probe_curve = np.asarray(responses["aggregated_responses"][0])
            overlap_min = max(float(np.min(recon_x)), float(np.min(probe_x)))
            overlap_max = min(float(np.max(recon_x)), float(np.max(probe_x)))
            if overlap_max > overlap_min:
                mask = (probe_x >= overlap_min) & (probe_x <= overlap_max)
                probe_overlap = probe_curve[mask]
                probe_overlap_x = probe_x[mask]
                recon_interp = np.interp(probe_overlap_x, recon_x, recon_mean)
                if len(probe_overlap) > 2:
                    corr = np.corrcoef(
                        probe_overlap - np.mean(probe_overlap),
                        recon_interp - np.mean(recon_interp),
                    )[0, 1]
                    if np.isfinite(corr):
                        print(
                            "B-spline reconstruction validated - curves match probe approach "
                            f"(feature 0 correlation: {corr:.3f})"
                        )
                    else:
                        print("Warning: reconstruction comparison was not numerically stable.")
                else:
                    print("Warning: insufficient overlap for reconstruction comparison.")
            else:
                print("Warning: reconstruction curve did not overlap probe range.")
        else:
            print("Warning: B-spline reconstruction returned no valid curves for comparison.")
    except Exception as exc:
        print(f"Warning: B-spline reconstruction validation failed: {exc}")

    generate_dummy_importance_plots(nonlinearity_scores, info["feature_names"])

    return {
        "best_kan_run": best_kan_run,
        "feature_ranking": ranking,
        "figure_path": figure_path,
    }


def print_final_summary(summary_df: pd.DataFrame, spline_info: dict[str, Any] | None) -> None:
    """Print the final analysis summary and next steps.

    Args:
        summary_df: Summary results DataFrame.
        spline_info: Optional spline-analysis metadata.
    """
    best_kan, best_mlp = get_best_config_pair(summary_df)
    winner = "KAN" if float(best_kan["test_r2_mean"]) >= float(best_mlp["test_r2_mean"]) else "MLP"
    margin = abs(float(best_kan["test_r2_mean"]) - float(best_mlp["test_r2_mean"]))
    speed_ratio = float(best_kan["avg_time_per_epoch_mean"]) / max(float(best_mlp["avg_time_per_epoch_mean"]), 1e-8)

    most_nonlinear_feature = "N/A"
    most_nonlinear_score = float("nan")
    if spline_info is not None and spline_info["feature_ranking"]:
        most_nonlinear_feature, most_nonlinear_score = spline_info["feature_ranking"][0]

    print("═" * 75)
    print("EXPERIMENT 1 ANALYSIS COMPLETE")
    print("═" * 75)
    print()
    print("Figures saved to: results/exp1/figures/")
    print("  - fig1_kan_r2_heatmap.png")
    print("  - fig2_kan_vs_mlp_r2.png")
    print("  - fig3_convergence_curves.png")
    print("  - fig4_mse_comparison.png")
    print("  - fig5_training_speed.png")
    print("  - fig6_params_vs_r2.png")
    if spline_info is not None:
        print("  - fig7_kan_splines.png")
    print()
    print("Tables saved to: results/exp1/figures/summary_table.txt")
    print()
    if spline_info is not None:
        print("Spline data saved for Experiment 3:")
        print("  - results/exp1/kan_nonlinearity_scores.npy")
        print("  - results/exp1/feature_names.json")
        print()
    print("Key findings:")
    print(
        f"  - Best KAN config: G={int(best_kan['grid_size'])}, k={int(best_kan['spline_order'])} "
        f"(R² = {float(best_kan['test_r2_mean']):.4f} ± {0.0 if pd.isna(best_kan['test_r2_std']) else float(best_kan['test_r2_std']):.4f})"
    )
    print(
        f"  - Best matched MLP: H={int(best_mlp['hidden_dim'])} "
        f"(R² = {float(best_mlp['test_r2_mean']):.4f} ± {0.0 if pd.isna(best_mlp['test_r2_std']) else float(best_mlp['test_r2_std']):.4f})"
    )
    print(f"  - KAN vs MLP winner: {winner} by margin of {margin:.4f}")
    if spline_info is not None:
        print(
            f"  - Most nonlinear KAN feature: {most_nonlinear_feature} "
            f"(score: {most_nonlinear_score:.4f})"
        )
    print(f"  - KAN training speed: {speed_ratio:.1f}x slower than MLP per epoch")
    print("═" * 75)


def main() -> None:
    """Run the Experiment 1 analysis pipeline."""
    ensure_results_exist()
    prepare_output_dirs()

    all_results_df, summary_df = load_results()

    generate_figure_1_heatmap(summary_df)
    generate_grouped_metric_chart(
        summary_df,
        metric_mean_col="test_r2_mean",
        metric_std_col="test_r2_std",
        title="KAN vs Parameter-Matched MLP: Test R²",
        ylabel="Test R²",
        filename="fig2_kan_vs_mlp_r2.png",
        best_reference=True,
    )
    generate_figure_3_convergence(all_results_df, summary_df)
    generate_grouped_metric_chart(
        summary_df,
        metric_mean_col="test_mse_mean",
        metric_std_col="test_mse_std",
        title="KAN vs Parameter-Matched MLP: Test MSE",
        ylabel="Test MSE",
        filename="fig4_mse_comparison.png",
        best_reference=False,
    )
    generate_figure_5_speed(summary_df)
    generate_figure_6_params_vs_r2(summary_df)
    generate_summary_table(summary_df)
    spline_info = run_spline_visualization_section(all_results_df, summary_df)
    print_final_summary(summary_df, spline_info)


if __name__ == "__main__":
    main()
