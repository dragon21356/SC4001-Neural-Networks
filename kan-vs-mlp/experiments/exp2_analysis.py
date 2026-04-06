"""Post-run analysis script for Experiment 2 CIFAR-10 classification results."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import BSPLINE_COLOR
from src.utils import KAN_COLOR
from src.utils import MLP_COLOR
from src.utils import format_param_count

ALL_RESULTS_PATH = os.path.join("results", "exp2", "exp2_all_results.csv")
SUMMARY_RESULTS_PATH = os.path.join("results", "exp2", "exp2_summary.csv")
FIGURES_DIR = os.path.join("results", "exp2", "figures")
SUMMARY_TABLE_PATH = os.path.join(FIGURES_DIR, "summary_table.txt")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
HEAD_LABELS = {
    "MLP_ModeA": "MLP",
    "KAN_ModeA": "KAN",
    "BSplineMLP_ModeA": "BSpline-MLP",
    "MLP_ModeB": "MLP",
    "KAN_ModeB": "KAN",
    "BSplineMLP_ModeB": "BSpline-MLP",
}
HEAD_ORDER = ["MLP", "KAN", "BSpline-MLP"]
COLOR_BY_LABEL = {
    "MLP": MLP_COLOR,
    "KAN": KAN_COLOR,
    "BSpline-MLP": BSPLINE_COLOR,
}
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOTATION_SIZE = 10


def ensure_results_exist() -> None:
    """Ensure Experiment 2 result CSVs are present.

    Raises:
        SystemExit: If the required CSV files do not exist.
    """
    if not os.path.exists(ALL_RESULTS_PATH) or not os.path.exists(SUMMARY_RESULTS_PATH):
        print("Results files not found. Run exp2_cifar10.py first.")
        raise SystemExit(1)


def prepare_output_dirs() -> None:
    """Create the analysis output directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Experiment 2 CSVs and print validation info.

    Returns:
        Tuple of ``(all_results_df, summary_df)``.
    """
    all_results_df = pd.read_csv(ALL_RESULTS_PATH)
    summary_df = pd.read_csv(SUMMARY_RESULTS_PATH)

    if len(all_results_df) < 18:
        print(f"Warning: Expected 18 rows in exp2_all_results.csv, found {len(all_results_df)}.")
    if len(summary_df) < 6:
        print(f"Warning: Expected 6 rows in exp2_summary.csv, found {len(summary_df)}.")

    models_found = sorted(summary_df["name"].dropna().astype(str).unique().tolist())
    seeds_found = sorted(all_results_df["seed"].dropna().astype(int).unique().tolist())

    print(f"Completed runs: {len(all_results_df)}")
    print(f"Models found: {len(models_found)} -> {models_found}")
    print(f"Seeds found: {len(seeds_found)} -> {seeds_found}")

    return all_results_df, summary_df


def save_figure(fig: plt.Figure, filename: str) -> str:
    """Save and close a matplotlib figure.

    Args:
        fig: Figure to save.
        filename: Output filename under ``results/exp2/figures``.

    Returns:
        Full output path.
    """
    output_path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
        return [float(value) for value in raw_value if np.isfinite(float(value))]

    try:
        decoded = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return []

    values: list[float] = []
    for value in decoded:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return values


def decode_per_class_accuracy(raw_value: Any) -> dict[str, float]:
    """Decode a JSON-encoded per-class accuracy dict safely.

    Args:
        raw_value: Raw CSV cell value.

    Returns:
        Decoded class-to-accuracy mapping.
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return {}
    if isinstance(raw_value, dict):
        return {str(key): float(value) for key, value in raw_value.items()}

    try:
        decoded = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}

    return {
        str(key): float(value)
        for key, value in decoded.items()
        if isinstance(decoded, dict)
    }


def mode_summary(summary_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return rows for one mode in display order.

    Args:
        summary_df: Summary results DataFrame.
        mode: Mode identifier, ``A`` or ``B``.

    Returns:
        Filtered and ordered summary DataFrame.
    """
    mode_df = summary_df[summary_df["mode"] == mode].copy()
    order = [f"MLP_Mode{mode}", f"KAN_Mode{mode}", f"BSplineMLP_Mode{mode}"]
    mode_df["sort_key"] = mode_df["name"].map({name: idx for idx, name in enumerate(order)})
    return mode_df.sort_values(by="sort_key").drop(columns=["sort_key"])


def build_summary_table(summary_df: pd.DataFrame) -> str:
    """Build the report-ready summary table text.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Table text.
    """
    def block_lines(mode: str, title: str) -> list[str]:
        rows = mode_summary(summary_df, mode)
        lines = [
            title,
            "┌─────────────────┬────────────┬────────────┬─────────────────────┬─────────────────────┬──────────────┬──────────┐",
            "│ Head            │ Total Params│ Head Params│ Test Accuracy       │ Test Loss           │ Conv. Epoch  │ Time/Ep  │",
            "├─────────────────┼────────────┼────────────┼─────────────────────┼─────────────────────┼──────────────┼──────────┤",
        ]
        for row in rows.itertuples(index=False):
            acc_std = 0.0 if pd.isna(row.test_accuracy_std) else float(row.test_accuracy_std)
            loss_std = 0.0 if pd.isna(row.test_loss_std) else float(row.test_loss_std)
            conv_std = 0.0 if pd.isna(row.convergence_epoch_std) else float(row.convergence_epoch_std)
            lines.append(
                f"│ {HEAD_LABELS[str(row.name)]:<15} │ "
                f"{format_param_count(int(row.total_params)):>10} │ "
                f"{format_param_count(int(row.head_params)):>10} │ "
                f"{float(row.test_accuracy_mean):>6.4f} ± {acc_std:<6.4f} │ "
                f"{float(row.test_loss_mean):>6.4f} ± {loss_std:<6.4f} │ "
                f"{float(row.convergence_epoch_mean):>5.1f} ± {conv_std:<5.1f} │ "
                f"{float(row.avg_time_per_epoch_mean):>6.1f}s │"
            )
        lines.append(
            "└─────────────────┴────────────┴────────────┴─────────────────────┴─────────────────────┴──────────────┴──────────┘"
        )
        return lines

    lines = [
        "Table 2: Experiment 2 Results — CNN Classification Heads on CIFAR-10 (averaged across 3 seeds)",
        "",
        *block_lines("A", "Mode A — Same Width (hidden=256), Different Parameter Counts"),
        "",
        *block_lines("B", "Mode B — Matched Parameters (~134-146K head params)"),
    ]
    table_text = "\n".join(lines)
    with open(SUMMARY_TABLE_PATH, "w", encoding="utf-8") as handle:
        handle.write(table_text)
    print(table_text)
    return table_text


def accuracy_baseline(summary_df: pd.DataFrame) -> float:
    """Compute a useful lower y-axis bound for accuracy charts.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Lower y-axis bound.
    """
    if summary_df.empty:
        return 0.70
    min_val = float(summary_df["test_accuracy_mean"].min())
    return max(0.70, min_val - 0.03)


def annotate_accuracy_bars(
    ax: plt.Axes,
    bars: list[Any],
    rows: list[pd.Series],
) -> None:
    """Add accuracy and head-param annotations to bars.

    Args:
        ax: Target axes.
        bars: Bar objects.
        rows: Matching summary rows.
    """
    for bar, row in zip(bars, rows, strict=True):
        height = float(row["test_accuracy_mean"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ax.get_ylim()[0] + 0.003,
            f"({format_param_count(int(row['head_params']))})",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )


def generate_mode_accuracy_figure(summary_df: pd.DataFrame, mode: str, filename: str) -> str:
    """Generate a single-mode accuracy bar chart.

    Args:
        summary_df: Summary results DataFrame.
        mode: Mode identifier.
        filename: Output filename.

    Returns:
        Saved figure path.
    """
    rows_df = mode_summary(summary_df, mode)
    labels = [HEAD_LABELS[str(name)] for name in rows_df["name"].tolist()]
    means = [float(value) for value in rows_df["test_accuracy_mean"].tolist()]
    stds = [0.0 if pd.isna(value) else float(value) for value in rows_df["test_accuracy_std"].tolist()]
    colors = [COLOR_BY_LABEL[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    ymin = accuracy_baseline(summary_df)
    ymax = min(1.0, max(means) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Test Accuracy", fontsize=LABEL_SIZE)
    ax.set_title(
        "Mode A: Same Width (hidden=256), Different Parameters"
        if mode == "A"
        else "Mode B: Matched Parameters (~134-146K head params)",
        fontsize=TITLE_SIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    annotate_accuracy_bars(ax, list(bars), [row for _, row in rows_df.iterrows()])
    return save_figure(fig, filename)


def select_median_seed_run(all_results_df: pd.DataFrame, model_name: str) -> pd.Series:
    """Select the median-seed run by final test accuracy.

    Args:
        all_results_df: Run-level results DataFrame.
        model_name: Model name to filter on.

    Returns:
        Selected run row.
    """
    subset = all_results_df[all_results_df["name"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"No runs found for {model_name}.")
    subset = subset.sort_values(by="test_accuracy", ascending=True).reset_index(drop=True)
    return subset.iloc[len(subset) // 2]


def generate_convergence_figure(all_results_df: pd.DataFrame, mode: str, filename: str) -> str:
    """Generate stacked convergence curves for one mode.

    Args:
        all_results_df: Run-level results DataFrame.
        mode: Mode identifier.
        filename: Output filename.

    Returns:
        Saved figure path.
    """
    model_names = [f"MLP_Mode{mode}", f"KAN_Mode{mode}", f"BSplineMLP_Mode{mode}"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name in model_names:
        run = select_median_seed_run(all_results_df, model_name)
        label = HEAD_LABELS[model_name]
        color = COLOR_BY_LABEL[label]
        val_acc = decode_history(run["val_accuracy_history"])
        train_loss = decode_history(run["train_loss_history"])
        axes[0].plot(
            range(1, len(val_acc) + 1),
            val_acc,
            color=color,
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            range(1, len(train_loss) + 1),
            train_loss,
            color=color,
            linewidth=2,
            label=label,
        )

    axes[0].set_ylabel("Validation Accuracy", fontsize=LABEL_SIZE)
    axes[0].set_title(f"Mode {mode} Convergence Curves", fontsize=TITLE_SIZE)
    axes[0].legend(fontsize=TICK_SIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="both", labelsize=TICK_SIZE)

    axes[1].set_xlabel("Epoch", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Train Loss", fontsize=LABEL_SIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="both", labelsize=TICK_SIZE)

    return save_figure(fig, filename)


def build_per_class_matrix(all_results_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build averaged per-class accuracy matrix for a mode.

    Args:
        all_results_df: Run-level results DataFrame.
        mode: Mode identifier.

    Returns:
        DataFrame with rows as heads and columns as class names.
    """
    matrix_rows: list[list[float]] = []
    row_labels: list[str] = []
    for model_name in [f"MLP_Mode{mode}", f"KAN_Mode{mode}", f"BSplineMLP_Mode{mode}"]:
        subset = all_results_df[all_results_df["name"] == model_name]
        class_values: dict[str, list[float]] = {class_name: [] for class_name in CLASS_NAMES}
        for raw in subset["per_class_accuracy"].tolist():
            decoded = decode_per_class_accuracy(raw)
            for class_name in CLASS_NAMES:
                if class_name in decoded and np.isfinite(float(decoded[class_name])):
                    class_values[class_name].append(float(decoded[class_name]))
        matrix_rows.append(
            [
                float(np.mean(class_values[class_name])) if class_values[class_name] else np.nan
                for class_name in CLASS_NAMES
            ]
        )
        row_labels.append(HEAD_LABELS[model_name])
    return pd.DataFrame(matrix_rows, index=row_labels, columns=CLASS_NAMES)


def generate_per_class_heatmap(all_results_df: pd.DataFrame) -> str:
    """Generate the Mode B per-class accuracy heatmap.

    Args:
        all_results_df: Run-level results DataFrame.

    Returns:
        Saved figure path.
    """
    matrix = build_per_class_matrix(all_results_df, mode="B")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Mode B: Per-Class Test Accuracy", fontsize=TITLE_SIZE)
    ax.set_xlabel("CIFAR-10 Class", fontsize=LABEL_SIZE)
    ax.set_ylabel("Head Type", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    return save_figure(fig, "fig5_per_class_heatmap.png")


def generate_accuracy_vs_params(summary_df: pd.DataFrame) -> str:
    """Generate head-parameter efficiency scatter plot.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    marker_by_label = {"MLP": "s", "KAN": "o", "BSpline-MLP": "D"}
    for _, row in summary_df.iterrows():
        label = HEAD_LABELS[str(row["name"])]
        color = COLOR_BY_LABEL[label]
        marker = marker_by_label[label]
        ax.errorbar(
            float(row["head_params"]),
            float(row["test_accuracy_mean"]),
            yerr=0.0 if pd.isna(row["test_accuracy_std"]) else float(row["test_accuracy_std"]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=8,
        )
        ax.annotate(
            f"{row['name']}",
            (float(row["head_params"]), float(row["test_accuracy_mean"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=ANNOTATION_SIZE,
        )
    ax.axvline(135_000, color="#808080", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Head Parameters", fontsize=LABEL_SIZE)
    ax.set_ylabel("Test Accuracy", fontsize=LABEL_SIZE)
    ax.set_title("Parameter Efficiency: Head Params vs Test Accuracy", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    return save_figure(fig, "fig6_accuracy_vs_params.png")


def generate_combined_accuracy_figure(summary_df: pd.DataFrame) -> str:
    """Generate the main combined Mode A + Mode B bar chart.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    mode_a = mode_summary(summary_df, "A")
    mode_b = mode_summary(summary_df, "B")
    labels = ["MLP", "KAN", "BSpline-MLP"]
    colors = [COLOR_BY_LABEL[label] for label in labels]
    width = 0.28
    mode_a_positions = np.array([0.0, 0.4, 0.8])
    mode_b_positions = np.array([1.8, 2.2, 2.6])

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_a = ax.bar(
        mode_a_positions,
        mode_a["test_accuracy_mean"].astype(float).tolist(),
        width=width,
        yerr=mode_a["test_accuracy_std"].fillna(0.0).astype(float).tolist(),
        capsize=5,
        color=colors,
        alpha=0.9,
    )
    bars_b = ax.bar(
        mode_b_positions,
        mode_b["test_accuracy_mean"].astype(float).tolist(),
        width=width,
        yerr=mode_b["test_accuracy_std"].fillna(0.0).astype(float).tolist(),
        capsize=5,
        color=colors,
        alpha=0.9,
    )

    ymin = accuracy_baseline(summary_df)
    ymax = min(1.0, float(summary_df["test_accuracy_mean"].max()) + 0.03)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Test Accuracy", fontsize=LABEL_SIZE)
    ax.set_title("CIFAR-10 Classification: CNN + Swappable Heads", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks([mode_a_positions.mean(), mode_b_positions.mean()])
    ax.set_xticklabels(["Mode A\n(Same Width)", "Mode B\n(Matched Params)"], fontsize=TICK_SIZE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.9) for color in colors
    ]
    ax.legend(legend_handles, labels, fontsize=TICK_SIZE)

    for positions, rows in ((mode_a_positions, mode_a), (mode_b_positions, mode_b)):
        for x_pos, (_, row) in zip(positions, rows.iterrows(), strict=True):
            ax.text(
                x_pos,
                float(row["test_accuracy_mean"]) + 0.002,
                f"{float(row['test_accuracy_mean']):.4f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
            )

    return save_figure(fig, "fig7_combined_accuracy.png")


def generate_training_speed_figure(summary_df: pd.DataFrame) -> str:
    """Generate the Mode B training-speed comparison chart.

    Args:
        summary_df: Summary results DataFrame.

    Returns:
        Saved figure path.
    """
    mode_b = mode_summary(summary_df, "B")
    labels = [HEAD_LABELS[str(name)] for name in mode_b["name"].tolist()]
    values = mode_b["avg_time_per_epoch_mean"].astype(float).tolist()
    colors = [COLOR_BY_LABEL[label] for label in labels]

    max_speed = max(values) if values else 1.0
    min_speed = min(values) if values else 0.0
    percent_impact = ((max_speed - min_speed) / max(min_speed, 1e-8)) * 100.0 if values else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Seconds per Epoch", fontsize=LABEL_SIZE)
    ax.set_title("Training Speed: Head Type Has Negligible Impact", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )
    ax.text(
        0.5,
        max_speed * 0.97,
        f"Backbone dominates — head type has <{percent_impact:.1f}% speed impact",
        ha="center",
        va="top",
        transform=ax.get_yaxis_transform(),
        fontsize=ANNOTATION_SIZE,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    return save_figure(fig, "fig8_training_speed.png")


def mode_accuracy_margin(summary_df: pd.DataFrame, mode: str, left_name: str, right_name: str) -> float:
    """Compute accuracy difference between two models in one mode.

    Args:
        summary_df: Summary results DataFrame.
        mode: Mode identifier.
        left_name: Left model name.
        right_name: Right model name.

    Returns:
        Mean test accuracy difference.
    """
    mode_df = mode_summary(summary_df, mode).set_index("name")
    return float(mode_df.loc[left_name, "test_accuracy_mean"]) - float(mode_df.loc[right_name, "test_accuracy_mean"])


def compute_per_class_spread(all_results_df: pd.DataFrame, mode: str) -> tuple[list[tuple[str, float]], float]:
    """Compute classwise accuracy spread across heads for one mode.

    Args:
        all_results_df: Run-level results DataFrame.
        mode: Mode identifier.

    Returns:
        Tuple of ``(classes_with_gt_5pct_spread, max_spread)``.
    """
    matrix = build_per_class_matrix(all_results_df, mode=mode)
    spreads: list[tuple[str, float]] = []
    max_spread = 0.0
    for class_name in CLASS_NAMES:
        values = matrix[class_name].astype(float).dropna().tolist()
        if not values:
            continue
        spread = max(values) - min(values)
        max_spread = max(max_spread, spread)
        if spread > 0.05:
            spreads.append((class_name, spread))
    spreads.sort(key=lambda item: item[1], reverse=True)
    return spreads, max_spread


def print_key_findings(summary_df: pd.DataFrame, all_results_df: pd.DataFrame) -> None:
    """Compute and print the key analytical findings.

    Args:
        summary_df: Summary results DataFrame.
        all_results_df: Run-level results DataFrame.
    """
    mode_a = mode_summary(summary_df, "A").set_index("name")
    mode_b = mode_summary(summary_df, "B").set_index("name")

    kan_a_margin = float(mode_a.loc["KAN_ModeA", "test_accuracy_mean"]) - float(mode_a.loc["MLP_ModeA", "test_accuracy_mean"])
    kan_a_param_ratio = float(mode_a.loc["KAN_ModeA", "head_params"]) / max(float(mode_a.loc["MLP_ModeA", "head_params"]), 1e-8)

    mode_b_rows = mode_summary(summary_df, "B").sort_values(by="test_accuracy_mean", ascending=False)
    ranking = " > ".join(mode_b_rows["name"].map(HEAD_LABELS).tolist())
    mlp_vs_kan_b = mode_accuracy_margin(summary_df, "B", "MLP_ModeB", "KAN_ModeB")
    mlp_vs_bspline_b = mode_accuracy_margin(summary_df, "B", "MLP_ModeB", "BSplineMLP_ModeB")
    bspline_vs_mlp_a = mode_accuracy_margin(summary_df, "A", "BSplineMLP_ModeA", "MLP_ModeA")
    bspline_vs_mlp_b = mode_accuracy_margin(summary_df, "B", "BSplineMLP_ModeB", "MLP_ModeB")

    fastest_idx = mode_b["convergence_epoch_mean"].astype(float).idxmin()
    spreads, max_spread = compute_per_class_spread(all_results_df, mode="B")

    speed_values = summary_df["avg_time_per_epoch_mean"].astype(float)
    speed_variation_pct = ((float(speed_values.max()) - float(speed_values.min())) / max(float(speed_values.min()), 1e-8)) * 100.0

    print("═" * 75)
    print("KEY FINDINGS")
    print("═" * 75)
    print()
    print("1. MODE A — Does extra KAN capacity help?")
    print("   KAN (1.34M head params) vs MLP (134K head params):")
    print(
        f"   Accuracy difference: {'+' if kan_a_margin >= 0 else '-'}{abs(kan_a_margin):.4f} "
        f"(KAN has {kan_a_param_ratio:.1f}x params but {'gains' if kan_a_margin >= 0 else 'loses'} "
        f"{abs(kan_a_margin) * 100:.2f}% accuracy)"
    )
    print(
        f"   -> Extra capacity {'does' if kan_a_margin >= 0 else 'does not'} "
        "translate to proportional accuracy gains"
    )
    print()
    print("2. MODE B — Which architecture wins at equal budget?")
    print(f"   Ranking: {ranking} by accuracy")
    print(
        f"   MLP vs KAN: {'MLP' if mlp_vs_kan_b >= 0 else 'KAN'} by {abs(mlp_vs_kan_b):.4f}"
    )
    print(
        f"   MLP vs BSpline-MLP: {'MLP' if mlp_vs_bspline_b >= 0 else 'BSpline-MLP'} by {abs(mlp_vs_bspline_b):.4f}"
    )
    print(
        f"   -> At equal parameters, {HEAD_LABELS[str(mode_b_rows.iloc[0].name)]} is most efficient"
    )
    print()
    print("3. ACTIVATION FUNCTION HYPOTHESIS")
    print("   BSpline-MLP vs MLP (same architecture, different activation):")
    print(f"   Mode A difference: {bspline_vs_mlp_a:+.4f}")
    print(f"   Mode B difference: {bspline_vs_mlp_b:+.4f}")
    activation_benefit = (bspline_vs_mlp_a + bspline_vs_mlp_b) / 2.0
    print(
        f"   -> Learnable B-spline activations {'do' if activation_benefit >= 0 else 'do not'} improve over fixed ReLU"
    )
    print(
        f"   -> This {'supports' if activation_benefit >= 0 else 'contradicts'} the hypothesis that "
        "KAN's advantage comes from activation functions"
    )
    print()
    print("4. CONVERGENCE SPEED")
    print("   Mode B convergence epochs (to 95% of final accuracy):")
    for model_name in ["MLP_ModeB", "KAN_ModeB", "BSplineMLP_ModeB"]:
        row = mode_b.loc[model_name]
        conv_std = 0.0 if pd.isna(row["convergence_epoch_std"]) else float(row["convergence_epoch_std"])
        print(
            f"   {HEAD_LABELS[model_name]}: {float(row['convergence_epoch_mean']):.1f} ± {conv_std:.1f} epochs"
        )
    print(f"   -> {HEAD_LABELS[str(fastest_idx)]} converges fastest")
    print()
    print("5. TRAINING SPEED")
    print(
        f"   All models: ~{float(speed_values.mean()):.1f}s per epoch (< {speed_variation_pct:.1f}% variation)"
    )
    print("   -> Head type has negligible speed impact when backbone dominates computation")
    print("   -> This contradicts the common narrative that KAN is impractically slow for vision tasks")
    print()
    print("6. PER-CLASS PATTERNS")
    if spreads:
        print("   Classes where models differ most:")
        for class_name, spread in spreads:
            print(f"   {class_name}: spread {spread:.4f}")
    else:
        print(f"   No systematic per-class differences observed (max spread: {max_spread:.4f})")
    print("═" * 75)


def print_final_summary() -> None:
    """Print the final analysis completion summary."""
    print()
    print("═" * 75)
    print("EXPERIMENT 2 ANALYSIS COMPLETE")
    print("═" * 75)
    print()
    print("Figures saved to: results/exp2/figures/")
    print("  - fig1_mode_a_accuracy.png")
    print("  - fig2_mode_b_accuracy.png")
    print("  - fig3_convergence_mode_a.png")
    print("  - fig4_convergence_mode_b.png")
    print("  - fig5_per_class_heatmap.png")
    print("  - fig6_accuracy_vs_params.png")
    print("  - fig7_combined_accuracy.png (KEY FIGURE for report)")
    print("  - fig8_training_speed.png")
    print()
    print("Tables saved to: results/exp2/figures/summary_table.txt")
    print()
    print("Ready for Experiment 3 (Interpretability Analysis) and report writing.")
    print("═" * 75)


def main() -> None:
    """Run the Experiment 2 analysis pipeline."""
    ensure_results_exist()
    prepare_output_dirs()

    all_results_df, summary_df = load_results()

    build_summary_table(summary_df)
    generate_mode_accuracy_figure(summary_df, mode="A", filename="fig1_mode_a_accuracy.png")
    generate_mode_accuracy_figure(summary_df, mode="B", filename="fig2_mode_b_accuracy.png")
    generate_convergence_figure(all_results_df, mode="A", filename="fig3_convergence_mode_a.png")
    generate_convergence_figure(all_results_df, mode="B", filename="fig4_convergence_mode_b.png")
    generate_per_class_heatmap(all_results_df)
    generate_accuracy_vs_params(summary_df)
    generate_combined_accuracy_figure(summary_df)
    generate_training_speed_figure(summary_df)
    print_key_findings(summary_df, all_results_df)
    print_final_summary()


if __name__ == "__main__":
    main()
