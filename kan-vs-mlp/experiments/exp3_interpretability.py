"""Experiment 3 interpretability analysis: SHAP vs KAN nonlinearity."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import shap
except ImportError:  # pragma: no cover - runtime dependency check
    shap = None

from src.data_utils import get_california_housing
from src.evaluate import evaluate_model
from src.evaluate import r2_metric
from src.kan_layer import KANLinear
from src.models import KANRegressor
from src.models import MLPRegressor
from src.spline_vis import compute_feature_nonlinearity
from src.spline_vis import plot_feature_importance_comparison
from src.spline_vis import plot_spline_vs_shap_scatter
from src.spline_vis import probe_kan_layer_responses
from src.utils import KAN_COLOR
from src.utils import MLP_COLOR
from src.utils import get_device
from src.utils import set_seed

EXP1_DIR = os.path.join("results", "exp1")
EXP1_ALL_RESULTS_PATH = os.path.join(EXP1_DIR, "exp1_all_results.csv")
KAN_SCORES_PATH = os.path.join(EXP1_DIR, "kan_nonlinearity_scores.npy")
FEATURE_NAMES_PATH = os.path.join(EXP1_DIR, "feature_names.json")
EXP1_CHECKPOINT_DIR = os.path.join(EXP1_DIR, "checkpoints")

EXP3_DIR = os.path.join("results", "exp3")
EXP3_FIGURES_DIR = os.path.join(EXP3_DIR, "figures")
SHAP_IMPORTANCE_PATH = os.path.join(EXP3_DIR, "shap_importance.npy")
SHAP_VALUES_FULL_PATH = os.path.join(EXP3_DIR, "shap_values_full.npy")

COMPARISON_FIG_PATH = os.path.join(EXP3_FIGURES_DIR, "fig1_kan_vs_shap_comparison.png")
SCATTER_FIG_PATH = os.path.join(EXP3_FIGURES_DIR, "fig2_kan_vs_shap_scatter.png")
BEESWARM_FIG_PATH = os.path.join(EXP3_FIGURES_DIR, "fig3_shap_beeswarm.png")
ANNOTATED_SPLINE_FIG_PATH = os.path.join(EXP3_FIGURES_DIR, "fig4_kan_splines_annotated.png")


def ensure_dependencies_and_paths() -> None:
    """Ensure required files and libraries are available."""
    if shap is None:
        print("SHAP library not found. Install with: pip install shap")
        raise SystemExit(1)

    if not os.path.exists(KAN_SCORES_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        print("Run exp1_analysis.py first to generate KAN nonlinearity scores.")
        raise SystemExit(1)

    if not os.path.exists(EXP1_ALL_RESULTS_PATH):
        print("Experiment 1 results not found. Run exp1_regression.py first.")
        raise SystemExit(1)

    os.makedirs(EXP3_FIGURES_DIR, exist_ok=True)


def load_kan_reference_data() -> tuple[np.ndarray, list[str]]:
    """Load precomputed KAN nonlinearity scores and feature names."""
    kan_nonlinearity = np.load(KAN_SCORES_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    assert kan_nonlinearity.shape == (8,), f"Expected KAN scores shape (8,), got {kan_nonlinearity.shape}"
    assert isinstance(feature_names, list) and len(feature_names) == 8
    assert all(isinstance(name, str) for name in feature_names)
    return kan_nonlinearity.astype(np.float64), feature_names


def print_feature_ranking(title: str, values: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """Print a descending ranking for one feature score vector."""
    order = np.argsort(values)[::-1]
    ranking = [(feature_names[idx], float(values[idx])) for idx in order]
    print(title)
    for rank, (feature_name, score) in enumerate(ranking, start=1):
        print(f"  {rank}. {feature_name}: {score:.4f}")
    print()
    return ranking


def load_experiment_1_results() -> pd.DataFrame:
    """Load the Experiment 1 all-results CSV."""
    return pd.read_csv(EXP1_ALL_RESULTS_PATH)


def get_best_row(results_df: pd.DataFrame, model_type: str) -> pd.Series:
    """Return the best row for the requested model type by test R^2."""
    subset = results_df[results_df["model_type"] == model_type].copy()
    if subset.empty:
        raise ValueError(f"No rows found for model_type={model_type}.")
    return subset.sort_values(by="test_r2", ascending=False).iloc[0]


def safe_load_state_dict(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load a checkpoint into a model with backward-compatible torch.load handling."""
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def build_mlp_checkpoint_path(best_mlp_row: pd.Series) -> str:
    """Construct the expected best-MLP checkpoint path from the CSV row."""
    config_name = str(best_mlp_row["config_name"])
    seed = int(best_mlp_row["seed"])
    checkpoint_name = f"mlp_{config_name}_seed{seed}_best.pt"
    return os.path.join(EXP1_CHECKPOINT_DIR, checkpoint_name)


def load_regression_loaders(seed: int, batch_size: int = 64) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load California Housing with the exact seed used during training."""
    return get_california_housing(batch_size=batch_size, seed=seed)


def collect_features(loader: Any, limit: int | None = None) -> np.ndarray:
    """Collect feature tensors from a regression loader into a numpy array."""
    feature_batches: list[torch.Tensor] = []
    count = 0
    for batch_x, _ in loader:
        feature_batches.append(batch_x.detach().cpu())
        count += batch_x.shape[0]
        if limit is not None and count >= limit:
            break

    if not feature_batches:
        return np.empty((0, 8), dtype=np.float32)

    features = torch.cat(feature_batches, dim=0)
    if limit is not None:
        features = features[:limit]
    return features.numpy()


def normalize_shap_output(raw_shap_values: Any) -> np.ndarray:
    """Normalize SHAP outputs across version differences to shape (N, F)."""
    if hasattr(raw_shap_values, "values"):
        values = np.asarray(raw_shap_values.values)
    else:
        values = np.asarray(raw_shap_values)

    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) == 1:
            values = np.asarray(raw_shap_values[0])
        else:
            values = np.asarray(raw_shap_values)

    if values.ndim == 3:
        if values.shape[-1] == 1:
            values = values[..., 0]
        elif values.shape[0] == 1:
            values = values[0]
        else:
            values = values.mean(axis=-1)

    if values.ndim != 2:
        raise ValueError(f"Expected SHAP values with 2 dimensions after normalization, got shape {values.shape}")

    return values.astype(np.float64)


def interpret_spearman(rho: float) -> str:
    """Return a plain-language interpretation of a Spearman correlation."""
    magnitude = abs(rho)
    if magnitude > 0.7:
        strength = "Strong"
    elif magnitude > 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"
    direction = "positive" if rho >= 0 else "negative"
    return f"{strength} {direction} correlation"


def verify_loaded_regressor(
    model: nn.Module,
    test_loader: Any,
    device: torch.device,
    original_r2: float,
    label: str,
) -> float:
    """Evaluate a loaded regressor and compare against the saved Experiment 1 R^2."""
    criterion = nn.MSELoss()
    evaluation = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        metric_fn=r2_metric,
    )
    verified_r2 = float(evaluation["test_metric"])
    difference = abs(original_r2 - verified_r2)

    print(f"{label} - Original R2 from Experiment 1: {original_r2:.6f}")
    print(f"{label} - Verified R2 after loading: {verified_r2:.6f}")
    print(f"{label} - Difference: {difference:.6f}")
    if difference > 0.01:
        print("WARNING: R2 mismatch exceeds tolerance. Check data loading.")
    else:
        print("R2 verified - model loaded correctly.")
    print()

    return verified_r2


def load_best_kan_model(results_df: pd.DataFrame, device: torch.device) -> tuple[KANRegressor, pd.Series]:
    """Rebuild and load the best KAN regression model from Experiment 1."""
    best_kan_row = get_best_row(results_df, model_type="kan")
    grid_size = int(best_kan_row["grid_size"])
    spline_order = int(best_kan_row["spline_order"])
    hidden_dim = int(best_kan_row["hidden_dim"])
    seed = int(best_kan_row["seed"])
    checkpoint_path = os.path.join(
        EXP1_CHECKPOINT_DIR,
        f"kan_G{grid_size}_K{spline_order}_seed{seed}_best.pt",
    )
    model = KANRegressor(
        in_features=8,
        hidden_dim=hidden_dim,
        out_features=1,
        grid_size=grid_size,
        spline_order=spline_order,
    ).to(device)
    safe_load_state_dict(model, checkpoint_path, device)
    model.eval()
    return model, best_kan_row


def create_shap_beeswarm(
    shap_values: np.ndarray,
    shap_test_samples: np.ndarray,
    feature_names: list[str],
) -> None:
    """Create and save the SHAP summary beeswarm figure."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_test_samples, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (MLP on California Housing)", fontsize=14)
    plt.tight_layout()
    plt.savefig(BEESWARM_FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def create_annotated_kan_spline_figure(
    results_df: pd.DataFrame,
    feature_names: list[str],
    shap_importance: np.ndarray,
    device: torch.device,
) -> None:
    """Create the annotated KAN spline plot with KAN/SHAP rank overlays."""
    try:
        kan_model, best_kan_row = load_best_kan_model(results_df, device)
        _, _, kan_test_loader, _ = load_regression_loaders(seed=int(best_kan_row["seed"]))
        verify_loaded_regressor(
            model=kan_model,
            test_loader=kan_test_loader,
            device=device,
            original_r2=float(best_kan_row["test_r2"]),
            label="KAN verification",
        )

        first_kan_layer = None
        for module in kan_model.modules():
            if isinstance(module, KANLinear):
                first_kan_layer = module
                break

        if first_kan_layer is None:
            raise RuntimeError("Could not find a KANLinear layer in the best KAN model.")

        responses = probe_kan_layer_responses(
            first_kan_layer,
            input_range=(-3.0, 3.0),
            num_points=200,
            reference_value=0.0,
            device=str(device),
        )
        kan_scores = compute_feature_nonlinearity(responses)
        input_values = np.asarray(responses["input_values"])
        aggregated_responses = np.asarray(responses["aggregated_responses"])

        kan_order = np.argsort(kan_scores)[::-1]
        shap_order = np.argsort(shap_importance)[::-1]
        kan_rank = {int(idx): rank for rank, idx in enumerate(kan_order, start=1)}
        shap_rank = {int(idx): rank for rank, idx in enumerate(shap_order, start=1)}

        fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
        flat_axes = axes.flatten()

        for feature_idx, ax in enumerate(flat_axes[: len(feature_names)]):
            curve = aggregated_responses[feature_idx]
            linear_coeffs = np.polyfit(input_values, curve, deg=1)
            linear_fit = np.polyval(linear_coeffs, input_values)

            ax.plot(input_values, curve, color=KAN_COLOR, linewidth=2)
            ax.plot(input_values, linear_fit, color="#757575", linestyle="--", linewidth=1.4)
            ax.set_title(feature_names[feature_idx], fontsize=11)
            ax.set_xlabel("Input value", fontsize=10)
            ax.set_ylabel("Activation output", fontsize=10)
            ax.grid(True, alpha=0.3)

            rank_gap = abs(kan_rank[feature_idx] - shap_rank[feature_idx])
            if rank_gap > 2:
                annotation_color = "#C62828"
            elif rank_gap <= 1:
                annotation_color = "#2E7D32"
            else:
                annotation_color = "#F9A825"

            ax.text(
                0.97,
                0.97,
                f"KAN rank: {kan_rank[feature_idx]}\nSHAP rank: {shap_rank[feature_idx]}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=annotation_color,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

        for ax in flat_axes[len(feature_names):]:
            ax.axis("off")

        fig.suptitle(
            "KAN Learned Splines with SHAP Rank Annotation",
            fontsize=14,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(ANNOTATED_SPLINE_FIG_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(
            "Annotated KAN splines created from "
            f"G={int(best_kan_row['grid_size'])}, k={int(best_kan_row['spline_order'])}, seed={int(best_kan_row['seed'])}."
        )
    except Exception as exc:
        print(f"Warning: could not create annotated KAN spline figure: {exc}")


def main() -> None:
    """Run the full Experiment 3 interpretability comparison."""
    ensure_dependencies_and_paths()
    set_seed(42)
    device = get_device()

    kan_nonlinearity, feature_names = load_kan_reference_data()
    kan_ranking = print_feature_ranking(
        "KAN Nonlinearity Ranking (loaded from Experiment 1):",
        kan_nonlinearity,
        feature_names,
    )

    results_df = load_experiment_1_results()

    best_mlp_row = get_best_row(results_df, model_type="mlp")
    best_mlp_hidden_dim = int(best_mlp_row["hidden_dim"])
    best_mlp_seed = int(best_mlp_row["seed"])
    best_mlp_checkpoint = build_mlp_checkpoint_path(best_mlp_row)
    train_loader, _, test_loader, _ = load_regression_loaders(seed=best_mlp_seed)

    print(
        "Best MLP config: "
        f"{best_mlp_row['config_name']}, hidden_dim={best_mlp_hidden_dim}, "
        f"seed={best_mlp_seed}, test_r2={float(best_mlp_row['test_r2']):.6f}"
    )

    mlp_model = MLPRegressor(
        in_features=8,
        hidden_dim=best_mlp_hidden_dim,
        out_features=1,
        dropout=0.1,
    ).to(device)
    safe_load_state_dict(mlp_model, best_mlp_checkpoint, device)
    mlp_model.eval()
    verified_mlp_r2 = verify_loaded_regressor(
        model=mlp_model,
        test_loader=test_loader,
        device=device,
        original_r2=float(best_mlp_row["test_r2"]),
        label="MLP verification",
    )
    print(
        f"Loaded best MLP checkpoint: {best_mlp_checkpoint}\n"
        f"Verified test R2 used for SHAP run: {verified_mlp_r2:.6f}"
    )
    print()

    test_features = collect_features(test_loader)
    bg_features = collect_features(train_loader, limit=100)
    shap_test_samples = test_features[:200]

    print(
        "Running SHAP KernelExplainer on 200 test samples with 100 background samples... "
        "(this may take 5-15 minutes)"
    )
    shap_start = time.time()

    def model_predict(x_numpy: np.ndarray) -> np.ndarray:
        """Wrapper that takes numpy input and returns numpy predictions."""
        mlp_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            output = mlp_model(x_tensor)
        return output.detach().cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, bg_features)
    raw_shap_values = explainer.shap_values(shap_test_samples, nsamples=100)
    shap_values = normalize_shap_output(raw_shap_values)
    shap_importance = np.abs(shap_values).mean(axis=0)

    print(f"SHAP computation finished in {(time.time() - shap_start) / 60.0:.1f} minutes.")
    print()

    shap_ranking = print_feature_ranking(
        "SHAP Feature Importance Ranking:",
        shap_importance,
        feature_names,
    )

    np.save(SHAP_IMPORTANCE_PATH, shap_importance)
    np.save(SHAP_VALUES_FULL_PATH, shap_values)

    plot_feature_importance_comparison(
        kan_nonlinearity=kan_nonlinearity,
        shap_importance=shap_importance,
        feature_names=feature_names,
        title="Feature Importance: KAN Nonlinearity vs SHAP",
        save_path=COMPARISON_FIG_PATH,
    )
    plot_spline_vs_shap_scatter(
        kan_nonlinearity=kan_nonlinearity,
        shap_importance=shap_importance,
        feature_names=feature_names,
        title="KAN Nonlinearity vs SHAP Importance",
        save_path=SCATTER_FIG_PATH,
    )
    create_shap_beeswarm(shap_values, shap_test_samples, feature_names)
    create_annotated_kan_spline_figure(
        results_df=results_df,
        feature_names=feature_names,
        shap_importance=shap_importance,
        device=device,
    )

    rho, p_value = spearmanr(kan_nonlinearity, shap_importance)
    interpretation = interpret_spearman(float(rho))

    print("=" * 75)
    print("EXPERIMENT 3: INTERPRETABILITY ANALYSIS")
    print("=" * 75)
    print()
    print("Feature Importance Rankings - Side by Side:")
    print("+------+----------------------------------+----------------------------------+")
    print("| Rank | KAN Nonlinearity                 | SHAP (MLP)                       |")
    print("+------+----------------------------------+----------------------------------+")
    for rank in range(8):
        kan_feature, kan_score = kan_ranking[rank]
        shap_feature, shap_score = shap_ranking[rank]
        print(
            f"| {rank + 1:<4} | "
            f"{kan_feature:<15} ({kan_score:>7.4f}) | "
            f"{shap_feature:<15} ({shap_score:>7.4f}) |"
        )
    print("+------+----------------------------------+----------------------------------+")
    print()
    print(f"Spearman Rank Correlation: rho = {rho:.3f} (p = {p_value:.3f})")
    print(f"Interpretation: {interpretation}")
    print("  - rho > 0.7: Strong agreement - both methods largely agree on feature importance")
    print("  - 0.4 < rho < 0.7: Moderate agreement - some shared signal but notable differences")
    print("  - rho < 0.4: Weak agreement - the methods capture fundamentally different aspects")
    print()
    print("Key observations:")
    print(f"  - SHAP's top feature: {shap_ranking[0][0]}")
    print(f"  - KAN's top feature: {kan_ranking[0][0]}")
    print("  - Critical insight: KAN nonlinearity measures COMPLEXITY of the learned function, not IMPORTANCE.")
    print("    A feature can be the most important predictor (high SHAP) while having a simple/linear relationship (low KAN nonlinearity).")
    print("    Conversely, a less important feature may have a complex nonlinear relationship (high KAN nonlinearity).")
    print("  - This means KAN spline visualization and SHAP answer DIFFERENT questions:")
    print("    * SHAP: How much does this feature affect the prediction? (importance)")
    print("    * KAN: How complex is the relationship between this feature and the prediction? (nonlinearity)")
    print("  - Both are valuable for interpretability, but they are NOT interchangeable.")
    print()
    print("Figures saved to: results/exp3/figures/")
    print("  - fig1_kan_vs_shap_comparison.png")
    print("  - fig2_kan_vs_shap_scatter.png")
    print("  - fig3_shap_beeswarm.png")
    print("  - fig4_kan_splines_annotated.png")
    print()
    print("Data saved:")
    print("  - results/exp3/shap_importance.npy")
    print("  - results/exp3/shap_values_full.npy")
    print()
    print("=" * 75)


if __name__ == "__main__":
    main()
