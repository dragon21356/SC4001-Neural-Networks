"""Experiment 1 regression sweep for KAN vs parameter-matched MLP baselines."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_california_housing
from src.evaluate import compute_convergence_epoch
from src.evaluate import evaluate_model
from src.evaluate import r2_metric
from src.models import KANRegressor
from src.models import MLPRegressor
from src.models import count_parameters
from src.train import train_model
from src.utils import format_param_count
from src.utils import get_device
from src.utils import save_results_csv
from src.utils import set_seed

CONFIG_PATH = os.path.join("experiments", "configs", "exp1_config.yaml")
ALL_RESULTS_PATH = os.path.join("results", "exp1", "exp1_all_results.csv")
SUMMARY_RESULTS_PATH = os.path.join("results", "exp1", "exp1_summary.csv")
PROMPT_FORMULA_WARNING_EMITTED = False


def load_config(config_path: str) -> dict[str, Any]:
    """Load the Experiment 1 YAML configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find Experiment 1 config at: {config_path}. "
            "Expected to load experiments/configs/exp1_config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_kan_parameter_count(
    kan_in: int,
    kan_hidden: int,
    kan_out: int,
    grid_size: int,
    spline_order: int,
) -> int:
    """Compute the KANRegressor parameter count from the layer formula.

    Args:
        kan_in: Number of KAN input features.
        kan_hidden: Hidden width of the KAN regressor.
        kan_out: Number of outputs.
        grid_size: KAN spline grid size.
        spline_order: KAN spline order.

    Returns:
        Total parameter count implied by the KAN layer formula.
    """
    per_edge_params = grid_size + spline_order + 2
    layer1 = kan_in * kan_hidden * per_edge_params
    layer2 = kan_hidden * kan_out * per_edge_params
    return layer1 + layer2


def compute_mlp_parameter_formula(
    mlp_in: int,
    hidden_dim: int,
    mlp_out: int,
) -> int:
    """Compute the true MLPRegressor parameter count from its architecture.

    Args:
        mlp_in: Number of input features.
        hidden_dim: Hidden width.
        mlp_out: Number of outputs.

    Returns:
        Total trainable parameters in the current ``MLPRegressor`` architecture.
    """
    layer1 = mlp_in * hidden_dim + hidden_dim
    layer2 = hidden_dim * hidden_dim + hidden_dim
    layer3 = hidden_dim * mlp_out + mlp_out
    return layer1 + layer2 + layer3


def compute_matched_mlp_hidden_dim(
    kan_in: int,
    kan_hidden: int,
    kan_out: int,
    grid_size: int,
    spline_order: int,
) -> int:
    """Compute an approximately parameter-matched MLP hidden dimension.

    Args:
        kan_in: Number of KAN input features.
        kan_hidden: Hidden width of the KAN regressor.
        kan_out: Number of KAN outputs.
        grid_size: KAN spline grid size.
        spline_order: KAN spline order.

    Returns:
        Hidden width for the matched MLP, clamped to at least 4.
    """
    global PROMPT_FORMULA_WARNING_EMITTED

    kan_total_params = compute_kan_parameter_count(
        kan_in=kan_in,
        kan_hidden=kan_hidden,
        kan_out=kan_out,
        grid_size=grid_size,
        spline_order=spline_order,
    )

    prompt_hidden = max(
        4,
        int(round((-10.0 + math.sqrt(max(96.0 + 4.0 * kan_total_params, 0.0))) / 2.0)),
    )
    prompt_formula_params = prompt_hidden * prompt_hidden + 10 * prompt_hidden + 1
    actual_formula_params = compute_mlp_parameter_formula(kan_in, prompt_hidden, kan_out)
    if prompt_formula_params != actual_formula_params and not PROMPT_FORMULA_WARNING_EMITTED:
        print(
            "Warning: prompt MLP formula (H^2 + 10H + 1) does not match the "
            "current MLPRegressor architecture. Using the actual architecture "
            "formula for matching."
        )
        PROMPT_FORMULA_WARNING_EMITTED = True

    discriminant = (kan_in + kan_out + 2) ** 2 - 4 * (1 - kan_out - kan_total_params)
    hidden_dim = int(
        round(
            (-(kan_in + kan_out + 2) + math.sqrt(max(discriminant, 0.0))) / 2.0
        )
    )
    return max(hidden_dim, 4)


def verify_parameter_match(
    grid_size: int,
    spline_order: int,
    kan_hidden: int,
    mlp_hidden: int,
    mlp_dropout: float,
    in_features: int = 8,
    out_features: int = 1,
) -> tuple[int, int]:
    """Verify parameter matching by instantiating both models.

    Args:
        grid_size: KAN spline grid size.
        spline_order: KAN spline order.
        kan_hidden: Hidden width of the KAN regressor.
        mlp_hidden: Proposed hidden width for the matched MLP.
        mlp_dropout: Dropout rate for the MLP.
        in_features: Number of model input features.
        out_features: Number of outputs.

    Returns:
        Tuple of ``(kan_params, mlp_params)`` from instantiated models.
    """
    kan_model = KANRegressor(
        in_features=in_features,
        hidden_dim=kan_hidden,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    mlp_model = MLPRegressor(
        in_features=in_features,
        hidden_dim=mlp_hidden,
        out_features=out_features,
        dropout=mlp_dropout,
    )

    kan_params = count_parameters(kan_model)
    mlp_params = count_parameters(mlp_model)
    relative_gap = abs(mlp_params - kan_params) / max(kan_params, 1)
    if relative_gap > 0.10:
        print(
            f"Warning: parameter match for G={grid_size}, K={spline_order} is "
            f"{relative_gap * 100:.2f}% apart (KAN={kan_params}, MLP={mlp_params})."
        )

    return kan_params, mlp_params


def build_configuration_table(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the parameter-matched configuration grid.

    Args:
        config: Experiment config dictionary.

    Returns:
        A list of configuration dictionaries for the full sweep.
    """
    kan_hidden = int(config["kan_grid_search"]["hidden_dim"])
    mlp_dropout = float(config["mlp"]["dropout"])
    configurations: list[dict[str, Any]] = []

    for grid_size in config["kan_grid_search"]["grid_sizes"]:
        for spline_order in config["kan_grid_search"]["spline_orders"]:
            matched_hidden = compute_matched_mlp_hidden_dim(
                kan_in=8,
                kan_hidden=kan_hidden,
                kan_out=1,
                grid_size=int(grid_size),
                spline_order=int(spline_order),
            )
            kan_params, mlp_params = verify_parameter_match(
                grid_size=int(grid_size),
                spline_order=int(spline_order),
                kan_hidden=kan_hidden,
                mlp_hidden=matched_hidden,
                mlp_dropout=mlp_dropout,
            )
            configurations.append(
                {
                    "config_name": f"G{grid_size}_K{spline_order}",
                    "grid_size": int(grid_size),
                    "spline_order": int(spline_order),
                    "kan_hidden_dim": kan_hidden,
                    "kan_params": kan_params,
                    "mlp_hidden_dim": matched_hidden,
                    "mlp_params": mlp_params,
                }
            )

    return configurations


def print_experiment_header(config: dict[str, Any]) -> None:
    """Print the experiment configuration header.

    Args:
        config: Experiment config dictionary.
    """
    print("=" * 80)
    print("EXPERIMENT 1: KAN vs MLP REGRESSION BENCHMARK")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    total_runs = (
        len(config["kan_grid_search"]["grid_sizes"])
        * len(config["kan_grid_search"]["spline_orders"])
        * len(config["seeds"])
        * 2
    )
    estimated_seconds_per_run = 30
    estimated_minutes = (total_runs * estimated_seconds_per_run) / 60.0
    print(
        f"Estimated total time: ~{estimated_minutes:.1f} minutes "
        f"(based on {total_runs} runs x ~{estimated_seconds_per_run}s each on CPU)"
    )
    print()


def print_configuration_table(configurations: list[dict[str, Any]]) -> None:
    """Print the parameter-matching table for all configurations.

    Args:
        configurations: Full sweep configuration entries.
    """
    print("Configuration Table:")
    print(
        "┌─────────┬───────────┬───────────┬────────────┬────────────┬────────────┬────────────┐"
    )
    print(
        "│ Config  │ Grid Size │ Spline K  │ KAN Hidden │ KAN Params │ MLP Hidden │ MLP Params │"
    )
    print(
        "├─────────┼───────────┼───────────┼────────────┼────────────┼────────────┼────────────┤"
    )
    for entry in configurations:
        print(
            f"│ {entry['config_name']:<7} │ "
            f"{entry['grid_size']:^9} │ "
            f"{entry['spline_order']:^9} │ "
            f"{entry['kan_hidden_dim']:^10} │ "
            f"{format_param_count(entry['kan_params']):>10} │ "
            f"{entry['mlp_hidden_dim']:^10} │ "
            f"{format_param_count(entry['mlp_params']):>10} │"
        )
    print(
        "└─────────┴───────────┴───────────┴────────────┴────────────┴────────────┴────────────┘"
    )
    print()


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict[str, Any],
) -> tuple[optim.Optimizer, Any | None]:
    """Create the optimizer and scheduler for a run.

    Args:
        model: Model being trained.
        config: Experiment config dictionary.

    Returns:
        Tuple of ``(optimizer, scheduler)``.
    """
    training_cfg = config["training"]
    optimizer_name = str(training_cfg["optimizer"]).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer in config: {training_cfg['optimizer']}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    scheduler_name = str(training_cfg["scheduler"]).lower()
    if scheduler_name == "reduce_on_plateau":
        scheduler: Any | None = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(training_cfg["scheduler_patience"]),
            factor=float(training_cfg["scheduler_factor"]),
        )
    elif scheduler_name in {"none", ""}:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler in config: {training_cfg['scheduler']}")

    return optimizer, scheduler


def load_existing_results(results_path: str) -> pd.DataFrame:
    """Load existing run results if resuming from a partial experiment.

    Args:
        results_path: Path to the all-results CSV.

    Returns:
        Existing results DataFrame or an empty one.
    """
    if not os.path.exists(results_path):
        return pd.DataFrame()
    return pd.read_csv(results_path)


def get_completed_keys(results_df: pd.DataFrame) -> set[tuple[str, str, int]]:
    """Extract completed run identifiers from existing results.

    Args:
        results_df: Existing results DataFrame.

    Returns:
        Set of ``(model_type, config_name, seed)`` tuples.
    """
    required_columns = {"model_type", "config_name", "seed"}
    if results_df.empty or not required_columns.issubset(results_df.columns):
        return set()

    completed: set[tuple[str, str, int]] = set()
    for row in results_df.itertuples(index=False):
        completed.add((str(row.model_type), str(row.config_name), int(row.seed)))
    return completed


def serialize_history_list(values: Iterable[float] | None) -> str:
    """Encode a history list as JSON for CSV storage.

    Args:
        values: Sequence of numeric history values.

    Returns:
        JSON string.
    """
    if values is None:
        return json.dumps([])
    return json.dumps([float(value) for value in values])


def convert_results_to_columns(results: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert a list of result dicts into column-oriented storage.

    Args:
        results: Row-wise results list.

    Returns:
        Column-oriented dictionary for CSV saving.
    """
    if not results:
        return {}

    keys = list(results[0].keys())
    return {key: [row.get(key) for row in results] for key in keys}


def save_all_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Persist all run-level results to CSV.

    Args:
        results: List of run result dictionaries.
        output_path: Destination CSV path.
    """
    columns = convert_results_to_columns(results)
    if columns:
        save_results_csv(columns, output_path)


def build_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level results across seeds into a summary table.

    Args:
        results_df: Full run-level results DataFrame.

    Returns:
        Summary DataFrame averaged across seeds.
    """
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "model_type",
                "config_name",
                "grid_size",
                "spline_order",
                "hidden_dim",
                "total_params",
                "test_mse_mean",
                "test_mse_std",
                "test_r2_mean",
                "test_r2_std",
                "convergence_epoch_mean",
                "convergence_epoch_std",
                "avg_time_per_epoch_mean",
                "total_training_time_mean",
            ]
        )

    summary = (
        results_df.groupby(
            ["model_type", "config_name", "grid_size", "spline_order", "hidden_dim", "total_params"],
            dropna=False,
        )
        .agg(
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            convergence_epoch_mean=("convergence_epoch_95", "mean"),
            convergence_epoch_std=("convergence_epoch_95", "std"),
            avg_time_per_epoch_mean=("avg_time_per_epoch", "mean"),
            total_training_time_mean=("total_training_time", "mean"),
        )
        .reset_index()
    )
    return summary


def print_summary_block(title: str, summary_df: pd.DataFrame) -> None:
    """Print one formatted summary block.

    Args:
        title: Block title.
        summary_df: Summary rows to print.
    """
    print(title)
    print("┌──────────────────┬────────┬──────────┬───────────────────┬──────────────────┬───────────┐")
    print("│ Config           │ Params │ Time/Ep  │ Test MSE          │ Test R²          │ Conv. Ep  │")
    print("├──────────────────┼────────┼──────────┼───────────────────┼──────────────────┼───────────┤")
    for row in summary_df.itertuples(index=False):
        config_name = str(row.config_name)
        params = format_param_count(int(row.total_params))
        time_per_epoch = float(row.avg_time_per_epoch_mean)
        mse_mean = float(row.test_mse_mean)
        mse_std = 0.0 if pd.isna(row.test_mse_std) else float(row.test_mse_std)
        r2_mean = float(row.test_r2_mean)
        r2_std = 0.0 if pd.isna(row.test_r2_std) else float(row.test_r2_std)
        conv_mean = float(row.convergence_epoch_mean)
        conv_std = 0.0 if pd.isna(row.convergence_epoch_std) else float(row.convergence_epoch_std)
        print(
            f"│ {config_name:<16} │ "
            f"{params:>6} │ "
            f"{time_per_epoch:>6.2f}s │ "
            f"{mse_mean:>6.4f} ± {mse_std:<6.4f} │ "
            f"{r2_mean:>6.4f} ± {r2_std:<6.4f} │ "
            f"{conv_mean:>5.1f} ± {conv_std:<4.1f} │"
        )
    print("└──────────────────┴────────┴──────────┴───────────────────┴──────────────────┴───────────┘")
    print()


def print_final_summary(
    summary_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    total_experiment_time: float,
    checkpoint_dir: str,
) -> None:
    """Print the final cross-seed summary and best-model report.

    Args:
        summary_df: Aggregated summary DataFrame.
        all_results_df: Run-level results DataFrame.
        total_experiment_time: Full wall-clock time for this script execution.
        checkpoint_dir: Directory containing saved checkpoints.
    """
    print("═" * 79)
    print("EXPERIMENT 1 RESULTS SUMMARY (averaged across 3 seeds)")
    print("═" * 79)
    print()

    kan_summary = summary_df[summary_df["model_type"] == "kan"].sort_values(
        by=["grid_size", "spline_order"]
    )
    mlp_summary = summary_df[summary_df["model_type"] == "mlp"].sort_values(
        by=["grid_size", "spline_order"]
    )

    print("KAN Models:")
    print_summary_block("", kan_summary)
    print("Matched MLP Models:")
    print_summary_block("", mlp_summary)

    best_kan = kan_summary.sort_values(by="test_r2_mean", ascending=False).iloc[0]
    best_mlp = mlp_summary.sort_values(by="test_r2_mean", ascending=False).iloc[0]
    winner = "KAN" if best_kan["test_r2_mean"] >= best_mlp["test_r2_mean"] else "MLP"
    margin = abs(float(best_kan["test_r2_mean"]) - float(best_mlp["test_r2_mean"]))

    print(
        f"Best KAN:  {best_kan['config_name']} | "
        f"R² = {best_kan['test_r2_mean']:.4f} ± {0.0 if pd.isna(best_kan['test_r2_std']) else best_kan['test_r2_std']:.4f} | "
        f"MSE = {best_kan['test_mse_mean']:.4f} ± {0.0 if pd.isna(best_kan['test_mse_std']) else best_kan['test_mse_std']:.4f}"
    )
    print(
        f"Best MLP:  matched to {best_mlp['config_name']} | "
        f"R² = {best_mlp['test_r2_mean']:.4f} ± {0.0 if pd.isna(best_mlp['test_r2_std']) else best_mlp['test_r2_std']:.4f} | "
        f"MSE = {best_mlp['test_mse_mean']:.4f} ± {0.0 if pd.isna(best_mlp['test_mse_std']) else best_mlp['test_mse_std']:.4f}"
    )
    print(f"Winner:    {winner} by R² margin of {margin:.4f}")
    minutes = int(total_experiment_time // 60)
    seconds = int(total_experiment_time % 60)
    print(f"Total experiment time: {minutes} minutes {seconds} seconds")

    best_kan_run = all_results_df[all_results_df["model_type"] == "kan"].sort_values(
        by="test_r2", ascending=False
    ).iloc[0]
    best_mlp_run = all_results_df[all_results_df["model_type"] == "mlp"].sort_values(
        by="test_r2", ascending=False
    ).iloc[0]

    best_kan_checkpoint = os.path.join(
        checkpoint_dir,
        f"kan_{best_kan_run['config_name']}_seed{int(best_kan_run['seed'])}_best.pt",
    )
    best_mlp_checkpoint = os.path.join(
        checkpoint_dir,
        f"mlp_{best_mlp_run['config_name']}_seed{int(best_mlp_run['seed'])}_best.pt",
    )
    print(f"Best KAN model saved at: {best_kan_checkpoint}")
    print(f"Best matched MLP model saved at: {best_mlp_checkpoint}")


def create_failure_result(
    model_type: str,
    config_name: str,
    grid_size: int | None,
    spline_order: int | None,
    hidden_dim: int,
    total_params: int,
    seed: int,
) -> dict[str, Any]:
    """Create a NaN-filled result row for a failed run.

    Args:
        model_type: Either ``kan`` or ``mlp``.
        config_name: Configuration identifier.
        grid_size: KAN grid size or ``None`` for MLP.
        spline_order: KAN spline order or ``None`` for MLP.
        hidden_dim: Model hidden dimension.
        total_params: Model parameter count.
        seed: Random seed for the run.

    Returns:
        Result dictionary with NaN metrics and empty histories.
    """
    nan = float("nan")
    return {
        "model_type": model_type,
        "grid_size": grid_size,
        "spline_order": spline_order,
        "config_name": config_name,
        "hidden_dim": hidden_dim,
        "total_params": total_params,
        "seed": seed,
        "test_mse": nan,
        "test_r2": nan,
        "best_val_loss": nan,
        "best_val_r2": nan,
        "best_epoch": nan,
        "epochs_trained": nan,
        "convergence_epoch_95": nan,
        "avg_time_per_epoch": nan,
        "total_training_time": nan,
        "train_loss_history": json.dumps([]),
        "val_loss_history": json.dumps([]),
        "val_r2_history": json.dumps([]),
    }


def run_single_experiment(
    model_type: str,
    config_entry: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run one training/evaluation job for Experiment 1.

    Args:
        model_type: ``kan`` or ``mlp``.
        config_entry: Sweep configuration entry.
        seed: Random seed for this run.
        config: Global experiment config.
        device: Training device.

    Returns:
        Run-level result dictionary.
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, info = get_california_housing(
        batch_size=int(config["dataset"]["batch_size"]),
        seed=seed,
        num_workers=0,
    )

    if model_type == "kan":
        hidden_dim = int(config_entry["kan_hidden_dim"])
        grid_size = int(config_entry["grid_size"])
        spline_order = int(config_entry["spline_order"])
        config_name = str(config_entry["config_name"])
        model = KANRegressor(
            in_features=int(info["n_features"]),
            hidden_dim=hidden_dim,
            out_features=1,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        total_params = count_parameters(model)
        checkpoint_stub = f"kan_{config_name}_seed{seed}"
    else:
        hidden_dim = int(config_entry["mlp_hidden_dim"])
        grid_size = None
        spline_order = None
        config_name = f"{config_entry['config_name']}_matched"
        model = MLPRegressor(
            in_features=int(info["n_features"]),
            hidden_dim=hidden_dim,
            out_features=1,
            dropout=float(config["mlp"]["dropout"]),
        )
        total_params = count_parameters(model)
        checkpoint_stub = f"mlp_{config_name}_seed{seed}"

    model = model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=int(config["training"]["epochs"]),
        device=device,
        metric_fn=r2_metric,
        scheduler=scheduler,
        early_stopping_patience=int(config["training"]["early_stopping_patience"]),
        checkpoint_dir=str(config["checkpoint_dir"]),
        model_name=checkpoint_stub,
        verbose=False,
    )

    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        metric_fn=r2_metric,
    )
    convergence_epoch = compute_convergence_epoch(history, threshold_fraction=0.95)

    result = {
        "model_type": model_type,
        "grid_size": grid_size,
        "spline_order": spline_order,
        "config_name": config_name,
        "hidden_dim": hidden_dim,
        "total_params": total_params,
        "seed": seed,
        "test_mse": float(test_results["test_loss"]),
        "test_r2": float(test_results["test_metric"]),
        "best_val_loss": float(history["best_val_loss"]),
        "best_val_r2": (
            float(history["best_val_metric"]) if history["best_val_metric"] is not None else float("nan")
        ),
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(history["epochs_trained"]),
        "convergence_epoch_95": int(convergence_epoch),
        "avg_time_per_epoch": float(np.mean(history["time_per_epoch"])),
        "total_training_time": float(history["total_time"]),
        "train_loss_history": serialize_history_list(history["train_loss"]),
        "val_loss_history": serialize_history_list(history["val_loss"]),
        "val_r2_history": serialize_history_list(history.get("val_metric", [])),
    }

    return result


def main() -> None:
    """Run the full Experiment 1 regression sweep."""
    script_start = time.time()

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    print_experiment_header(config)
    configurations = build_configuration_table(config)
    print_configuration_table(configurations)

    existing_results_df = load_existing_results(ALL_RESULTS_PATH)
    existing_results = (
        existing_results_df.to_dict(orient="records") if not existing_results_df.empty else []
    )
    completed_keys = get_completed_keys(existing_results_df)

    total_runs = len(configurations) * len(config["seeds"]) * 2
    if completed_keys:
        print(f"Resuming from previous run: {len(completed_keys)}/{total_runs} runs already complete")
    else:
        print(f"Starting fresh run: 0/{total_runs} runs complete")

    device = get_device()
    all_results: list[dict[str, Any]] = list(existing_results)
    run_counter = len(completed_keys)

    for config_entry in configurations:
        for seed in config["seeds"]:
            for model_type in ("kan", "mlp"):
                config_name = (
                    str(config_entry["config_name"])
                    if model_type == "kan"
                    else f"{config_entry['config_name']}_matched"
                )
                run_key = (model_type, config_name, int(seed))
                if run_key in completed_keys:
                    print(f"Skipping completed run: {model_type} {config_name} seed={seed}")
                    continue

                run_counter += 1
                run_start = time.time()
                try:
                    result = run_single_experiment(
                        model_type=model_type,
                        config_entry=config_entry,
                        seed=int(seed),
                        config=config,
                        device=device,
                    )
                    print(
                        f"[{run_counter}/{total_runs}] "
                        f"{model_type.upper()} G={config_entry['grid_size']} "
                        f"K={config_entry['spline_order']} seed={seed} | "
                        f"Test MSE: {result['test_mse']:.4f} | "
                        f"Test R²: {result['test_r2']:.4f} | "
                        f"Epochs: {result['epochs_trained']}/{config['training']['epochs']} | "
                        f"Time: {time.time() - run_start:.1f}s"
                    )
                except Exception as exc:
                    print(
                        f"Warning: run failed for {model_type} {config_name} seed={seed}: {exc}"
                    )
                    if model_type == "kan":
                        hidden_dim = int(config_entry["kan_hidden_dim"])
                        total_params = int(config_entry["kan_params"])
                        grid_size = int(config_entry["grid_size"])
                        spline_order = int(config_entry["spline_order"])
                    else:
                        hidden_dim = int(config_entry["mlp_hidden_dim"])
                        total_params = int(config_entry["mlp_params"])
                        grid_size = None
                        spline_order = None
                    result = create_failure_result(
                        model_type=model_type,
                        config_name=config_name,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        hidden_dim=hidden_dim,
                        total_params=total_params,
                        seed=int(seed),
                    )

                all_results.append(result)
                completed_keys.add(run_key)
                save_all_results(all_results, ALL_RESULTS_PATH)
                results_df = pd.DataFrame(all_results)
                summary_df = build_summary_dataframe(results_df)
                summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    all_results_df = pd.DataFrame(all_results)
    summary_df = build_summary_dataframe(all_results_df)
    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    print()
    print_final_summary(
        summary_df=summary_df,
        all_results_df=all_results_df,
        total_experiment_time=time.time() - script_start,
        checkpoint_dir=str(config["checkpoint_dir"]),
    )
    print()
    print(f"Saved run-level results to: {ALL_RESULTS_PATH}")
    print(f"Saved summary results to: {SUMMARY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
