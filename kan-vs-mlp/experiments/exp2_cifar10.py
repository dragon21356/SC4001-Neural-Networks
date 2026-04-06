"""Experiment 2 CIFAR-10 classification sweep for CNN + swappable heads."""

from __future__ import annotations

import json
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

from src.data_utils import get_cifar10
from src.evaluate import accuracy_metric
from src.evaluate import compute_convergence_epoch
from src.evaluate import evaluate_model
from src.evaluate import per_class_accuracy
from src.models import build_cifar10_model
from src.models import count_head_parameters
from src.models import count_parameters
from src.train import train_model
from src.utils import format_param_count
from src.utils import get_device
from src.utils import save_results_csv
from src.utils import set_seed

CONFIG_PATH = os.path.join("experiments", "configs", "exp2_config.yaml")
DIAGNOSTIC_HPARAMS_PATH = os.path.join(
    "results",
    "exp2",
    "diagnostic",
    "recommended_hyperparams.yaml",
)
ALL_RESULTS_PATH = os.path.join("results", "exp2", "exp2_all_results.csv")
SUMMARY_RESULTS_PATH = os.path.join("results", "exp2", "exp2_summary.csv")


def load_config(config_path: str) -> dict[str, Any]:
    """Load the Experiment 2 YAML configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find Experiment 2 config at: {config_path}. "
            "Expected to load experiments/configs/exp2_config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_diagnostic_hyperparams(filepath: str) -> dict[str, Any] | None:
    """Load diagnostic hyperparameter recommendations when available.

    Args:
        filepath: Path to the diagnostic YAML.

    Returns:
        Parsed recommendation dictionary, or ``None`` if the file is missing.
    """
    if not os.path.exists(filepath):
        print(
            "Note: diagnostic hyperparameter file not found at "
            f"{filepath}. Proceeding with the learning rates already stored in the config."
        )
        return None

    with open(filepath, "r", encoding="utf-8") as handle:
        recommendations = yaml.safe_load(handle)
    print(f"Loaded diagnostic recommendations from: {filepath}")
    return recommendations


def print_experiment_header(config: dict[str, Any]) -> None:
    """Print the Experiment 2 header and runtime estimate.

    Args:
        config: Experiment config dictionary.
    """
    total_runs = len(config["models"]) * len(config["seeds"])
    estimated_seconds_per_epoch = 100
    estimated_total_seconds = total_runs * config["training"]["epochs"] * estimated_seconds_per_epoch
    estimated_hours = estimated_total_seconds / 3600.0

    print("=" * 79)
    print("EXPERIMENT 2: CIFAR-10 CLASSIFICATION — CNN + SWAPPABLE HEADS")
    print("=" * 79)
    print(json.dumps(config, indent=2))
    print(
        "Estimated runtime: "
        f"{total_runs} runs x ~{estimated_seconds_per_epoch}s/epoch x "
        f"{config['training']['epochs']} epochs = ~{estimated_hours:.1f} hours on CPU"
    )
    print("Tip: Use GPU or NTU cluster to reduce this to ~5-6 hours")
    print()


def instantiate_model_from_config(model_config: dict[str, Any]) -> torch.nn.Module:
    """Build a CIFAR-10 model from one model config entry.

    Args:
        model_config: Single model configuration entry.

    Returns:
        Instantiated CNN classifier.
    """
    return build_cifar10_model(
        head_type=str(model_config["head_type"]),
        hidden_dim=int(model_config["hidden_dim"]),
        grid_size=int(model_config.get("grid_size", 5)),
        spline_order=int(model_config.get("spline_order", 3)),
        dropout=float(model_config.get("dropout", 0.3)),
    )


def build_configuration_table(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Instantiate each configured model once to record exact parameter counts.

    Args:
        config: Experiment config dictionary.

    Returns:
        List of enriched model configuration dictionaries.
    """
    table_rows: list[dict[str, Any]] = []
    for model_config in config["models"]:
        model = instantiate_model_from_config(model_config)
        total_params = count_parameters(model)
        head_params = count_head_parameters(model)
        table_rows.append(
            {
                "name": str(model_config["name"]),
                "head_type": str(model_config["head_type"]),
                "mode": str(model_config["mode"]),
                "hidden_dim": int(model_config["hidden_dim"]),
                "learning_rate": float(model_config["learning_rate"]),
                "dropout": float(model_config.get("dropout", 0.3)),
                "grid_size": int(model_config.get("grid_size", 5)),
                "spline_order": int(model_config.get("spline_order", 3)),
                "total_params": int(total_params),
                "head_params": int(head_params),
            }
        )
        del model
    return table_rows


def print_configuration_table(table_rows: list[dict[str, Any]]) -> None:
    """Print the Experiment 2 configuration table.

    Args:
        table_rows: Enriched configuration rows with parameter counts.
    """
    print("Experiment 2 Configuration")
    print("═" * 75)
    print("┌──────────────────┬─────────────┬──────┬──────────────┬────────────┬─────────┐")
    print("│ Name             │ Head Type   │ Mode │ Total Params │ Head Params│ LR      │")
    print("├──────────────────┼─────────────┼──────┼──────────────┼────────────┼─────────┤")
    for row in table_rows:
        print(
            f"│ {row['name']:<16} │ "
            f"{row['head_type']:<11} │ "
            f"{row['mode']:^4} │ "
            f"{format_param_count(row['total_params']):>12} │ "
            f"{format_param_count(row['head_params']):>10} │ "
            f"{row['learning_rate']:>7.1e} │"
        )
    print("└──────────────────┴─────────────┴──────┴──────────────┴────────────┴─────────┘")
    print()
    print("Mode A: Same hidden width (256), different parameter counts")
    print("  -> Tests whether KAN's extra expressiveness justifies its parameter cost")
    print("Mode B: Approximately matched parameters (~134-146K head params)")
    print("  -> Tests which architecture makes best use of a fixed parameter budget")
    print("═" * 75)
    print()


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


def get_completed_keys(results_df: pd.DataFrame) -> set[tuple[str, int]]:
    """Extract completed run identifiers from existing results.

    Args:
        results_df: Existing results DataFrame.

    Returns:
        Set of ``(name, seed)`` tuples.
    """
    required_columns = {"name", "seed"}
    if results_df.empty or not required_columns.issubset(results_df.columns):
        return set()

    completed: set[tuple[str, int]] = set()
    for row in results_df.itertuples(index=False):
        completed.add((str(row.name), int(row.seed)))
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


def serialize_per_class_accuracy(per_class: dict[str, float] | None) -> str:
    """Encode per-class accuracy as JSON.

    Args:
        per_class: Per-class accuracy mapping.

    Returns:
        JSON string.
    """
    if per_class is None:
        return json.dumps({})
    return json.dumps({str(key): float(value) for key, value in per_class.items()})


def convert_results_to_columns(results: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row-oriented results into a column-oriented dictionary.

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
                "name",
                "head_type",
                "mode",
                "hidden_dim",
                "total_params",
                "head_params",
                "test_accuracy_mean",
                "test_accuracy_std",
                "test_loss_mean",
                "test_loss_std",
                "convergence_epoch_mean",
                "convergence_epoch_std",
                "avg_time_per_epoch_mean",
                "total_training_time_mean",
            ]
        )

    summary = (
        results_df.groupby(
            ["name", "head_type", "mode", "hidden_dim", "total_params", "head_params"],
            dropna=False,
        )
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            convergence_epoch_mean=("convergence_epoch_95", "mean"),
            convergence_epoch_std=("convergence_epoch_95", "std"),
            avg_time_per_epoch_mean=("avg_time_per_epoch", "mean"),
            total_training_time_mean=("total_training_time", "mean"),
        )
        .reset_index()
    )
    return summary


def create_failure_result(
    model_row: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Create a NaN-filled result row for a failed run.

    Args:
        model_row: Enriched model configuration row.
        seed: Random seed for the run.

    Returns:
        Result dictionary with NaN metrics and empty histories.
    """
    nan = float("nan")
    return {
        "name": str(model_row["name"]),
        "head_type": str(model_row["head_type"]),
        "mode": str(model_row["mode"]),
        "hidden_dim": int(model_row["hidden_dim"]),
        "total_params": int(model_row["total_params"]),
        "head_params": int(model_row["head_params"]),
        "learning_rate": float(model_row["learning_rate"]),
        "seed": int(seed),
        "test_loss": nan,
        "test_accuracy": nan,
        "best_val_loss": nan,
        "best_val_accuracy": nan,
        "best_epoch": nan,
        "epochs_trained": nan,
        "convergence_epoch_95": nan,
        "avg_time_per_epoch": nan,
        "total_training_time": nan,
        "per_class_accuracy": json.dumps({}),
        "train_loss_history": json.dumps([]),
        "val_loss_history": json.dumps([]),
        "val_accuracy_history": json.dumps([]),
    }


def run_single_experiment(
    model_row: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run one training/evaluation job for Experiment 2.

    Args:
        model_row: Enriched model configuration row.
        seed: Random seed for this run.
        config: Global experiment config.
        device: Training device.

    Returns:
        Run-level result dictionary.
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, info = get_cifar10(
        batch_size=int(config["dataset"]["batch_size"]),
        data_dir=str(config["dataset"]["data_dir"]),
        num_workers=int(config["dataset"]["num_workers"]),
        seed=seed,
    )

    model = build_cifar10_model(
        head_type=str(model_row["head_type"]),
        hidden_dim=int(model_row["hidden_dim"]),
        grid_size=int(model_row["grid_size"]),
        spline_order=int(model_row["spline_order"]),
        dropout=float(model_row["dropout"]),
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(model_row["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_stub = f"{model_row['name']}_seed{seed}"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=int(config["training"]["epochs"]),
        device=device,
        metric_fn=accuracy_metric,
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
        metric_fn=accuracy_metric,
    )
    class_accuracy = per_class_accuracy(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=list(info["class_names"]),
    )
    convergence_epoch = compute_convergence_epoch(history, threshold_fraction=0.95)

    result = {
        "name": str(model_row["name"]),
        "head_type": str(model_row["head_type"]),
        "mode": str(model_row["mode"]),
        "hidden_dim": int(model_row["hidden_dim"]),
        "total_params": int(model_row["total_params"]),
        "head_params": int(model_row["head_params"]),
        "learning_rate": float(model_row["learning_rate"]),
        "seed": int(seed),
        "test_loss": float(test_results["test_loss"]),
        "test_accuracy": float(test_results["test_metric"]),
        "best_val_loss": float(history["best_val_loss"]),
        "best_val_accuracy": (
            float(history["best_val_metric"])
            if history["best_val_metric"] is not None
            else float("nan")
        ),
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(history["epochs_trained"]),
        "convergence_epoch_95": int(convergence_epoch),
        "avg_time_per_epoch": float(np.mean(history["time_per_epoch"])),
        "total_training_time": float(history["total_time"]),
        "per_class_accuracy": serialize_per_class_accuracy(class_accuracy),
        "train_loss_history": serialize_history_list(history["train_loss"]),
        "val_loss_history": serialize_history_list(history["val_loss"]),
        "val_accuracy_history": serialize_history_list(history.get("val_metric", [])),
    }
    return result


def print_mode_summary(
    title: str,
    summary_df: pd.DataFrame,
) -> None:
    """Print one mode-level summary block.

    Args:
        title: Block title.
        summary_df: Summary rows for the mode.
    """
    print(title)
    print("┌──────────────────┬──────────────┬────────────┬─────────────────────┬──────────────────────┬───────────┐")
    print("│ Model            │ Total Params │ Head Params│ Test Accuracy       │ Test Loss            │ Time/Ep   │")
    print("├──────────────────┼──────────────┼────────────┼─────────────────────┼──────────────────────┼───────────┤")
    for row in summary_df.itertuples(index=False):
        test_acc_std = 0.0 if pd.isna(row.test_accuracy_std) else float(row.test_accuracy_std)
        test_loss_std = 0.0 if pd.isna(row.test_loss_std) else float(row.test_loss_std)
        print(
            f"│ {str(row.name):<16} │ "
            f"{format_param_count(int(row.total_params)):>12} │ "
            f"{format_param_count(int(row.head_params)):>10} │ "
            f"{float(row.test_accuracy_mean):>6.4f} ± {test_acc_std:<6.4f} │ "
            f"{float(row.test_loss_mean):>6.4f} ± {test_loss_std:<6.4f} │ "
            f"{float(row.avg_time_per_epoch_mean):>6.1f}s │"
        )
    print("└──────────────────┴──────────────┴────────────┴─────────────────────┴──────────────────────┴───────────┘")
    print()


def get_best_summary_row(summary_df: pd.DataFrame, mode: str) -> pd.Series:
    """Get the best model summary row for a given mode.

    Args:
        summary_df: Summary results DataFrame.
        mode: Mode identifier, ``A`` or ``B``.

    Returns:
        Best summary row by mean test accuracy.
    """
    mode_rows = summary_df[summary_df["mode"] == mode].copy()
    if mode_rows.empty:
        raise ValueError(f"No summary rows found for mode {mode}.")
    return mode_rows.sort_values(by="test_accuracy_mean", ascending=False).iloc[0]


def get_best_checkpoint_for_name(
    all_results_df: pd.DataFrame,
    model_name: str,
    checkpoint_dir: str,
) -> str:
    """Find the best saved checkpoint path for a summary-level model name.

    Args:
        all_results_df: Run-level results DataFrame.
        model_name: Model name to filter on.
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Best checkpoint path for the given model name.
    """
    model_runs = all_results_df[all_results_df["name"] == model_name].copy()
    if model_runs.empty:
        return "N/A"
    best_run = model_runs.sort_values(by="test_accuracy", ascending=False).iloc[0]
    return os.path.join(checkpoint_dir, f"{model_name}_seed{int(best_run['seed'])}_best.pt")


def print_final_summary(
    summary_df: pd.DataFrame,
    all_results_df: pd.DataFrame,
    total_experiment_time: float,
    checkpoint_dir: str,
) -> None:
    """Print the final Experiment 2 summary.

    Args:
        summary_df: Aggregated summary DataFrame.
        all_results_df: Run-level results DataFrame.
        total_experiment_time: Full wall-clock time for this script execution.
        checkpoint_dir: Directory containing saved checkpoints.
    """
    print("═" * 75)
    print("EXPERIMENT 2 RESULTS SUMMARY (averaged across 3 seeds)")
    print("═" * 75)
    print()

    mode_a = summary_df[summary_df["mode"] == "A"].sort_values(by="name")
    mode_b = summary_df[summary_df["mode"] == "B"].sort_values(by="name")

    print("Mode A — Same Width (hidden=256), Different Parameters:")
    print_mode_summary("", mode_a)
    print("Mode B — Matched Parameters (~134-146K head params):")
    print_mode_summary("", mode_b)

    best_mode_a = get_best_summary_row(summary_df, "A")
    best_mode_b = get_best_summary_row(summary_df, "B")
    mlp_mode_b = mode_b[mode_b["name"] == "MLP_ModeB"].iloc[0]
    kan_mode_b = mode_b[mode_b["name"] == "KAN_ModeB"].iloc[0]
    bspline_mode_b = mode_b[mode_b["name"] == "BSplineMLP_ModeB"].iloc[0]
    mlp_mode_a = mode_a[mode_a["name"] == "MLP_ModeA"].iloc[0]
    kan_mode_a = mode_a[mode_a["name"] == "KAN_ModeA"].iloc[0]

    bspline_vs_mlp_margin = float(bspline_mode_b["test_accuracy_mean"]) - float(mlp_mode_b["test_accuracy_mean"])
    kan_vs_mlp_mode_b_margin = float(kan_mode_b["test_accuracy_mean"]) - float(mlp_mode_b["test_accuracy_mean"])
    kan_vs_mlp_mode_a_margin = float(kan_mode_a["test_accuracy_mean"]) - float(mlp_mode_a["test_accuracy_mean"])

    print("Key Findings:")
    print(
        f"  Mode A winner: {best_mode_a['name']} "
        f"(accuracy {float(best_mode_a['test_accuracy_mean']):.4f} ± "
        f"{0.0 if pd.isna(best_mode_a['test_accuracy_std']) else float(best_mode_a['test_accuracy_std']):.4f})"
    )
    print(
        f"  Mode B winner: {best_mode_b['name']} "
        f"(accuracy {float(best_mode_b['test_accuracy_mean']):.4f} ± "
        f"{0.0 if pd.isna(best_mode_b['test_accuracy_std']) else float(best_mode_b['test_accuracy_std']):.4f})"
    )
    print(
        f"  BSpline-MLP vs MLP (Mode B): "
        f"{'BSpline' if bspline_vs_mlp_margin >= 0 else 'MLP'} wins by {abs(bspline_vs_mlp_margin):.4f}"
    )
    print(
        "    -> Learnable activations "
        f"{'do' if bspline_vs_mlp_margin >= 0 else 'do not'} provide a benefit over fixed ReLU"
    )
    print(
        f"  KAN vs MLP (Mode B): {'KAN' if kan_vs_mlp_mode_b_margin >= 0 else 'MLP'} "
        f"wins by {abs(kan_vs_mlp_mode_b_margin):.4f}"
    )
    print(
        "    -> Full KAN redesign "
        f"{'does' if kan_vs_mlp_mode_b_margin >= 0 else 'does not'} justify itself at equal parameter budget"
    )
    print(
        f"  KAN Mode A vs MLP Mode A: {'KAN' if kan_vs_mlp_mode_a_margin >= 0 else 'MLP'} "
        f"wins by {abs(kan_vs_mlp_mode_a_margin):.4f}"
    )
    print(
        "    -> Extra KAN parameters "
        f"{'do' if kan_vs_mlp_mode_a_margin >= 0 else 'do not'} translate to accuracy gains"
    )
    print()

    total_hours = int(total_experiment_time // 3600)
    total_minutes = int((total_experiment_time % 3600) // 60)
    print(f"Total experiment time: {total_hours} hours {total_minutes} minutes")
    print("═" * 75)

    best_mode_a_checkpoint = get_best_checkpoint_for_name(
        all_results_df=all_results_df,
        model_name=str(best_mode_a["name"]),
        checkpoint_dir=checkpoint_dir,
    )
    best_mode_b_checkpoint = get_best_checkpoint_for_name(
        all_results_df=all_results_df,
        model_name=str(best_mode_b["name"]),
        checkpoint_dir=checkpoint_dir,
    )
    print()
    print(
        f"Best model (Mode A): {best_mode_a['name']} — checkpoint at {best_mode_a_checkpoint}"
    )
    print(
        f"Best model (Mode B): {best_mode_b['name']} — checkpoint at {best_mode_b_checkpoint}"
    )


def main() -> None:
    """Run the full Experiment 2 CIFAR-10 sweep."""
    script_start = time.time()

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    print_experiment_header(config)
    _ = load_diagnostic_hyperparams(DIAGNOSTIC_HPARAMS_PATH)

    table_rows = build_configuration_table(config)
    print_configuration_table(table_rows)

    existing_results_df = load_existing_results(ALL_RESULTS_PATH)
    existing_results = (
        existing_results_df.to_dict(orient="records") if not existing_results_df.empty else []
    )
    completed_keys = get_completed_keys(existing_results_df)
    total_runs = len(table_rows) * len(config["seeds"])

    if completed_keys:
        print(f"Resuming from previous run: {len(completed_keys)}/{total_runs} runs already complete")
    else:
        print(f"Starting fresh run: 0/{total_runs} runs complete")
    print()

    device = get_device()
    all_results: list[dict[str, Any]] = list(existing_results)
    run_counter = len(completed_keys)

    for model_row in table_rows:
        for seed in config["seeds"]:
            run_key = (str(model_row["name"]), int(seed))
            if run_key in completed_keys:
                print(f"Skipping completed run: {model_row['name']} seed={seed}")
                continue

            run_counter += 1
            run_start = time.time()
            try:
                result = run_single_experiment(
                    model_row=model_row,
                    seed=int(seed),
                    config=config,
                    device=device,
                )
                elapsed_minutes = (time.time() - run_start) / 60.0
                print(
                    f"[{run_counter:02d}/{total_runs}] {model_row['name']} seed={seed} | "
                    f"Test Acc: {result['test_accuracy']:.4f} | "
                    f"Best Val Acc: {result['best_val_accuracy']:.4f} | "
                    f"Epochs: {result['epochs_trained']}/{config['training']['epochs']} | "
                    f"Time: {elapsed_minutes:.1f}min"
                )
            except Exception as exc:
                print(f"Warning: run failed for {model_row['name']} seed={seed}: {exc}")
                result = create_failure_result(model_row=model_row, seed=int(seed))

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
