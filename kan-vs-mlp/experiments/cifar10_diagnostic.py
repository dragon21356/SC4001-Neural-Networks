"""CIFAR-10 diagnostic script for pipeline validation and LR selection."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import get_cifar10
from src.evaluate import accuracy_metric
from src.models import build_cifar10_model
from src.models import count_head_parameters
from src.models import count_parameters
from src.train import train_model
from src.utils import BSPLINE_COLOR
from src.utils import KAN_COLOR
from src.utils import MLP_COLOR
from src.utils import format_param_count
from src.utils import get_device
from src.utils import save_results_csv
from src.utils import set_seed

OUTPUT_DIR = os.path.join("results", "exp2", "diagnostic")
PIPELINE_CURVES_PATH = os.path.join(OUTPUT_DIR, "pipeline_validation_curves.png")
KAN_LR_SWEEP_PATH = os.path.join(OUTPUT_DIR, "kan_lr_sweep.png")
LR_SWEEP_RESULTS_PATH = os.path.join(OUTPUT_DIR, "lr_sweep_results.csv")
RECOMMENDED_HPARAMS_PATH = os.path.join(OUTPUT_DIR, "recommended_hyperparams.yaml")


def ensure_output_dir() -> None:
    """Create the diagnostic output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_part_header(title: str, estimate: str) -> None:
    """Print a formatted part header.

    Args:
        title: Title of the diagnostic section.
        estimate: Runtime estimate string.
    """
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"Estimated runtime: {estimate}")


def history_has_nan(history: dict[str, Any]) -> bool:
    """Check whether a training history contains NaN or inf values.

    Args:
        history: History dictionary returned by ``train_model``.

    Returns:
        Whether any tracked loss or metric is non-finite.
    """
    for key in ("train_loss", "val_loss", "val_metric"):
        values = history.get(key, [])
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return True
            if not np.isfinite(numeric):
                return True
    return False


def cleanup_model(model: torch.nn.Module | None) -> None:
    """Release model memory between sequential diagnostic runs.

    Args:
        model: Model to clean up.
    """
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_cifar_model(
    *,
    name: str,
    head_type: str,
    hidden_dim: int,
    lr: float,
    epochs: int,
    device: torch.device,
    batch_size: int = 128,
    grid_size: int = 5,
    spline_order: int = 3,
    dropout: float = 0.3,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train a single CIFAR-10 model for diagnostic purposes.

    Args:
        name: Human-readable run name.
        head_type: Model head type passed to ``build_cifar10_model``.
        hidden_dim: Hidden dimension for the head.
        lr: Learning rate.
        epochs: Number of training epochs.
        device: Device on which to train.
        batch_size: CIFAR-10 batch size.
        grid_size: KAN/BSpline grid size.
        spline_order: KAN/BSpline spline order.
        dropout: Dropout for MLP-style heads.
        verbose: Whether to show per-epoch logging from ``train_model``.

    Returns:
        Dictionary containing the model metadata and training history.
    """
    set_seed(42)
    train_loader, val_loader, _, info = get_cifar10(
        batch_size=batch_size,
        data_dir="./data",
        num_workers=0,
    )

    model = build_cifar10_model(
        head_type=head_type,
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        dropout=dropout,
    ).to(device)
    total_params = count_parameters(model)
    head_params = count_head_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    set_seed(42)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        metric_fn=accuracy_metric,
        scheduler=None,
        early_stopping_patience=0,
        checkpoint_dir=None,
        model_name=name,
        verbose=verbose,
    )

    status = "ok"
    if history_has_nan(history):
        status = "nan"

    result = {
        "name": name,
        "head_type": head_type,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "epochs": epochs,
        "total_params": total_params,
        "head_params": head_params,
        "final_val_acc": float(history["val_metric"][-1]) if history.get("val_metric") else float("nan"),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else float("nan"),
        "avg_time_per_epoch": float(np.mean(history["time_per_epoch"])) if history["time_per_epoch"] else float("nan"),
        "history": history,
        "status": status,
        "n_classes": info["n_classes"],
    }

    cleanup_model(model)
    return result


def print_pipeline_validation_table(results: list[dict[str, Any]]) -> None:
    """Print the Part 1 comparison table.

    Args:
        results: List of validation run dictionaries.
    """
    print("CIFAR-10 Pipeline Validation (10 epochs)")
    print("┌──────────────────────┬──────────┬────────────┬──────────────┬──────────────┬──────────┐")
    print("│ Model                │ Params   │ Head Params │ Val Acc (ep10)│ Train Loss   │ Time/Ep  │")
    print("├──────────────────────┼──────────┼────────────┼──────────────┼──────────────┼──────────┤")
    for row in results:
        print(
            f"│ {row['name']:<20} │ "
            f"{format_param_count(int(row['total_params'])):>8} │ "
            f"{format_param_count(int(row['head_params'])):>10} │ "
            f"{row['final_val_acc']:>12.4f} │ "
            f"{row['final_train_loss']:>12.4f} │ "
            f"{row['avg_time_per_epoch']:>6.1f}s │"
        )
    print("└──────────────────────┴──────────┴────────────┴──────────────┴──────────────┴──────────┘")


def print_pipeline_checks(results: list[dict[str, Any]]) -> dict[str, bool]:
    """Print PASS/FAIL checks for pipeline validation.

    Args:
        results: List of validation run dictionaries.

    Returns:
        Mapping from model name to validation pass flag.
    """
    status_map: dict[str, bool] = {}
    print("\nValidation checks:")
    for row in results:
        acc_ok = row["final_val_acc"] > 0.50
        loss_hist = row["history"]["train_loss"]
        loss_ok = bool(loss_hist) and loss_hist[-1] < loss_hist[0]
        nan_ok = row["status"] == "ok"
        passed = acc_ok and loss_ok and nan_ok
        status_map[row["name"]] = passed
        print(
            f"  - {row['name']}: {'PASS' if passed else 'FAIL'} | "
            f"acc>0.50={acc_ok} | loss↓={loss_ok} | finite={nan_ok}"
        )
    return status_map


def save_pipeline_validation_plot(results: list[dict[str, Any]]) -> None:
    """Plot validation accuracy curves for the three pipeline-validation models.

    Args:
        results: List of validation run dictionaries.
    """
    color_map = {
        "CNN+MLPHead": MLP_COLOR,
        "CNN+KANHead_ModeB": KAN_COLOR,
        "CNN+BSplineMLPHead": BSPLINE_COLOR,
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in results:
        ax.plot(
            range(1, len(row["history"]["val_metric"]) + 1),
            row["history"]["val_metric"],
            label=row["name"],
            color=color_map[row["name"]],
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("CIFAR-10 Pipeline Validation: Val Accuracy (10 epochs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PIPELINE_CURVES_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def is_better_lr(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    """Return whether a candidate LR run is better under the selection rule.

    Args:
        candidate: Candidate LR result.
        current_best: Current best LR result.

    Returns:
        Whether the candidate should replace the current best.
    """
    if current_best is None:
        return True
    if candidate["status"] != "ok":
        return False
    if current_best["status"] != "ok":
        return True

    candidate_acc = float(candidate["final_val_acc"])
    current_acc = float(current_best["final_val_acc"])
    if candidate_acc > current_acc + 0.005:
        return True
    if abs(candidate_acc - current_acc) <= 0.005 and float(candidate["lr"]) < float(current_best["lr"]):
        return True
    return False


def select_recommended_lr(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pick the best and runner-up LR based on final validation accuracy.

    Args:
        results: LR sweep results for one model family.

    Returns:
        Tuple of ``(best_result, runner_up_result_or_none)``.
    """
    valid_results = [row for row in results if row["status"] == "ok" and np.isfinite(row["final_val_acc"])]
    if not valid_results:
        fallback = sorted(results, key=lambda row: float(row["lr"]))[0]
        return fallback, None

    best: dict[str, Any] | None = None
    for row in sorted(valid_results, key=lambda item: float(item["lr"])):
        if is_better_lr(row, best):
            best = row

    assert best is not None
    remaining = [row for row in valid_results if row is not best]
    runner_up = None
    if remaining:
        runner_up = sorted(
            remaining,
            key=lambda item: (-float(item["final_val_acc"]), float(item["lr"])),
        )[0]
    return best, runner_up


def save_kan_lr_sweep_plot(
    mode_b_results: list[dict[str, Any]],
    mode_a_results: list[dict[str, Any]],
) -> None:
    """Save the side-by-side KAN LR sweep figure.

    Args:
        mode_b_results: Sweep results for KAN Mode B.
        mode_a_results: Sweep results for KAN Mode A.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title, rows, cmap_name in (
        (axes[0], "KAN Head Mode B LR Sweep", mode_b_results, "viridis"),
        (axes[1], "KAN Head Mode A LR Sweep", mode_a_results, "plasma"),
    ):
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(np.linspace(0.1, 0.9, len(rows)))
        for color, row in zip(colors, rows, strict=True):
            ax.plot(
                range(1, len(row["history"]["val_metric"]) + 1),
                row["history"]["val_metric"],
                label=f"LR={row['lr']:.0e}",
                linewidth=2,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(KAN_LR_SWEEP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_pipeline_validation(device: torch.device) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Run the 10-epoch pipeline validation for all three head types.

    Args:
        device: Training device.

    Returns:
        Tuple of validation results and pass/fail flags.
    """
    print_part_header(
        "PART 1: PIPELINE VALIDATION",
        "~45-90 minutes on GPU, ~3-5 hours on CPU for the full diagnostic script",
    )
    set_seed(42)
    _ = get_cifar10(batch_size=128, data_dir="./data", num_workers=0)

    models_to_test = [
        {"name": "CNN+MLPHead", "head_type": "mlp", "hidden_dim": 256, "lr": 1e-3},
        {"name": "CNN+KANHead_ModeB", "head_type": "kan", "hidden_dim": 28, "lr": 5e-4},
        {"name": "CNN+BSplineMLPHead", "head_type": "bspline_mlp", "hidden_dim": 256, "lr": 1e-3},
    ]

    results: list[dict[str, Any]] = []
    for config in models_to_test:
        result = train_cifar_model(
            name=config["name"],
            head_type=config["head_type"],
            hidden_dim=int(config["hidden_dim"]),
            lr=float(config["lr"]),
            epochs=10,
            device=device,
            verbose=True,
        )
        print(
            f"{result['name']} | Total Params: {format_param_count(result['total_params'])} | "
            f"Head Params: {format_param_count(result['head_params'])} | "
            f"Final Val Acc: {result['final_val_acc']:.4f} | "
            f"Final Train Loss: {result['final_train_loss']:.4f}"
        )
        results.append(result)

    print()
    print_pipeline_validation_table(results)
    status_map = print_pipeline_checks(results)
    save_pipeline_validation_plot(results)
    return results, status_map


def run_kan_lr_sweep(device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the KAN head LR sweeps for Mode B and Mode A.

    Args:
        device: Training device.

    Returns:
        Mode B results, Mode A results, and a summary dictionary.
    """
    print_part_header(
        "PART 2: KAN HEAD LEARNING RATE SWEEP",
        "~25-50 minutes on GPU, ~2-3 hours on CPU depending on KAN speed",
    )

    lr_candidates_mode_b = [2e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5]
    lr_candidates_mode_a = [1e-3, 5e-4, 3e-4, 1e-4]

    mode_b_results: list[dict[str, Any]] = []
    for lr in lr_candidates_mode_b:
        result = train_cifar_model(
            name=f"KAN_ModeB_LR_{lr:.0e}",
            head_type="kan",
            hidden_dim=28,
            lr=float(lr),
            epochs=15,
            device=device,
            verbose=False,
        )
        print(
            f"LR={lr:.0e} | Final Val Acc: {result['final_val_acc']:.4f} | "
            f"Final Train Loss: {result['final_train_loss']:.4f} | "
            f"Time/Ep: {result['avg_time_per_epoch']:.1f}s | "
            f"Status: {'OK' if result['status'] == 'ok' else 'NaN/Diverged'}"
        )
        mode_b_results.append(result)

    print()

    mode_a_results: list[dict[str, Any]] = []
    for lr in lr_candidates_mode_a:
        result = train_cifar_model(
            name=f"KAN_ModeA_LR_{lr:.0e}",
            head_type="kan",
            hidden_dim=256,
            lr=float(lr),
            epochs=10,
            device=device,
            verbose=False,
        )
        print(
            f"Mode A LR={lr:.0e} | Final Val Acc: {result['final_val_acc']:.4f} | "
            f"Final Train Loss: {result['final_train_loss']:.4f} | "
            f"Time/Ep: {result['avg_time_per_epoch']:.1f}s | "
            f"Status: {'OK' if result['status'] == 'ok' else 'NaN/Diverged'}"
        )
        mode_a_results.append(result)

    save_kan_lr_sweep_plot(mode_b_results, mode_a_results)

    best_b, runner_b = select_recommended_lr(mode_b_results)
    best_a, runner_a = select_recommended_lr(mode_a_results)

    diverged_b = [f"{row['lr']:.0e}" for row in mode_b_results if row["status"] != "ok"]
    diverged_a = [f"{row['lr']:.0e}" for row in mode_a_results if row["status"] != "ok"]

    print("\nKAN Learning Rate Sweep Results:\n")
    print("Mode B (hidden=28):")
    print(f"  Best LR: {best_b['lr']:.0e} (Val Acc: {best_b['final_val_acc']:.4f} after 15 epochs)")
    if runner_b is not None:
        print(f"  Runner-up: {runner_b['lr']:.0e} (Val Acc: {runner_b['final_val_acc']:.4f})")
    print(f"  Diverged/NaN: {diverged_b if diverged_b else 'None'}")
    print()
    print("Mode A (hidden=256):")
    print(f"  Best LR: {best_a['lr']:.0e} (Val Acc: {best_a['final_val_acc']:.4f} after 10 epochs)")
    if runner_a is not None:
        print(f"  Runner-up: {runner_a['lr']:.0e} (Val Acc: {runner_a['final_val_acc']:.4f})")
    print(f"  Diverged/NaN: {diverged_a if diverged_a else 'None'}")

    csv_rows = []
    for mode, hidden_dim, rows in (("B", 28, mode_b_results), ("A", 256, mode_a_results)):
        for row in rows:
            csv_rows.append(
                {
                    "mode": mode,
                    "hidden_dim": hidden_dim,
                    "lr": float(row["lr"]),
                    "final_val_acc": float(row["final_val_acc"]),
                    "final_train_loss": float(row["final_train_loss"]),
                    "status": row["status"],
                    "val_acc_history": json.dumps([float(v) for v in row["history"]["val_metric"]]),
                }
            )

    save_results_csv(
        {key: [entry[key] for entry in csv_rows] for key in csv_rows[0].keys()},
        LR_SWEEP_RESULTS_PATH,
    )

    summary = {
        "mode_b_best": best_b,
        "mode_b_runner_up": runner_b,
        "mode_a_best": best_a,
        "mode_a_runner_up": runner_a,
    }
    return mode_b_results, mode_a_results, summary


def run_standard_head_lr_checks(device: torch.device) -> dict[str, dict[str, Any]]:
    """Run the quick LR checks for MLP and BSpline-MLP heads.

    Args:
        device: Training device.

    Returns:
        Mapping of model family name to selected LR result.
    """
    print_part_header(
        "PART 3: MLP / BSPLINE-MLP LEARNING RATE QUICK CHECK",
        "~10-20 minutes on GPU, ~45-90 minutes on CPU",
    )
    lr_candidates = [2e-3, 1e-3, 5e-4]
    recommendations: dict[str, dict[str, Any]] = {}

    for label, head_type in (("MLP", "mlp"), ("BSpline-MLP", "bspline_mlp")):
        rows: list[dict[str, Any]] = []
        for lr in lr_candidates:
            rows.append(
                train_cifar_model(
                    name=f"{label.replace('-', '')}_LR_{lr:.0e}",
                    head_type=head_type,
                    hidden_dim=256,
                    lr=float(lr),
                    epochs=10,
                    device=device,
                    verbose=False,
                )
            )
        best, _ = select_recommended_lr(rows)
        recommendations[label] = best
        result_parts = [f"LR={row['lr']:.0e}: Val Acc {row['final_val_acc']:.4f}" for row in rows]
        print(f"{label} LR Check:")
        print(f"  {' | '.join(result_parts)}")
        print(f"  Recommended: {best['lr']:.0e}\n")

    return recommendations


def save_recommended_hyperparams(
    *,
    mlp_lr: float,
    kan_mode_a_lr: float,
    kan_mode_b_lr: float,
    bspline_lr: float,
) -> None:
    """Save the final recommended hyperparameters YAML.

    Args:
        mlp_lr: Recommended LR for the MLP head.
        kan_mode_a_lr: Recommended LR for KAN Mode A.
        kan_mode_b_lr: Recommended LR for KAN Mode B.
        bspline_lr: Recommended LR for BSpline-MLP.
    """
    payload = {
        "mlp_head": {
            "hidden_dim": 256,
            "learning_rate": float(mlp_lr),
            "dropout": 0.3,
        },
        "kan_head_mode_a": {
            "hidden_dim": 256,
            "learning_rate": float(kan_mode_a_lr),
            "grid_size": 5,
            "spline_order": 3,
        },
        "kan_head_mode_b": {
            "hidden_dim": 28,
            "learning_rate": float(kan_mode_b_lr),
            "grid_size": 5,
            "spline_order": 3,
        },
        "bspline_mlp_head": {
            "hidden_dim": 256,
            "learning_rate": float(bspline_lr),
            "dropout": 0.3,
            "grid_size": 5,
            "spline_order": 3,
        },
        "training": {
            "epochs": 50,
            "batch_size": 128,
            "scheduler": "cosine_annealing",
            "early_stopping_patience": 10,
            "seeds": [42, 123, 456],
        },
    }
    with open(RECOMMENDED_HPARAMS_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def print_final_summary(
    pipeline_results: list[dict[str, Any]],
    pipeline_status: dict[str, bool],
    kan_sweep_summary: dict[str, Any],
    standard_lr_recs: dict[str, dict[str, Any]],
) -> None:
    """Print the overall diagnostic summary.

    Args:
        pipeline_results: Part 1 run results.
        pipeline_status: Part 1 pass/fail map.
        kan_sweep_summary: Part 2 recommended LR summary.
        standard_lr_recs: Part 3 recommendations.
    """
    pipeline_map = {row["name"]: row for row in pipeline_results}
    mlp_speed = pipeline_map["CNN+MLPHead"]["avg_time_per_epoch"]
    kan_b_speed = pipeline_map["CNN+KANHead_ModeB"]["avg_time_per_epoch"]
    bspline_speed = pipeline_map["CNN+BSplineMLPHead"]["avg_time_per_epoch"]
    kan_a_speed = kan_sweep_summary["mode_a_best"]["avg_time_per_epoch"]

    mlp_run_minutes = mlp_speed * 50 / 60.0
    kan_a_run_minutes = kan_a_speed * 50 / 60.0
    kan_b_run_minutes = kan_b_speed * 50 / 60.0
    bspline_run_minutes = bspline_speed * 50 / 60.0
    total_minutes = 3 * (mlp_run_minutes + kan_a_run_minutes + kan_b_run_minutes + bspline_run_minutes)

    print("\n" + "═" * 75)
    print("CIFAR-10 DIAGNOSTIC COMPLETE")
    print("═" * 75)
    print()
    print(
        "Pipeline Validation: "
        f"{'ALL PASS' if all(pipeline_status.values()) else 'SOME FAILED'}"
    )
    print(
        f"  - MLPHead: {'PASS' if pipeline_status['CNN+MLPHead'] else 'FAIL'} "
        f"({pipeline_map['CNN+MLPHead']['final_val_acc']:.4f} acc @ 10 epochs)"
    )
    print(
        f"  - KANHead ModeB: {'PASS' if pipeline_status['CNN+KANHead_ModeB'] else 'FAIL'} "
        f"({pipeline_map['CNN+KANHead_ModeB']['final_val_acc']:.4f} acc @ 10 epochs)"
    )
    print(
        f"  - BSplineMLPHead: {'PASS' if pipeline_status['CNN+BSplineMLPHead'] else 'FAIL'} "
        f"({pipeline_map['CNN+BSplineMLPHead']['final_val_acc']:.4f} acc @ 10 epochs)"
    )
    print()
    print("Recommended Learning Rates:")
    print(f"  - MLPHead:          {standard_lr_recs['MLP']['lr']:.0e}")
    print(f"  - KANHead Mode A:   {kan_sweep_summary['mode_a_best']['lr']:.0e}")
    print(f"  - KANHead Mode B:   {kan_sweep_summary['mode_b_best']['lr']:.0e}")
    print(f"  - BSplineMLPHead:   {standard_lr_recs['BSpline-MLP']['lr']:.0e}")
    print()
    print("Training Speed (per epoch):")
    print(f"  - MLPHead:          {mlp_speed:.1f}s")
    print(f"  - KANHead Mode B:   {kan_b_speed:.1f}s ({kan_b_speed / max(mlp_speed, 1e-8):.1f}x slower than MLP)")
    print(f"  - KANHead Mode A:   {kan_a_speed:.1f}s ({kan_a_speed / max(mlp_speed, 1e-8):.1f}x slower than MLP)")
    print(f"  - BSplineMLPHead:   {bspline_speed:.1f}s ({bspline_speed / max(mlp_speed, 1e-8):.1f}x slower than MLP)")
    print()
    print("Estimated Phase 2 total time:")
    print("  - 12 runs (MLP + KAN Mode A/B + BSpline, across 3 seeds) x 50 epochs")
    print(f"  - MLPHead runs:          ~{mlp_run_minutes:.1f} min each x 3 runs = ~{mlp_run_minutes * 3:.1f} min")
    print(f"  - KANHead Mode A runs:   ~{kan_a_run_minutes:.1f} min each x 3 runs = ~{kan_a_run_minutes * 3:.1f} min")
    print(f"  - KANHead Mode B runs:   ~{kan_b_run_minutes:.1f} min each x 3 runs = ~{kan_b_run_minutes * 3:.1f} min")
    print(f"  - BSplineMLPHead runs:   ~{bspline_run_minutes:.1f} min each x 3 runs = ~{bspline_run_minutes * 3:.1f} min")
    print(f"  - Total estimated:       ~{total_minutes / 60.0:.2f} hours")
    print()
    print("Files saved:")
    print(f"  - {PIPELINE_CURVES_PATH}")
    print(f"  - {KAN_LR_SWEEP_PATH}")
    print(f"  - {LR_SWEEP_RESULTS_PATH}")
    print(f"  - {RECOMMENDED_HPARAMS_PATH}")
    print()
    print("Ready for Phase 2 full runs.")
    print("═" * 75)


def main() -> None:
    """Run the full CIFAR-10 diagnostic workflow."""
    ensure_output_dir()
    device = get_device()
    print(
        "This is a one-time diagnostic script. Expected runtime: ~45-90 minutes on GPU, "
        "~3-5 hours on CPU."
    )

    pipeline_results, pipeline_status = run_pipeline_validation(device)
    mode_b_results, mode_a_results, kan_summary = run_kan_lr_sweep(device)
    standard_lr_recs = run_standard_head_lr_checks(device)

    save_recommended_hyperparams(
        mlp_lr=float(standard_lr_recs["MLP"]["lr"]),
        kan_mode_a_lr=float(kan_summary["mode_a_best"]["lr"]),
        kan_mode_b_lr=float(kan_summary["mode_b_best"]["lr"]),
        bspline_lr=float(standard_lr_recs["BSpline-MLP"]["lr"]),
    )

    _ = mode_b_results, mode_a_results
    print_final_summary(
        pipeline_results=pipeline_results,
        pipeline_status=pipeline_status,
        kan_sweep_summary=kan_summary,
        standard_lr_recs=standard_lr_recs,
    )


if __name__ == "__main__":
    main()
