"""Phase 2 pre-flight verification for the CIFAR-10 experiment pipeline."""

from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
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
from src.evaluate import get_classification_report
from src.evaluate import per_class_accuracy
from src.models import build_cifar10_model
from src.models import count_head_parameters
from src.models import count_parameters
from src.train import train_model
from src.utils import format_param_count
from src.utils import get_device
from src.utils import set_seed

CONFIG_PATH = os.path.join("experiments", "configs", "exp2_config.yaml")
PREFLIGHT_DIR = os.path.join("results", "exp2", "preflight")
PREFLIGHT_CSV = os.path.join(PREFLIGHT_DIR, "preflight_results.csv")
PREFLIGHT_CHECKPOINT_DIR = os.path.join(PREFLIGHT_DIR, "checkpoints")
EXPECTED_RESULT_COLUMNS = [
    "name",
    "head_type",
    "mode",
    "hidden_dim",
    "total_params",
    "head_params",
    "learning_rate",
    "seed",
    "test_loss",
    "test_accuracy",
    "best_val_loss",
    "best_val_accuracy",
    "best_epoch",
    "epochs_trained",
    "convergence_epoch_95",
    "avg_time_per_epoch",
    "total_training_time",
    "per_class_accuracy",
    "train_loss_history",
    "val_loss_history",
    "val_accuracy_history",
]

results: dict[str, str] = {}
CONFIG: dict[str, Any] | None = None
CONFIG_ROWS: list[dict[str, Any]] = []
TRAIN_HISTORIES: dict[str, dict[str, Any]] = {}
TRAIN_RESULTS: list[dict[str, Any]] = []
EVAL_RESULTS: dict[str, dict[str, float]] = {}
CHECKPOINT_PATHS: dict[str, str] = {}
TEST_INFO: dict[str, Any] | None = None
TEST_LOADERS: tuple[Any, Any, Any] | None = None
PREFLIGHT_LOADED_DF: pd.DataFrame | None = None


def run_test(name: str, fn: Any) -> None:
    """Run a test function, catch any exception, and record pass/fail."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")
    try:
        fn()
        results[name] = "PASSED"
        print(f"[PASS] {name}")
    except Exception as exc:
        results[name] = f"FAILED: {exc}"
        print(f"[FAIL] {name}: {exc}")
        traceback.print_exc()


def load_config() -> dict[str, Any]:
    """Load the Experiment 2 config.

    Returns:
        Parsed config dictionary.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(model_config: dict[str, Any]) -> torch.nn.Module:
    """Build a CIFAR-10 model from one config entry.

    Args:
        model_config: Single model config entry.

    Returns:
        Instantiated model.
    """
    return build_cifar10_model(
        head_type=str(model_config["head_type"]),
        hidden_dim=int(model_config["hidden_dim"]),
        grid_size=int(model_config.get("grid_size", 5)),
        spline_order=int(model_config.get("spline_order", 3)),
        dropout=float(model_config.get("dropout", 0.3)),
    )


def checkpoint_path_for(model_name: str) -> str:
    """Return the checkpoint path used by preflight runs.

    Args:
        model_name: Model name.

    Returns:
        Checkpoint path.
    """
    return os.path.join(PREFLIGHT_CHECKPOINT_DIR, f"preflight_{model_name}_best.pt")


def cleanup_model(model: torch.nn.Module | None) -> None:
    """Delete model and clear CUDA cache if needed.

    Args:
        model: Model to clean up.
    """
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def serialize_results_entry(
    model_row: dict[str, Any],
    seed: int,
    history: dict[str, Any],
    evaluation: dict[str, float],
    class_accuracy: dict[str, float],
) -> dict[str, Any]:
    """Create one CSV row matching the full Experiment 2 schema.

    Args:
        model_row: Enriched model metadata.
        seed: Seed used for the run.
        history: History dict from training.
        evaluation: Test evaluation dict.
        class_accuracy: Per-class accuracy mapping.

    Returns:
        Serialized result row.
    """
    return {
        "name": str(model_row["name"]),
        "head_type": str(model_row["head_type"]),
        "mode": str(model_row["mode"]),
        "hidden_dim": int(model_row["hidden_dim"]),
        "total_params": int(model_row["total_params"]),
        "head_params": int(model_row["head_params"]),
        "learning_rate": float(model_row["learning_rate"]),
        "seed": int(seed),
        "test_loss": float(evaluation["test_loss"]),
        "test_accuracy": float(evaluation["test_metric"]),
        "best_val_loss": float(history["best_val_loss"]),
        "best_val_accuracy": (
            float(history["best_val_metric"])
            if history["best_val_metric"] is not None
            else float("nan")
        ),
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(history["epochs_trained"]),
        "convergence_epoch_95": int(compute_convergence_epoch(history, 0.95)),
        "avg_time_per_epoch": float(np.mean(history["time_per_epoch"])),
        "total_training_time": float(history["total_time"]),
        "per_class_accuracy": json.dumps({key: float(value) for key, value in class_accuracy.items()}),
        "train_loss_history": json.dumps([float(v) for v in history["train_loss"]]),
        "val_loss_history": json.dumps([float(v) for v in history["val_loss"]]),
        "val_accuracy_history": json.dumps([float(v) for v in history["val_metric"]]),
    }


def test_config_loading() -> None:
    """Verify config structure for Experiment 2."""
    global CONFIG, CONFIG_ROWS

    CONFIG = load_config()
    assert "models" in CONFIG and len(CONFIG["models"]) == 6
    assert "training" in CONFIG
    assert {"epochs", "scheduler", "early_stopping_patience"}.issubset(CONFIG["training"])
    assert "seeds" in CONFIG and isinstance(CONFIG["seeds"], list)

    required_model_keys = {"name", "head_type", "mode", "hidden_dim", "learning_rate"}
    CONFIG_ROWS = []
    for model_config in CONFIG["models"]:
        assert required_model_keys.issubset(model_config)
        CONFIG_ROWS.append(dict(model_config))

    print("Model configs found:")
    for model_config in CONFIG_ROWS:
        print(f"  - {model_config['name']}")
    print("Config structure validated")


def test_model_construction() -> None:
    """Build and sanity-check all 6 model configurations."""
    assert CONFIG_ROWS, "Config rows not loaded yet."

    param_map: dict[str, tuple[int, int]] = {}
    for model_config in CONFIG_ROWS:
        model = build_model_from_config(model_config)
        total_params = count_parameters(model)
        head_params = count_head_parameters(model)
        param_map[str(model_config["name"])] = (total_params, head_params)

        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        assert y.shape == (2, 10), f"{model_config['name']} produced shape {tuple(y.shape)}"
        assert not torch.isnan(y).any(), f"{model_config['name']} produced NaN outputs"
        print(
            f"  {model_config['name']}: {format_param_count(total_params)} total, "
            f"{format_param_count(head_params)} head - OK"
        )
        cleanup_model(model)

    assert param_map["MLP_ModeA"] == param_map["MLP_ModeB"]
    assert param_map["BSplineMLP_ModeA"] == param_map["BSplineMLP_ModeB"]
    assert param_map["KAN_ModeA"][1] >= 5 * param_map["KAN_ModeB"][1]

    mlp_mode_b_head = param_map["MLP_ModeB"][1]
    kan_mode_b_head = param_map["KAN_ModeB"][1]
    relative_gap = abs(kan_mode_b_head - mlp_mode_b_head) / mlp_mode_b_head
    assert relative_gap <= 0.20, f"KAN_ModeB head params gap too large: {relative_gap:.3f}"


def test_data_loading_determinism() -> None:
    """Verify CIFAR-10 loader reproducibility with the same seed."""
    set_seed(42)
    train_loader_1, _, _, _ = get_cifar10(
        batch_size=32,
        data_dir="./data",
        seed=42,
        num_workers=0,
    )
    images1, labels1 = next(iter(train_loader_1))

    set_seed(42)
    train_loader_2, _, _, _ = get_cifar10(
        batch_size=32,
        data_dir="./data",
        seed=42,
        num_workers=0,
    )
    images2, labels2 = next(iter(train_loader_2))

    assert torch.allclose(images1, images2)
    assert torch.equal(labels1, labels2)
    print("Data loading is deterministic across reloads with same seed")


def test_mini_training_run() -> None:
    """Run the miniature 3-epoch version of the full sweep."""
    global TRAIN_HISTORIES, TRAIN_RESULTS, TEST_INFO, TEST_LOADERS

    assert CONFIG is not None and CONFIG_ROWS, "Config must be loaded first."

    if os.path.exists(PREFLIGHT_DIR):
        shutil.rmtree(PREFLIGHT_DIR)
    os.makedirs(PREFLIGHT_CHECKPOINT_DIR, exist_ok=True)

    device = get_device()
    seed = 42
    set_seed(seed)
    train_loader, val_loader, test_loader, info = get_cifar10(
        batch_size=128,
        data_dir="./data",
        seed=seed,
        num_workers=0,
    )
    TEST_INFO = info
    TEST_LOADERS = (train_loader, val_loader, test_loader)
    TRAIN_HISTORIES = {}
    TRAIN_RESULTS = []
    CHECKPOINT_PATHS.clear()

    criterion = nn.CrossEntropyLoss()

    for model_config in CONFIG_ROWS:
        model_name = str(model_config["name"])
        set_seed(seed)
        model = build_model_from_config(model_config).to(device)
        total_params = count_parameters(model)
        head_params = count_head_parameters(model)

        optimizer = optim.Adam(model.parameters(), lr=float(model_config["learning_rate"]))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=3,
            device=device,
            metric_fn=accuracy_metric,
            scheduler=scheduler,
            early_stopping_patience=0,
            checkpoint_dir=PREFLIGHT_CHECKPOINT_DIR,
            model_name=f"preflight_{model_name}",
            verbose=True,
        )

        assert {"train_loss", "val_loss", "val_metric", "time_per_epoch", "best_epoch", "best_val_loss", "best_val_metric", "total_time", "epochs_trained"}.issubset(history)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_metric"]) == 3
        assert all(np.isfinite(history["train_loss"]))
        assert all(np.isfinite(history["val_loss"]))
        assert all(np.isfinite(history["val_metric"]))
        assert history["train_loss"][-1] < history["train_loss"][0], f"{model_name} train loss did not decrease"
        assert history["val_metric"][-1] > 0.10, f"{model_name} val accuracy did not beat chance"

        avg_time = sum(history["time_per_epoch"]) / 3.0
        print(
            f"  {model_name}: Val Acc={history['val_metric'][-1]:.4f}, "
            f"Train Loss={history['train_loss'][-1]:.4f}, Time/Ep={avg_time:.1f}s"
        )

        evaluation = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            metric_fn=accuracy_metric,
        )
        class_acc = per_class_accuracy(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=info["class_names"],
        )

        serialized_row = serialize_results_entry(
            {
                **model_config,
                "total_params": total_params,
                "head_params": head_params,
            },
            seed=seed,
            history=history,
            evaluation=evaluation,
            class_accuracy=class_acc,
        )
        TRAIN_RESULTS.append(serialized_row)
        TRAIN_HISTORIES[model_name] = history
        CHECKPOINT_PATHS[model_name] = checkpoint_path_for(model_name)

        cleanup_model(model)

    print("Preflight Training Results (3 epochs, seed=42)")
    print("+------------------+--------------+--------------+----------+")
    print("| Model            | Val Acc (ep3)| Train Loss   | Time/Ep  |")
    print("+------------------+--------------+--------------+----------+")
    for row in TRAIN_RESULTS:
        val_history = json.loads(row["val_accuracy_history"])
        train_history = json.loads(row["train_loss_history"])
        print(
            f"| {row['name']:<16} | "
            f"{float(val_history[-1]):>12.4f} | "
            f"{float(train_history[-1]):>12.4f} | "
            f"{float(row['avg_time_per_epoch']):>7.1f}s |"
        )
    print("+------------------+--------------+--------------+----------+")


def test_evaluation_pipeline() -> None:
    """Verify evaluation on the saved checkpoints for all 6 models."""
    assert TEST_LOADERS is not None, "Mini training must run first."
    _, _, test_loader = TEST_LOADERS
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    for model_config in CONFIG_ROWS:
        model_name = str(model_config["name"])
        checkpoint_path = CHECKPOINT_PATHS[model_name]
        assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"

        model = build_model_from_config(model_config).to(device)
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        result = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            metric_fn=accuracy_metric,
        )
        assert {"test_loss", "test_metric"}.issubset(result)
        assert float(result["test_metric"]) > 0.10
        assert np.isfinite(float(result["test_loss"]))
        EVAL_RESULTS[model_name] = result
        print(
            f"  {model_name}: Test Acc={float(result['test_metric']):.4f}, "
            f"Test Loss={float(result['test_loss']):.4f}"
        )
        cleanup_model(model)


def test_per_class_accuracy() -> None:
    """Verify per-class accuracy structure on MLP_ModeA."""
    assert TEST_LOADERS is not None and TEST_INFO is not None
    _, _, test_loader = TEST_LOADERS
    device = get_device()

    target_config = next(config for config in CONFIG_ROWS if config["name"] == "MLP_ModeA")
    model = build_model_from_config(target_config).to(device)
    checkpoint_path = CHECKPOINT_PATHS["MLP_ModeA"]
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    class_acc = per_class_accuracy(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=TEST_INFO["class_names"],
    )
    assert isinstance(class_acc, dict)
    assert len(class_acc) == 10
    for class_name in TEST_INFO["class_names"]:
        assert class_name in class_acc
        assert 0.0 <= float(class_acc[class_name]) <= 1.0
    print(class_acc)
    print("Per-class accuracy structure is correct")
    cleanup_model(model)


def test_convergence_epoch() -> None:
    """Verify convergence epoch computation for all histories."""
    assert TRAIN_HISTORIES, "Mini training must run first."
    for model_name, history in TRAIN_HISTORIES.items():
        conv_epoch = compute_convergence_epoch(history, 0.95)
        assert isinstance(conv_epoch, int)
        assert 1 <= conv_epoch <= 3
        print(f"  {model_name}: convergence epoch = {conv_epoch}")


def test_csv_round_trip() -> None:
    """Verify CSV save/load and JSON decoding for preflight results."""
    global PREFLIGHT_LOADED_DF

    assert TRAIN_RESULTS, "Mini training results missing."
    pd.DataFrame(TRAIN_RESULTS).to_csv(PREFLIGHT_CSV, index=False)
    loaded = pd.read_csv(PREFLIGHT_CSV)
    PREFLIGHT_LOADED_DF = loaded

    assert len(loaded) == 6
    for column in EXPECTED_RESULT_COLUMNS:
        assert column in loaded.columns, f"Missing column: {column}"

    json_type_map = {
        "per_class_accuracy": dict,
        "train_loss_history": list,
        "val_loss_history": list,
        "val_accuracy_history": list,
    }
    for column, expected_type in json_type_map.items():
        decoded = json.loads(loaded.iloc[0][column])
        assert isinstance(decoded, expected_type), f"{column} decoded to {type(decoded)}"
        if isinstance(decoded, list):
            assert len(decoded) == 3, f"{column} should have 3 entries"
        if isinstance(decoded, dict):
            assert len(decoded) == 10, f"{column} should have 10 classes"

    print("CSV round-trip verified - all columns present and JSON decodable")


def test_resume_logic() -> None:
    """Simulate resume logic on the preflight CSV."""
    assert PREFLIGHT_LOADED_DF is not None, "Preflight CSV must be loaded first."

    completed = {
        (str(row["name"]), int(row["seed"]))
        for _, row in PREFLIGHT_LOADED_DF.iterrows()
    }
    assert len(completed) == 6

    for model_config in CONFIG_ROWS:
        assert (str(model_config["name"]), 42) in completed

    for model_config in CONFIG_ROWS:
        assert (str(model_config["name"]), 123) not in completed

    print("Resume logic: 6/6 completed runs detected, 6 new runs would be needed for seed=123")
    print("Resume logic works correctly")


def test_analysis_compatibility() -> None:
    """Verify the CSV schema is compatible with exp2_analysis.py."""
    assert PREFLIGHT_LOADED_DF is not None, "Preflight CSV must be loaded first."

    loaded = PREFLIGHT_LOADED_DF.copy()
    for column in EXPECTED_RESULT_COLUMNS:
        assert column in loaded.columns

    numeric_columns = [
        "hidden_dim",
        "total_params",
        "head_params",
        "learning_rate",
        "seed",
        "test_loss",
        "test_accuracy",
        "best_val_loss",
        "best_val_accuracy",
        "best_epoch",
        "epochs_trained",
        "convergence_epoch_95",
        "avg_time_per_epoch",
        "total_training_time",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(loaded[column], errors="coerce")
        assert np.isfinite(values).all(), f"Non-finite values found in {column}"

    summary = (
        loaded.groupby("name")
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
        )
        .reset_index()
    )
    assert len(summary) == 6
    print("Analysis-compatible CSV format verified")


def test_scheduler_compatibility() -> None:
    """Verify CosineAnnealingLR works correctly with train_model."""
    device = get_device()
    set_seed(42)
    train_loader, val_loader, _, _ = get_cifar10(
        batch_size=128,
        data_dir="./data",
        seed=42,
        num_workers=0,
    )
    model = build_cifar10_model("mlp", hidden_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=5,
        device=device,
        metric_fn=accuracy_metric,
        scheduler=scheduler,
        early_stopping_patience=0,
        checkpoint_dir=None,
        model_name="preflight_scheduler_check",
        verbose=True,
    )
    assert len(history["train_loss"]) == 5
    assert history["train_loss"][-1] < history["train_loss"][0]
    print("CosineAnnealingLR compatible with train_model")
    cleanup_model(model)


def test_checkpoint_save_load() -> None:
    """Verify checkpoint save/load round-trips exactly."""
    device = get_device()
    set_seed(42)
    train_loader, val_loader, test_loader, _ = get_cifar10(
        batch_size=128,
        data_dir="./data",
        seed=42,
        num_workers=0,
    )

    model = build_cifar10_model("mlp", hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = os.path.join(PREFLIGHT_DIR, "checkpoint_test")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_identity_best.pt")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=2,
        device=device,
        metric_fn=accuracy_metric,
        scheduler=None,
        early_stopping_patience=0,
        checkpoint_dir=checkpoint_dir,
        model_name="checkpoint_identity",
        verbose=True,
    )
    assert os.path.exists(checkpoint_path)

    original_eval = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        metric_fn=accuracy_metric,
    )

    fresh_model = build_cifar10_model("mlp", hidden_dim=256).to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    fresh_model.load_state_dict(state_dict)
    loaded_eval = evaluate_model(
        model=fresh_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        metric_fn=accuracy_metric,
    )

    assert abs(float(original_eval["test_metric"]) - float(loaded_eval["test_metric"])) <= 1e-5
    print("Checkpoint save/load produces identical model")
    cleanup_model(model)
    cleanup_model(fresh_model)


def cleanup() -> None:
    """Remove the preflight directory after all tests."""
    if os.path.exists(PREFLIGHT_DIR):
        shutil.rmtree(PREFLIGHT_DIR)
        print("Cleaned up preflight directory")


def print_final_summary() -> None:
    """Print the final preflight summary."""
    print("\n" + "=" * 75)
    print("PHASE 2 PRE-FLIGHT SUMMARY")
    print("=" * 75)
    passed = sum(1 for value in results.values() if value == "PASSED")
    failed = len(results) - passed
    for name, outcome in results.items():
        status = "[PASS]" if outcome == "PASSED" else "[FAIL]"
        print(f"  {status} {name}: {outcome}")
    print(f"\n{passed}/{len(results)} tests passed, {failed} failed\n")
    if failed == 0:
        print("ALL PRE-FLIGHT CHECKS PASSED - safe to launch full Experiment 2 sweep.")
        print("Estimated full run time: 18 runs x 50 epochs x ~100s/epoch ~= 25 hours on CPU")
        print("Recommendation: Use GPU or NTU cluster to reduce to ~5-6 hours.")
    else:
        print("SOME CHECKS FAILED - fix issues before running the full sweep.")
    print("=" * 75)


def main() -> None:
    """Run all Phase 2 pre-flight tests."""
    run_test("Config Loading", test_config_loading)
    run_test("Model Construction - All 6 Configs", test_model_construction)
    run_test("Data Loading Determinism", test_data_loading_determinism)
    run_test("Mini Training Run - All 6 Models, 3 Epochs", test_mini_training_run)
    run_test("Evaluation Pipeline - All 6 Models", test_evaluation_pipeline)
    run_test("Per-Class Accuracy", test_per_class_accuracy)
    run_test("Convergence Epoch Computation", test_convergence_epoch)
    run_test("CSV Save/Load Round-Trip", test_csv_round_trip)
    run_test("Resume Logic Simulation", test_resume_logic)
    run_test("Analysis Script Compatibility", test_analysis_compatibility)
    run_test("Scheduler Compatibility", test_scheduler_compatibility)
    run_test("Checkpoint Save/Load", test_checkpoint_save_load)
    cleanup()
    print_final_summary()


if __name__ == "__main__":
    main()
