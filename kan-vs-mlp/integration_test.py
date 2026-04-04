"""End-to-end integration test for the Phase 0 project scaffold.

Run this script from the project root with:
    python integration_test.py

Note:
    The CIFAR-10 training tests can take significantly longer on CPU-only
    machines. The script keeps the default classification batch size at 128
    for speed, but you can lower it manually if your machine runs out of
    memory or becomes too slow.
"""

from __future__ import annotations

import os
import traceback
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam

from src.bspline_activation import BSplineActivation
from src.data_utils import get_california_housing
from src.data_utils import get_cifar10
from src.data_utils import get_dataset
from src.evaluate import accuracy_metric
from src.evaluate import compute_convergence_epoch
from src.evaluate import evaluate_model
from src.evaluate import mse_metric
from src.evaluate import per_class_accuracy
from src.evaluate import r2_metric
from src.kan_layer import KANLinear
from src.models import BSplineMLPHead
from src.models import CNNBackbone
from src.models import CNNWithHead
from src.models import KANHead
from src.models import KANRegressor
from src.models import MLPHead
from src.models import MLPRegressor
from src.models import build_cifar10_model
from src.models import count_head_parameters
from src.models import count_parameters
from src.models import model_summary
from src.spline_vis import compute_feature_nonlinearity
from src.spline_vis import plot_feature_importance_comparison
from src.spline_vis import plot_kan_splines
from src.spline_vis import probe_kan_layer_responses
from src.train import EarlyStopping
from src.train import train_model
from src.utils import format_param_count
from src.utils import get_device
from src.utils import load_results_csv
from src.utils import plot_bar_comparison
from src.utils import plot_training_curves
from src.utils import save_results_csv
from src.utils import set_seed

results: dict[str, str] = {}
created_files: list[str] = []


def run_test(name: str, fn: Callable[[], None]) -> None:
    """Run a test function, catch any exception, and record the result.

    Args:
        name: Name of the integration test.
        fn: Test function to execute.
    """
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")
    try:
        fn()
        results[name] = "PASSED"
        print(f"[PASS] {name} PASSED")
    except Exception as exc:
        results[name] = f"FAILED: {exc}"
        print(f"[FAIL] {name} FAILED: {exc}")
        traceback.print_exc()


def register_artifact(path: str) -> None:
    """Track a file created by the integration test.

    Args:
        path: File path to track for later cleanup.
    """
    if path not in created_files:
        created_files.append(path)


def cleanup_artifacts() -> None:
    """Remove files created during integration testing."""
    print(f"\n{'=' * 60}")
    print("CLEANUP")
    print(f"{'=' * 60}")
    for path in created_files:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed: {path}")
            except OSError as exc:
                print(f"Could not remove {path}: {exc}")


def test_imports() -> None:
    """Verify that all key project imports succeed."""
    _ = (
        set_seed,
        get_device,
        plot_training_curves,
        plot_bar_comparison,
        save_results_csv,
        load_results_csv,
        format_param_count,
        KANLinear,
        BSplineActivation,
        MLPRegressor,
        KANRegressor,
        CNNBackbone,
        MLPHead,
        KANHead,
        BSplineMLPHead,
        CNNWithHead,
        build_cifar10_model,
        count_parameters,
        count_head_parameters,
        model_summary,
        get_california_housing,
        get_cifar10,
        get_dataset,
        train_model,
        EarlyStopping,
        accuracy_metric,
        r2_metric,
        mse_metric,
        evaluate_model,
        per_class_accuracy,
        compute_convergence_epoch,
        probe_kan_layer_responses,
        compute_feature_nonlinearity,
        plot_kan_splines,
        plot_feature_importance_comparison,
    )


def test_seed_reproducibility() -> None:
    """Verify deterministic outputs across repeated seeded runs."""
    set_seed(42)
    mlp1 = MLPRegressor()
    x1 = torch.randn(4, 8)
    y1 = mlp1(x1)

    set_seed(42)
    mlp2 = MLPRegressor()
    x2 = torch.randn(4, 8)
    y2 = mlp2(x2)

    assert torch.equal(x1, x2), "MLP inputs differed after reseeding."
    assert torch.allclose(y1, y2, atol=0.0), "MLP outputs were not exactly reproducible."

    set_seed(42)
    kan1 = KANRegressor()
    kx1 = torch.randn(4, 8)
    ky1 = kan1(kx1)

    set_seed(42)
    kan2 = KANRegressor()
    kx2 = torch.randn(4, 8)
    ky2 = kan2(kx2)

    assert torch.equal(kx1, kx2), "KAN inputs differed after reseeding."
    assert torch.allclose(ky1, ky2, atol=0.0), "KAN outputs were not exactly reproducible."


def test_california_housing_data_pipeline() -> None:
    """Verify the California Housing data pipeline end to end."""
    set_seed(42)
    train_loader, val_loader, test_loader, info = get_california_housing(
        batch_size=64,
        num_workers=0,
    )
    assert info["n_features"] == 8, f"Expected 8 features, found {info['n_features']}."
    total_samples = info["n_train"] + info["n_val"] + info["n_test"]
    assert total_samples == 20640, f"Expected 20640 samples, found {total_samples}."

    features, targets = next(iter(train_loader))
    assert features.shape[1] == 8, f"Expected feature dimension 8, got {tuple(features.shape)}."
    assert targets.shape[1] == 1, f"Expected target dimension 1, got {tuple(targets.shape)}."
    assert features.dtype == torch.float32, f"Expected float32 features, got {features.dtype}."
    assert targets.dtype == torch.float32, f"Expected float32 targets, got {targets.dtype}."

    feature_means = features.mean(dim=0)
    feature_stds = features.std(dim=0, unbiased=False)
    print("California batch means:", feature_means)
    print("California batch stds:", feature_stds)
    assert torch.all(feature_means.abs() < 1.0), "Feature means were not roughly centered."
    full_train_features = train_loader.dataset.tensors[0]
    full_train_means = full_train_features.mean(dim=0)
    full_train_stds = full_train_features.std(dim=0, unbiased=False)
    print("California full-train means:", full_train_means)
    print("California full-train stds:", full_train_stds)

    assert torch.all((full_train_means.abs() < 0.1)), "Full training features were not centered."
    assert torch.all((full_train_stds > 0.9) & (full_train_stds < 1.1)), (
        "Full training features were not properly standardized."
    )

    _ = val_loader, test_loader


def test_cifar10_data_pipeline() -> None:
    """Verify the CIFAR-10 data pipeline end to end."""
    set_seed(42)
    train_loader, _, test_loader, info = get_cifar10(
        batch_size=32,
        data_dir="./data",
        num_workers=0,
    )
    assert info["n_classes"] == 10, f"Expected 10 classes, found {info['n_classes']}."
    assert info["n_train"] == 45000, f"Expected 45000 training samples, found {info['n_train']}."
    assert info["n_val"] == 5000, f"Expected 5000 validation samples, found {info['n_val']}."
    assert info["n_test"] == 10000, f"Expected 10000 test samples, found {info['n_test']}."

    images, labels = next(iter(train_loader))
    assert images.shape == (32, 3, 32, 32), f"Unexpected image batch shape: {tuple(images.shape)}."
    assert labels.shape == (32,), f"Unexpected label batch shape: {tuple(labels.shape)}."
    assert images.min().item() < 0.0, "Expected normalized images with negative values."
    assert labels.min().item() >= 0, "Found negative class label."
    assert labels.max().item() <= 9, "Found class label above 9."

    _ = test_loader


def test_regression_models_forward_pass() -> None:
    """Verify forward passes for regression models."""
    device = get_device()
    mlp_model = MLPRegressor(8, 64).to(device)
    kan_model = KANRegressor(8, 64, grid_size=5, spline_order=3).to(device)
    x = torch.randn(16, 8, device=device)

    mlp_output = mlp_model(x)
    kan_output = kan_model(x)
    assert mlp_output.shape == (16, 1), f"Unexpected MLP output shape: {tuple(mlp_output.shape)}."
    assert kan_output.shape == (16, 1), f"Unexpected KAN output shape: {tuple(kan_output.shape)}."
    assert not torch.isnan(mlp_output).any(), "MLPRegressor output contains NaN."
    assert not torch.isnan(kan_output).any(), "KANRegressor output contains NaN."

    print(
        f"MLPRegressor params: {count_parameters(mlp_model):,} "
        f"({format_param_count(count_parameters(mlp_model))})"
    )
    print(
        f"KANRegressor params: {count_parameters(kan_model):,} "
        f"({format_param_count(count_parameters(kan_model))})"
    )


def test_classification_models_forward_pass() -> None:
    """Verify forward passes and parameter relationships for CIFAR models."""
    device = get_device()
    model_mlp = build_cifar10_model("mlp", hidden_dim=256).to(device)
    model_kan_a = build_cifar10_model("kan", hidden_dim=256).to(device)
    model_kan_b = build_cifar10_model("kan", hidden_dim=28).to(device)
    model_bspline = build_cifar10_model("bspline_mlp", hidden_dim=256).to(device)

    x = torch.randn(4, 3, 32, 32, device=device)
    outputs = {
        "MLPHead (256)": model_mlp(x),
        "KANHead ModeA (256)": model_kan_a(x),
        "KANHead ModeB (28)": model_kan_b(x),
        "BSplineMLPHead (256)": model_bspline(x),
    }
    for name, output in outputs.items():
        assert output.shape == (4, 10), f"{name} produced shape {tuple(output.shape)} instead of (4, 10)."
        assert not torch.isnan(output).any(), f"{name} output contains NaN."

    print("Model               | Total Params | Head Params")
    print("------------------------------------------------")
    models = {
        "MLPHead (256)": model_mlp,
        "KANHead ModeA (256)": model_kan_a,
        "KANHead ModeB (28)": model_kan_b,
        "BSplineMLPHead (256)": model_bspline,
    }
    for name, model in models.items():
        print(
            f"{name:<20} | "
            f"{count_parameters(model):>11,} | "
            f"{count_head_parameters(model):>11,}"
        )

    mlp_total = count_parameters(model_mlp)
    bspline_total = count_parameters(model_bspline)
    kan_a_head = count_head_parameters(model_kan_a)
    mlp_head = count_head_parameters(model_mlp)

    assert abs(bspline_total - mlp_total) / mlp_total < 0.05, (
        "BSplineMLP total parameters were not within 5% of the MLP total."
    )
    assert kan_a_head >= 5 * mlp_head, "KAN Mode A head was not at least 5x larger than the MLP head."


def test_gradient_flow_all_models() -> None:
    """Verify non-zero gradients across all regression and classification models."""
    device = get_device()
    regression_models = {
        "MLPRegressor": MLPRegressor(8, 64).to(device),
        "KANRegressor": KANRegressor(8, 64, grid_size=5, spline_order=3).to(device),
    }
    classification_models = {
        "CNN+MLPHead": build_cifar10_model("mlp", hidden_dim=256).to(device),
        "CNN+KANHead ModeA": build_cifar10_model("kan", hidden_dim=256).to(device),
        "CNN+KANHead ModeB": build_cifar10_model("kan", hidden_dim=28).to(device),
        "CNN+BSplineMLPHead": build_cifar10_model("bspline_mlp", hidden_dim=256).to(device),
    }

    for name, model in regression_models.items():
        model.zero_grad(set_to_none=True)
        x = torch.randn(8, 8, device=device)
        loss = model(x).sum()
        loss.backward()
        non_zero = sum(
            1 for param in model.parameters()
            if param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        )
        assert non_zero > 0, f"{name} had no non-zero gradients."

    criterion = nn.CrossEntropyLoss()
    for name, model in classification_models.items():
        model.zero_grad(set_to_none=True)
        x = torch.randn(8, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (8,), device=device)
        loss = criterion(model(x), labels)
        loss.backward()
        non_zero = sum(
            1 for param in model.parameters()
            if param.grad is not None and torch.count_nonzero(param.grad).item() > 0
        )
        assert non_zero > 0, f"{name} had no non-zero gradients."

    print("All 6 models have valid gradient flow")


def test_train_regression_mlp_end_to_end() -> None:
    """Train and evaluate the MLP regressor end to end."""
    set_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader, _ = get_california_housing(batch_size=64, num_workers=0)
    model = MLPRegressor(8, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=3,
        device=device,
        metric_fn=r2_metric,
        early_stopping_patience=0,
        model_name="integration_test_mlp_reg",
        verbose=True,
    )
    assert len(history["train_loss"]) == 3, "Expected 3 training-loss entries."
    assert len(history["val_loss"]) == 3, "Expected 3 validation-loss entries."
    assert len(history["val_metric"]) == 3, "Expected 3 validation-metric entries."
    assert all(epoch_time > 0 for epoch_time in history["time_per_epoch"]), "Non-positive epoch time recorded."
    assert history["train_loss"][2] < history["train_loss"][0], "MLP train loss did not decrease over 3 epochs."

    test_results = evaluate_model(model, test_loader, criterion, device, metric_fn=r2_metric)
    assert "test_loss" in test_results, "Missing test_loss in evaluation results."
    assert "test_metric" in test_results, "Missing test_metric in evaluation results."
    print(
        f"MLP Regression - Test MSE: {test_results['test_loss']:.4f}, "
        f"Test R^2: {test_results['test_metric']:.4f}"
    )


def test_train_regression_kan_end_to_end() -> None:
    """Train and evaluate the KAN regressor end to end."""
    set_seed(42)
    device = get_device()
    train_loader, val_loader, test_loader, _ = get_california_housing(batch_size=64, num_workers=0)
    model = KANRegressor(8, 64, grid_size=5, spline_order=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=3,
        device=device,
        metric_fn=r2_metric,
        early_stopping_patience=0,
        model_name="integration_test_kan_reg",
        verbose=True,
    )
    assert len(history["train_loss"]) == 3, "Expected 3 training-loss entries."
    assert history["train_loss"][2] < history["train_loss"][0], "KAN train loss did not decrease over 3 epochs."

    test_results = evaluate_model(model, test_loader, criterion, device, metric_fn=r2_metric)
    assert "test_loss" in test_results and "test_metric" in test_results
    print(
        f"KAN Regression - Test MSE: {test_results['test_loss']:.4f}, "
        f"Test R^2: {test_results['test_metric']:.4f}"
    )


def _classification_batch_size() -> int:
    """Return a CPU-aware batch size for the heavier CNN training tests."""
    return 128


def test_train_classification_mlp_end_to_end() -> None:
    """Train the CNN+MLP classifier for two epochs."""
    set_seed(42)
    device = get_device()
    batch_size = _classification_batch_size()
    train_loader, val_loader, _, _ = get_cifar10(
        batch_size=batch_size,
        data_dir="./data",
        num_workers=0,
    )
    model = build_cifar10_model("mlp", hidden_dim=256).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=2,
        device=device,
        metric_fn=accuracy_metric,
        early_stopping_patience=0,
        model_name="integration_test_cnn_mlp",
        verbose=True,
    )
    assert len(history["train_loss"]) == 2, "Expected 2 training epochs for CNN+MLP."
    assert history["val_metric"][1] > 0.15, (
        f"Expected validation accuracy above 0.15, got {history['val_metric'][1]:.4f}."
    )
    print(
        f"Time per epoch: {history['time_per_epoch'][0]:.1f}s, "
        f"{history['time_per_epoch'][1]:.1f}s"
    )
    print(f"CNN+MLP - Val Accuracy after 2 epochs: {history['val_metric'][1]:.4f}")


def test_train_classification_kan_mode_b_end_to_end() -> None:
    """Train the CNN+KAN Mode B classifier for two epochs."""
    set_seed(42)
    device = get_device()
    batch_size = _classification_batch_size()
    train_loader, val_loader, _, _ = get_cifar10(
        batch_size=batch_size,
        data_dir="./data",
        num_workers=0,
    )
    model = build_cifar10_model("kan", hidden_dim=28).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=2,
        device=device,
        metric_fn=accuracy_metric,
        early_stopping_patience=0,
        model_name="integration_test_cnn_kan_mode_b",
        verbose=True,
    )
    assert len(history["train_loss"]) == 2, "Expected 2 training epochs for CNN+KAN Mode B."
    assert history["val_metric"][1] > 0.10, (
        f"Expected validation accuracy above 0.10, got {history['val_metric'][1]:.4f}."
    )
    print(
        f"Time per epoch: {history['time_per_epoch'][0]:.1f}s, "
        f"{history['time_per_epoch'][1]:.1f}s"
    )
    print(f"CNN+KAN ModeB - Val Accuracy after 2 epochs: {history['val_metric'][1]:.4f}")


def test_train_classification_bspline_end_to_end() -> None:
    """Train the CNN+BSplineMLP classifier for two epochs."""
    set_seed(42)
    device = get_device()
    batch_size = _classification_batch_size()
    train_loader, val_loader, _, _ = get_cifar10(
        batch_size=batch_size,
        data_dir="./data",
        num_workers=0,
    )
    model = build_cifar10_model("bspline_mlp", hidden_dim=256).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=2,
        device=device,
        metric_fn=accuracy_metric,
        early_stopping_patience=0,
        model_name="integration_test_cnn_bspline",
        verbose=True,
    )
    assert len(history["train_loss"]) == 2, "Expected 2 training epochs for CNN+BSplineMLP."
    assert history["val_metric"][1] > 0.15, (
        f"Expected validation accuracy above 0.15, got {history['val_metric'][1]:.4f}."
    )
    print(
        f"Time per epoch: {history['time_per_epoch'][0]:.1f}s, "
        f"{history['time_per_epoch'][1]:.1f}s"
    )
    print(f"CNN+BSplineMLP - Val Accuracy after 2 epochs: {history['val_metric'][1]:.4f}")


def test_spline_visualization_pipeline() -> None:
    """Train a small KAN regressor and verify the spline visualization stack."""
    set_seed(42)
    device = torch.device("cpu")
    train_loader, val_loader, _, info = get_california_housing(batch_size=64, num_workers=0)
    model = KANRegressor(8, 32, grid_size=5, spline_order=3).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=10,
        device=device,
        metric_fn=r2_metric,
        early_stopping_patience=0,
        model_name="integration_test_spline_reg",
        verbose=False,
    )

    first_kan_layer = None
    for module in model.modules():
        if isinstance(module, KANLinear):
            first_kan_layer = module
            break

    assert first_kan_layer is not None, "Could not find KANLinear layer in KANRegressor."
    responses = probe_kan_layer_responses(first_kan_layer, device="cpu")
    assert responses["aggregated_responses"].shape == (8, 200), (
        f"Unexpected aggregated response shape: {responses['aggregated_responses'].shape}"
    )

    nonlinearity_scores = compute_feature_nonlinearity(responses)
    assert nonlinearity_scores.shape == (8,), "Expected 8 nonlinearity scores."
    assert np_all_non_negative(nonlinearity_scores), "Found negative nonlinearity score."

    output_path = os.path.join("results", "exp3", "integration_test_splines.png")
    plot_kan_splines(
        responses,
        feature_names=info["feature_names"],
        save_path=output_path,
    )
    register_artifact(output_path)
    assert os.path.exists(output_path), f"Spline plot was not created at {output_path}."

    ranking = sorted(
        zip(info["feature_names"], nonlinearity_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print("Feature nonlinearity ranking:")
    for feature_name, score in ranking:
        print(f"  {feature_name}: {score:.4f}")
    print("Spline visualization pipeline works end-to-end")


def np_all_non_negative(values: object) -> bool:
    """Return True if all numeric values are non-negative.

    Args:
        values: Array-like numeric object.

    Returns:
        Whether all entries are greater than or equal to zero.
    """
    tensor = torch.as_tensor(values)
    return bool(torch.all(tensor >= 0))


def test_save_load_results_csv() -> None:
    """Verify CSV save/load roundtrips."""
    results_dict = {
        "model": ["mlp", "kan"],
        "mse": [0.5, 0.3],
        "r2": [0.85, 0.91],
    }
    path = os.path.join("results", "integration_test.csv")
    save_results_csv(results_dict, path)
    register_artifact(path)

    df = load_results_csv(path)
    assert df.shape == (2, 3), f"Expected DataFrame shape (2, 3), got {df.shape}."
    assert df.loc[0, "model"] == "mlp"
    assert abs(float(df.loc[1, "mse"]) - 0.3) < 1e-9
    print("CSV save/load works")


def test_plotting_utilities() -> None:
    """Verify training-curve plotting utility."""
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "val_metric": [0.5, 0.7, 0.8],
    }
    path = os.path.join("results", "integration_test_curves.png")
    plot_training_curves(history, "Integration Test", save_path=path)
    register_artifact(path)
    assert os.path.exists(path), f"Expected plot file at {path}."
    print("Plotting works")


def test_convergence_epoch_computation() -> None:
    """Verify convergence epoch calculation."""
    history = {"val_metric": [0.3, 0.5, 0.7, 0.85, 0.88, 0.90, 0.91, 0.91]}
    convergence = compute_convergence_epoch(history, threshold_fraction=0.95)
    assert convergence == 5, f"Expected convergence epoch 5, got {convergence}."
    print(f"Convergence epoch: {convergence}")


def test_early_stopping_behavior() -> None:
    """Verify early stopping trigger behavior."""
    es = EarlyStopping(patience=3, min_delta=0.01)
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]
    trigger_index = None

    for idx, loss in enumerate(losses):
        if es(loss):
            trigger_index = idx
            break

    assert trigger_index is not None, "Early stopping never triggered."
    assert trigger_index in {5, 6}, f"Unexpected trigger index {trigger_index}."
    print("Early stopping triggers correctly")


def test_per_class_accuracy() -> None:
    """Verify per-class accuracy output for an untrained CIFAR model."""
    set_seed(42)
    device = get_device()
    batch_size = _classification_batch_size()
    _, _, test_loader, info = get_cifar10(
        batch_size=batch_size,
        data_dir="./data",
        num_workers=0,
    )
    model = build_cifar10_model("mlp", hidden_dim=256).to(device)
    class_acc = per_class_accuracy(model, test_loader, device, class_names=info["class_names"])

    assert len(class_acc) == 10, f"Expected 10 class entries, got {len(class_acc)}."
    values = list(class_acc.values())
    for class_name, value in class_acc.items():
        assert 0.0 <= value <= 1.0, f"Unexpected per-class accuracy for {class_name}: {value:.4f}"

    mean_per_class_accuracy = sum(values) / len(values)
    assert 0.0 <= mean_per_class_accuracy <= 0.3, (
        f"Expected mean per-class accuracy to stay near random-chance levels, got {mean_per_class_accuracy:.4f}."
    )

    print("Per-class accuracies:")
    for class_name, value in class_acc.items():
        print(f"  {class_name}: {value:.4f}")
    print(f"Mean per-class accuracy: {mean_per_class_accuracy:.4f}")
    print("Per-class accuracy works on untrained model")


def main() -> None:
    """Run all integration tests and print a summary."""
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "exp3"), exist_ok=True)

    print("Integration test note: CIFAR-10 training uses batch size 128 by default for speed.")

    tests: list[tuple[str, Callable[[], None]]] = [
        ("Imports", test_imports),
        ("Seed Reproducibility", test_seed_reproducibility),
        ("California Housing Data Pipeline", test_california_housing_data_pipeline),
        ("CIFAR-10 Data Pipeline", test_cifar10_data_pipeline),
        ("Regression Models Forward Pass", test_regression_models_forward_pass),
        ("Classification Models Forward Pass", test_classification_models_forward_pass),
        ("Gradient Flow - All Models", test_gradient_flow_all_models),
        ("Train Regression Model End-to-End (MLPRegressor, 3 epochs)", test_train_regression_mlp_end_to_end),
        ("Train Regression Model End-to-End (KANRegressor, 3 epochs)", test_train_regression_kan_end_to_end),
        ("Train Classification Model End-to-End (CNN+MLPHead, 2 epochs)", test_train_classification_mlp_end_to_end),
        ("Train Classification Model End-to-End (CNN+KANHead Mode B, 2 epochs)", test_train_classification_kan_mode_b_end_to_end),
        ("Train Classification Model End-to-End (CNN+BSplineMLPHead, 2 epochs)", test_train_classification_bspline_end_to_end),
        ("Spline Visualization Pipeline", test_spline_visualization_pipeline),
        ("Utilities - Save/Load Results CSV", test_save_load_results_csv),
        ("Utilities - Plotting", test_plotting_utilities),
        ("Convergence Epoch Computation", test_convergence_epoch_computation),
        ("Early Stopping", test_early_stopping_behavior),
        ("Per-Class Accuracy", test_per_class_accuracy),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'=' * 60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(1 for value in results.values() if value == "PASSED")
    failed = len(results) - passed
    for name, result in results.items():
        status = "[PASS]" if result == "PASSED" else "[FAIL]"
        print(f"  {status} {name}: {result}")
    print(f"\n{passed}/{len(results)} tests passed, {failed} failed")

    if failed == 0:
        print("ALL INTEGRATION TESTS PASSED - Phase 0 is complete. Ready for Phase 1.")
    else:
        print("SOME TESTS FAILED - fix the issues above before proceeding to Phase 1.")

    cleanup_artifacts()


if __name__ == "__main__":
    main()
