"""Evaluation metrics and test-set utilities for project experiments."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def accuracy_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy from raw logits.

    Args:
        predictions: Logits of shape ``(N, n_classes)``.
        targets: Integer class labels of shape ``(N,)``.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    predicted_labels = predictions.argmax(dim=1)
    return (predicted_labels == targets).float().mean().item()


def r2_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute the coefficient of determination (R²).

    Args:
        predictions: Model predictions of shape ``(N,)`` or ``(N, 1)``.
        targets: Regression targets of shape ``(N,)`` or ``(N, 1)``.

    Returns:
        R² score as a float.
    """
    preds = predictions.reshape(-1).float()
    trgs = targets.reshape(-1).float()

    ss_res = torch.sum((trgs - preds) ** 2)
    target_mean = torch.mean(trgs)
    ss_tot = torch.sum((trgs - target_mean) ** 2)

    if ss_tot.item() == 0.0:
        return 0.0

    return (1.0 - ss_res / ss_tot).item()


def mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean squared error.

    Args:
        predictions: Model predictions.
        targets: Ground-truth targets.

    Returns:
        Mean squared error as a float.
    """
    return ((predictions.float() - targets.float()) ** 2).mean().item()


def evaluate_model(
    model: nn.Module,
    test_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    metric_fn: MetricFn | None = None,
) -> dict[str, float]:
    """Evaluate a model on the test set.

    Args:
        model: Model to evaluate.
        test_loader: DataLoader yielding ``(inputs, targets)`` batches.
        criterion: Loss function for the current task.
        device: Device on which evaluation is executed.
        metric_fn: Optional metric function operating on all predictions and targets.

    Returns:
        Dictionary containing ``test_loss`` and optionally ``test_metric``.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            batch_size = inputs.shape[0]
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if metric_fn is not None:
                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())

    results: dict[str, float] = {
        "test_loss": running_loss / total_samples if total_samples > 0 else float("inf")
    }

    if metric_fn is not None and all_predictions:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        results["test_metric"] = metric_fn(predictions_tensor, targets_tensor)

    return results


def per_class_accuracy(
    model: nn.Module,
    test_loader: Any,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str | int, float]:
    """Compute per-class accuracy for a classification model.

    Args:
        model: Classification model producing logits.
        test_loader: DataLoader yielding ``(inputs, targets)`` batches.
        device: Device on which evaluation is executed.
        class_names: Optional class names aligned with label indices.

    Returns:
        Mapping from class label or class name to per-class accuracy.
    """
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            all_predictions.append(logits.argmax(dim=1).detach())
            all_targets.append(targets.detach())

    if not all_predictions:
        return {}

    predictions_tensor = torch.cat(all_predictions, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    class_indices = torch.unique(targets_tensor).tolist()
    results: dict[str | int, float] = {}
    for class_idx in class_indices:
        class_mask = targets_tensor == class_idx
        class_accuracy = (
            (predictions_tensor[class_mask] == targets_tensor[class_mask]).float().mean().item()
        )
        label: str | int
        if class_names is not None:
            label = class_names[int(class_idx)]
        else:
            label = int(class_idx)
        results[label] = class_accuracy

    return results


def get_classification_report(
    model: nn.Module,
    test_loader: Any,
    device: torch.device,
    class_names: list[str] | None = None,
) -> str:
    """Generate a scikit-learn classification report.

    Args:
        model: Classification model producing logits.
        test_loader: DataLoader yielding ``(inputs, targets)`` batches.
        device: Device on which evaluation is executed.
        class_names: Optional class names aligned with label indices.

    Returns:
        The classification report as a formatted string.
    """
    model.eval()
    predicted_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predicted_batches.append(logits.argmax(dim=1).detach().cpu())
            target_batches.append(targets.detach().cpu())

    if not predicted_batches:
        return "No samples available."

    predictions = torch.cat(predicted_batches, dim=0).numpy()
    targets = torch.cat(target_batches, dim=0).numpy()

    if class_names is not None:
        return classification_report(targets, predictions, target_names=class_names, digits=4)
    return classification_report(targets, predictions, digits=4)


def compute_convergence_epoch(
    history: dict[str, Any],
    threshold_fraction: float = 0.95,
) -> int:
    """Compute the epoch where the model first reaches near-final performance.

    Args:
        history: History dictionary returned by ``train_model``.
        threshold_fraction: Fraction of best validation performance used as the threshold.

    Returns:
        The 1-indexed epoch where the threshold is first met.
    """
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in the interval (0, 1].")

    if "val_metric" in history and history["val_metric"]:
        metrics = np.asarray(history["val_metric"], dtype=np.float64)
        target_value = threshold_fraction * np.nanmax(metrics)
        for epoch_idx, metric_value in enumerate(metrics, start=1):
            if metric_value >= target_value:
                return epoch_idx
        return len(metrics)

    losses = np.asarray(history["val_loss"], dtype=np.float64)
    target_loss = (1.0 / threshold_fraction) * np.nanmin(losses)
    for epoch_idx, loss_value in enumerate(losses, start=1):
        if loss_value <= target_loss:
            return epoch_idx
    return len(losses)


def _run_smoke_test() -> None:
    """Run smoke tests for evaluation metrics and helpers."""
    predictions = torch.tensor(
        [[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [0.5, 0.1, 2.5]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    assert accuracy_metric(predictions, targets) == 1.0

    imperfect_targets = torch.tensor([1, 1, 2], dtype=torch.long)
    assert abs(accuracy_metric(predictions, imperfect_targets) - (2.0 / 3.0)) < 1e-6

    regression_targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    regression_predictions = regression_targets.clone()
    assert abs(r2_metric(regression_predictions, regression_targets) - 1.0) < 1e-6

    mse_predictions = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    mse_targets = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert mse_metric(mse_predictions, mse_targets) == 0.0

    history = {"val_metric": [0.5, 0.7, 0.85, 0.90, 0.92, 0.93, 0.93]}
    assert compute_convergence_epoch(history, threshold_fraction=0.95) == 4

    print("evaluate.py smoke test passed!")


if __name__ == "__main__":
    _run_smoke_test()
