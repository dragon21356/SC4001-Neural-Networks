"""Generic training utilities shared across project experiments."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import get_device
from src.utils import set_seed

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


class EarlyStopping:
    """Track validation loss improvements and trigger early stopping.

    Args:
        patience: Number of epochs to wait without improvement.
        min_delta: Minimum decrease in validation loss to count as improvement.
        mode: Optimization mode. Only ``"min"`` is supported in this project.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode != "min":
            raise ValueError("EarlyStopping currently supports mode='min' only.")
        if patience < 0:
            raise ValueError("patience must be non-negative.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """Update the early-stopping state.

        Args:
            val_loss: Validation loss from the current epoch.

        Returns:
            ``True`` if training should stop, otherwise ``False``.
        """
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train a model for a single epoch.

    Args:
        model: Model to train.
        train_loader: DataLoader yielding ``(inputs, targets)`` batches.
        optimizer: Optimizer used for parameter updates.
        criterion: Loss function for the current task.
        device: Device on which training is executed.

    Returns:
        Dictionary containing the mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = running_loss / total_samples if total_samples > 0 else float("inf")
    return {"train_loss": average_loss}


def validate(
    model: nn.Module,
    val_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    metric_fn: MetricFn | None = None,
) -> dict[str, float]:
    """Evaluate a model on a validation or test split.

    Args:
        model: Model to evaluate.
        val_loader: DataLoader yielding ``(inputs, targets)`` batches.
        criterion: Loss function for the current task.
        device: Device on which evaluation is executed.
        metric_fn: Optional metric function operating on all predictions and targets.

    Returns:
        Dictionary containing average validation loss and, when requested,
        a validation metric.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in val_loader:
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

    average_loss = running_loss / total_samples if total_samples > 0 else float("inf")
    results: dict[str, float] = {"val_loss": average_loss}

    if metric_fn is not None and all_predictions:
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        results["val_metric"] = metric_fn(predictions_tensor, targets_tensor)

    return results


def train_model(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    metric_fn: MetricFn | None = None,
    scheduler: Any | None = None,
    early_stopping_patience: int | None = 10,
    checkpoint_dir: str | None = None,
    model_name: str = "model",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a model for multiple epochs with optional early stopping.

    Args:
        model: Model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer used for parameter updates.
        criterion: Loss function for the current task.
        epochs: Maximum number of epochs to train.
        device: Device on which training is executed.
        metric_fn: Optional metric function operating on all predictions and targets.
        scheduler: Optional learning rate scheduler.
        early_stopping_patience: Patience for early stopping. Set to ``0`` or
            ``None`` to disable it.
        checkpoint_dir: Optional directory used to store the best checkpoint.
        model_name: Prefix used for checkpoint naming and logging.
        verbose: Whether to print per-epoch progress.

    Returns:
        Training history dictionary containing losses, optional metrics, and
        timing information.
    """
    model.to(device)

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "time_per_epoch": [],
    }
    if metric_fn is not None:
        history["val_metric"] = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metric: float | None = None
    checkpoint_path: str | None = None

    early_stopper = None
    if early_stopping_patience is not None and early_stopping_patience > 0:
        early_stopper = EarlyStopping(patience=early_stopping_patience)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_results = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_results = validate(model, val_loader, criterion, device, metric_fn)

            epoch_time = time.time() - epoch_start
            history["train_loss"].append(train_results["train_loss"])
            history["val_loss"].append(val_results["val_loss"])
            history["time_per_epoch"].append(epoch_time)
            if metric_fn is not None:
                history["val_metric"].append(val_results.get("val_metric", float("nan")))

            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_results["val_loss"])
                else:
                    scheduler.step()

            if val_results["val_loss"] < best_val_loss:
                best_val_loss = val_results["val_loss"]
                best_epoch = epoch
                best_val_metric = val_results.get("val_metric")
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)

            if verbose:
                current_lr = optimizer.param_groups[0]["lr"]
                message = (
                    f"Epoch [{epoch:02d}/{epochs}] | "
                    f"Train Loss: {train_results['train_loss']:.4f} | "
                    f"Val Loss: {val_results['val_loss']:.4f}"
                )
                if "val_metric" in val_results:
                    message += f" | Val Metric: {val_results['val_metric']:.4f}"
                message += f" | Time: {epoch_time:.1f}s | LR: {current_lr:.2e}"
                print(message)

            if early_stopper is not None and early_stopper(val_results["val_loss"]):
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break
    except KeyboardInterrupt:
        if verbose:
            print("Training interrupted by user. Returning collected history so far.")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["best_val_metric"] = best_val_metric
    history["total_time"] = sum(history["time_per_epoch"])
    history["epochs_trained"] = len(history["train_loss"])

    return history


def _run_smoke_test() -> None:
    """Run a smoke test for the training loop."""
    set_seed(42)
    device = get_device()

    x = torch.randn(200, 8)
    y = torch.randn(200, 1)

    train_dataset = torch.utils.data.TensorDataset(x[:160], y[:160])
    val_dataset = torch.utils.data.TensorDataset(x[160:], y[160:])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=5,
        device=device,
        metric_fn=None,
        scheduler=None,
        early_stopping_patience=0,
        checkpoint_dir=None,
        model_name="smoke_model",
        verbose=True,
    )

    expected_keys = {
        "train_loss",
        "val_loss",
        "time_per_epoch",
        "best_epoch",
        "best_val_loss",
        "best_val_metric",
        "total_time",
        "epochs_trained",
    }
    assert expected_keys.issubset(history.keys())
    assert len(history["train_loss"]) == 5
    assert len(history["val_loss"]) == 5
    assert len(history["time_per_epoch"]) == 5
    assert all(epoch_time > 0 for epoch_time in history["time_per_epoch"])

    print("train.py smoke test passed!")


if __name__ == "__main__":
    _run_smoke_test()
