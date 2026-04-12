"""Model definitions for regression and CIFAR-10 experiments."""

from __future__ import annotations

try:
    from typing import TypeAlias
except ImportError:  # Python < 3.10
    from typing_extensions import TypeAlias

import torch
import torch.nn as nn

from src.bspline_activation import BSplineActivation
from src.kan_layer import KANLinear

HeadType: TypeAlias = nn.Module


def _format_param_count(num_params: int) -> str:
    """Format a parameter count into a compact human-readable string.

    Args:
        num_params: Number of parameters.

    Returns:
        A formatted string such as ``132.1K`` or ``1.31M``.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


class MLPRegressor(nn.Module):
    """Standard multilayer perceptron regressor for tabular data.

    Args:
        in_features: Number of input features.
        hidden_dim: Width of the hidden layers.
        out_features: Number of regression outputs.
        dropout: Dropout probability used after each hidden activation.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 64,
        out_features: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the regression model.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Regression predictions of shape ``(batch_size, out_features)``.
        """
        return self.network(x)


class KANRegressor(nn.Module):
    """KAN-based regressor for tabular data.

    Args:
        in_features: Number of input features.
        hidden_dim: Width of the hidden KAN layer.
        out_features: Number of regression outputs.
        grid_size: Number of spline intervals.
        spline_order: B-spline order.
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 64,
        out_features: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.layer1 = KANLinear(
            in_features,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.layer2 = KANLinear(
            hidden_dim,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the KAN regressor.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Regression predictions of shape ``(batch_size, out_features)``.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class CNNBackbone(nn.Module):
    """Four-block convolutional backbone for CIFAR-10 feature extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self._initialize_weights()

    @property
    def output_dim(self) -> int:
        """Return the feature dimension produced by the backbone."""
        return 512

    def _initialize_weights(self) -> None:
        """Initialize convolutional and batch-normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 512-dimensional feature vector from an image batch.

        Args:
            x: Input tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Feature tensor of shape ``(batch_size, 512)``.
        """
        x = self.features(x)
        return self.flatten(x)


class MLPHead(nn.Module):
    """Standard MLP classification head with ReLU activation.

    Args:
        in_features: Number of input features.
        hidden_dim: Width of the hidden layer.
        n_classes: Number of output classes.
        dropout: Dropout probability used after the hidden activation.
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers with Kaiming normal weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            x: Feature tensor of shape ``(batch_size, in_features)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        return self.network(x)


class KANHead(nn.Module):
    """KAN-based classification head with two KANLinear layers.

    Args:
        in_features: Number of input features.
        hidden_dim: Width of the hidden KAN layer.
        n_classes: Number of output classes.
        grid_size: Number of spline intervals.
        spline_order: B-spline order.
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 10,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.layer1 = KANLinear(
            in_features,
            hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.layer2 = KANLinear(
            hidden_dim,
            n_classes,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits with a KAN head.

        Args:
            x: Feature tensor of shape ``(batch_size, in_features)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class BSplineMLPHead(nn.Module):
    """MLP head that replaces ReLU with a learnable B-spline activation.

    Args:
        in_features: Number of input features.
        hidden_dim: Width of the hidden layer.
        n_classes: Number of output classes.
        dropout: Dropout probability used after the spline activation.
        grid_size: Number of spline intervals.
        spline_order: B-spline order.
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 10,
        dropout: float = 0.3,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            BSplineActivation(
                hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            ),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers with Kaiming normal weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            x: Feature tensor of shape ``(batch_size, in_features)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        return self.network(x)


class CNNWithHead(nn.Module):
    """Compose a CNN backbone with a classification head.

    Args:
        backbone: Feature extractor returning ``(batch_size, feature_dim)``.
        head: Classification head mapping features to logits.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the composed image classifier.

        Args:
            x: Input tensor of shape ``(batch_size, 3, 32, 32)``.

        Returns:
            Logits of shape ``(batch_size, n_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)


def build_cifar10_model(
    head_type: str,
    hidden_dim: int = 256,
    grid_size: int = 5,
    spline_order: int = 3,
    dropout: float = 0.3,
) -> CNNWithHead:
    """Build a CIFAR-10 model with a fresh backbone and selected head.

    Args:
        head_type: Head variant to build. One of ``"mlp"``, ``"kan"``, or
            ``"bspline_mlp"``.
        hidden_dim: Width of the hidden layer for the selected head.
        grid_size: Number of spline intervals used by spline-based heads.
        spline_order: B-spline order used by spline-based heads.
        dropout: Dropout probability for MLP-style heads.

    Returns:
        A composed ``CNNWithHead`` model.

    Raises:
        ValueError: If ``head_type`` is unknown.
    """
    backbone = CNNBackbone()

    if head_type == "mlp":
        head: HeadType = MLPHead(
            in_features=backbone.output_dim,
            hidden_dim=hidden_dim,
            n_classes=10,
            dropout=dropout,
        )
    elif head_type == "kan":
        head = KANHead(
            in_features=backbone.output_dim,
            hidden_dim=hidden_dim,
            n_classes=10,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    elif head_type == "bspline_mlp":
        head = BSplineMLPHead(
            in_features=backbone.output_dim,
            hidden_dim=hidden_dim,
            n_classes=10,
            dropout=dropout,
            grid_size=grid_size,
            spline_order=spline_order,
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    return CNNWithHead(backbone=backbone, head=head)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: Model to inspect.

    Returns:
        Number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_head_parameters(model: CNNWithHead) -> int:
    """Count trainable parameters in the classification head only.

    Args:
        model: Composed CNN classifier.

    Returns:
        Number of trainable parameters in ``model.head``.
    """
    return sum(param.numel() for param in model.head.parameters() if param.requires_grad)


def model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> None:
    """Print a compact model summary and verify a forward pass.

    Args:
        model: Model to summarize.
        input_shape: Shape of the synthetic input tensor used for testing.
    """
    total_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {_format_param_count(total_params)} ({total_params:,})")

    if isinstance(model, CNNWithHead):
        head_params = count_head_parameters(model)
        backbone_params = count_parameters(model.backbone)
        print(
            f"Backbone parameters: {_format_param_count(backbone_params)} "
            f"({backbone_params:,})"
        )
        print(f"Head parameters: {_format_param_count(head_params)} ({head_params:,})")

    device = next(model.parameters()).device
    sample = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        output = model(sample)

    print(f"Output shape: {tuple(output.shape)}")
    print("-" * 50)


def _run_regression_smoke_test() -> None:
    """Run smoke tests for the tabular regression models."""
    x = torch.randn(32, 8)
    mlp_regressor = MLPRegressor(in_features=8, hidden_dim=64)
    kan_regressor = KANRegressor(in_features=8, hidden_dim=64, grid_size=5, spline_order=3)

    mlp_output = mlp_regressor(x)
    kan_output = kan_regressor(x)
    assert mlp_output.shape == (32, 1)
    assert kan_output.shape == (32, 1)

    mlp_params = count_parameters(mlp_regressor)
    kan_params = count_parameters(kan_regressor)
    print("Regression parameter comparison:")
    print(f"  MLPRegressor: {mlp_params:,} params")
    print(f"  KANRegressor: {kan_params:,} params")
    print("Regression models smoke test passed!")


def _run_backbone_smoke_test() -> None:
    """Run smoke tests for the CNN backbone."""
    backbone = CNNBackbone()
    x = torch.randn(4, 3, 32, 32)
    y = backbone(x)
    assert y.shape == (4, 512)
    print(f"CNNBackbone params: {count_parameters(backbone):,}")
    print("CNNBackbone smoke test passed!")


def _run_head_smoke_test() -> None:
    """Run smoke tests for standalone classification heads."""
    x = torch.randn(4, 512)
    heads = {
        "MLPHead (hidden=256)": MLPHead(512, 256, 10),
        "KANHead Mode A (hidden=256)": KANHead(512, 256, 10),
        "KANHead Mode B (hidden=28)": KANHead(512, 28, 10),
        "BSplineMLPHead (hidden=256)": BSplineMLPHead(512, 256, 10),
    }

    param_counts: dict[str, int] = {}
    for name, head in heads.items():
        logits = head(x)
        assert logits.shape == (4, 10)
        param_counts[name] = count_parameters(head)
        print(f"{name}: {param_counts[name]:,} params")

    print("Head Comparison:")
    print(f"MLPHead (hidden=256):         {param_counts['MLPHead (hidden=256)']:,} params")
    print(
        "KANHead Mode A (hidden=256):  "
        f"{param_counts['KANHead Mode A (hidden=256)']:,} params"
    )
    print(
        "KANHead Mode B (hidden=28):   "
        f"{param_counts['KANHead Mode B (hidden=28)']:,} params"
    )
    print(
        "BSplineMLPHead (hidden=256):  "
        f"{param_counts['BSplineMLPHead (hidden=256)']:,} params"
    )

    mlp_params = param_counts["MLPHead (hidden=256)"]
    kan_mode_b_params = param_counts["KANHead Mode B (hidden=28)"]
    bspline_params = param_counts["BSplineMLPHead (hidden=256)"]

    assert abs(kan_mode_b_params - mlp_params) / mlp_params < 0.2
    assert abs(bspline_params - mlp_params) / mlp_params < 0.1
    print("Classification heads smoke test passed!")


def _run_full_model_smoke_test() -> dict[str, CNNWithHead]:
    """Run smoke tests for composed CIFAR-10 models.

    Returns:
        The composed models built during the smoke test.
    """
    models = {
        "mlp": build_cifar10_model("mlp", hidden_dim=256),
        "kan_a": build_cifar10_model("kan", hidden_dim=256),
        "kan_b": build_cifar10_model("kan", hidden_dim=28),
        "bspline_mlp": build_cifar10_model("bspline_mlp", hidden_dim=256),
    }

    x = torch.randn(4, 3, 32, 32)
    for name, model in models.items():
        model.zero_grad(set_to_none=True)
        logits = model(x)
        assert logits.shape == (4, 10)
        logits.sum().backward()
        print(f"{name}:")
        print(f"  total params: {count_parameters(model):,}")
        print(f"  backbone params: {count_parameters(model.backbone):,}")
        print(f"  head params: {count_head_parameters(model):,}")

    print("Full model smoke test passed!")
    return models


if __name__ == "__main__":
    _run_regression_smoke_test()
    _run_backbone_smoke_test()
    _run_head_smoke_test()
    composed_models = _run_full_model_smoke_test()

    for model_name, model in composed_models.items():
        print(f"Summary for {model_name}:")
        model_summary(model, input_shape=(4, 3, 32, 32))
