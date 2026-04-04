"""
BSplineActivation: Learnable B-Spline Activation Function

A drop-in replacement for nn.ReLU() that applies an independent learnable
B-spline activation function to each feature dimension. Unlike KANLinear,
this module does NOT mix features — it only transforms each feature
independently through its own learned nonlinearity.

This is used in our BSpline-MLP variant to test the hypothesis from
Yu et al. (2024) "KAN or MLP: A Fairer Comparison" — that KAN's
advantage stems primarily from its learnable activation functions,
not from its structural redesign of the weight-activation relationship.

Usage:
    # As a drop-in replacement for ReLU in an MLP:
    mlp_head = nn.Sequential(
        nn.Linear(512, 256),
        BSplineActivation(256),  # instead of nn.ReLU()
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_grid(
    num_features: int,
    grid_size: int,
    spline_order: int,
    grid_range: Sequence[float],
) -> torch.Tensor:
    """Construct an extended knot grid for each feature.

    Args:
        num_features: Number of independent feature dimensions.
        grid_size: Number of intervals in the interior grid.
        spline_order: B-spline order.
        grid_range: Inclusive minimum and maximum values covered by the grid.

    Returns:
        A tensor of shape ``(num_features, grid_size + 2 * spline_order + 1)``.
    """
    start, end = float(grid_range[0]), float(grid_range[1])
    if end <= start:
        raise ValueError("grid_range must satisfy grid_range[1] > grid_range[0].")

    step = (end - start) / grid_size
    base_grid = torch.linspace(
        start - spline_order * step,
        end + spline_order * step,
        steps=grid_size + 2 * spline_order + 1,
        dtype=torch.float32,
    )
    return base_grid.unsqueeze(0).repeat(num_features, 1)


def _compute_b_splines(
    x: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Compute B-spline basis functions with Cox-de Boor recursion.

    Args:
        x: Input tensor of shape ``(batch_size, num_features)``.
        grid: Knot tensor of shape
            ``(num_features, grid_size + 2 * spline_order + 1)``.
        spline_order: B-spline order.

    Returns:
        A tensor of shape ``(batch_size, num_features, grid_size + spline_order)``.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (batch_size, num_features).")
    if x.shape[1] != grid.shape[0]:
        raise ValueError("x.shape[1] must match the number of feature grids.")

    x_expanded = x.unsqueeze(-1)
    left = grid[:, :-1].unsqueeze(0)
    right = grid[:, 1:].unsqueeze(0)

    bases = ((x_expanded >= left) & (x_expanded < right)).to(dtype=x.dtype)

    right_endpoint = grid[:, -1].unsqueeze(0)
    right_mask = x.eq(right_endpoint)
    if right_mask.any():
        bases[right_mask, -1] = 1.0

    for order in range(1, spline_order + 1):
        left_num = x_expanded - grid[:, : -(order + 1)].unsqueeze(0)
        left_den = grid[:, order:-1].unsqueeze(0) - grid[:, : -(order + 1)].unsqueeze(0)
        left_term = torch.where(
            left_den.abs() > 1e-7,
            (left_num / left_den) * bases[:, :, :-1],
            torch.zeros_like(bases[:, :, :-1]),
        )

        right_num = grid[:, order + 1 :].unsqueeze(0) - x_expanded
        right_den = grid[:, order + 1 :].unsqueeze(0) - grid[:, 1:-order].unsqueeze(0)
        right_term = torch.where(
            right_den.abs() > 1e-7,
            (right_num / right_den) * bases[:, :, 1:],
            torch.zeros_like(bases[:, :, 1:]),
        )

        bases = left_term + right_term

    return bases.contiguous()


class BSplineActivation(nn.Module):
    """Feature-wise learnable B-spline activation module.

    Each feature receives its own independent spline nonlinearity, making this
    module a direct nonlinear replacement for activations such as ``nn.ReLU``.
    A LayerNorm + SiLU residual path is included to improve stability and allow
    identity-like behavior.

    Args:
        num_features: Number of input features.
        grid_size: Number of intervals in the spline grid.
        spline_order: B-spline order, where ``3`` corresponds to cubic splines.
        grid_range: Inclusive minimum and maximum covered by the spline grid.
    """

    def __init__(
        self,
        num_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()

        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if spline_order < 0:
            raise ValueError("spline_order must be non-negative.")

        self.num_features = num_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        self.norm = nn.LayerNorm(num_features)
        self.spline_weight = nn.Parameter(
            torch.empty(num_features, grid_size + spline_order)
        )
        self.base_weight = nn.Parameter(torch.ones(num_features))
        self.base_activation = nn.SiLU()

        self.register_buffer(
            "grid",
            _build_grid(num_features, grid_size, spline_order, grid_range),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable activation parameters."""
        with torch.no_grad():
            self.spline_weight.uniform_(-0.1, 0.1)
            self.base_weight.fill_(1.0)

        self.norm.reset_parameters()

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions for each feature.

        Args:
            x: Input tensor of shape ``(batch_size, num_features)``.

        Returns:
            Basis activations with shape
            ``(batch_size, num_features, grid_size + spline_order)``.
        """
        return _compute_b_splines(x, self.grid, self.spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learnable B-spline activation.

        Args:
            x: Input tensor whose last dimension equals ``num_features``.

        Returns:
            Tensor with the same shape as ``x``.
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected input with last dimension {self.num_features}, "
                f"but received {tuple(x.shape)}."
            )

        original_shape = x.shape
        x_flat = x.reshape(-1, self.num_features)

        normed_input = self.norm(x_flat)
        bases = self.b_splines(normed_input)
        spline_output = (bases * self.spline_weight.unsqueeze(0)).sum(dim=-1)
        residual_output = self.base_weight.unsqueeze(0) * self.base_activation(normed_input)

        output = spline_output + residual_output
        return output.reshape(original_shape)


def _run_smoke_test() -> None:
    """Validate the BSplineActivation module with a small battery of checks."""
    activation = BSplineActivation(256, grid_size=5, spline_order=3)
    total_params = sum(param.numel() for param in activation.parameters())

    print("BSplineActivation(256)")
    print(f"  total params: {total_params}")

    x = torch.randn(32, 256, requires_grad=True)
    y = activation(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert activation.spline_weight.grad is not None
    assert activation.base_weight.grad is not None

    large_magnitude = (torch.rand(32, 256) * 20.0 - 10.0).requires_grad_(True)
    large_output = activation(large_magnitude)
    assert torch.isfinite(large_output).all()
    large_output.sum().backward()

    single_batch = torch.randn(1, 256, requires_grad=True)
    single_output = activation(single_batch)
    assert single_output.shape == single_batch.shape
    single_output.sum().backward()

    print("BSplineActivation smoke test passed!")


if __name__ == "__main__":
    _run_smoke_test()
