"""
KANLinear: Kolmogorov-Arnold Network Linear Layer

Adapted from efficient-kan by Blealtan (Huanqi Cao)
Source: https://github.com/Blealtan/efficient-kan
License: MIT

Original paper: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
https://arxiv.org/abs/2404.19756

The key efficiency insight: instead of expanding intermediate variables to
shape (batch, out_features, in_features) for per-edge activations, we compute
B-spline bases first, then combine via matrix multiplication. This makes
the computation equivalent to: activate input with basis functions, then
linearly combine — a standard matmul.
"""

from __future__ import annotations

import math
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

    # Include the rightmost endpoint in the final interval to avoid dropping
    # the last grid value due to the half-open interval definition.
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


def _fit_spline_coefficients(
    sample_points: torch.Tensor,
    sample_values: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Fit spline coefficients to match sampled function values.

    Args:
        sample_points: Query points with shape ``(num_samples, num_features)``.
        sample_values: Target values with shape
            ``(num_samples, num_features, out_features)``.
        grid: Knot tensor for the new spline basis.
        spline_order: B-spline order.

    Returns:
        Coefficients with shape ``(out_features, num_features, n_bases)``.
    """
    bases = _compute_b_splines(sample_points, grid, spline_order)
    num_samples, num_features, num_bases = bases.shape
    _, target_features, out_features = sample_values.shape

    if num_features != target_features:
        raise ValueError("sample_values must have the same num_features as sample_points.")

    coeffs = sample_values.new_zeros(out_features, num_features, num_bases)
    for feature_idx in range(num_features):
        design = bases[:, feature_idx, :]
        target = sample_values[:, feature_idx, :]
        solution = torch.linalg.lstsq(design, target).solution
        coeffs[:, feature_idx, :] = solution.transpose(0, 1)

    return coeffs


class KANLinear(nn.Module):
    """Efficient Kolmogorov-Arnold Network linear layer.

    This module combines a standard activated linear projection with an
    additive spline-based projection. The spline component places a learnable
    univariate function on each input-output edge, while still evaluating the
    layer through efficient matrix multiplication.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of intervals in the spline grid.
        spline_order: B-spline order, where ``3`` corresponds to cubic splines.
        scale_noise: Noise scale used to initialize spline coefficients.
        scale_base: Multiplicative factor applied to the initialized base weight.
        scale_spline: Multiplicative factor applied to the initialized spline weights.
        enable_standalone_scale_spline: Whether to learn an extra per-edge spline scale.
        base_activation: Activation module class applied before the base projection.
        grid_eps: Blending factor between adaptive quantile grids and uniform grids
            during ``update_grid``.
        grid_range: Inclusive minimum and maximum covered by the initial grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Sequence[float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if spline_order < 0:
            raise ValueError("spline_order must be non-negative.")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.base_activation = base_activation()

        n_bases = grid_size + spline_order
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_bases))

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)

        self.register_buffer(
            "grid",
            _build_grid(in_features, grid_size, spline_order, self.grid_range),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize module parameters."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)

            noise = torch.empty_like(self.spline_weight).uniform_(-1.0, 1.0)
            noise = noise * (self.scale_noise / max(self.grid_size, 1))
            self.spline_weight.copy_(noise * self.scale_spline)

        if self.spline_scaler is not None:
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Return spline coefficients after applying the optional learned scale."""
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions for the input tensor.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Basis activations with shape
            ``(batch_size, in_features, grid_size + spline_order)``.
        """
        return _compute_b_splines(x, self.grid, self.spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the KAN linear layer.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Output tensor of shape ``(batch_size, out_features)``.
        """
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input of shape (batch_size, {self.in_features}), "
                f"but received {tuple(x.shape)}."
            )

        base_output = F.linear(self.base_activation(x), self.base_weight)

        spline_bases = self.b_splines(x).reshape(x.shape[0], -1)
        spline_weight = self.scaled_spline_weight.reshape(self.out_features, -1)
        spline_output = F.linear(spline_bases, spline_weight)

        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the spline grid to the input distribution.

        The new grid is computed from feature-wise quantiles blended with a
        uniform grid. Spline coefficients are then refit so the effective spline
        function is approximately preserved under the new basis.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)`` used to adapt the grid.
            margin: Extra padding added to the observed min and max values.
        """
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input of shape (batch_size, {self.in_features}), "
                f"but received {tuple(x.shape)}."
            )
        if x.shape[0] < 2:
            return

        x_detached = x.detach()
        effective_weight = self.scaled_spline_weight.detach().clone()
        old_grid = self.grid.detach().clone()

        x_sorted = x_detached.sort(dim=0).values
        quantile_positions = torch.linspace(
            0,
            x_sorted.shape[0] - 1,
            steps=self.grid_size + 1,
            device=x_detached.device,
            dtype=x_detached.dtype,
        )

        lower_idx = quantile_positions.floor().long()
        upper_idx = quantile_positions.ceil().long()
        interp_weight = (quantile_positions - lower_idx).unsqueeze(-1)

        quantile_grid = (
            (1.0 - interp_weight) * x_sorted[lower_idx]
            + interp_weight * x_sorted[upper_idx]
        ).transpose(0, 1)

        data_min = x_detached.min(dim=0).values - margin
        data_max = x_detached.max(dim=0).values + margin
        uniform_positions = torch.linspace(
            0.0,
            1.0,
            steps=self.grid_size + 1,
            device=x_detached.device,
            dtype=x_detached.dtype,
        )
        uniform_grid = data_min.unsqueeze(1) + (data_max - data_min).unsqueeze(1) * uniform_positions

        blended_grid = self.grid_eps * uniform_grid + (1.0 - self.grid_eps) * quantile_grid
        step = (blended_grid[:, -1] - blended_grid[:, 0]).unsqueeze(1) / self.grid_size

        left_offsets = torch.arange(
            self.spline_order,
            0,
            -1,
            device=x_detached.device,
            dtype=x_detached.dtype,
        ).unsqueeze(0)
        right_offsets = torch.arange(
            1,
            self.spline_order + 1,
            device=x_detached.device,
            dtype=x_detached.dtype,
        ).unsqueeze(0)

        left_extension = blended_grid[:, :1] - left_offsets * step
        right_extension = blended_grid[:, -1:] + right_offsets * step
        new_grid = torch.cat((left_extension, blended_grid, right_extension), dim=1)

        num_samples = max(2 * (self.grid_size + self.spline_order), 16)
        sample_positions = torch.linspace(
            0.0,
            1.0,
            steps=num_samples,
            device=x_detached.device,
            dtype=x_detached.dtype,
        )
        interior_start = new_grid[:, self.spline_order]
        interior_end = new_grid[:, -(self.spline_order + 1)]
        sample_points = interior_start.unsqueeze(0) + (
            interior_end - interior_start
        ).unsqueeze(0) * sample_positions.unsqueeze(1)

        old_bases = _compute_b_splines(sample_points, old_grid, self.spline_order)
        sample_values = torch.einsum("bin,oin->bio", old_bases, effective_weight)

        new_coefficients = _fit_spline_coefficients(
            sample_points=sample_points,
            sample_values=sample_values,
            grid=new_grid,
            spline_order=self.spline_order,
        )

        self.grid.copy_(new_grid)
        if self.spline_scaler is None:
            self.spline_weight.copy_(new_coefficients)
        else:
            scaler = self.spline_scaler.unsqueeze(-1)
            restored = torch.where(
                scaler.abs() > 1e-7,
                new_coefficients / scaler,
                new_coefficients,
            )
            self.spline_weight.copy_(restored)


def _format_count(num_params: int) -> str:
    """Format parameter counts for readable smoke-test output.

    Args:
        num_params: Number of parameters.

    Returns:
        Human-readable parameter count string.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def _run_smoke_test(layer: KANLinear, batch_size: int = 32) -> None:
    """Run a forward and backward pass for a KANLinear instance.

    Args:
        layer: Layer to validate.
        batch_size: Batch size used in the random input.
    """
    x = torch.randn(batch_size, layer.in_features, requires_grad=True)
    y = layer(x)
    assert y.shape == (batch_size, layer.out_features)
    y.sum().backward()
    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None


if __name__ == "__main__":
    small_layer = KANLinear(8, 16, grid_size=5, spline_order=3)
    total_params = sum(param.numel() for param in small_layer.parameters())
    base_params = small_layer.base_weight.numel()
    spline_params = small_layer.spline_weight.numel()
    scaler_params = 0 if small_layer.spline_scaler is None else small_layer.spline_scaler.numel()

    print("KANLinear(8, 16)")
    print(f"  total params: {total_params} ({_format_count(total_params)})")
    print(f"  base_weight params: {base_params}")
    print(f"  spline_weight params: {spline_params}")
    print(f"  spline_scaler params: {scaler_params}")

    _run_smoke_test(small_layer, batch_size=32)

    large_layer = KANLinear(512, 256, grid_size=5, spline_order=3)
    large_params = sum(param.numel() for param in large_layer.parameters())
    print("KANLinear(512, 256)")
    print(f"  total params: {large_params} ({_format_count(large_params)})")
    _run_smoke_test(large_layer, batch_size=8)

    compact_layer = KANLinear(512, 28, grid_size=5, spline_order=3)
    compact_params = sum(param.numel() for param in compact_layer.parameters())
    print("KANLinear(512, 28)")
    print(f"  total params: {compact_params} ({_format_count(compact_params)})")
    _run_smoke_test(compact_layer, batch_size=8)

    print("KANLinear smoke test passed!")
