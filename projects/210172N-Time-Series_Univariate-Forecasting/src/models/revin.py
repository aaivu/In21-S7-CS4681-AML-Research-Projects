"""
Reversible Instance Normalization (RevIN) for Time Series.

RevIN helps mitigate distribution shift in time series by normalizing each
instance (time series) independently. It's particularly useful for handling
heterogeneous time series with different scales.

Reference:
    Kim et al., "Reversible Instance Normalization for Accurate Time-Series
    Forecasting against Distribution Shift", ICLR 2022.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (ONNX-compatible version).

    This implementation is designed for ONNX export compatibility by storing
    normalization statistics as buffers that can be properly serialized.

    Args:
        num_features: Number of features/channels in the time series
        eps: Small value for numerical stability (default: 1e-5)
        affine: If True, use learnable affine parameters (default: True)

    Shape:
        - Input: (batch_size, seq_len, num_features)
        - Output: (batch_size, seq_len, num_features)

    Examples:
        >>> revin = RevIN(num_features=7, affine=True)
        >>> x = torch.randn(32, 336, 7)  # (batch, seq_len, features)
        >>> x_norm = revin(x, mode='norm')  # Normalize
        >>> x_denorm = revin(x_norm, mode='denorm')  # Denormalize
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # Learnable affine parameters
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        # Register buffers for storing statistics (required for ONNX export)
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('stdev', torch.ones(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass with normalization or denormalization.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            mode: Operation mode, either 'norm' or 'denorm'

        Returns:
            Normalized or denormalized tensor

        Raises:
            NotImplementedError: If mode is not 'norm' or 'denorm'
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise NotImplementedError(
                f"Mode '{mode}' not supported. Use 'norm' or 'denorm'."
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input time series.

        Computes mean and standard deviation along the time dimension and
        normalizes each instance independently.

        Args:
            x: Input tensor (batch, seq_len, num_features)

        Returns:
            Normalized tensor
        """
        # Compute statistics along time dimension (dim=1)
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

        # Normalize
        x = (x - self.mean) / self.stdev

        # Apply learnable affine transformation
        if self.affine:
            x = x * self.affine_weight + self.affine_bias

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the output back to original scale.

        Uses the stored mean and standard deviation from normalization step.

        Args:
            x: Normalized tensor (batch, pred_len, num_features)

        Returns:
            Denormalized tensor
        """
        # Reverse affine transformation
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        # Denormalize
        x = x * self.stdev + self.mean

        return x

    def extra_repr(self) -> str:
        """Extra information for print representation."""
        return f'num_features={self.num_features}, eps={self.eps}, affine={self.affine}'


class RevINInline(nn.Module):
    """
    Inline RevIN implementation for full ONNX compatibility.

    This version performs normalization/denormalization inline without storing
    statistics in separate buffers, making it fully compatible with ONNX export
    for deployment.

    Note: This is used internally in PatchTSTModel for ONNX export.
    The statistics (mean, stdev) are computed on-the-fly and passed through
    the forward pass.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(RevINInline, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable affine parameters
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize and return statistics.

        Args:
            x: Input tensor (batch, seq_len, num_features)

        Returns:
            Tuple of (normalized_tensor, mean, stdev)
        """
        # Compute statistics
        dim2reduce = tuple(range(1, x.ndim - 1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        )

        # Normalize
        x_norm = (x - mean) / stdev
        x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm, mean, stdev

    def denormalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize using provided statistics.

        Args:
            x: Normalized tensor
            mean: Mean tensor from normalization
            stdev: Standard deviation tensor from normalization

        Returns:
            Denormalized tensor
        """
        # Reverse affine
        x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        # Denormalize
        x = x * stdev + mean

        return x
