"""
PatchTST: A Time Series is Worth 64 Words

Production-ready implementation of PatchTST for M4 Competition with ONNX
compatibility and real-time optimization support.

Reference:
    Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers", ICLR 2023.

Key Features:
    - Channel-independent processing for multivariate series
    - Patching mechanism to reduce computational complexity
    - ONNX-compatible architecture for deployment
    - RevIN for distribution shift mitigation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class PatchTSTModel(nn.Module):
    """
    PatchTST model with ONNX-compatible architecture.

    This implementation uses:
    - Manual patching (ONNX compatible, no unfold)
    - Inline RevIN normalization
    - Channel-independent processing
    - Standard Transformer encoder layers

    Args:
        c_in: Number of input channels/features
        seq_len: Input sequence length (look-back window)
        pred_len: Prediction horizon (forecast length)
        patch_len: Length of each patch
        stride: Stride between patches
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        d_ff: Feed-forward network dimension
        dropout: Dropout rate
        use_revin: Whether to use RevIN normalization
        revin_affine: Whether to use affine parameters in RevIN

    Shape:
        - Input: (batch, seq_len, c_in)
        - Output: (batch, pred_len, c_in)

    Examples:
        >>> model = PatchTSTModel(
        ...     c_in=7, seq_len=336, pred_len=96,
        ...     patch_len=16, stride=8,
        ...     d_model=128, n_heads=16, e_layers=3
        ... )
        >>> x = torch.randn(32, 336, 7)
        >>> y = model(x)
        >>> y.shape
        torch.Size([32, 96, 7])
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        use_revin: bool = True,
        revin_affine: bool = True,
    ):
        super().__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin
        self.eps = 1e-5

        # Compute number of patches
        self.patch_num = (seq_len - patch_len) // stride + 1

        # RevIN parameters (inline for ONNX compatibility)
        if use_revin and revin_affine:
            self.affine_weight = nn.Parameter(torch.ones(c_in))
            self.affine_bias = nn.Parameter(torch.zeros(c_in))
        else:
            self.register_buffer('affine_weight', torch.ones(c_in))
            self.register_buffer('affine_bias', torch.zeros(c_in))

        # Patch embedding: Linear projection from patch to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        self._init_pos_encoding()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Prediction head: Project from flattened patches to prediction
        self.head = nn.Linear(d_model * self.patch_num, pred_len)

        # Initialize weights
        self._init_weights()

    def _init_pos_encoding(self):
        """Initialize positional encoding with sinusoidal patterns."""
        position = torch.arange(0, self.patch_num, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pos_encoding.size(-1), 2).float()
            * (-math.log(10000.0) / self.pos_encoding.size(-1))
        )

        self.pos_encoding.data[:, :, 0::2] = torch.sin(position * div_term)
        self.pos_encoding.data[:, :, 1::2] = torch.cos(position * div_term)

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patches from input using ONNX-compatible manual extraction.

        Args:
            x: Input tensor (batch, c_in, seq_len)

        Returns:
            Patches tensor (batch * c_in, patch_num, patch_len)
        """
        batch_size, c_in, seq_len = x.shape
        patches = []

        # Extract patches manually (ONNX compatible)
        for i in range(self.patch_num):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, :, start_idx:end_idx]  # (batch, c_in, patch_len)
            patches.append(patch)

        # Stack patches: (patch_num, batch, c_in, patch_len)
        patches = torch.stack(patches, dim=0)

        # Reshape to: (batch, c_in, patch_num, patch_len)
        patches = patches.permute(1, 2, 0, 3)

        # Flatten batch and channels: (batch * c_in, patch_num, patch_len)
        patches = patches.reshape(batch_size * c_in, self.patch_num, self.patch_len)

        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PatchTST.

        Args:
            x: Input tensor (batch, seq_len, c_in)

        Returns:
            Predictions (batch, pred_len, c_in)
        """
        batch_size = x.shape[0]

        # ====================================================================
        # RevIN Normalization (inline for ONNX compatibility)
        # ====================================================================
        if self.use_revin:
            # Compute statistics along time dimension
            mean = torch.mean(x, dim=1, keepdim=True)
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
            )

            # Normalize
            x_norm = (x - mean) / stdev
            x_norm = x_norm * self.affine_weight + self.affine_bias
        else:
            x_norm = x
            mean = torch.zeros(batch_size, 1, self.c_in, device=x.device)
            stdev = torch.ones(batch_size, 1, self.c_in, device=x.device)

        # ====================================================================
        # Patching: (batch, seq_len, c_in) -> (batch * c_in, patch_num, patch_len)
        # ====================================================================
        # Transpose to (batch, c_in, seq_len) for patching
        x = x_norm.permute(0, 2, 1)

        # Create patches
        x = self.create_patches(x)  # (batch * c_in, patch_num, patch_len)

        # ====================================================================
        # Transformer Processing
        # ====================================================================
        # Patch embedding
        x = self.patch_embedding(x)  # (batch * c_in, patch_num, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoder
        x = self.transformer(x)  # (batch * c_in, patch_num, d_model)

        # ====================================================================
        # Prediction Head
        # ====================================================================
        # Flatten patches
        x = x.reshape(batch_size * self.c_in, -1)  # (batch * c_in, patch_num * d_model)

        # Project to prediction
        x = self.head(x)  # (batch * c_in, pred_len)

        # Reshape to (batch, c_in, pred_len)
        x = x.view(batch_size, self.c_in, self.pred_len)

        # Transpose to (batch, pred_len, c_in)
        x = x.permute(0, 2, 1)

        # ====================================================================
        # RevIN Denormalization
        # ====================================================================
        if self.use_revin:
            # Reverse affine transformation
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

            # Denormalize
            x = x * stdev + mean

        return x

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of model architecture."""
        return (
            f"PatchTSTModel(\n"
            f"  Input: ({self.seq_len}, {self.c_in})\n"
            f"  Output: ({self.pred_len}, {self.c_in})\n"
            f"  Patching: patch_len={self.patch_len}, stride={self.stride}, "
            f"patches={self.patch_num}\n"
            f"  Architecture: d_model={self.patch_embedding.out_features}, "
            f"layers={len(self.transformer.layers)}, "
            f"heads={self.transformer.layers[0].self_attn.num_heads}\n"
            f"  Parameters: {self.get_num_params():,} "
            f"({self.get_num_trainable_params():,} trainable)\n"
            f"  RevIN: {self.use_revin}\n"
            f")"
        )
