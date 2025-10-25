"""
Base configuration class for PatchTST models.

This module provides the base configuration structure for all PatchTST models,
including model architecture, training hyperparameters, and system settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
from pathlib import Path


@dataclass
class BaseConfig:
    """
    Base configuration for PatchTST models.

    This class contains all common configuration parameters that can be
    inherited and customized by specific dataset configurations.
    """

    # ========================================================================
    # Model Architecture
    # ========================================================================
    d_model: int = 128
    """Dimensionality of the model (embedding dimension)."""

    n_heads: int = 16
    """Number of attention heads in multi-head attention."""

    e_layers: int = 3
    """Number of encoder layers in the Transformer."""

    d_ff: int = 256
    """Dimensionality of the feed-forward network."""

    dropout: float = 0.2
    """Dropout rate for regularization."""

    # ========================================================================
    # Patching Parameters
    # ========================================================================
    patch_len: int = 16
    """Length of each patch (P in the paper)."""

    stride: int = 8
    """Stride between consecutive patches (S in the paper)."""

    # ========================================================================
    # Time Series Parameters
    # ========================================================================
    seq_len: int = 336
    """Input sequence length (look-back window)."""

    pred_len: int = 96
    """Prediction horizon (forecast length)."""

    c_in: int = 1
    """Number of input channels (features)."""

    # ========================================================================
    # RevIN Parameters
    # ========================================================================
    use_revin: bool = True
    """Whether to use Reversible Instance Normalization."""

    revin_affine: bool = True
    """Whether to use learnable affine parameters in RevIN."""

    # ========================================================================
    # Training Parameters
    # ========================================================================
    batch_size: int = 128
    """Batch size for training."""

    epochs: int = 20
    """Maximum number of training epochs."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    patience: int = 5
    """Early stopping patience (epochs without improvement)."""

    optimizer: str = 'adamw'
    """Optimizer to use ('adam', 'adamw', 'sgd')."""

    weight_decay: float = 0.0
    """Weight decay (L2 regularization) coefficient."""

    grad_clip: Optional[float] = None
    """Gradient clipping threshold (None for no clipping)."""

    # ========================================================================
    # System Parameters
    # ========================================================================
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    """Device to run computations on ('cuda' or 'cpu')."""

    num_workers: int = 0
    """Number of workers for data loading."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ========================================================================
    # Data Paths
    # ========================================================================
    data_root: str = 'data'
    """Root directory for datasets."""

    checkpoint_dir: str = 'checkpoints'
    """Directory to save model checkpoints."""

    results_dir: str = 'results'
    """Directory to save evaluation results."""

    # ========================================================================
    # Optimization Parameters
    # ========================================================================
    enable_onnx: bool = True
    """Whether to export to ONNX format."""

    enable_quantization: bool = True
    """Whether to apply post-training quantization."""

    onnx_opset_version: int = 14
    """ONNX opset version for export."""

    # ========================================================================
    # Logging Parameters
    # ========================================================================
    log_interval: int = 50
    """Interval (in batches) for logging training progress."""

    save_best_only: bool = True
    """Whether to save only the best model checkpoint."""

    verbose: bool = True
    """Whether to print detailed logging information."""

    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Compute number of patches
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1

        # Validate parameters
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

        if self.patch_len > self.seq_len:
            raise ValueError(
                f"patch_len ({self.patch_len}) cannot be greater than "
                f"seq_len ({self.seq_len})"
            )

        # Convert string paths to Path objects
        self.data_root = Path(self.data_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.results_dir = Path(self.results_dir)

        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        """String representation of configuration."""
        config_str = "Configuration:\n"
        config_str += "=" * 80 + "\n"

        sections = {
            "Model Architecture": ['d_model', 'n_heads', 'e_layers', 'd_ff', 'dropout'],
            "Patching": ['patch_len', 'stride', 'patch_num'],
            "Time Series": ['seq_len', 'pred_len', 'c_in'],
            "Training": ['batch_size', 'epochs', 'learning_rate', 'patience'],
            "System": ['device', 'seed']
        }

        for section, params in sections.items():
            config_str += f"\n{section}:\n"
            config_str += "-" * 40 + "\n"
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    config_str += f"  {param:20s}: {value}\n"

        config_str += "=" * 80
        return config_str
