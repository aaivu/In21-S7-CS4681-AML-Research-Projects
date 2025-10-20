"""
Helper utility functions for common operations.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value

    Examples:
        >>> set_seed(42)
        >>> # Now all random operations are deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters

    Examples:
        >>> model = PatchTSTModel(...)
        >>> total_params = count_parameters(model)
        >>> trainable_params = count_parameters(model, trainable_only=True)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Device specification ('cuda', 'cpu', or None for auto-detect)

    Returns:
        torch.device object

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force CUDA
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = 'cpu'

    return torch.device(device)


def print_model_summary(model: nn.Module, input_size: tuple = None):
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Optional input tensor size for testing

    Examples:
        >>> model = PatchTSTModel(...)
        >>> print_model_summary(model, input_size=(32, 336, 7))
    """
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(model)
    print("-" * 80)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print("=" * 80)

    if input_size is not None:
        dummy_input = torch.randn(*input_size)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print("=" * 80)
        except Exception as e:
            print(f"Could not run forward pass: {e}")
            print("=" * 80)
