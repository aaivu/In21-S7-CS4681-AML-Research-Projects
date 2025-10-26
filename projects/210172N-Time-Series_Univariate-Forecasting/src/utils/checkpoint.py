"""Checkpoint management for model saving and loading."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import shutil


class CheckpointManager:
    """
    Manage model checkpoints with best model tracking.

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_best_only: If True, only save when model improves
        max_checkpoints: Maximum number of checkpoints to keep

    Examples:
        >>> manager = CheckpointManager('checkpoints/')
        >>> manager.save(model, epoch=5, metrics={'loss': 0.15})
        >>> model = manager.load_best(model)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        max_checkpoints: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('inf')
        self.checkpoints = []

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Path:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save regular checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)

            # Remove old checkpoints
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                old_checkpoint.unlink(missing_ok=True)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)

        return checkpoint_path if not self.save_best_only or is_best else None

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best model checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location='cpu'))
        return model

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None):
        """Load full checkpoint with optimizer state."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint.get('metrics', {})
