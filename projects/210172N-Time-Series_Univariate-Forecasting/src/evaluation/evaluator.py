"""Model evaluation module."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from src.evaluation.metrics import calculate_standard_metrics, calculate_m4_metrics


class Evaluator:
    """
    Model evaluator for forecasting tasks.

    Args:
        model: PyTorch model
        device: Device to run evaluation on
        use_m4_metrics: Whether to use M4 competition metrics

    Examples:
        >>> evaluator = Evaluator(model, device='cuda')
        >>> metrics = evaluator.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_m4_metrics: bool = False
    ):
        self.model = model
        self.device = torch.device(device)
        self.use_m4_metrics = use_m4_metrics
        self.model.to(self.device)

    def evaluate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: Data loader for evaluation
            verbose: Whether to show progress bar

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        predictions = []
        targets = []

        iterator = tqdm(dataloader, desc='Evaluating') if verbose else dataloader

        with torch.no_grad():
            for batch in iterator:
                if len(batch) == 3:  # M4 dataset with series IDs
                    batch_x, batch_y, _ = batch
                else:
                    batch_x, batch_y = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                output = self.model(batch_x)

                predictions.append(output.cpu().numpy())
                targets.append(batch_y.cpu().numpy())

        # Concatenate all predictions and targets
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Calculate metrics
        if self.use_m4_metrics:
            metrics = calculate_m4_metrics(targets, predictions)
        else:
            metrics = calculate_standard_metrics(targets, predictions)

        return metrics

    def predict(
        self,
        dataloader: DataLoader,
        return_targets: bool = False
    ) -> np.ndarray:
        """
        Generate predictions without calculating metrics.

        Args:
            dataloader: Data loader
            return_targets: Whether to return targets as well

        Returns:
            Predictions array, optionally with targets
        """
        self.model.eval()

        predictions = []
        targets = [] if return_targets else None

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    batch_x, batch_y, _ = batch
                else:
                    batch_x, batch_y = batch

                batch_x = batch_x.to(self.device)

                output = self.model(batch_x)
                predictions.append(output.cpu().numpy())

                if return_targets:
                    targets.append(batch_y.numpy())

        predictions = np.concatenate(predictions, axis=0)

        if return_targets:
            targets = np.concatenate(targets, axis=0)
            return predictions, targets

        return predictions
