"""Training module for PatchTST models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm
import time

from src.utils.checkpoint import CheckpointManager
from src.evaluation.evaluator import Evaluator


class Trainer:
    """
    Trainer for PatchTST models.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training
        checkpoint_manager: Checkpoint manager
        max_epochs: Maximum number of epochs
        patience: Early stopping patience
        grad_clip: Gradient clipping threshold

    Examples:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=nn.MSELoss(),
        ...     optimizer=optimizer,
        ...     device='cuda'
        ... )
        >>> trainer.train(epochs=20)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_epochs: int = 20,
        patience: int = 5,
        grad_clip: Optional[float] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.checkpoint_manager = checkpoint_manager
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.training_history = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.max_epochs} [Train]')

        for batch in pbar:
            if len(batch) == 3:  # M4 dataset with IDs
                batch_x, batch_y, _ = batch
            else:
                batch_x, batch_y = batch

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1}/{self.max_epochs} [Val]'):
                if len(batch) == 3:
                    batch_x, batch_y, _ = batch
                else:
                    batch_x, batch_y = batch

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)

                total_loss += loss.item()

        return total_loss / num_batches

    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs (overrides max_epochs)

        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = self.max_epochs

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Log
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")

            # Save history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

            # Check improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                print(f"  ✓ New best model! (Val Loss: {val_loss:.4f})")

                # Save checkpoint
                if self.checkpoint_manager:
                    self.checkpoint_manager.save(
                        model=self.model,
                        epoch=epoch + 1,
                        metrics={'val_loss': val_loss, 'train_loss': train_loss},
                        is_best=True,
                        optimizer=self.optimizer
                    )
            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s)")

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 80)

        return {
            'history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time
        }
