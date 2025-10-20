"""
Example: Standard Dataset (Weather, Traffic, etc.) Training Pipeline

This script demonstrates training PatchTST on standard LTSF benchmarks.

Usage:
    python example_standard_training.py --dataset weather --pred_len 96
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.standard_config import StandardConfig
from src.models.patchtst import PatchTSTModel
from src.data.standard_dataset import create_standard_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.utils.helpers import set_seed, print_model_summary
from src.utils.checkpoint import CheckpointManager


def main(args):
    """Main training pipeline for standard datasets."""

    print("=" * 80)
    print(f"Training PatchTST on {args.dataset.upper()} Dataset")
    print("=" * 80)

    set_seed(42)

    # Configuration
    config = StandardConfig(dataset=args.dataset, pred_len=args.pred_len)
    print(config)

    # Model
    model = PatchTSTModel(
        c_in=config.c_in,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        dropout=config.dropout
    )

    print_model_summary(model, input_size=(32, config.seq_len, config.c_in))

    # Data
    data_file = config.get_data_file()
    train_loader, val_loader, test_loader = create_standard_dataloaders(
        data_file=data_file,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        c_in=config.c_in,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Train
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(config.checkpoint_dir),
        save_best_only=True
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.MSELoss(),
        optimizer=optim.AdamW(model.parameters(), lr=config.learning_rate),
        device=config.device,
        checkpoint_manager=checkpoint_manager,
        max_epochs=config.epochs,
        patience=config.patience
    )

    history = trainer.train()

    # Evaluate
    model = checkpoint_manager.load_best(model)
    evaluator = Evaluator(model, device=config.device)
    test_metrics = evaluator.evaluate(test_loader)

    print("\n" + "=" * 80)
    print("Test Set Performance")
    print("=" * 80)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name:>15s}: {value:>10.4f}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standard Dataset Training')
    parser.add_argument(
        '--dataset',
        type=str,
        default='weather',
        help='Dataset name'
    )
    parser.add_argument(
        '--pred_len',
        type=int,
        default=96,
        choices=[96, 192, 336, 720],
        help='Prediction horizon'
    )

    args = parser.parse_args()
    main(args)
