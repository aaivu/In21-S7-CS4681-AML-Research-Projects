#!/usr/bin/env python3
"""
Generate sample training data for testing visualization scripts.
This creates mock training logs that can be used to test the visualization.
"""

import json
import numpy as np
import os
from pathlib import Path


def generate_sample_logs(num_epochs=10, save_dir="./sample_results"):
    """Generate sample training logs for testing."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Simulate realistic training curves
    epochs = list(range(1, num_epochs + 1))
    
    # Training loss (decreasing with some noise)
    base_loss = 4.0
    train_losses = []
    for epoch in epochs:
        # Exponential decay with noise
        loss = base_loss * np.exp(-0.1 * epoch) + np.random.normal(0, 0.05)
        loss = max(0.1, loss)  # Minimum loss
        train_losses.append(loss)
    
    # Validation accuracy (increasing with some noise)
    val_top1 = []
    val_top5 = []
    for epoch in epochs:
        # Logarithmic growth with noise
        top1 = 20 + 40 * np.log(epoch + 1) / np.log(num_epochs + 1) + np.random.normal(0, 2)
        top1 = max(0, min(100, top1))  # Clamp between 0-100
        
        top5 = top1 + 15 + np.random.normal(0, 1)  # Top5 is usually higher
        top5 = max(top1, min(100, top5))
        
        val_top1.append(top1)
        val_top5.append(top5)
    
    # Training times (with some variation)
    train_times = [120 + np.random.normal(0, 10) for _ in epochs]
    train_times = [max(60, t) for t in train_times]  # Minimum 60 seconds
    
    # Per-iteration data
    train_losses_per_iter = []
    iter_count = 0
    for epoch in epochs:
        epoch_loss = train_losses[epoch - 1]
        # Simulate 50 iterations per epoch
        for i in range(50):
            iter_count += 1
            # Loss decreases within epoch with noise
            iter_loss = epoch_loss + np.random.normal(0, 0.1)
            train_losses_per_iter.append({
                'epoch': epoch,
                'iteration': iter_count,
                'loss': max(0.01, iter_loss),
                'lr': 1e-4 * (0.95 ** (epoch - 1))  # Learning rate decay
            })
    
    # Create logs dictionary
    logs = {
        'epochs': epochs,
        'train_losses': train_losses,
        'train_losses_per_iter': train_losses_per_iter,
        'val_top1': val_top1,
        'val_top5': val_top5,
        'train_times': train_times,
        'learning_rates': [1e-4 * (0.95 ** (epoch - 1)) for epoch in epochs],
        'iteration_times': []
    }
    
    # Save logs
    log_file = os.path.join(save_dir, "training_logs.json")
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    # Create sample config
    config = {
        'model': 'swin_tiny_2d_temporalavg',
        'num_classes': 101,
        'frames': 32,
        'stride': 2,
        'size': 224,
        'data': {
            'root': 'data/UCF101',
            'split': 1,
            'trainlist': 'data/UCF101/splits/trainlist01.txt',
            'testlist': 'data/UCF101/splits/testlist01.txt',
            'classind': 'data/UCF101/splits/classInd.txt'
        },
        'optim': {
            'name': 'adamw',
            'lr_backbone': 1.0e-4,
            'lr_head': 2.0e-4,
            'weight_decay': 0.05
        },
        'train': {
            'batch_size': 8,
            'accum_steps': 1,
            'epochs': num_epochs,
            'num_workers': 4,
            'amp': True,
            'save_dir': save_dir
        }
    }
    
    config_file = os.path.join(save_dir, "config.yaml")
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create summary
    summary = {
        'config': config,
        'best_top1_accuracy': max(val_top1),
        'total_epochs': num_epochs,
        'total_training_time': sum(train_times),
        'final_metrics': {
            'train_loss': train_losses[-1],
            'val_top1': val_top1[-1],
            'val_top5': val_top5[-1],
        }
    }
    
    summary_file = os.path.join(save_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Sample data generated in: {save_dir}")
    print(f"Files created:")
    print(f"  - {log_file}")
    print(f"  - {config_file}")
    print(f"  - {summary_file}")
    
    return save_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to simulate")
    parser.add_argument("--output", default="./sample_results", help="Output directory")
    
    args = parser.parse_args()
    
    generate_sample_logs(args.epochs, args.output)
