"""
Example: Complete M4 Competition Training and Optimization Pipeline

This script demonstrates the full workflow for training PatchTST on M4 data,
exporting to ONNX, quantizing, and evaluating all model variants.

Usage:
    python example_m4_training.py --frequency Monthly
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from src.config.m4_config import M4Config
from src.models.patchtst import PatchTSTModel
from src.data.m4_dataset import create_m4_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.optimization.onnx_export import export_to_onnx
from src.optimization.quantization import quantize_onnx_model
from src.inference.predictor import Predictor
from src.utils.helpers import set_seed, count_parameters
from src.utils.checkpoint import CheckpointManager


def main(args):
    """Main training and optimization pipeline."""

    print("=" * 80)
    print("M4 COMPETITION - PatchTST Training and Optimization")
    print("=" * 80)

    # Set random seed for reproducibility
    set_seed(42)

    # ========================================================================
    # 1. Configuration
    # ========================================================================
    print("\n[1/8] Loading configuration...")
    config = M4Config(frequency=args.frequency)
    print(config)

    # ========================================================================
    # 2. Create Model
    # ========================================================================
    print("\n[2/8] Creating PatchTST model...")
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
        dropout=config.dropout,
        use_revin=True
    )

    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"Trainable Parameters: {count_parameters(model, trainable_only=True):,}")

    # ========================================================================
    # 3. Load Data
    # ========================================================================
    print("\n[3/8] Loading M4 data...")
    train_file, test_file = config.get_data_files()

    train_loader, test_loader = create_m4_dataloaders(
        train_file=train_file,
        test_file=test_file,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # ========================================================================
    # 4. Train Model
    # ========================================================================
    print("\n[4/8] Training model...")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(config.checkpoint_dir),
        save_best_only=True
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=nn.MSELoss(),
        optimizer=optim.AdamW(model.parameters(), lr=config.learning_rate),
        device=config.device,
        checkpoint_manager=checkpoint_manager,
        max_epochs=config.epochs,
        patience=config.patience
    )

    history = trainer.train()

    # Load best model
    model = checkpoint_manager.load_best(model)

    # ========================================================================
    # 5. Export to ONNX
    # ========================================================================
    print("\n[5/8] Exporting to ONNX...")

    onnx_fp32_path = config.checkpoint_dir / f'm4_{config.frequency.lower()}_fp32.onnx'
    export_to_onnx(
        model=model,
        output_path=str(onnx_fp32_path),
        input_shape=(1, config.seq_len, config.c_in),
        opset_version=config.onnx_opset_version
    )

    # ========================================================================
    # 6. Quantize Model
    # ========================================================================
    print("\n[6/8] Quantizing model to INT8...")

    onnx_int8_path = config.checkpoint_dir / f'm4_{config.frequency.lower()}_int8.onnx'
    quantize_onnx_model(
        onnx_model_path=str(onnx_fp32_path),
        output_path=str(onnx_int8_path)
    )

    # ========================================================================
    # 7. Evaluate All Models
    # ========================================================================
    print("\n[7/8] Evaluating all model variants...")

    model_paths = {
        'PyTorch': ('pytorch', str(config.checkpoint_dir / 'best_model.pth')),
        'ONNX_FP32': ('onnx', str(onnx_fp32_path)),
        'ONNX_INT8': ('onnx_int8', str(onnx_int8_path))
    }

    results = {}

    for model_name, (model_type, model_path) in model_paths.items():
        print(f"\n   Evaluating {model_name}...")

        # Create predictor
        if model_type == 'pytorch':
            predictor = Predictor(model_path, model_type=model_type,
                                pytorch_model=model, device=config.device)
        else:
            predictor = Predictor(model_path, model_type=model_type)

        # Warmup and benchmark
        preds, targets, latencies = [], [], []

        for batch in test_loader:
            if len(batch) == 3:
                batch_x, batch_y, _ = batch
            else:
                batch_x, batch_y = batch

            # Measure inference time
            start = time.time()
            pred = predictor.predict(batch_x)
            latencies.append(time.time() - start)

            preds.append(pred)
            targets.append(batch_y.numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Calculate metrics
        from src.evaluation.metrics import calculate_m4_metrics
        metrics = calculate_m4_metrics(targets, preds)

        # Add latency
        metrics['Latency_ms'] = np.mean(latencies) * 1000

        # Get model size
        model_size = Path(model_path).stat().st_size / (1024 * 1024)
        metrics['Size_MB'] = model_size

        results[model_name] = metrics

    # ========================================================================
    # 8. Print Results
    # ========================================================================
    print("\n[8/8] Final Results")
    print("=" * 100)
    print(f"M4 {config.frequency} - PatchTST Performance Comparison")
    print("=" * 100)
    print(f"{'Model':<15} {'sMAPE ↓':<12} {'MASE ↓':<12} {'MAE ↓':<12} "
          f"{'Latency (ms)':<15} {'Size (MB)':<12}")
    print("-" * 100)

    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['sMAPE']:<12.4f} {metrics['MASE']:<12.4f} "
              f"{metrics['MAE']:<12.4f} {metrics['Latency_ms']:<15.2f} "
              f"{metrics['Size_MB']:<12.2f}")

    print("-" * 100)

    # Summary
    fp32_metrics = results['ONNX_FP32']
    int8_metrics = results['ONNX_INT8']

    compression = fp32_metrics['Size_MB'] / int8_metrics['Size_MB']
    speedup = fp32_metrics['Latency_ms'] / int8_metrics['Latency_ms']
    smape_degradation = ((int8_metrics['sMAPE'] - fp32_metrics['sMAPE'])
                         / fp32_metrics['sMAPE']) * 100

    print(f"\nOptimization Summary:")
    print(f"  Compression:          {compression:.2f}x")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  sMAPE Degradation:    {smape_degradation:+.2f}%")
    print("=" * 100)

    print(f"\n✅ Pipeline completed successfully!")
    print(f"   Checkpoints saved to: {config.checkpoint_dir}")
    print(f"   Best model: {config.checkpoint_dir / 'best_model.pth'}")
    print(f"   ONNX FP32: {onnx_fp32_path}")
    print(f"   ONNX INT8: {onnx_int8_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M4 Competition Training Pipeline')
    parser.add_argument(
        '--frequency',
        type=str,
        default='Monthly',
        choices=['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly'],
        help='M4 frequency'
    )

    args = parser.parse_args()
    main(args)
