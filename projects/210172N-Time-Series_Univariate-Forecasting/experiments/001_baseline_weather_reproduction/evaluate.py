"""
Evaluation script for PatchTST weather forecasting
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import config
from data_loader import get_dataloaders
from model import PatchTST_backbone


def evaluate(checkpoint_path):
    """Evaluate model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    _, _, test_loader, scaler = get_dataloaders()
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nLoading model...")
    model = PatchTST_backbone(
        c_in=config.ENC_IN,
        context_window=config.SEQ_LEN,
        target_window=config.PRED_LEN,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        n_layers=config.E_LAYERS,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        fc_dropout=config.FC_DROPOUT,
        head_dropout=config.HEAD_DROPOUT,
        individual=config.INDIVIDUAL,
        revin=config.REVIN,
        affine=config.AFFINE,
        subtract_last=config.SUBTRACT_LAST,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate
    model.eval()
    preds = []
    trues = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Test'):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # PatchTST expects [batch, features, seq_len]
            batch_x = batch_x.permute(0, 2, 1)
            batch_y = batch_y.permute(0, 2, 1)

            outputs = model(batch_x)  # [batch, features, pred_len]

            # Permute back for metric calculation
            outputs = outputs.permute(0, 2, 1).cpu().numpy()  # [batch, pred_len, features]
            batch_y = batch_y.permute(0, 2, 1).cpu().numpy()

            preds.append(outputs)
            trues.append(batch_y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize
    print("\nDenormalizing predictions...")
    preds_denorm = scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues_denorm = scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

    # Calculate metrics
    mse = np.mean((preds_denorm - trues_denorm) ** 2)
    mae = np.mean(np.abs(preds_denorm - trues_denorm))

    print("\n" + "="*50)
    print(f"Test Results (pred_len={config.PRED_LEN}):")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print("="*50)

    return mse, mae


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate PatchTST on Weather dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--pred_len', type=int, default=None,
                       help='Prediction length (must match checkpoint)')
    args = parser.parse_args()

    if args.pred_len is not None:
        config.PRED_LEN = args.pred_len

    evaluate(args.checkpoint)
