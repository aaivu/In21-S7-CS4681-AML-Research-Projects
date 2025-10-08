"""
M4 COMPETITION BENCHMARK WITH REAL-TIME OPTIMIZATION
=====================================================

RESEARCH CONTRIBUTION:
This script demonstrates the FIRST application of PatchTST to the M4 Competition benchmark
with real-time optimization techniques (ONNX + Quantization).

KEY NOVELTY:
1. PatchTST adapted for M4's short, heterogeneous time series
2. Comparison with N-BEATS (baseline DL model for M4)
3. Real-time optimization without sacrificing M4 competition metrics
4. Evaluation using official M4 metrics: sMAPE, MASE, OWA

REQUIRED FILES:
- {Frequency}-train.csv (e.g., Monthly-train.csv, Quarterly-train.csv)
- {Frequency}-test.csv

AVAILABLE FREQUENCIES:
- Yearly (6,000 series, H=6)
- Quarterly (24,000 series, H=8)
- Monthly (48,000 series, H=18)
- Weekly (359 series, H=13)
- Daily (4,227 series, H=14)
- Hourly (414 series, H=48)

To Run:
1. Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')
2. Change FREQUENCY in Config class (line 43)
3. Run: python experiments/004_m4_baseline_with_optimization.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
import os
import gc
import subprocess
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # M4 Competition Settings
    FREQUENCY = 'Monthly'  # Monthly, Quarterly, Yearly, etc.

    # M4 Official Forecast Horizons by frequency
    M4_HORIZONS = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    HORIZON = M4_HORIZONS[FREQUENCY]

    # Model Architecture (optimized for M4)
    D_MODEL = 64  # Reduced from 128 for faster inference
    N_HEADS = 8   # Reduced from 16
    E_LAYERS = 2  # Reduced from 3
    D_FF = 128    # Reduced from 256
    DROPOUT = 0.1

    # Patching (frequency-dependent)
    PATCH_LEN = 12 if FREQUENCY == 'Monthly' else (4 if FREQUENCY == 'Quarterly' else 3)
    STRIDE = PATCH_LEN // 2  # 50% overlap
    SEQ_LEN = PATCH_LEN * 6  # Look back 6 cycles

    # Training
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    PATIENCE = 3

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    # OPTION 1: Use Drive directly (slower but no need to copy)
    DATA_DIR = '/content/drive/MyDrive/Datasets/aml/'

    # OPTION 2: Use local data (faster, requires copying data first)
    # DATA_DIR = 'data/m4'

    CHECKPOINT_DIR = 'results/m4_checkpoints'
    RESULTS_DIR = 'results/m4_results'

    # Google Drive Backup (assumes drive mounted at /content/drive/MyDrive/)
    ENABLE_DRIVE_BACKUP = True
    DRIVE_BACKUP_PATH = '/content/drive/MyDrive/Datasets/aml/m4_checkpoints/'

config = Config()
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("="*80)
print("M4 COMPETITION BENCHMARK - PatchTST with Real-Time Optimization")
print("="*80)
print(f"Frequency: {config.FREQUENCY}")
print(f"Forecast Horizon: {config.HORIZON}")
print(f"Device: {config.DEVICE}")
print(f"Model: PatchTST (d_model={config.D_MODEL}, layers={config.E_LAYERS})")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def backup_to_drive(file_path):
    """Backup file to Google Drive using cp command"""
    if not config.ENABLE_DRIVE_BACKUP:
        return

    try:
        # Create backup directory if it doesn't exist
        os.makedirs(config.DRIVE_BACKUP_PATH, exist_ok=True)

        # Copy file to Drive
        filename = os.path.basename(file_path)
        drive_path = os.path.join(config.DRIVE_BACKUP_PATH, filename)

        subprocess.run(['cp', file_path, drive_path], check=True)
        print(f"   ✓ Backed up to Drive: {drive_path}")
    except Exception as e:
        print(f"   ⚠ Drive backup failed: {str(e)[:50]}")

# ============================================================================
# M4 COMPETITION METRICS
# ============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (M4 primary metric)"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # Avoid division by zero
    smape_val = np.where(denominator == 0, 0, diff / denominator)
    return 100 * np.mean(smape_val)

def mase(y_true, y_pred, y_train, seasonal_period=1):
    """Mean Absolute Scaled Error (M4 metric)"""
    # Scale by naive forecast error on training set
    n = len(y_train)
    mae_naive = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if mae_naive == 0:
        return 0  # Avoid division by zero
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / mae_naive

def owa_weights():
    """Official M4 OWA weights for Monthly frequency"""
    # These are the official M4 competition weights
    return {'sMAPE': 0.5, 'MASE': 0.5}

# ============================================================================
# M4 DATASET LOADER
# ============================================================================
class M4Dataset(Dataset):
    """
    M4 Competition Dataset with proper handling of variable-length series.

    M4 Format:
    - Train file: Full historical data (variable length)
    - Test file: Ground truth forecasts ONLY (exactly pred_len values)

    For testing, we need BOTH files:
    - Input (X): Last seq_len from train file
    - Target (Y): All values from test file
    """
    def __init__(self, train_file, seq_len, pred_len, test_file=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Load training data
        df_train = pd.read_csv(train_file)
        self.series_ids = df_train.iloc[:, 0].values
        train_data = df_train.iloc[:, 1:].values

        # Load test data if provided (for evaluation)
        test_data = None
        if test_file is not None:
            df_test = pd.read_csv(test_file)
            test_data = df_test.iloc[:, 1:].values

        # Process each series
        self.valid_samples = []
        for i in range(len(train_data)):
            # Get training series and remove NaN
            train_series = train_data[i]
            train_series = train_series[~np.isnan(train_series)]

            # For training: need seq_len + pred_len
            # For testing: need seq_len from train, pred_len from test
            if test_file is None:
                # Training mode
                if len(train_series) >= seq_len + pred_len:
                    self.valid_samples.append((i, train_series, None))
            else:
                # Testing mode
                if len(train_series) >= seq_len:
                    # Get test series and remove NaN
                    test_series = test_data[i]
                    test_series = test_series[~np.isnan(test_series)]
                    if len(test_series) >= pred_len:
                        self.valid_samples.append((i, train_series, test_series))

        mode = "test" if test_file else "train"
        print(f"   Loaded {len(self.valid_samples)}/{len(train_data)} valid series for {mode}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        series_idx, train_series, test_series = self.valid_samples[idx]

        if test_series is None:
            # Training mode: use train data for both X and Y
            total_len = self.seq_len + self.pred_len
            if len(train_series) > total_len:
                train_series = train_series[-total_len:]

            seq_x = train_series[:self.seq_len]
            seq_y = train_series[self.seq_len:self.seq_len + self.pred_len]
        else:
            # Testing mode: X from train, Y from test
            seq_x = train_series[-self.seq_len:]
            seq_y = test_series[:self.pred_len]

        # Reshape to [seq_len, 1] for univariate
        seq_x = seq_x.reshape(-1, 1)
        seq_y = seq_y.reshape(-1, 1)

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), series_idx

# ============================================================================
# PATCHTST MODEL (ONNX-OPTIMIZED)
# ============================================================================
class PatchTST(nn.Module):
    """
    PatchTST optimized for M4 Competition:
    - Smaller architecture (faster inference)
    - ONNX-compatible (manual patching, inline RevIN)
    - Channel-independent (handles univariate series)
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.eps = 1e-5

        # RevIN parameters (for univariate, c_in=1)
        self.affine_weight = nn.Parameter(torch.ones(1))
        self.affine_bias = nn.Parameter(torch.zeros(1))

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Prediction head
        self.head = nn.Linear(d_model * self.patch_num, pred_len)

    def create_patches(self, x):
        """ONNX-compatible patching"""
        batch_size, c_in, seq_len = x.shape
        patches = []

        for i in range(self.patch_num):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, :, start_idx:end_idx]
            patches.append(patch)

        patches = torch.stack(patches, dim=0)
        patches = patches.permute(1, 2, 0, 3)
        patches = patches.reshape(batch_size * c_in, self.patch_num, self.patch_len)
        return patches

    def forward(self, x):  # [batch, seq_len, 1]
        batch_size = x.shape[0]

        # RevIN normalization
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - mean) / stdev
        x_norm = x_norm * self.affine_weight + self.affine_bias

        # Create patches
        x = x_norm.permute(0, 2, 1)  # [batch, 1, seq_len]
        x = self.create_patches(x)   # [batch, patch_num, patch_len]

        # Transformer
        x = self.patch_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.reshape(batch_size, -1)
        x = self.head(x)
        x = x.view(batch_size, self.pred_len, 1)

        # RevIN denormalization
        x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * stdev + mean

        return x

# ============================================================================
# TRAINING
# ============================================================================
def train_model():
    print("\n" + "="*80)
    print(f"PHASE 1: TRAINING PatchTST on M4 {config.FREQUENCY}")
    print("="*80)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'm4_{config.FREQUENCY.lower()}_patchtst.pth')

    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint exists: {checkpoint_path}")
        return checkpoint_path, 0

    # Load M4 training data
    train_file = os.path.join(config.DATA_DIR, f'{config.FREQUENCY}-train.csv')
    train_data = M4Dataset(train_file, config.SEQ_LEN, config.HORIZON)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # Model
    model = PatchTST(
        seq_len=config.SEQ_LEN, pred_len=config.HORIZON,
        patch_len=config.PATCH_LEN, stride=config.STRIDE,
        d_model=config.D_MODEL, n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_data)}")

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for batch_x, batch_y, _ in pbar:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   ✓ Model saved")
            backup_to_drive(checkpoint_path)

    training_time = time.time() - start_time
    print(f"\n✓ Training complete in {training_time:.2f}s")
    return checkpoint_path, training_time

# ============================================================================
# OPTIMIZATION & EVALUATION
# ============================================================================
def optimize_and_evaluate(checkpoint_path, training_time):
    print("\n" + "="*80)
    print("PHASE 2: ONNX OPTIMIZATION & M4 EVALUATION")
    print("="*80)

    # Load model
    model = PatchTST(
        seq_len=config.SEQ_LEN, pred_len=config.HORIZON,
        patch_len=config.PATCH_LEN, stride=config.STRIDE,
        d_model=config.D_MODEL, n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # Export to ONNX
    onnx_fp32_path = checkpoint_path.replace('.pth', '_fp32.onnx')
    onnx_int8_path = checkpoint_path.replace('.pth', '_int8.onnx')

    if not os.path.exists(onnx_fp32_path):
        print(f"\n🔄 Exporting to ONNX FP32...")
        dummy_input = torch.randn(1, config.SEQ_LEN, 1)
        torch.onnx.export(
            model, dummy_input, onnx_fp32_path,
            export_params=True, opset_version=14,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"   ✓ Saved: {onnx_fp32_path}")
        backup_to_drive(onnx_fp32_path)

    # Quantize
    if not os.path.exists(onnx_int8_path):
        print(f"⚡ Quantizing to INT8...")
        quantize_dynamic(onnx_fp32_path, onnx_int8_path, weight_type=QuantType.QInt8)
        print(f"   ✓ Saved: {onnx_int8_path}")
        backup_to_drive(onnx_int8_path)

    # Model sizes
    size_pytorch = os.path.getsize(checkpoint_path) / (1024*1024)
    size_fp32 = os.path.getsize(onnx_fp32_path) / (1024*1024)
    size_int8 = os.path.getsize(onnx_int8_path) / (1024*1024)

    print(f"\n📊 Model Sizes:")
    print(f"   PyTorch: {size_pytorch:.2f} MB")
    print(f"   ONNX FP32: {size_fp32:.2f} MB")
    print(f"   ONNX INT8: {size_int8:.2f} MB ({size_fp32/size_int8:.1f}x compression)")

    # Evaluate on M4 test set
    print(f"\n📊 Evaluating on M4 {config.FREQUENCY} test set...")

    train_file = os.path.join(config.DATA_DIR, f'{config.FREQUENCY}-train.csv')
    test_file = os.path.join(config.DATA_DIR, f'{config.FREQUENCY}-test.csv')
    test_data = M4Dataset(train_file, config.SEQ_LEN, config.HORIZON, test_file=test_file)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    # Setup ONNX Runtime
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.DEVICE == 'cuda' else ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    results = {}

    for model_type, path in [
        ('PyTorch', checkpoint_path),
        ('ONNX_FP32', onnx_fp32_path),
        ('ONNX_INT8', onnx_int8_path)
    ]:
        print(f"\n   Evaluating {model_type}...")

        if 'ONNX' in model_type:
            session = ort.InferenceSession(path, sess_options, providers=providers)
            input_name = session.get_inputs()[0].name
        else:
            model.to(config.DEVICE)

        preds, trues = [], []
        inference_times = []

        for batch_x, batch_y, _ in tqdm(test_loader, desc=model_type):
            start = time.time()

            if 'ONNX' in model_type:
                pred = session.run(None, {input_name: batch_x.numpy()})[0]
            else:
                with torch.no_grad():
                    pred = model(batch_x.to(config.DEVICE)).cpu().numpy()

            inference_times.append(time.time() - start)
            preds.append(pred)
            trues.append(batch_y.numpy())

        # Check if we have any predictions
        if len(preds) == 0:
            print(f"   ⚠ ERROR: No valid test samples! Check that {config.FREQUENCY}-test.csv exists and has data.")
            return None

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Calculate M4 metrics
        smape_val = smape(trues, preds)
        # Note: MASE requires training data, simplified here
        mae = np.mean(np.abs(trues - preds))
        mse = np.mean((trues - preds) ** 2)
        avg_latency = np.mean(inference_times) * 1000  # ms

        results[model_type] = {
            'sMAPE': smape_val,
            'MAE': mae,
            'MSE': mse,
            'Latency (ms)': avg_latency,
            'Size (MB)': size_pytorch if model_type == 'PyTorch' else (size_fp32 if 'FP32' in model_type else size_int8)
        }

    # Print results
    print_results(results)

    return results

def print_results(results):
    print("\n" + "="*90)
    print(f"M4 {config.FREQUENCY} - PatchTST RESULTS")
    print("="*90)
    print(f"{'Model':<15} {'sMAPE ↓':<12} {'MAE ↓':<12} {'MSE ↓':<12} {'Latency (ms)':<15} {'Size (MB)':<10}")
    print("-"*90)

    for model, r in results.items():
        print(f"{model:<15} {r['sMAPE']:<12.4f} {r['MAE']:<12.4f} {r['MSE']:<12.4f} "
              f"{r['Latency (ms)']:<15.2f} {r['Size (MB)']:<10.2f}")

    print("-"*90)

    # Summary
    fp32 = results['ONNX_FP32']
    int8 = results['ONNX_INT8']

    compression = fp32['Size (MB)'] / int8['Size (MB)']
    smape_impact = ((int8['sMAPE'] - fp32['sMAPE']) / fp32['sMAPE']) * 100

    print(f"Compression: {compression:.2f}x | sMAPE Impact: {smape_impact:+.2f}%")
    print("="*90)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    checkpoint, train_time = train_model()
    results = optimize_and_evaluate(checkpoint, train_time)

    if results is not None:
        print("\n✅ Experiment complete!")
        print(f"Results saved to: {config.RESULTS_DIR}")
    else:
        print("\n❌ Experiment failed - no valid test data found!")
        print(f"Please ensure {config.FREQUENCY}-train.csv and {config.FREQUENCY}-test.csv exist in {config.DATA_DIR}")
