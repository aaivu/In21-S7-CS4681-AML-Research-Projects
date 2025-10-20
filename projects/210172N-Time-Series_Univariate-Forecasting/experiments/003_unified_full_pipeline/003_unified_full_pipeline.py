
"""
PHASE 1 & 2: UNIFIED FULL PIPELINE FOR STANDARD DATASETS
********************************************************
All-in-one script for Google Colab to run a full train-and-evaluate pipeline
on standard time series datasets (Weather, Traffic, Electricity, etc.).

This script combines all necessary steps:
1.  **Setup**: Installs dependencies.
2.  **Configuration**: Easily select which dataset to run.
3.  **Data Loading**: Implements a standard train/val/test split for CSVs.
4.  **Model**: Defines the PatchTST model with a proper RevIN layer and correct channel-independent architecture.
5.  **Phase 1: Training**: Trains a model for each forecast horizon and saves checkpoints locally.
6.  **Phase 2: Optimization & Evaluation**:
    - Exports trained models to ONNX (FP32).
    - Applies post-training dynamic quantization (INT8).
    - Evaluates and benchmarks PyTorch, FP32, and INT8 models on the test set.
    - Prints comprehensive results and comparisons, including training time.

To Run in Colab:
1.  **Change `DATASET_NAME` in the Configuration section if needed.**
2.  Upload the required dataset CSV (e.g., 'weather.csv') and any existing checkpoints to the root of your Colab environment.
3.  Run all cells.
"""

# ============================================================================
# 1. SETUP (Run this cell first in Colab)
# ============================================================================
# !pip install -q torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# !pip install -q numpy pandas scikit-learn tqdm onnx==1.14.0 onnxruntime-gpu==1.14.1

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

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

# CHOOSE YOUR DATASET HERE
DATASET_NAME = 'weather'  # Options: 'weather', 'traffic', 'electricity'

DATASET_CONFIGS = {
    'weather': {
        'file': 'weather.csv',
        'c_in': 21,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'patch_len': 16,
        'stride': 8
    },
    'traffic': {
        'file': 'traffic.csv',
        'c_in': 862,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'patch_len': 16,
        'stride': 8
    },
    'electricity': {
        'file': 'electricity.csv',
        'c_in': 321,
        'seq_len': 336,
        'pred_lens': [96, 192, 336, 720],
        'patch_len': 16,
        'stride': 8
    }
}

class Config:
    # General Model Params
    D_MODEL = 128
    N_HEADS = 16
    E_LAYERS = 3
    D_FF = 256
    DROPOUT = 0.2

    # Training Params
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    PATIENCE = 5

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Google Drive Backup (assumes drive mounted at /content/drive/MyDrive/)
    ENABLE_DRIVE_BACKUP = True
    DRIVE_BACKUP_PATH = '/content/drive/MyDrive/Datasets/aml/checkpoints/'

# Load selected dataset config
config = Config()
DS_CONFIG = DATASET_CONFIGS[DATASET_NAME]

print("="*50)
print("Configuration Loaded")
print(f"Device: {config.DEVICE}")
print(f"Dataset: {DATASET_NAME.upper()}")
print("="*50)

# ============================================================================
# 3. HELPER FUNCTIONS
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
# 4. EVALUATION METRICS
# ============================================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ============================================================================
# 4. DATA LOADING
# ============================================================================
class UnifiedDataset(Dataset):
    def __init__(self, file_path, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        df_raw = pd.read_csv(file_path)
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(columns=['date'])

        # Clean Train/Val/Test split: 70% / 10% / 20%
        train_border = int(len(df_raw) * 0.7)
        val_border = int(len(df_raw) * 0.8)
        
        if flag == 'train':
            data_df = df_raw.iloc[:train_border]
        elif flag == 'val':
            data_df = df_raw.iloc[train_border:val_border]
        else: # test
            data_df = df_raw.iloc[val_border:]
            
        self.data = data_df.values

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

# ============================================================================
# 5. MODEL DEFINITION (PatchTST with RevIN) - CORRECTED ARCHITECTURE
# ============================================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization (ONNX-compatible version)

    Modified to store statistics as buffers for proper ONNX export.
    Statistics are computed during normalization and stored for denormalization.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

        # Register buffers for ONNX compatibility
        self.register_buffer('mean', torch.zeros(1, 1, num_features))
        self.register_buffer('stdev', torch.ones(1, 1, num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _normalize(self, x):
        # Compute statistics and store in buffers
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        # Normalize
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        # Denormalize using stored statistics
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x

class PatchTST(nn.Module):
    """
    PatchTST model with corrected channel-independent architecture.
    Modified for ONNX compatibility with explicit RevIN statistics handling
    and manual patching implementation.
    """
    def __init__(self, c_in, seq_len, pred_len, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.eps = 1e-5

        # RevIN affine parameters
        self.affine_weight = nn.Parameter(torch.ones(c_in))
        self.affine_bias = nn.Parameter(torch.zeros(c_in))

        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.head = nn.Linear(d_model * self.patch_num, pred_len)

    def create_patches(self, x):
        """
        ONNX-compatible patching using manual extraction.
        Input: [batch, c_in, seq_len]
        Output: [batch * c_in, patch_num, patch_len]
        """
        batch_size, c_in, seq_len = x.shape
        patches = []

        # Extract patches manually
        for i in range(self.patch_num):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, :, start_idx:end_idx]  # [batch, c_in, patch_len]
            patches.append(patch)

        # Stack patches: [patch_num, batch, c_in, patch_len]
        patches = torch.stack(patches, dim=0)
        # Permute to: [batch, c_in, patch_num, patch_len]
        patches = patches.permute(1, 2, 0, 3)
        # Reshape to: [batch * c_in, patch_num, patch_len]
        patches = patches.reshape(batch_size * c_in, self.patch_num, self.patch_len)

        return patches

    def forward(self, x): # x: [batch_size, seq_len, c_in]
        batch_size = x.shape[0]

        # RevIN normalization (inline for ONNX compatibility)
        dim2reduce = tuple(range(1, x.ndim-1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)
        x_normed = (x - mean) / stdev
        x_normed = x_normed * self.affine_weight + self.affine_bias

        # Patching: [batch, seq_len, c_in] -> [batch, c_in, seq_len]
        x = x_normed.permute(0, 2, 1)

        # Create patches using ONNX-compatible method
        x = self.create_patches(x)  # [batch * c_in, patch_num, patch_len]

        # Patch embedding and Transformer processing
        x = self.patch_embedding(x)  # [batch * c_in, patch_num, d_model]
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.reshape(batch_size * self.c_in, -1)
        x = self.head(x)
        x = x.view(batch_size, self.c_in, self.pred_len)
        x = x.permute(0, 2, 1)

        # RevIN denormalization (inline for ONNX compatibility)
        x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * stdev + mean

        return x

# ============================================================================
# 6. PHASE 1: TRAINING
# ============================================================================
def train_model(pred_len):
    checkpoint_path = f'{DATASET_NAME}_pred{pred_len}.pth'
    print()
    print("="*50)
    print(f"PHASE 1: TRAINING (pred_len={pred_len})")
    print("="*50)

    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint '{checkpoint_path}' already exists. Skipping training.")
        return checkpoint_path, 0

    train_data = UnifiedDataset(DS_CONFIG['file'], DS_CONFIG['seq_len'], pred_len, flag='train')
    val_data = UnifiedDataset(DS_CONFIG['file'], DS_CONFIG['seq_len'], pred_len, flag='val')
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = PatchTST(
        c_in=DS_CONFIG['c_in'], seq_len=DS_CONFIG['seq_len'], pred_len=pred_len,
        patch_len=DS_CONFIG['patch_len'], stride=DS_CONFIG['stride'], d_model=config.D_MODEL,
        n_heads=config.N_HEADS, e_layers=config.E_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    training_start_time = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch_x, batch_y in train_pbar:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]")
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Model saved to {checkpoint_path}")
            backup_to_drive(checkpoint_path)
        else: epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE: print(f"Early stopping at epoch {epoch+1}"); break
    
    total_training_time = time.time() - training_start_time
    print(f"✓ Training complete in {total_training_time:.2f} seconds.")
    return checkpoint_path, total_training_time

# ============================================================================
# 7. PHASE 2: OPTIMIZATION & EVALUATION
# ============================================================================
def evaluate_and_optimize(checkpoint_path, pred_len, training_time):
    print()
    print("="*50)
    print(f"PHASE 2: EVALUATION (pred_len={pred_len})")
    print("="*50)

    model = PatchTST(
        c_in=DS_CONFIG['c_in'], seq_len=DS_CONFIG['seq_len'], pred_len=pred_len,
        patch_len=DS_CONFIG['patch_len'], stride=DS_CONFIG['stride'], d_model=config.D_MODEL,
        n_heads=config.N_HEADS, e_layers=config.E_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # PyTorch checkpoint size (state_dict only - just weights)
    size_pytorch = os.path.getsize(checkpoint_path) / (1024*1024)

    onnx_fp32_path = checkpoint_path.replace('.pth', '_fp32.onnx')
    print(f"🔄 Exporting to ONNX FP32: {onnx_fp32_path}")
    dummy_input = torch.randn(1, DS_CONFIG['seq_len'], DS_CONFIG['c_in'])

    # Export with optimizations enabled
    torch.onnx.export(
        model, dummy_input, onnx_fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    backup_to_drive(onnx_fp32_path)

    size_fp32 = os.path.getsize(onnx_fp32_path) / (1024*1024)

    onnx_int8_path = checkpoint_path.replace('.pth', '_int8.onnx')
    print(f"⚡ Quantizing to ONNX INT8: {onnx_int8_path}")
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QInt8)
    backup_to_drive(onnx_int8_path)

    size_int8 = os.path.getsize(onnx_int8_path) / (1024*1024)

    print()
    print("📊 Starting evaluation on test set...")

    # Detect available ONNX Runtime providers
    available_providers = ort.get_available_providers()
    has_cuda = 'CUDAExecutionProvider' in available_providers and config.DEVICE == 'cuda'

    if has_cuda:
        gpu_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"✓ GPU Mode: PyTorch + ONNX FP32 will use CUDA")
        print(f"✓ CPU Mode: ONNX INT8 will use CPU (optimized for edge devices)")
    else:
        gpu_providers = ['CPUExecutionProvider']
        print(f"✓ CPU-only mode: All models will run on CPU")

    test_data = UnifiedDataset(DS_CONFIG['file'], DS_CONFIG['seq_len'], pred_len, flag='test')
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    results = {}
    for model_type, path in [('PyTorch (GPU)', checkpoint_path), ('ONNX_FP32 (GPU)', onnx_fp32_path), ('ONNX_INT8 (CPU)', onnx_int8_path)]:
        preds, trues = [], []
        total_time = 0

        # Select provider based on model type
        if 'ONNX' in model_type:
            if 'INT8' in model_type:
                # INT8: Force CPU for optimal dynamic quantization performance
                providers = ['CPUExecutionProvider']
                print(f"\nEvaluating {model_type} on CPU (edge device simulation)...")
            else:
                # FP32: Use GPU if available
                providers = gpu_providers
                print(f"\nEvaluating {model_type}...")

            # Create session with optimized settings
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(path, sess_options, providers=providers)
            input_name = session.get_inputs()[0].name

            # WARMUP: Run 10 iterations to allow JIT optimization
            print(f"   Running warmup iterations...")
            warmup_data = next(iter(test_loader))[0].numpy()
            for _ in range(10):
                _ = session.run(None, {input_name: warmup_data})
        else:
            model.to(config.DEVICE)
            print(f"\nEvaluating {model_type}...")

            # WARMUP for PyTorch too
            print(f"   Running warmup iterations...")
            warmup_data = next(iter(test_loader))[0].to(config.DEVICE)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(warmup_data)

        for batch_x, batch_y in tqdm(test_loader, desc=f'Evaluating {model_type}'):
            start_time = time.time()
            if 'ONNX' in model_type: 
                output = session.run(None, {input_name: batch_x.numpy()})[0]
            else: 
                with torch.no_grad(): output = model(batch_x.to(config.DEVICE)).cpu().numpy()
            total_time += (time.time() - start_time)
            preds.append(output); trues.append(batch_y.numpy())

        preds, trues = np.concatenate(preds, 0), np.concatenate(trues, 0)
        avg_batch_latency_ms = (total_time / len(test_loader)) * 1000
        
        # Store results with cleaned model name (remove hardware suffix for dict key)
        clean_name = model_type.split(' (')[0]  # 'PyTorch (GPU)' -> 'PyTorch'
        results[clean_name] = {
            'Full Name': model_type,  # Keep full name for display
            'MSE': mse(trues, preds),
            'MAE': mae(trues, preds),
            'Latency (ms/batch)': avg_batch_latency_ms,
            'Size (MB)': size_pytorch if 'PyTorch' in model_type else (size_fp32 if 'FP32' in model_type else size_int8)
        }
        if 'PyTorch' in model_type:
            results[clean_name]['Training Time (s)'] = training_time

    print_results_table(results, pred_len)
    return results

def print_results_table(results, pred_len):
    print()
    print("="*95)
    print(f"RESULTS: {DATASET_NAME.upper()} (pred_len={pred_len})")
    print("="*95)
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'Latency (ms)':<15} {'Size (MB)':<12}")
    print("-"*95)

    for model_key, r in results.items():
        display_name = r.get('Full Name', model_key)
        print(f"{display_name:<20} {r['MSE']:<12.4f} {r['MAE']:<12.4f} {r['Latency (ms/batch)']:<15.2f} {r['Size (MB)']:<12.2f}")

    print("-"*95)

    # Summary metrics
    fp32_res = results['ONNX_FP32']
    int8_res = results['ONNX_INT8']

    compression = fp32_res['Size (MB)'] / int8_res['Size (MB)']
    mae_deg = ((int8_res['MAE'] - fp32_res['MAE']) / fp32_res['MAE']) * 100

    print(f"Compression: {compression:.2f}x | Accuracy Impact: {mae_deg:+.2f}% MAE")
    print("="*95)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    if not os.path.exists(DS_CONFIG['file']):
        print(f"! ERROR: Dataset file '{DS_CONFIG['file']}' not found!")
    else:
        all_results = {}
        for p_len in DS_CONFIG['pred_lens']:
            # Run Phase 1
            chk_path, train_time = train_model(p_len)
            # Clear memory
            gc.collect()
            if config.DEVICE == 'cuda': torch.cuda.empty_cache()
            # Run Phase 2
            all_results[p_len] = evaluate_and_optimize(chk_path, p_len, train_time)
