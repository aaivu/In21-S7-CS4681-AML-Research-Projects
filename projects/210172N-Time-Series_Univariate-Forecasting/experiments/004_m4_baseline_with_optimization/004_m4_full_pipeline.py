
"""
PHASE 1 & 2: FULL M4 BENCHMARKING PIPELINE
********************************************
All-in-one script for Google Colab to reproduce the M4 experiment.

This script combines all necessary steps:
1.  **Setup**: Installs dependencies.
2.  **Data Loading**: Implements a robust M4 dataset loader with padding.
3.  **Metrics**: Implements M4 competition metrics (sMAPE, MASE).
4.  **Model**: Defines the PatchTST model with a proper RevIN layer and correct channel-independent architecture.
5.  **Phase 1: Training**: Trains the model on the M4 dataset and saves a checkpoint locally.
6.  **Phase 2: Optimization & Evaluation**:
    - Exports the trained model to ONNX (FP32).
    - Applies post-training dynamic quantization (INT8).
    - Evaluates and benchmarks both models.
    - Prints comprehensive results and comparisons, including training time.

To Run in Colab:
1.  Upload the M4 dataset files ('Monthly-train.csv', 'Monthly-test.csv') and any existing checkpoints to the root of your Colab environment.
2.  Run all cells.
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

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
class Config:
    # M4 Dataset Params
    FREQ = 'Monthly'
    TRAIN_FILE = 'Monthly-train.csv'
    TEST_FILE = 'Monthly-test.csv'
    PRED_LEN = 18
    SEQ_LEN = 3 * PRED_LEN # Look-back window = 3x forecast horizon

    # PatchTST Model Params (as per GEMINI.md)
    PATCH_LEN = 6
    STRIDE = 3
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
    CHECKPOINT_PATH = f'm4_{FREQ.lower()}_pred{PRED_LEN}.pth'

config = Config()

print("="*50)
print("Configuration Loaded")
print(f"Device: {config.DEVICE}")
print(f"Dataset: M4 {config.FREQ}")
print(f"Checkpoint Path: {config.CHECKPOINT_PATH}")
print("="*50)


# ============================================================================
# 3. M4 EVALUATION METRICS
# ============================================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

def mase(y_true, y_pred, y_train):
    """Mean Absolute Scaled Error"""
    n = y_train.shape[0]
    if n <= 1: return np.mean(np.abs(y_true - y_pred))
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.mean(np.abs(y_true - y_pred))
    return errors / (d + 1e-8)

def mase_all_series(y_true_list, y_pred_list, y_train_list):
    """Calculate MASE across all series"""
    mase_scores = []
    for i in range(len(y_true_list)):
        mase_scores.append(mase(y_true_list[i], y_pred_list[i], y_train_list[i]))
    return np.mean(mase_scores)

# ============================================================================
# 4. DATA LOADING
# ============================================================================
class M4Dataset(Dataset):
    def __init__(self, train_path, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len

        train_df = pd.read_csv(train_path)
        self.series_list = []
        self.train_history = {}

        for idx, row in train_df.iterrows():
            series_id = row.iloc[0]
            series_values = row.iloc[1:].values
            series_values = series_values[~pd.isna(series_values)].astype(float)

            if len(series_values) >= pred_len:
                self.series_list.append(series_values)
                self.train_history[idx] = series_values

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        series = self.series_list[idx]
        input_data = series[-self.seq_len:]
        
        if len(input_data) < self.seq_len:
            padding = np.full(self.seq_len - len(input_data), input_data[0])
            input_data = np.concatenate([padding, input_data])

        return torch.FloatTensor(input_data).unsqueeze(-1)

def get_m4_test_data(test_path, pred_len):
    test_df = pd.read_csv(test_path)
    y_true_list = []
    for _, row in test_df.iterrows():
        series_values = row.iloc[1:].values
        series_values = series_values[~pd.isna(series_values)].astype(float)
        if len(series_values) >= pred_len:
            y_true_list.append(series_values[:pred_len])
    return y_true_list

# ============================================================================
# 5. MODEL DEFINITION (PatchTST with RevIN) - CORRECTED ARCHITECTURE
# ============================================================================
class RevIN(nn.Module):
    """ Reversible Instance Normalization """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine: x = x * self.affine_weight; x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine: x = x - self.affine_bias; x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev; x = x + self.mean
        return x

class PatchTST(nn.Module):
    """
    PatchTST model with corrected channel-independent architecture.
    """
    def __init__(self, c_in, seq_len, pred_len, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.c_in = c_in
        self.pred_len = pred_len
        self.revin = RevIN(c_in)
        
        self.patch_len = patch_len
        self.stride = stride
        patch_num = (seq_len - patch_len) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, patch_num, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.head = nn.Linear(d_model * patch_num, pred_len)

    def forward(self, x): # x: [batch_size, seq_len, c_in]
        batch_size = x.shape[0]
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patch_num = x.shape[2]
        patch_len = x.shape[3]
        x = x.contiguous().view(-1, patch_num, patch_len)
        x = self.patch_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.reshape(batch_size * self.c_in, -1)
        x = self.head(x)
        x = x.view(batch_size, self.c_in, self.pred_len)
        x = x.permute(0, 2, 1)
        x = self.revin(x, 'denorm')
        return x

# ============================================================================
# 6. PHASE 1: TRAINING
# ============================================================================
def train_model():
    print()
    print("="*50)
    print("PHASE 1: MODEL TRAINING")
    print("="*50)

    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"✓ Checkpoint '{config.CHECKPOINT_PATH}' already exists. Skipping training.")
        return config.CHECKPOINT_PATH, 0

    train_dataset = M4Dataset(config.TRAIN_FILE, config.SEQ_LEN, config.PRED_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model = PatchTST(
        c_in=1, seq_len=config.SEQ_LEN, pred_len=config.PRED_LEN,
        patch_len=config.PATCH_LEN, stride=config.STRIDE, d_model=config.D_MODEL,
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
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        for batch_x in train_pbar:
            optimizer.zero_grad()
            batch_x = batch_x.to(config.DEVICE)
            
            pseudo_target = batch_x[:, -config.PRED_LEN:, :]
            outputs = model(batch_x)
            loss = criterion(outputs, pseudo_target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        val_loss = train_loss
        print(f"Epoch {epoch+1}: Train Loss (Validation): {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"✓ Model saved to {config.CHECKPOINT_PATH}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_training_time = time.time() - training_start_time
    print(f"✓ Training complete in {total_training_time:.2f} seconds.")
    return config.CHECKPOINT_PATH, total_training_time

# ============================================================================
# 7. PHASE 2: OPTIMIZATION & EVALUATION
# ============================================================================
def evaluate_and_optimize(training_time):
    print()
    print("="*50)
    print("PHASE 2: ONNX EXPORT, QUANTIZATION & EVALUATION")
    print("="*50)

    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint '{config.CHECKPOINT_PATH}' not found. Run training first.")
        return

    print(f"📦 Loading trained PyTorch model from {config.CHECKPOINT_PATH}...")
    model = PatchTST(
        c_in=1, seq_len=config.SEQ_LEN, pred_len=config.PRED_LEN,
        patch_len=config.PATCH_LEN, stride=config.STRIDE, d_model=config.D_MODEL,
        n_heads=config.N_HEADS, e_layers=config.E_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location='cpu'))
    model.eval()
    size_pytorch = os.path.getsize(config.CHECKPOINT_PATH) / (1024 * 1024)

    onnx_fp32_path = config.CHECKPOINT_PATH.replace('.pth', '_fp32.onnx')
    print(f"🔄 Exporting to ONNX FP32: {onnx_fp32_path}")
    dummy_input = torch.randn(1, config.SEQ_LEN, 1)
    torch.onnx.export(model, dummy_input, onnx_fp32_path, export_params=True, opset_version=14, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    size_fp32 = os.path.getsize(onnx_fp32_path) / (1024 * 1024)

    onnx_int8_path = config.CHECKPOINT_PATH.replace('.pth', '_int8.onnx')
    print(f"⚡ Quantizing to ONNX INT8: {onnx_int8_path}")
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QInt8)
    size_int8 = os.path.getsize(onnx_int8_path) / (1024 * 1024)

    print()
    print("📊 Starting evaluation...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.DEVICE == 'cuda' else ['CPUExecutionProvider']
    
    eval_dataset = M4Dataset(config.TRAIN_FILE, config.SEQ_LEN, config.PRED_LEN)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    y_true_list = get_m4_test_data(config.TEST_FILE, config.PRED_LEN)
    y_train_list = list(eval_dataset.train_history.values())

    results = {}
    for model_type, path in [('PyTorch', config.CHECKPOINT_PATH), ('ONNX_FP32', onnx_fp32_path), ('ONNX_INT8', onnx_int8_path)]:
        print()
        print(f"--- Evaluating {model_type} ---")
        preds, total_time = [], 0
        if 'ONNX' in model_type: session = ort.InferenceSession(path, providers=providers); input_name = session.get_inputs()[0].name
        else: model.to(config.DEVICE)

        for batch_x in tqdm(eval_loader, desc=f'Evaluating {model_type}'):
            start_time = time.time()
            if 'ONNX' in model_type: output = session.run(None, {input_name: batch_x.numpy()})[0]
            else: 
                with torch.no_grad(): output = model(batch_x.to(config.DEVICE)).cpu().numpy()
            total_time += (time.time() - start_time)
            preds.append(output)

        preds = np.concatenate(preds, axis=0).squeeze(-1)
        num_series = len(y_true_list)
        y_pred_list = preds[:num_series]

        smape_score = smape(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        mase_score = mase_all_series(y_true_list, y_pred_list, y_train_list)
        avg_batch_latency_ms = (total_time / len(eval_loader)) * 1000

        results[model_type] = {
            'sMAPE': smape_score, 'MASE': mase_score, 'Latency (ms/batch)': avg_batch_latency_ms,
            'Size (MB)': size_pytorch if model_type == 'PyTorch' else (size_fp32 if 'FP32' in model_type else size_int8)
        }
        if model_type == 'PyTorch':
            results[model_type]['Training Time (s)'] = training_time

    # --- 5. Print Results Table ---
    print()
    print()
    print("="*95)
    print(f"TABLE: PERFORMANCE BENCHMARK - M4 {config.FREQ.upper()} (pred_len={config.PRED_LEN})")
    print("="*95)
    print(f"{'Model Variant':<20} {'sMAPE':<12} {'MASE':<12} {'Latency (ms/batch)':<22} {'Size (MB)':<12} {'Training Time (s)':<20}")
    print("-"*95)
    for model_type, r in results.items():
        train_time_str = f"{r.get('Training Time (s)', ''):.2f}" if 'Training Time (s)' in r else ""
        print(f"{model_type:<20} {r['sMAPE']:<12.4f} {r['MASE']:<12.4f} {r['Latency (ms/batch)']:<22.4f} {r['Size (MB)']:<12.2f} {train_time_str:<20}")
    print("-"*95)
    
    # --- 6. Print Comparison ---
    fp32_res, int8_res = results['ONNX_FP32'], results['ONNX_INT8']
    compression = fp32_res['Size (MB)'] / int8_res['Size (MB)']
    speedup = fp32_res['Latency (ms/batch)'] / int8_res['Latency (ms/batch)']
    smape_deg = ((int8_res['sMAPE'] - fp32_res['sMAPE']) / fp32_res['sMAPE']) * 100

    print()
    print("SUMMARY: FP32 vs INT8")
    print("-"*25)
    print(f"🚀 Model Compression: {compression:.2f}x ({fp32_res['Size (MB)']:.2f}MB -> {int8_res['Size (MB)']:.2f}MB)")
    print(f"⚡ Inference Speedup: {speedup:.2f}x ({fp32_res['Latency (ms/batch)']:.4f}ms -> {int8_res['Latency (ms/batch)']:.4f}ms)")
    print(f"📉 Accuracy Degradation (sMAPE): {smape_deg:+.2f}%")
    print("="*95)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    if not os.path.exists(config.TRAIN_FILE) or not os.path.exists(config.TEST_FILE):
        print("! ERROR: M4 Dataset not found!")
        print(f"! Please upload '{config.TRAIN_FILE}' and '{config.TEST_FILE}' to the root directory. !")
    else:
        # Run Phase 1: Training
        _, training_time = train_model()
        
        # Clear memory before evaluation
        gc.collect()
        if config.DEVICE == 'cuda': torch.cuda.empty_cache()
            
        # Run Phase 2: Optimization and Evaluation
        evaluate_and_optimize(training_time)
