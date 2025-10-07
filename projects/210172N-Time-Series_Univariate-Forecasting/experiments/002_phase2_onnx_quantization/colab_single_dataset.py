"""
PHASE 2: ONNX & INT8 Quantization - Single Dataset Mode
All-in-one script for Google Colab

Assumes:
- PyTorch checkpoint from Phase 1 (e.g., weather_pred96.pth)
- Dataset file in root (weather.csv, traffic.csv, or Monthly-train.csv)

Usage:
  DATASET = 'weather'  # or 'traffic' or 'm4_monthly'

Default: Weather dataset
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
import os

# ============================================================================
# CONFIGURATION - CHANGE DATASET HERE
# ============================================================================

DATASET = 'weather'  # OPTIONS: 'weather', 'traffic', 'm4_monthly'

# Dataset configurations
DATASET_CONFIG = {
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
    'm4_monthly': {
        'file': 'Monthly-train.csv',
        'c_in': 1,
        'seq_len': 54,
        'pred_lens': [18],
        'patch_len': 6,
        'stride': 3
    }
}

class Config:
    D_MODEL = 128
    N_HEADS = 16
    E_LAYERS = 3
    D_FF = 256
    DROPOUT = 0.2
    BATCH_SIZE = 128

config = Config()

# ============================================================================
# METRICS
# ============================================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ============================================================================
# DATASET (Same as Phase 1)
# ============================================================================

class UnifiedDataset(Dataset):
    def __init__(self, file_path, dataset_type, seq_len, pred_len, flag='test'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()

        if dataset_type == 'm4_monthly':
            self._load_m4(file_path)
        else:
            self._load_standard(file_path, flag)

    def _load_standard(self, file_path, flag):
        df_raw = pd.read_csv(file_path)
        cols = list(df_raw.columns)
        if 'date' in cols:
            cols.remove('date')
            df_raw = df_raw[cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        border1 = len(df_raw) - num_test - self.seq_len
        border2 = len(df_raw)

        train_data = df_raw[:num_train].values
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_raw.values)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.use_revin = False

    def _load_m4(self, file_path):
        df = pd.read_csv(file_path)
        self.series_list = []
        for idx, row in df.iterrows():
            series = row.iloc[1:].values
            series = series[~pd.isna(series)].astype(float)
            if len(series) >= self.pred_len:
                self.series_list.append(series)
        self.use_revin = True

    def __len__(self):
        if self.dataset_type == 'm4_monthly':
            return len(self.series_list)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        if self.dataset_type == 'm4_monthly':
            series = self.series_list[idx].copy()
            if len(series) < self.seq_len + self.pred_len:
                series = np.pad(series, (self.seq_len + self.pred_len - len(series), 0), 'edge')
            series = series[-(self.seq_len + self.pred_len):]
            input_seq = series[:self.seq_len]
            target_seq = series[self.seq_len:]

            mean = input_seq.mean()
            std = input_seq.std() + 1e-5
            input_normalized = (input_seq - mean) / std

            return (
                torch.FloatTensor(input_normalized).unsqueeze(-1),
                torch.FloatTensor(target_seq).unsqueeze(-1),
                torch.FloatTensor([mean]),
                torch.FloatTensor([std])
            )
        else:
            s_begin = idx
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]

            return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

# ============================================================================
# MODEL (Same as Phase 1)
# ============================================================================

class PatchTST(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, patch_len, stride):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        patch_num = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, config.D_MODEL)
        self.pos_encoding = nn.Parameter(torch.zeros(1, patch_num, config.D_MODEL))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_FF,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.E_LAYERS)

        self.head = nn.Linear(config.D_MODEL * patch_num * c_in, pred_len * c_in)
        self.c_in = c_in
        self.pred_len = pred_len

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_size, c_in, patch_num, patch_len = x.shape

        x = x.reshape(batch_size * c_in, patch_num, patch_len)
        x = self.patch_embedding(x)
        x = x + self.pos_encoding
        x = self.transformer(x)

        x = x.reshape(batch_size, -1)
        x = self.head(x)
        x = x.reshape(batch_size, self.pred_len, self.c_in)

        return x

# ============================================================================
# ONNX PIPELINE
# ============================================================================

def export_and_quantize(checkpoint_path, dataset_type, c_in, seq_len, pred_len, patch_len, stride):
    """Export PyTorch -> ONNX -> Quantized ONNX"""

    # Load PyTorch model
    print(f"\n📦 Loading PyTorch checkpoint: {checkpoint_path}")
    model = PatchTST(c_in, seq_len, pred_len, patch_len, stride)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # ONNX paths
    base_name = checkpoint_path.replace('.pth', '')
    onnx_fp32 = f'{base_name}_fp32.onnx'
    onnx_int8 = f'{base_name}_int8.onnx'

    # Export to ONNX
    print(f"\n🔄 Exporting to ONNX FP32...")
    dummy_input = torch.randn(1, seq_len, c_in)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_fp32,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    onnx_model = onnx.load(onnx_fp32)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX FP32 exported: {onnx_fp32}")

    # Quantize to INT8
    print(f"\n⚡ Applying INT8 quantization...")
    quantize_dynamic(
        model_input=onnx_fp32,
        model_output=onnx_int8,
        weight_type=QuantType.QInt8
    )
    print(f"✓ ONNX INT8 quantized: {onnx_int8}")

    return onnx_fp32, onnx_int8

def evaluate_onnx(session, dataset, dataset_type):
    """Evaluate ONNX model"""
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    preds, trues = [], []

    eval_pbar = tqdm(loader, desc='Evaluating', leave=False)
    for batch_data in eval_pbar:
        if len(batch_data) == 4:  # M4
            batch_x, batch_y, mean, std = batch_data
            input_np = batch_x.numpy()
            mean_np, std_np = mean.numpy(), std.numpy()

            outputs = session.run(None, {session.get_inputs()[0].name: input_np})
            output_denorm = outputs[0] * std_np[:, np.newaxis, :] + mean_np[:, np.newaxis, :]

            preds.append(output_denorm)
            trues.append(batch_y.numpy())
        else:  # Weather/Traffic
            batch_x, batch_y = batch_data
            outputs = session.run(None, {session.get_inputs()[0].name: batch_x.numpy()})
            preds.append(outputs[0])
            trues.append(batch_y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize for Weather/Traffic
    if hasattr(dataset, 'scaler') and not dataset.use_revin:
        preds_shape = preds.shape
        preds = dataset.scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds_shape)
        trues = dataset.scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(preds_shape)

    return {'mse': mse(trues, preds), 'mae': mae(trues, preds)}

def benchmark_onnx(session, seq_len, c_in):
    """Benchmark ONNX inference speed"""
    dummy_input = np.random.randn(1, seq_len, c_in).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in tqdm(range(10), desc='Warmup', leave=False):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    start = time.time()
    for _ in tqdm(range(100), desc='Benchmarking', leave=False):
        _ = session.run(None, {input_name: dummy_input})

    return (time.time() - start) / 100 * 1000

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*100)
    print(f"PHASE 2: ONNX & INT8 QUANTIZATION - {DATASET.upper()} DATASET")
    print("="*100)

    # Check available execution providers for ONNX Runtime
    available_providers = ort.get_available_providers()
    use_gpu = 'CUDAExecutionProvider' in available_providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    print(f"\nONNX Runtime Execution Providers: {available_providers}")
    print(f"Using GPU: {'Yes (CUDA)' if use_gpu else 'No (CPU only)'}")

    cfg = DATASET_CONFIG[DATASET]
    print(f"\nDataset: {DATASET}")
    print(f"Forecast Horizons: {cfg['pred_lens']}\n")

    results = {}

    for pred_len in cfg['pred_lens']:
        print("\n" + "="*100)
        print(f"Processing pred_len={pred_len}")
        print("="*100)

        checkpoint = f'{DATASET}_pred{pred_len}.pth'

        if not os.path.exists(checkpoint):
            print(f"ERROR: {checkpoint} not found! Run Phase 1 first.")
            continue

        # Export and quantize
        onnx_fp32, onnx_int8 = export_and_quantize(
            checkpoint, DATASET, cfg['c_in'], cfg['seq_len'],
            pred_len, cfg['patch_len'], cfg['stride']
        )

        # Load dataset
        dataset = UnifiedDataset(cfg['file'], DATASET, cfg['seq_len'], pred_len, 'test')

        # Evaluate FP32
        print("\n📊 Evaluating ONNX FP32...")
        session_fp32 = ort.InferenceSession(onnx_fp32, providers=providers)
        metrics_fp32 = evaluate_onnx(session_fp32, dataset, DATASET)
        print(f"  MSE: {metrics_fp32['mse']:.4f}, MAE: {metrics_fp32['mae']:.4f}")

        print("⏱️  Benchmarking FP32 speed...")
        latency_fp32 = benchmark_onnx(session_fp32, cfg['seq_len'], cfg['c_in'])
        size_fp32 = os.path.getsize(onnx_fp32) / (1024 * 1024)
        print(f"  Latency: {latency_fp32:.2f}ms, Size: {size_fp32:.2f}MB")

        # Evaluate INT8
        print("\n📊 Evaluating ONNX INT8...")
        session_int8 = ort.InferenceSession(onnx_int8, providers=providers)
        metrics_int8 = evaluate_onnx(session_int8, dataset, DATASET)
        print(f"  MSE: {metrics_int8['mse']:.4f}, MAE: {metrics_int8['mae']:.4f}")

        print("⏱️  Benchmarking INT8 speed...")
        latency_int8 = benchmark_onnx(session_int8, cfg['seq_len'], cfg['c_in'])
        size_int8 = os.path.getsize(onnx_int8) / (1024 * 1024)
        print(f"  Latency: {latency_int8:.2f}ms, Size: {size_int8:.2f}MB")

        # PyTorch baseline size
        size_pytorch = os.path.getsize(checkpoint) / (1024 * 1024)

        results[pred_len] = {
            'pytorch': {'size': size_pytorch},
            'fp32': {'metrics': metrics_fp32, 'latency': latency_fp32, 'size': size_fp32},
            'int8': {'metrics': metrics_int8, 'latency': latency_int8, 'size': size_int8}
        }

    # ========================================================================
    # TABLE I: COMPLETE PERFORMANCE BENCHMARK
    # ========================================================================
    print("\n" + "="*100)
    print(f"TABLE I: PERFORMANCE BENCHMARK - {DATASET.upper()} DATASET")
    print("="*100)
    print(f"\n{'Model Variant':<30} {'Pred Len':<12} {'MSE':<12} {'MAE':<12} {'Latency (ms)':<15} {'Size (MB)':<12}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        print(f"{'PyTorch (Baseline)':<30} {pred_len:<12} {'[Phase 1]':<12} {'[Phase 1]':<12} {'[Phase 1]':<15} {r['pytorch']['size']:<12.2f}")
        print(f"{'ONNX (FP32)':<30} {pred_len:<12} {r['fp32']['metrics']['mse']:<12.4f} {r['fp32']['metrics']['mae']:<12.4f} {r['fp32']['latency']:<15.2f} {r['fp32']['size']:<12.2f}")
        print(f"{'ONNX (INT8 Quantized)':<30} {pred_len:<12} {r['int8']['metrics']['mse']:<12.4f} {r['int8']['metrics']['mae']:<12.4f} {r['int8']['latency']:<15.2f} {r['int8']['size']:<12.2f}")
        print()

    # ========================================================================
    # PATCHTST COMPARISON TABLE
    # ========================================================================
    print("\n" + "="*100)
    print(f"PATCHTST PERFORMANCE COMPARISON - {DATASET.upper()}")
    print("="*100)
    print(f"\n{'Pred Len':<12} {'Baseline':<25} {'ONNX FP32':<25} {'ONNX INT8':<25}")
    print(f"{'':12} {'MSE':<12} {'MAE':<12} {'MSE':<12} {'MAE':<12} {'MSE':<12} {'MAE':<12}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        print(f"{pred_len:<12} {'[Phase 1]':<12} {'[Phase 1]':<12} "
              f"{r['fp32']['metrics']['mse']:<12.3f} {r['fp32']['metrics']['mae']:<12.3f} "
              f"{r['int8']['metrics']['mse']:<12.3f} {r['int8']['metrics']['mae']:<12.3f}")

    # ========================================================================
    # COMPRESSION & SPEEDUP
    # ========================================================================
    print("\n" + "="*100)
    print("COMPRESSION & SPEEDUP ANALYSIS")
    print("="*100)
    print(f"\n{'Pred Len':<12} {'Compression (FP32→INT8)':<30} {'Speedup (FP32→INT8)':<30}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        compression = r['fp32']['size'] / r['int8']['size']
        speedup = r['fp32']['latency'] / r['int8']['latency']
        print(f"{pred_len:<12} {compression:<30.2f}x {speedup:<30.2f}x")

    # ========================================================================
    # ACCURACY DEGRADATION
    # ========================================================================
    print("\n" + "="*100)
    print("ACCURACY DEGRADATION (FP32 vs INT8)")
    print("="*100)
    print(f"\n{'Pred Len':<12} {'MSE Degradation':<25} {'MAE Degradation':<25}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        mse_deg = ((r['int8']['metrics']['mse'] - r['fp32']['metrics']['mse']) / r['fp32']['metrics']['mse']) * 100
        mae_deg = ((r['int8']['metrics']['mae'] - r['fp32']['metrics']['mae']) / r['fp32']['metrics']['mae']) * 100
        print(f"{pred_len:<12} {mse_deg:<25.2f}% {mae_deg:<25.2f}%")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("PHASE 2 COMPLETE - KEY ACHIEVEMENTS")
    print("="*100)

    # Use first pred_len for summary
    first_pred = cfg['pred_lens'][0]
    r = results[first_pred]
    compression = r['fp32']['size'] / r['int8']['size']
    speedup = r['fp32']['latency'] / r['int8']['latency']
    mse_loss = ((r['int8']['metrics']['mse'] - r['fp32']['metrics']['mse']) / r['fp32']['metrics']['mse']) * 100

    print(f"\n✓ Model Compression: {compression:.2f}x ({r['fp32']['size']:.2f}MB → {r['int8']['size']:.2f}MB)")
    print(f"✓ Inference Speedup: {speedup:.2f}x ({r['fp32']['latency']:.2f}ms → {r['int8']['latency']:.2f}ms)")
    print(f"✓ Accuracy Impact: {mse_loss:+.2f}% MSE change")

    print("\nGenerated Models:")
    for pred_len in cfg['pred_lens']:
        print(f"  ✓ {DATASET}_pred{pred_len}_fp32.onnx")
        print(f"  ✓ {DATASET}_pred{pred_len}_int8.onnx ⭐ (FINAL)")

    print("\n" + "="*100)

if __name__ == '__main__':
    main()
