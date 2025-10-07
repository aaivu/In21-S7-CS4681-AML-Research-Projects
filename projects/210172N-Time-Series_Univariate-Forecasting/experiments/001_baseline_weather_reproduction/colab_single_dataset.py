"""
PHASE 1: Baseline PatchTST - Single Dataset Mode
All-in-one script for Google Colab

Assumes in root folder: weather.csv, traffic.csv, OR Monthly-train.csv

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
        'c_in': 21,  # number of features
        'seq_len': 336,
        'pred_lens': [96, 192],
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
    # Model
    D_MODEL = 128
    N_HEADS = 16
    E_LAYERS = 3
    D_FF = 256
    DROPOUT = 0.2

    # Training
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001
    EPOCHS = 20
    PATIENCE = 5
    SEED = 2021

config = Config()

# ============================================================================
# METRICS
# ============================================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# ============================================================================
# DATASET
# ============================================================================

class UnifiedDataset(Dataset):
    def __init__(self, file_path, dataset_type, seq_len, pred_len, flag='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dataset_type = dataset_type
        self.scaler = StandardScaler()

        if dataset_type == 'm4_monthly':
            self._load_m4(file_path, flag)
        else:
            self._load_standard(file_path, flag)

    def _load_standard(self, file_path, flag):
        """Load Weather/Traffic datasets"""
        df_raw = pd.read_csv(file_path)

        # Remove date column if exists
        cols = list(df_raw.columns)
        if 'date' in cols:
            cols.remove('date')
            df_raw = df_raw[cols]

        # 70/10/20 split
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        # Scale
        train_data = df_raw[border1s[0]:border2s[0]].values
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_raw.values)

        self.data_x = data[border1s[set_type]:border2s[set_type]]
        self.data_y = data[border1s[set_type]:border2s[set_type]]
        self.use_revin = False

    def _load_m4(self, file_path, flag):
        """Load M4 Monthly dataset"""
        df = pd.read_csv(file_path)
        self.series_list = []

        for idx, row in df.iterrows():
            series = row.iloc[1:].values
            series = series[~pd.isna(series)].astype(float)
            if len(series) >= self.pred_len:
                self.series_list.append(series)

        # Split for train/val (90/10)
        if flag == 'train':
            self.series_list = self.series_list[:int(len(self.series_list) * 0.9)]
        elif flag == 'val':
            self.series_list = self.series_list[int(len(self.series_list) * 0.9):]

        self.use_revin = True

    def __len__(self):
        if self.dataset_type == 'm4_monthly':
            return len(self.series_list)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        if self.dataset_type == 'm4_monthly':
            return self._getitem_m4(idx)
        return self._getitem_standard(idx)

    def _getitem_standard(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def _getitem_m4(self, idx):
        series = self.series_list[idx].copy()

        if len(series) < self.seq_len + self.pred_len:
            series = np.pad(series, (self.seq_len + self.pred_len - len(series), 0), 'edge')

        series = series[-(self.seq_len + self.pred_len):]
        input_seq = series[:self.seq_len]
        target_seq = series[self.seq_len:]

        # RevIN
        mean = input_seq.mean()
        std = input_seq.std() + 1e-5
        input_normalized = (input_seq - mean) / std

        return (
            torch.FloatTensor(input_normalized).unsqueeze(-1),
            torch.FloatTensor(target_seq).unsqueeze(-1),
            torch.FloatTensor([mean]),
            torch.FloatTensor([std])
        )

# ============================================================================
# MODEL
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

        # Patching
        x = x.permute(0, 2, 1)  # [batch, c_in, seq_len]
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
# TRAIN & EVALUATE
# ============================================================================

def train_and_evaluate(dataset_type, c_in, seq_len, pred_len, patch_len, stride, resume=True):
    """Train and evaluate model, optionally resume from saved checkpoint"""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if enough free memory; if not, fallback to CPU
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_mem < 2*1024**3:  # <2GB free
            device = torch.device('cpu')
            print("⚠ Not enough GPU memory, switching to CPU for evaluation")
    cfg = DATASET_CONFIG[dataset_type]

    # Load datasets
    train_ds = UnifiedDataset(cfg['file'], dataset_type, seq_len, pred_len, 'train')
    val_ds = UnifiedDataset(cfg['file'], dataset_type, seq_len, pred_len, 'val')
    test_ds = val_ds if dataset_type == 'm4_monthly' else UnifiedDataset(cfg['file'], dataset_type, seq_len, pred_len, 'test')

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = PatchTST(c_in, seq_len, pred_len, patch_len, stride).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model_file = f'{dataset_type}_pred{pred_len}.pth'
    best_val_loss = float('inf')
    patience_counter = 0

    # If resume is True and model exists, load it
    if resume and os.path.exists(model_file):
        print(f"\n⚡ Resuming from saved model: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        # Optional: skip training completely
        print("✓ Loaded model, skipping training.")
    else:
        print(f"\n{'='*80}")
        print(f"Training {dataset_type.upper()} (pred_len={pred_len})")
        print(f"Total epochs: {config.EPOCHS}, Early stopping patience: {config.PATIENCE}")
        print(f"{'='*80}\n")

        for epoch in range(1, config.EPOCHS + 1):
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}/{config.EPOCHS} [Train]', leave=False)
            for batch_data in train_pbar:
                if len(batch_data) == 4:  # M4 with RevIN
                    batch_x, batch_y, mean, std = batch_data
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    mean, std = mean.to(device), std.to(device)
                else:  # Standard
                    batch_x, batch_y = batch_data
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = model(batch_x)
                if len(batch_data) == 4:
                    output = output * std.unsqueeze(1) + mean.unsqueeze(1)

                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 4:
                        batch_x, batch_y, mean, std = batch_data
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        mean, std = mean.to(device), std.to(device)
                    else:
                        batch_x, batch_y = batch_data
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)

                    output = model(batch_x)
                    if len(batch_data) == 4:
                        output = output * std.unsqueeze(1) + mean.unsqueeze(1)
                    val_loss += criterion(output, batch_y).item()

            val_loss /= len(val_loader)
            improvement = '✓ NEW BEST!' if val_loss < best_val_loss else ''
            print(f'Epoch {epoch:03d}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{config.PATIENCE} {improvement}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_file)
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f'\n⚠ Early stopping triggered at epoch {epoch}')
                    break

        print(f'\n✓ Training complete! Best validation loss: {best_val_loss:.4f}')

    # Test evaluation (always run)
    print(f'\nEvaluating on test set...')
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:
                batch_x, batch_y, mean, std = batch_data
                batch_x = batch_x.to(device)
                mean, std = mean.to(device), std.to(device)
                output = model(batch_x)
                output = output * std.unsqueeze(1) + mean.unsqueeze(1)
            else:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(device)
                output = model(batch_x)

            preds.append(output.cpu().numpy())
            trues.append(batch_y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize if needed
    if hasattr(test_ds, 'scaler') and not test_ds.use_revin:
        preds_shape = preds.shape
        preds = test_ds.scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds_shape)
        trues = test_ds.scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(preds_shape)

    # Metrics
    mse_val = mse(trues, preds)
    mae_val = mae(trues, preds)
    rmse_val = rmse(trues, preds)

    # Benchmark latency
    print(f'Benchmarking inference speed...')
    dummy_input = torch.randn(1, seq_len, c_in).to(device)
    model.eval()
    for _ in range(10):  # Warmup
        _ = model(dummy_input)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    latency = (time.time() - start) / 100 * 1000
    size_mb = os.path.getsize(model_file) / (1024 * 1024)

    return {
        'mse': mse_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'latency': latency,
        'size': size_mb
    }

# ============================================================================
# MAIN
# ============================================================================

def evaluate_model(model, c_in, seq_len, pred_len, dataset_type):
    """Evaluate a pre-trained PatchTST model and return metrics."""
    import torch
    import time
    import os
    import numpy as np
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device

    cfg = DATASET_CONFIG[dataset_type]
    test_ds = UnifiedDataset(cfg['file'], dataset_type, seq_len, pred_len, 'test')

    # Reduce batch size to avoid OOM
    eval_batch_size = min(config.BATCH_SIZE, 16)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 4:  # M4 with RevIN
                batch_x, batch_y, mean, std = batch_data
                batch_x = batch_x.to(device)
                mean, std = mean.to(device), std.to(device)
                output = model(batch_x)
                output = output * std.unsqueeze(1) + mean.unsqueeze(1)
            else:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(device)
                output = model(batch_x)

            preds.append(output.cpu().numpy())
            trues.append(batch_y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Denormalize for Weather/Traffic
    if hasattr(test_ds, 'scaler') and not test_ds.use_revin:
        preds_shape = preds.shape
        preds = test_ds.scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds_shape)
        trues = test_ds.scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(preds_shape)

    # Metrics
    mse_val = mse(trues, preds)
    mae_val = mae(trues, preds)
    rmse_val = rmse(trues, preds)

    # Latency benchmark
    dummy_input = torch.randn(1, seq_len, c_in).to(device)
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    latency = (time.time() - start) / 100 * 1000

    # Model size
    size_mb = os.path.getsize(f'{dataset_type}_pred{pred_len}.pth') / (1024 * 1024)

    return {
        'mse': mse_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'latency': latency,
        'size': size_mb
    }

def main():
    print("="*100)
    print(f"PHASE 1: BASELINE PATCHTST - {DATASET.upper()} DATASET")
    print("="*100)

    # GPU Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Optimize for T4 GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✓ CUDA optimizations enabled for T4 GPU")
    else:
        print("⚠ No GPU available, using CPU (slower training)")

    cfg = DATASET_CONFIG[DATASET]
    print(f"\nDataset: {DATASET}")
    print(f"File: {cfg['file']}")
    print(f"Features: {cfg['c_in']}")
    print(f"Sequence Length: {cfg['seq_len']}")
    print(f"Forecast Horizons: {cfg['pred_lens']}\n")

    results = {}

    for pred_len in cfg['pred_lens']:
      model_file = f'{DATASET}_pred{pred_len}.pth'

      if os.path.exists(model_file):
          print(f"✅ Model for pred_len={pred_len} already exists. Skipping training and resuming evaluation.")
          
          # Free GPU memory
          torch.cuda.empty_cache()
          
          # Load model
          model = PatchTST(cfg['c_in'], cfg['seq_len'], pred_len, cfg['patch_len'], cfg['stride']).to(device)
          model.load_state_dict(torch.load(model_file, map_location=device))
          model.eval()

          # Run evaluation and benchmark
          result = evaluate_model(model, cfg['c_in'], cfg['seq_len'], pred_len, DATASET)
          results[pred_len] = result

          # Delete model to free GPU memory
          del model
          torch.cuda.empty_cache()
      else:
          # Train and evaluate as usual
          result = train_and_evaluate(
              DATASET,
              cfg['c_in'],
              cfg['seq_len'],
              pred_len,
              cfg['patch_len'],
              cfg['stride']
          )
          results[pred_len] = result


    # ========================================================================
    # TABLE I: PERFORMANCE BENCHMARK
    # ========================================================================
    print("\n" + "="*100)
    print(f"TABLE I: PRELIMINARY PERFORMANCE BENCHMARK - {DATASET.upper()}")
    print("="*100)
    print(f"\n{'Model Variant':<30} {'Pred Len':<12} {'MSE':<12} {'MAE':<12} {'Latency (ms)':<15} {'Size (MB)':<12}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        print(f"{'PyTorch (Baseline)':<30} {pred_len:<12} {r['mse']:<12.4f} {r['mae']:<12.4f} {r['latency']:<15.2f} {r['size']:<12.2f}")
        print(f"{'ONNX (FP32)':<30} {pred_len:<12} {'[Phase 2]':<12} {'[Phase 2]':<12} {'[Phase 2]':<15} {'[Phase 2]':<12}")
        print(f"{'ONNX (INT8 Quantized)':<30} {pred_len:<12} {'[Phase 2]':<12} {'[Phase 2]':<12} {'[Phase 2]':<15} {'[Phase 2]':<12}")
        print()

    # ========================================================================
    # PATCHTST PERFORMANCE TABLE
    # ========================================================================
    print("\n" + "="*100)
    print(f"PATCHTST PERFORMANCE - {DATASET.upper()} DATASET")
    print("="*100)
    print(f"\n{'Pred Len':<12} {'PatchTST (Baseline)':<35} {'PatchTST (Enhanced - Phase 2)':<35}")
    print(f"{'':12} {'MSE':<17} {'MAE':<17} {'MSE':<17} {'MAE':<17}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        print(f"{pred_len:<12} {r['mse']:<17.3f} {r['mae']:<17.3f} {'[Phase 2]':<17} {'[Phase 2]':<17}")

    # ========================================================================
    # DETAILED METRICS
    # ========================================================================
    print("\n" + "="*100)
    print("DETAILED METRICS")
    print("="*100)
    print(f"\n{'Pred Len':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Latency(ms)':<15} {'Size(MB)':<12}")
    print("-"*100)

    for pred_len in cfg['pred_lens']:
        r = results[pred_len]
        print(f"{pred_len:<12} {r['mse']:<12.4f} {r['mae']:<12.4f} {r['rmse']:<12.4f} {r['latency']:<15.2f} {r['size']:<12.2f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("PHASE 1 COMPLETE - BASELINE ESTABLISHED")
    print("="*100)
    print("\nSaved Models:")
    for pred_len in cfg['pred_lens']:
        print(f"  ✓ {DATASET}_pred{pred_len}.pth")

    print("\nTo run other datasets:")
    print("  1. Change: DATASET = 'traffic' or 'weather' or 'm4_monthly'")
    print("  2. Re-run the script")
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
