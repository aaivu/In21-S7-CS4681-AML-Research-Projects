# PatchTST Production Source Code

Production-ready implementation of PatchTST for real-time M4 Competition forecasting with ONNX optimization and quantization support.

## 📁 Directory Structure

```
src/
├── config/              # Configuration management
│   ├── base_config.py   # Base configuration class
│   ├── m4_config.py     # M4 Competition configuration
│   └── standard_config.py # Standard datasets configuration
│
├── models/              # Model architecture
│   ├── patchtst.py      # Main PatchTST model (ONNX-compatible)
│   └── revin.py         # Reversible Instance Normalization
│
├── data/                # Data loading and processing
│   ├── m4_dataset.py    # M4 Competition dataset loader
│   └── standard_dataset.py # Standard LTSF dataset loader
│
├── training/            # Training pipeline
│   └── trainer.py       # Model trainer with early stopping
│
├── evaluation/          # Evaluation and metrics
│   ├── evaluator.py     # Model evaluator
│   └── metrics.py       # Forecasting metrics (MSE, MAE, sMAPE, MASE)
│
├── optimization/        # Model optimization
│   ├── onnx_export.py   # ONNX export functionality
│   └── quantization.py  # Post-training quantization
│
├── inference/           # Inference and deployment
│   └── predictor.py     # Unified predictor (PyTorch/ONNX)
│
└── utils/               # Utility functions
    ├── helpers.py       # Helper functions
    ├── logger.py        # Logging utilities
    └── checkpoint.py    # Checkpoint management
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scikit-learn tqdm onnx onnxruntime
```

### 2. M4 Competition Example

```python
from src.config.m4_config import M4Config
from src.models.patchtst import PatchTSTModel
from src.data.m4_dataset import create_m4_dataloaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.optimization.onnx_export import export_to_onnx
from src.optimization.quantization import quantize_onnx_model
import torch.nn as nn
import torch.optim as optim

# 1. Configuration
config = M4Config(frequency='Monthly')
print(config)

# 2. Create Model
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

# 3. Load Data
train_file, test_file = config.get_data_files()
train_loader, test_loader = create_m4_dataloaders(
    train_file=train_file,
    test_file=test_file,
    seq_len=config.seq_len,
    pred_len=config.pred_len,
    batch_size=config.batch_size
)

# 4. Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=nn.MSELoss(),
    optimizer=optim.AdamW(model.parameters(), lr=config.learning_rate),
    device=config.device,
    max_epochs=config.epochs,
    patience=config.patience
)

history = trainer.train()

# 5. Evaluate
evaluator = Evaluator(model, device=config.device, use_m4_metrics=True)
metrics = evaluator.evaluate(test_loader)
print(f"Test Metrics: {metrics}")

# 6. Export to ONNX
onnx_fp32_path = export_to_onnx(
    model=model,
    output_path='model_fp32.onnx',
    input_shape=(1, config.seq_len, config.c_in)
)

# 7. Quantize
onnx_int8_path = quantize_onnx_model(
    onnx_model_path=str(onnx_fp32_path),
    output_path='model_int8.onnx'
)
```

### 3. Standard Dataset Example

```python
from src.config.standard_config import StandardConfig
from src.models.patchtst import PatchTSTModel
from src.data.standard_dataset import create_standard_dataloaders
from src.training.trainer import Trainer
import torch.nn as nn
import torch.optim as optim

# Configuration
config = StandardConfig(dataset='weather', pred_len=96)

# Model
model = PatchTSTModel(
    c_in=config.c_in,
    seq_len=config.seq_len,
    pred_len=config.pred_len,
    patch_len=config.patch_len,
    stride=config.stride,
    d_model=config.d_model,
    n_heads=config.n_heads,
    e_layers=config.e_layers
)

# Data
data_file = config.get_data_file()
train_loader, val_loader, test_loader = create_standard_dataloaders(
    data_file=data_file,
    seq_len=config.seq_len,
    pred_len=config.pred_len,
    c_in=config.c_in,
    batch_size=config.batch_size
)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=optim.AdamW(model.parameters(), lr=config.learning_rate),
    device=config.device
)

history = trainer.train()
```

## 🔧 Key Features

### ONNX-Compatible Architecture
- Manual patching implementation (no unfold operation)
- Inline RevIN normalization
- Full compatibility with ONNX export

### Real-Time Optimization
- ONNX FP32 export with graph optimizations
- Post-training dynamic quantization (INT8)
- 3-4x model compression with minimal accuracy loss

### M4 Competition Support
- Variable-length series handling
- Official M4 metrics (sMAPE, MASE, OWA)
- Frequency-specific configurations

### Production-Ready
- Type hints throughout
- Comprehensive documentation
- Error handling and validation
- Checkpoint management
- Logging utilities

## 📊 Configuration System

### Base Configuration
```python
from src.config.base_config import BaseConfig

config = BaseConfig()
config.d_model = 128
config.n_heads = 16
config.epochs = 20
```

### M4 Configuration
```python
from src.config.m4_config import M4Config

# Monthly frequency
config = M4Config(frequency='Monthly')  # Horizon: 18

# Yearly frequency
config = M4Config(frequency='Yearly')   # Horizon: 6
```

### Standard Configuration
```python
from src.config.standard_config import StandardConfig

# Weather dataset, 96-step prediction
config = StandardConfig(dataset='weather', pred_len=96)

# Available datasets
StandardConfig.get_available_datasets()
# ['weather', 'traffic', 'electricity', 'illness', 'exchange_rate',
#  'etth1', 'etth2', 'ettm1', 'ettm2']
```

## 🔬 Model Architecture

### PatchTST Model
```python
from src.models.patchtst import PatchTSTModel

model = PatchTSTModel(
    c_in=7,              # Number of features
    seq_len=336,         # Input sequence length
    pred_len=96,         # Prediction horizon
    patch_len=16,        # Patch length
    stride=8,            # Patch stride
    d_model=128,         # Model dimension
    n_heads=16,          # Attention heads
    e_layers=3,          # Encoder layers
    d_ff=256,            # Feed-forward dimension
    dropout=0.2,         # Dropout rate
    use_revin=True       # Use RevIN normalization
)

# Model info
print(f"Parameters: {model.get_num_params():,}")
print(model)
```

## 📈 Evaluation Metrics

### Standard Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### M4 Competition Metrics
- **sMAPE**: Symmetric MAPE (primary M4 metric)
- **MASE**: Mean Absolute Scaled Error
- **OWA**: Overall Weighted Average (ranking metric)

```python
from src.evaluation.metrics import calculate_m4_metrics, calculate_standard_metrics

# Standard metrics
metrics = calculate_standard_metrics(y_true, y_pred)
print(metrics)  # {'MSE': ..., 'MAE': ..., 'RMSE': ..., 'MAPE': ...}

# M4 metrics
metrics = calculate_m4_metrics(y_true, y_pred, y_train)
print(metrics)  # {'sMAPE': ..., 'MASE': ..., 'MAE': ..., 'MSE': ...}
```

## 🚀 Deployment

### Export to ONNX
```python
from src.optimization.onnx_export import export_to_onnx

onnx_path = export_to_onnx(
    model=model,
    output_path='model_fp32.onnx',
    input_shape=(1, 336, 7),
    opset_version=14,
    verify=True
)
```

### Quantization
```python
from src.optimization.quantization import quantize_onnx_model

quantized_path = quantize_onnx_model(
    onnx_model_path='model_fp32.onnx',
    output_path='model_int8.onnx',
    weight_type='int8'
)
```

### Inference
```python
from src.inference.predictor import Predictor
import numpy as np

# PyTorch inference
predictor = Predictor('model.pth', model_type='pytorch', pytorch_model=model)
predictions = predictor.predict(input_data)

# ONNX FP32 inference
predictor = Predictor('model_fp32.onnx', model_type='onnx')
predictions = predictor.predict(input_data)

# ONNX INT8 inference
predictor = Predictor('model_int8.onnx', model_type='onnx_int8')
predictions = predictor.predict(input_data)
```

## 🔗 References

For more information about the underlying research and methodologies, please refer to the academic literature on:
- PatchTST architecture for long-term time series forecasting
- M4 Competition benchmark and evaluation metrics
- Reversible Instance Normalization (RevIN) for distribution shift
- ONNX Runtime for model optimization and deployment
