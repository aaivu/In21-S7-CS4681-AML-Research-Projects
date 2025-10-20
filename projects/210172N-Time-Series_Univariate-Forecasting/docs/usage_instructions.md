# Usage Instructions

## Project: PatchTST for Real-Time M4 Forecasting (210172N)

**Author:** Galappaththi A. S.
**Course:** CS4681 - Advanced Machine Learning Research
**Branch:** TS001

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Project Structure](#project-structure)
3. [Using the Production Source Code](#using-the-production-source-code)
4. [Running Experiments](#running-experiments)
5. [Data Preparation](#data-preparation)
6. [Training Models](#training-models)
7. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
8. [ONNX Export and Quantization](#onnx-export-and-quantization)

---

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 16GB RAM minimum

### Installation

1. **Clone and Navigate to Project:**
   ```bash
   cd projects/210172N-Time-Series_Univariate-Forecasting
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

### Verify Installation

```python
import torch
import onnx
import onnxruntime

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
```

---

## Project Structure

```
210172N-Time-Series_Univariate-Forecasting/
├── src/                          # Production source code
│   ├── config/                   # Configuration management
│   ├── models/                   # PatchTST model architecture
│   ├── data/                     # Data loaders
│   ├── training/                 # Training pipeline
│   ├── evaluation/               # Metrics and evaluation
│   ├── optimization/             # ONNX export & quantization
│   ├── inference/                # Prediction and deployment
│   ├── utils/                    # Utility functions
│   ├── examples/                 # Example scripts
│   └── README.md                 # API documentation
│
├── experiments/                  # Research experiments
│   ├── 001_baseline_weather_reproduction/
│   ├── 002_phase2_onnx_quantization/
│   ├── 003_unified_full_pipeline/
│   └── 004_m4_baseline_with_optimization/
│
├── data/                         # Datasets (gitignored)
│   ├── m4/                       # M4 Competition data
│   ├── secondary/                # Benchmark datasets
│   └── PatchTST/                 # Original PatchTST code
│
├── checkpoints/                  # Model checkpoints (gitignored)
├── results/                      # Experiment results (gitignored)
├── docs/                         # Documentation
└── README.md                     # Project overview
```

---

## Using the Production Source Code

The `src/` directory contains production-ready, modular code for all components.

### Quick Start Example

```python
import sys
sys.path.insert(0, 'src')

from src.config.m4_config import M4Config
from src.models.patchtst import PatchTSTModel
from src.data.m4_dataset import create_m4_dataloaders
from src.training.trainer import Trainer
import torch.nn as nn
import torch.optim as optim

# 1. Configuration
config = M4Config(frequency='Monthly')

# 2. Model
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

# 3. Data
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
    device=config.device
)

history = trainer.train()
```

### Configuration Options

**M4 Competition (6 frequencies):**
```python
from src.config.m4_config import M4Config

# Available frequencies: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
config = M4Config(frequency='Monthly')
```

**Standard Datasets (9 benchmarks):**
```python
from src.config.standard_config import StandardConfig

# Available: weather, traffic, electricity, illness, exchange_rate,
#            etth1, etth2, ettm1, ettm2
config = StandardConfig(dataset='weather', pred_len=96)
```

---

## Running Experiments

### Method 1: Using Example Scripts

**M4 Competition:**
```bash
cd src
python examples/example_m4_training.py --frequency Monthly
```

**Standard Datasets:**
```bash
python examples/example_standard_training.py --dataset weather --pred_len 96
```

### Method 2: Custom Experiments

Create a new experiment script in `experiments/`:

```python
# experiments/my_experiment.py
import sys
sys.path.insert(0, '../src')

from src.config.m4_config import M4Config
from src.models.patchtst import PatchTSTModel
# ... rest of your experiment code
```

---

## Data Preparation

### M4 Competition Data

1. **Download M4 Data:**
   - Visit M4 Competition website or use Kaggle
   - Download frequency-specific files (e.g., Monthly-train.csv, Monthly-test.csv)

2. **Place in Data Folder:**
   ```
   data/m4/
   ├── Monthly-train.csv
   ├── Monthly-test.csv
   ├── Quarterly-train.csv
   ├── Quarterly-test.csv
   └── ...
   ```

3. **Verify Data:**
   ```python
   from src.data.m4_dataset import M4Dataset

   dataset = M4Dataset(
       train_file='data/m4/Monthly-train.csv',
       seq_len=72,
       pred_len=18
   )
   print(f"Valid series: {len(dataset)}")
   ```

### Standard Datasets

1. **Download Datasets:**
   - Weather: 21 features, 52,696 timesteps
   - Traffic: 862 sensors, 17,544 timesteps
   - Electricity: 321 customers, 26,304 timesteps

2. **Place in Secondary Folder:**
   ```
   data/secondary/
   ├── weather/weather.csv
   ├── traffic/traffic.csv
   ├── electricity/electricity.csv
   └── ...
   ```

---

## Training Models

### Basic Training

```python
from src.training.trainer import Trainer
from src.utils.checkpoint import CheckpointManager

# Setup checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir='checkpoints/my_experiment',
    save_best_only=True
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=optimizer,
    device='cuda',
    checkpoint_manager=checkpoint_manager,
    max_epochs=20,
    patience=5
)

# Train
history = trainer.train()
```

### Advanced Training Options

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    optimizer=optimizer,
    device='cuda',
    max_epochs=50,
    patience=10,
    grad_clip=1.0  # Gradient clipping
)
```

---

## Evaluation and Benchmarking

### Standard Evaluation

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, device='cuda')
metrics = evaluator.evaluate(test_loader)

print(f"MSE: {metrics['MSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
```

### M4 Competition Metrics

```python
evaluator = Evaluator(model, device='cuda', use_m4_metrics=True)
metrics = evaluator.evaluate(test_loader)

print(f"sMAPE: {metrics['sMAPE']:.4f}")
print(f"MASE: {metrics['MASE']:.4f}")
```

### Generating Predictions

```python
# Get predictions without metrics
predictions = evaluator.predict(test_loader)

# Get predictions with targets
predictions, targets = evaluator.predict(test_loader, return_targets=True)
```

---

## ONNX Export and Quantization

### Export to ONNX FP32

```python
from src.optimization.onnx_export import export_to_onnx

onnx_path = export_to_onnx(
    model=model,
    output_path='checkpoints/model_fp32.onnx',
    input_shape=(1, config.seq_len, config.c_in),
    opset_version=14,
    verify=True
)
```

### Quantize to INT8

```python
from src.optimization.quantization import quantize_onnx_model

quantized_path = quantize_onnx_model(
    onnx_model_path='checkpoints/model_fp32.onnx',
    output_path='checkpoints/model_int8.onnx',
    weight_type='int8'
)
```

### Inference with Different Models

```python
from src.inference.predictor import Predictor

# PyTorch inference
predictor_pt = Predictor('checkpoints/best_model.pth',
                         model_type='pytorch',
                         pytorch_model=model)

# ONNX FP32 inference
predictor_fp32 = Predictor('checkpoints/model_fp32.onnx',
                           model_type='onnx')

# ONNX INT8 inference
predictor_int8 = Predictor('checkpoints/model_int8.onnx',
                           model_type='onnx_int8')

# Make predictions
predictions = predictor_int8.predict(input_data)
```

---

## Common Tasks

### Task 1: Train on New Dataset

1. Add dataset configuration to `src/config/standard_config.py`
2. Place data in `data/secondary/your_dataset/`
3. Run training:
   ```python
   config = StandardConfig(dataset='your_dataset', pred_len=96)
   # ... rest of training code
   ```

### Task 2: Customize Model Architecture

```python
# Modify configuration
config = M4Config(frequency='Monthly')
config.d_model = 256  # Increase model size
config.n_heads = 32
config.e_layers = 6

# Create model with custom config
model = PatchTSTModel(
    c_in=config.c_in,
    seq_len=config.seq_len,
    pred_len=config.pred_len,
    d_model=config.d_model,  # Custom value
    n_heads=config.n_heads,
    e_layers=config.e_layers
)
```

### Task 3: Benchmark Model Performance

```python
import time
import numpy as np

# Benchmark inference speed
latencies = []
for batch in test_loader:
    start = time.time()
    predictions = predictor.predict(batch[0])
    latencies.append(time.time() - start)

print(f"Average latency: {np.mean(latencies)*1000:.2f}ms")
print(f"Throughput: {len(test_loader.dataset)/sum(latencies):.2f} samples/s")
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size
```python
config.batch_size = 32  # Reduce from 64/128
```

### Issue: ONNX Export Fails

**Solution:** Check model compatibility
- Ensure no dynamic control flow
- Use manual patching (already implemented)
- Check ONNX opset version compatibility

### Issue: Data Not Found

**Solution:** Verify paths
```python
from pathlib import Path

data_path = Path('data/m4/Monthly-train.csv')
if not data_path.exists():
    print(f"File not found: {data_path}")
    print(f"Expected location: {data_path.absolute()}")
```

---

## Best Practices

1. **Always use configuration classes** for reproducibility
2. **Save checkpoints regularly** using CheckpointManager
3. **Set random seeds** for reproducible results
4. **Monitor GPU memory** during training
5. **Use validation set** for hyperparameter tuning
6. **Export to ONNX** for deployment
7. **Benchmark all model variants** (PyTorch, FP32, INT8)

---

## Additional Resources

- **API Documentation:** `src/README.md`
- **Example Scripts:** `src/examples/`
- **Research Proposal:** `docs/research_proposal.md`
- **Literature Review:** `docs/literature_review.md`
- **Methodology:** `docs/methodology.md`

For questions or issues, please refer to the project documentation or contact the supervisor.
