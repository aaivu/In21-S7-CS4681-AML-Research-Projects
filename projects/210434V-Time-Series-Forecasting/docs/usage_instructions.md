# Enhanced TS2Vec Usage Instructions

**Student:** 210434V  
**Research:** Enhanced TS2Vec with Hybrid Ensemble Architecture  
**Updated:** 2025-10-13

## 📋 Quick Start Guide

### Prerequisites
- Python 3.7+ with GPU support
- CUDA-compatible environment (Kaggle/Colab/Local GPU)
- Required datasets in `datasets/` folder

### Basic Usage
```bash
python train.py <dataset> <run_name> --loader <loader_type> --repr-dims 320 --seed 42 --eval
```

---

## 🚀 Complete Setup Instructions

### 1. Environment Setup

#### Option A: Kaggle Notebooks (Recommended)
```python
# Clone the repository
!git clone https://github.com/Niroshan2001/ts2vec.git
%cd ts2vec

# Switch to enhanced branch
!git fetch origin
!git checkout optimized-ensemble

# Install dependencies
!pip install bottleneck statsmodels scikit-learn
```

#### Option B: Local Setup
```bash
# Clone repository
git clone https://github.com/Niroshan2001/ts2vec.git
cd ts2vec
git checkout optimized-ensemble

# Install dependencies
pip install -r requirements.txt
pip install bottleneck statsmodels
```

### 2. Dataset Preparation

#### ETT Datasets Setup
```python
import os
import shutil

# For Kaggle environment
source_paths = [
    "/kaggle/input/ettsmall/ETTh1.csv",
    "/kaggle/input/ettsmall/ETTh2.csv", 
    "/kaggle/input/ettsmall/ETTm1.csv"
]

# Create datasets directory
os.makedirs("datasets", exist_ok=True)

# Copy datasets
for source_path in source_paths:
    filename = os.path.basename(source_path)
    dest_path = os.path.join("datasets", filename)
    shutil.copyfile(source_path, dest_path)
    print(f"Copied {filename}")
```

---

## 🎯 Running Experiments

### ETT Dataset Experiments

#### Complete ETT Evaluation (Recommended)
```bash
# Run all ETT experiments using the provided script
bash scripts/ett.sh
```

#### Individual Dataset Commands

##### ETTh1 (Hourly)
```bash
# Multivariate forecasting
python -u train.py ETTh1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# Univariate forecasting
python -u train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
```

##### ETTh2 (Hourly)
```bash
# Multivariate forecasting
python -u train.py ETTh2 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# Univariate forecasting
python -u train.py ETTh2 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
```

##### ETTm1 (15-minute)
```bash
# Multivariate forecasting
python -u train.py ETTm1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

# Univariate forecasting
python -u train.py ETTm1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
```

---

## ⚙️ Configuration Options

### Command Line Arguments

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `dataset` | Dataset name | Required | ETTh1, ETTh2, ETTm1 |
| `run_name` | Experiment identifier | Required | Any string |
| `--loader` | Data loading method | Required | forecast_csv, forecast_csv_univar |
| `--repr-dims` | TS2Vec representation dimension | 320 | Integer (recommended: 320) |
| `--batch-size` | Training batch size | 8 | Integer |
| `--lr` | Learning rate | 0.001 | Float |
| `--max-threads` | CPU threads for training | None | Integer (recommended: 8) |
| `--seed` | Random seed | None | Integer (recommended: 42) |
| `--eval` | Run evaluation after training | False | Flag |
| `--epochs` | Number of training epochs | None | Integer |
| `--max-train-length` | Maximum sequence length | 3000 | Integer |

### Enhanced TS2Vec Specific Parameters

#### Automatic Configurations (No manual tuning needed)
- **Ensemble weights:** Automatically optimized per dataset/horizon
- **Ridge regularization:** Grid search over α ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
- **Time features:** Daily cycle encoding automatically applied
- **Feature scaling:** StandardScaler applied per dataset

---

## 📊 Understanding Output

### Training Output
```
Loading data... done
Training time: 0:45:23

TS2Vec infer time: 124.56s
Ridge train time: {24: 12.3s, 48: 15.7s, 168: 18.2s, 336: 21.4s, 720: 25.8s}
Ridge infer time: {24: 2.1s, 48: 2.3s, 168: 2.8s, 336: 3.2s, 720: 3.7s}

Evaluation result: {
    'ours': {
        24: {'norm': {'MSE': 0.345, 'MAE': 0.432}, 'raw': {'MSE': 12.45, 'MAE': 2.87}},
        48: {'norm': {'MSE': 0.456, 'MAE': 0.523}, 'raw': {'MSE': 16.78, 'MAE': 3.21}},
        ...
    }
}
```

### Result Interpretation

#### Metrics Explained
- **MSE (Mean Squared Error):** Lower is better, emphasizes large errors
- **MAE (Mean Absolute Error):** Lower is better, robust to outliers
- **norm:** Metrics on standardized data (for model comparison)
- **raw:** Metrics on original scale (for practical interpretation)

#### Horizons Tested
- **24:** 1 day ahead (short-term)
- **48:** 2 days ahead
- **168:** 1 week ahead (medium-term)
- **336:** 2 weeks ahead
- **720:** 1 month ahead (long-term)

### Output Files
Results are saved to `training/{dataset}__{run_name}_{timestamp}/`:
```
training/ETTh1__forecast_multivar_20251013_143022/
├── model.pkl          # Trained TS2Vec model
├── out.pkl            # Detailed predictions and ground truth
└── eval_res.pkl       # Evaluation metrics and timing
```

---

## 🔬 Advanced Usage

### Custom Hyperparameters
```bash
# Custom learning rate and batch size
python train.py ETTh1 custom_run \
    --loader forecast_csv \
    --repr-dims 320 \
    --lr 0.0005 \
    --batch-size 16 \
    --epochs 100 \
    --seed 42 \
    --eval
```

### Debugging Mode
```bash
# Verbose output with timing information
python -u train.py ETTh1 debug_run --loader forecast_csv --repr-dims 320 --seed 42 --eval
```

### Memory-Efficient Training
```bash
# For limited GPU memory
python train.py ETTh1 efficient_run \
    --loader forecast_csv \
    --batch-size 4 \
    --max-train-length 2000 \
    --seed 42 \
    --eval
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Dataset Not Found
```
Error: FileNotFoundError: datasets/ETTh1.csv not found
```
**Solution:** Ensure datasets are copied to `datasets/` folder

#### 2. CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solutions:**
- Reduce `--batch-size` to 4 or 2
- Reduce `--max-train-length` to 2000
- Use CPU training (not recommended)

#### 3. Ridge Regression Warnings
```
Warning: LinAlgWarning: Ill-conditioned matrix
```
**Note:** These warnings are automatically suppressed and don't affect results

#### 4. Slow Training
**Solutions:**
- Increase `--max-threads` to match CPU cores
- Use GPU acceleration
- Consider cloud platforms (Kaggle/Colab)

### Performance Optimization

#### For Faster Training
```bash
# Optimized for speed
python train.py ETTh1 fast_run \
    --loader forecast_csv \
    --batch-size 16 \
    --max-threads 16 \
    --seed 42 \
    --eval
```

#### For Better Accuracy
```bash
# Optimized for accuracy
python train.py ETTh1 accurate_run \
    --loader forecast_csv \
    --repr-dims 512 \
    --lr 0.0005 \
    --max-train-length 5000 \
    --seed 42 \
    --eval
```

---

## 📈 Expected Results

### Performance Improvements (vs. Baseline TS2Vec)

| Dataset | Horizon | Expected MSE Improvement |
|---------|---------|-------------------------|
| ETTh1   | 24h     | 5-10% reduction        |
| ETTh1   | 168h    | 8-15% reduction        |
| ETTh1   | 720h    | 10-20% reduction       |
| ETTh2   | 24h     | 5-10% reduction        |
| ETTm1   | 96      | 8-12% reduction        |
| ETTm1   | 672     | 15-25% reduction       |

### Runtime Estimates

| Dataset | Training Time | Evaluation Time | Total |
|---------|---------------|-----------------|-------|
| ETTh1   | 30-45 min    | 5-10 min       | 35-55 min |
| ETTh2   | 30-45 min    | 5-10 min       | 35-55 min |
| ETTm1   | 2-3 hours    | 15-30 min      | 2.5-3.5 hrs |

---

## 🎓 Research Usage

### Reproducing Paper Results
```bash
# Run all experiments from the paper
bash scripts/ett.sh > results_log.txt 2>&1

# Extract metrics for paper
python analyze_results.py training/
```

### Ablation Studies
```python
# Modify tasks/forecasting.py to disable ensemble
# Set weights = [1.0, 0.0] for baseline TS2Vec only
# Set weights = [0.0, 1.0] for time features only
```

### Custom Datasets
1. Add dataset to `datasets/` folder
2. Modify `datautils.py` to add loading function
3. Update `detect_dataset_name()` in `forecasting.py`

---

## 📚 Additional Resources

### Documentation
- `README_Enhanced_TS2Vec.md` - Research overview
- `README_Technical_Details.md` - Complete technical specs
- `METHODOLOGY.md` - Research methodology
- `Algorithm_Enhanced_TS2Vec.md` - Algorithm specification

### Code Organization
```
├── train.py                    # Main training script
├── ts2vec.py                  # Core TS2Vec model
├── datautils.py               # Data loading utilities
├── tasks/
│   ├── forecasting.py         # Enhanced ensemble forecasting
│   └── _eval_protocols.py     # Ridge regression protocols
├── models/                    # TS2Vec architecture
└── scripts/
    └── ett.sh                # ETT experiment script
```

### Support
- **Issues:** Create GitHub issue with error logs
- **Questions:** Review documentation files
- **Contributions:** Fork repository and submit pull requests

---

**Note:** This enhanced version automatically optimizes ensemble weights and hyperparameters. For best results, use the default parameters shown in the examples above.

**Research Status:** ✅ Implementation Complete | ✅ Experiments Complete | 📋 Ready for Deployment