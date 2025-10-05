# Experiment 001: Baseline Weather Dataset Reproduction

**Objective:** Reproduce exact PatchTST baseline results on Weather dataset to validate implementation before M4 experiments.

## Dataset
- **Source:** `../../data/secondary/weather/weather.csv`
- **Size:** 52,696 observations × 21 features
- **Target:** OT (Oil Temperature)
- **Frequency:** 10-minute intervals
- **Task:** Multivariate forecasting (features='M', all 21 channels)
- **Split:** 70% train, 10% validation, 20% test (with seq_len border offset)

## Configuration
Exactly matches PatchTST paper weather.sh script:
- **Look-back (seq_len):** 336
- **Forecast horizons (pred_len):** 96, 192, 336, 720
- **Model:** PatchTST with RevIN
  - enc_in: 21 (all weather features)
  - d_model: 128
  - e_layers: 3
  - n_heads: 16
  - d_ff: 256
  - patch_len: 16
  - stride: 8
  - dropout: 0.2
  - fc_dropout: 0.2
  - head_dropout: 0.0
- **Training:** 100 epochs, patience: 20, batch_size: 128, lr: 0.0001, seed: 2021

## Files
- `config.py` - Experiment configuration (matches weather.sh exactly)
- `data_loader.py` - Weather dataset loader (matches Dataset_Custom)
- `model.py` - Complete PatchTST implementation (backbone + layers + RevIN)
- `patchtst_wrapper.py` - Model creation wrapper
- `train.py` - Training script with early stopping
- `evaluate.py` - Evaluation script with MSE/MAE
- `COMPARISON.md` - Detailed comparison with original implementation
- `results/` - Training logs, checkpoints, metrics

## Implementation Validation

### Data Loader (verified against PatchTST):
- ✅ Border calculation with seq_len offset
- ✅ Multivariate features (21 channels)
- ✅ Normalization on train data only
- ✅ Column reordering (date, features, target)

### Model (verified against PatchTST):
- ✅ All hyperparameters match weather.sh
- ✅ RevIN normalization enabled
- ✅ Channel-independent processing (individual=False)
- ✅ Patch length 16, stride 8

## Usage

### Train Model
```bash
# Install dependencies first
pip install torch numpy pandas scikit-learn tqdm tensorboard

# Train with default pred_len=96
python train.py

# Train with specific forecast horizon
python train.py --pred_len 192

# Run all forecast horizons
python train.py --pred_len 96
python train.py --pred_len 192
python train.py --pred_len 336
python train.py --pred_len 720
```

### Evaluate Model
```bash
# Evaluate specific checkpoint
python evaluate.py --checkpoint results/checkpoints/best_model_pred96.pt --pred_len 96
```

### Test Model Creation
```bash
# Verify model architecture
python patchtst_wrapper.py
```

### Test Data Loader
```bash
# Verify data loading
python data_loader.py
```

## Expected Results
Target MSE and MAE comparable to PatchTST paper Table 3 (Weather row):
- **pred_len=96:** MSE ≈ 0.17-0.20, MAE ≈ 0.20-0.23 (approximate from paper)

## Directory Structure
```
001_baseline_weather_reproduction/
├── config.py                  # Configuration
├── data_loader.py            # Dataset loader
├── model.py                  # PatchTST complete (RevIN + layers + backbone)
├── patchtst_wrapper.py       # Model creation
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── COMPARISON.md             # Implementation comparison
├── README.md                 # This file
└── results/
    ├── checkpoints/          # Model checkpoints
    └── logs/                 # TensorBoard logs
```

## Notes
- Start with pred_len=96 for fastest validation
- Model checkpoints auto-saved to `results/checkpoints/best_model_pred{pred_len}.pt`
- TensorBoard logs saved to `results/logs/pred_len_{pred_len}/`
- Early stopping patience: 20 epochs
- See COMPARISON.md for detailed implementation differences vs original

## Next Steps
1. Run baseline experiment (pred_len=96)
2. Validate MSE/MAE matches paper
3. Create experiment 002 with improvements
4. Compare results between baseline and improved versions
