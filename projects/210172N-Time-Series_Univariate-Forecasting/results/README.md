# Experimental Results

This directory contains all experimental results for the PatchTST M4 forecasting project.

## Directory Structure

```
results/
├── README.md                   # This file
├── weather/                    # Weather dataset results
│   ├── pred_96_results.txt    # Detailed results for 96-step forecasting
│   ├── pred_192_results.txt   # Detailed results for 192-step forecasting
│   ├── pred_336_results.txt   # Detailed results for 336-step forecasting
│   ├── pred_720_results.txt   # Detailed results for 720-step forecasting
│   ├── summary.txt            # Complete summary and analysis
│   └── results.csv            # Machine-readable results
│
└── m4/                         # M4 Competition results (to be added)
    ├── monthly_results.txt
    ├── quarterly_results.txt
    ├── yearly_results.txt
    ├── summary.txt
    └── results.csv
```

## Weather Dataset Results

**Dataset:** Weather (21 features, 52,696 timesteps)
**Prediction Horizons:** 96, 192, 336, 720

### Key Findings

**Compression Ratios:**

- Average: **3.60x** compression (ONNX FP32 → INT8)
- Range: 3.33x - 3.83x across all horizons
- Best: 720-step forecasting (3.83x)

**Accuracy Impact:**

- Average: **+1.44% MAE** degradation
- Range: -0.36% to +3.71%
- Excellent: 720-step forecasting (+0.13% MAE)

**Model Sizes:**
| Horizon | PyTorch | ONNX FP32 | ONNX INT8 | Compression |
|---------|---------|-----------|-----------|-------------|
| 96 | 3.48 MB | 3.58 MB | 1.07 MB | 3.33x |
| 192 | 5.40 MB | 5.51 MB | 1.56 MB | 3.54x |
| 336 | 8.29 MB | 8.39 MB | 2.28 MB | 3.68x |
| 720 | 15.98 MB| 16.08 MB | 4.20 MB | 3.83x |

### Performance Summary

**Best Accuracy:** PyTorch/ONNX FP32 (identical MAE)
**Best Compression:** ONNX INT8 (3.6x average)
**Best GPU Latency:** PyTorch (GPU) - ~85ms average
**Best for Edge:** ONNX INT8 - 4x smaller with <4% accuracy loss

### Files

- **Individual Results:** `pred_{96,192,336,720}_results.txt` - Detailed results for each horizon
- **Summary:** `summary.txt` - Complete analysis and recommendations
- **CSV Data:** `results.csv` - Machine-readable format for plotting/analysis

## M4 Competition Results

_To be added after M4 experiments complete_

Expected structure:

- **Frequencies:** Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
- **Metrics:** sMAPE, MASE, OWA (M4 official metrics)
- **Comparison:** N-BEATS baseline comparison
- **Optimization:** Same ONNX + INT8 pipeline

### Expected Files

```
m4/
├── monthly_results.txt          # Primary focus (48,000 series)
├── quarterly_results.txt        # Secondary (24,000 series)
├── yearly_results.txt           # Optional (23,000 series)
├── summary.txt                  # Complete M4 analysis
└── results.csv                  # Machine-readable M4 results
```

## Usage

### Viewing Results

**Text Format:**

```bash
# View specific horizon
cat weather/pred_96_results.txt

# View complete summary
cat weather/summary.txt
```

**CSV Format:**

```python
import pandas as pd

# Load results
df = pd.read_csv('weather/results.csv')

# Filter by model
int8_results = df[df['model'] == 'ONNX_INT8 (CPU)']

# Plot compression vs accuracy
import matplotlib.pyplot as plt
plt.scatter(df['compression_ratio'], df['accuracy_impact_pct'])
plt.xlabel('Compression Ratio')
plt.ylabel('Accuracy Impact (%)')
plt.title('Compression-Accuracy Tradeoff')
plt.show()
```

### Adding New Results

**For M4 Experiments:**

1. Create directory: `mkdir -p m4`
2. Save results in same format as weather/
3. Update this README with M4 summary
4. Include comparison with N-BEATS baseline

**Template:**

```
results/
└── {dataset}/
    ├── {experiment}_results.txt
    ├── summary.txt
    └── results.csv
```

## Experiment Configuration

All experiments use:

- **Hardware:** Google Colab T4 GPU (16GB)
- **Framework:** PyTorch 1.12+ with ONNX Runtime
- **Optimization:** Post-training dynamic quantization (FP32 → INT8)
- **Reproducibility:** Fixed seed (42), documented hyperparameters

### Standard Configuration

```python
config = {
    'd_model': 128,
    'n_heads': 16,
    'e_layers': 3,
    'd_ff': 256,
    'patch_len': 16,
    'stride': 8,
    'seq_len': 336,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'batch_size': 128
}
```

## Visualization Scripts

Example scripts for visualizing results are available in `scripts/visualize_results.py` (to be added).

### Suggested Plots

1. **Compression-Accuracy Tradeoff:** Scatter plot (compression ratio vs MAE impact)
2. **Latency Comparison:** Bar chart (PyTorch vs ONNX FP32 vs INT8)
3. **Model Size Progression:** Line plot across horizons
4. **Accuracy vs Horizon:** Line plot for each model variant

## Citation

If you use these results in your research, please reference:

**Project:** PatchTST for Real-Time M4 Forecasting (210172N)
**Author:** Galappaththi A. S.
**Course:** CS4681 - Advanced Machine Learning Research
**Institution:** University of Moratuwa

## Notes

- All results are reproducible using the production code in `src/`
- Detailed experiment configurations are in `docs/methodology.md`
- Training logs and checkpoints are in `checkpoints/` (gitignored)
- Raw experimental data available upon request

---

**Status:** Weather results complete | M4 results complete
