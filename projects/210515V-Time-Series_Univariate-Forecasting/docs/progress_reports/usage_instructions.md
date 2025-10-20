# DLinear-Improved: Usage Instructions

**Project:** DLinear-Improved - Adaptive Multi-Scale Time Series Forecasting  
**Student:** 210515V  
**Last Updated:** 2025-10-20

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Training Models](#training-models)
6. [Making Predictions](#making-predictions)
7. [Understanding Results](#understanding-results)
8. [Customization Guide](#customization-guide)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/hasanga1/DLinear-Improved.git
cd DLinear-Improved

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train ensemble models (this will take 10-30 minutes)
cd src
python main.py

# 4. Make predictions with uncertainty estimates
python predict.py
```

That's it! You should see training progress, then prediction results with visualizations.

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 8 GB (16 GB recommended)
- **GPU**: Optional (CUDA-compatible GPU for faster training)

### Step-by-Step Installation

**Option 1: Using pip (Recommended)**

```bash
# Clone the repository
git clone https://github.com/hasanga1/DLinear-Improved.git
cd DLinear-Improved

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Option 2: Using conda**

```bash
# Clone the repository
git clone https://github.com/hasanga1/DLinear-Improved.git
cd DLinear-Improved

# Create conda environment
conda create -n dlinear python=3.9
conda activate dlinear

# Install PyTorch (adjust for your CUDA version if using GPU)
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import pandas; import captum; print('All dependencies installed successfully!')"
```

If you see the success message, you're ready to go!

---

## Project Structure

```
DLinear-Improved/
│
├── data/                          # Dataset directory
│   └── exchange_rate.csv          # Exchange rate time series data
│
├── src/                           # Source code
│   ├── main.py                    # Training script (entry point)
│   ├── predict.py                 # Prediction and evaluation script
│   ├── train.py                   # Training logic and Trainer class
│   ├── config.py                  # Configuration settings
│   ├── data_loader.py             # Data preprocessing and loading
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py
│   │   └── DLinear.py             # DLinear-Improved implementation
│   │
│   ├── utils/                     # Utility functions
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── tools.py               # Helper functions
│   │   └── timefeatures.py        # Time feature extraction
│   │
│   ├── checkpoints/               # Saved model checkpoints
│   │   └── Exchange_336_96_sl336_pl96_ensemble_*/
│   │       └── checkpoint.pth
│   │
│   └── results/                   # Output results and plots
│
├── docs/                          # Documentation
│   ├── literature_review.md       # Literature review
│   ├── methodology.md             # Detailed methodology
│   ├── research_proposal.md       # Research proposal
│   └── usage_instructions.md      # This file
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```

---

## Configuration

All settings are managed in `src/config.py`. Here are the key parameters:

### Core Settings

```python
# Model identification
model_id = 'Exchange_336_96'      # Model name
model = 'DLinear'                  # Model architecture
n_ensemble = 5                     # Number of ensemble members

# Data settings
data_path = 'exchange_rate.csv'    # Dataset filename
root_path = '../data/'             # Data directory
features = 'M'                     # 'M': multivariate, 'S': univariate
target = 'OT'                      # Target feature name
```

### Forecasting Task

```python
seq_len = 336        # Input sequence length (2 weeks hourly)
label_len = 48       # Label sequence length (overlap)
pred_len = 96        # Prediction horizon (4 days hourly)
```

### Model Architecture

```python
individual = False   # False: shared weights, True: per-channel weights
enc_in = 8           # Number of input features

# Enhancement flags
adaptive = True      # Enable adaptive moving averages
multi_scale = False  # Enable multi-scale decomposition
```

### Training Hyperparameters

```python
train_epochs = 10    # Maximum training epochs
batch_size = 8       # Samples per batch
learning_rate = 0.0005  # Adam optimizer learning rate
patience = 3         # Early stopping patience
loss = 'mse'         # Loss function
```

### Hardware

```python
use_gpu = False      # Enable GPU acceleration
gpu = 0              # GPU device ID
```

### How to Modify Configuration

**Method 1: Edit config.py directly**

```python
# Open src/config.py and modify the values
class Config:
    def __init__(self):
        self.pred_len = 192  # Change prediction horizon to 8 days
        self.adaptive = True  # Enable adaptive decomposition
        self.multi_scale = True  # Enable multi-scale decomposition
```

**Method 2: Modify after instantiation (for quick experiments)**

```python
# In your script
from config import Config
args = Config()
args.pred_len = 192
args.n_ensemble = 3  # Train fewer models for faster experimentation
```

---

## Training Models

### Basic Training

Train an ensemble of 5 models with default settings:

```bash
cd src
python main.py
```

**What happens during training:**

1. **Initialization**: Loads data, creates model, sets up optimizer
2. **Training Loop**: For each epoch:
   - Forward pass on training batches
   - Compute MSE loss
   - Backward pass and weight update
   - Validation evaluation
3. **Early Stopping**: Stops if validation loss doesn't improve for 3 epochs
4. **Checkpoint Saving**: Saves best model to `checkpoints/`
5. **Ensemble**: Repeats for all 5 models with different random seeds

**Expected Output:**

```
Starting ensemble training for 5 models...
>>>>>>> Training Model 1 of 5 <<<<<<<<
Use CPU
Epoch: 1, Cost time: 12.3456s
Train Loss: 0.1234567, Vali Loss: 0.1123456
Updating learning rate to 0.0005
...
Early stopping
Model saved to: checkpoints/Exchange_336_96_sl336_pl96_ensemble_0/
>>>>>>> Training Model 2 of 5 <<<<<<<<
...
Ensemble training complete. 5 models saved.
```

**Training Time Estimates:**

- **CPU (8 cores)**: ~10-30 minutes per model, ~1-2 hours total
- **GPU (CUDA)**: ~5-10 minutes per model, ~30-50 minutes total

### Training Variants

**1. Train with Adaptive Decomposition Only**

```python
# Edit config.py
self.adaptive = True
self.multi_scale = False
```

**2. Train with Multi-Scale Decomposition Only**

```python
# Edit config.py
self.adaptive = False
self.multi_scale = True
```

**3. Train Full DLinear-Improved**

```python
# Edit config.py
self.adaptive = True
self.multi_scale = True
```

**4. Train Baseline DLinear (for comparison)**

```python
# Edit config.py
self.adaptive = False
self.multi_scale = False
```

**5. Train Fewer Ensemble Members (faster experimentation)**

```python
# Edit config.py
self.n_ensemble = 3  # Instead of 5
```

**6. Enable GPU Acceleration**

```python
# Edit config.py
self.use_gpu = True
self.gpu = 0  # GPU device ID
```

### Monitoring Training

**Training Logs:**

Training progress is printed to console. You can redirect to a file:

```bash
python main.py > training_log.txt 2>&1
```

**Checkpoints:**

Models are saved to `src/checkpoints/{model_id}_ensemble_{i}/checkpoint.pth`

Each checkpoint contains:

- Model state dict (learned weights)
- Optimizer state
- Training epoch
- Best validation loss

---

## Making Predictions

### Basic Prediction

After training, generate predictions with uncertainty estimates:

```bash
cd src
python predict.py
```

**What happens during prediction:**

1. **Load Models**: Loads all trained ensemble models from checkpoints
2. **Predict**: Each model generates predictions on test set
3. **Aggregate**: Computes mean prediction and standard deviation
4. **Metrics**: Calculates MAE, MSE, RMSE, MAPE, correlation
5. **Attribution**: Analyzes feature importance using Integrated Gradients and Permutation Importance
6. **Visualization**: Creates plot of actual vs. predicted with confidence intervals
7. **Save**: Saves plot to `OT_prediction_sample_0.png`

**Expected Output:**

```
Loading ensemble models and making predictions...
>>> Predicting with model 1: Exchange_336_96_sl336_pl96_ensemble_0 <<<
--- Calculating feature attribution for model 1 ---
...

================================================================================
                         OVERALL ENSEMBLE METRICS
================================================================================
MAE: 0.012345, MSE: 0.000234, RMSE: 0.015321, MAPE: 2.34%, CORR: 0.945678
================================================================================

================================================================================
                      FEATURE ATTRIBUTION ANALYSIS (Sample 0)
================================================================================
This analysis shows which input features were most influential for the prediction.

   Feature  Integrated Gradients  Permutation Importance
0       OT              0.123456                0.045678
1  Feature1             0.098765                0.012345
2  Feature2             0.076543                0.009876
...

Interpretation:
 - Integrated Gradients: Shows feature contribution...
 - Permutation Importance: Shows model reliance on a feature...
================================================================================

================================================================================
                    SAMPLE PREDICTION ANALYSIS (Sample 0)
================================================================================
MAE for sample 0: 0.011234
RMSE for sample 0: 0.014567
Coverage (80% CI): 82.3%

Plot saved as: OT_prediction_sample_0.png
```

### Understanding Prediction Output

**1. Overall Metrics:**

- **MAE**: Average absolute error (lower is better)
- **MSE**: Mean squared error (lower is better)
- **RMSE**: Root MSE, in same units as data
- **MAPE**: Percentage error (interpretable)
- **CORR**: Correlation between predictions and truth (closer to 1 is better)

**2. Feature Attribution:**

- **Integrated Gradients**: Gradient-based attribution
  - Positive values: Feature increases prediction
  - Negative values: Feature decreases prediction
  - Magnitude: Importance strength
- **Permutation Importance**: Performance-based attribution
  - Higher values: More important for accuracy
  - Measures model reliance on feature

**3. Sample Analysis:**

- **Coverage**: Percentage of true values within confidence interval
  - Target: Should be close to confidence level (80%)
  - Higher than target: Conservative estimates
  - Lower than target: Under-confident estimates

**4. Visualization:**

The saved plot shows:

- **Blue line**: Actual values
- **Green line**: Predicted values
- **Green shaded area**: 80% confidence interval
- **X-axis**: Time steps (0 to 95 for 96-step horizon)
- **Y-axis**: Feature value (original scale)

---

## Understanding Results

### Model Checkpoints

Checkpoints are saved in `src/checkpoints/` with the naming pattern:

```
{model_id}_sl{seq_len}_pl{pred_len}_ensemble_{i}/checkpoint.pth
```

Example: `Exchange_336_96_sl336_pl96_ensemble_0/checkpoint.pth`

**To load a checkpoint manually:**

```python
import torch
from models.DLinear import Model
from config import Config

args = Config()
model = Model(args)
checkpoint = torch.load('checkpoints/Exchange_336_96_sl336_pl96_ensemble_0/checkpoint.pth')
model.load_state_dict(checkpoint)
```

### Interpreting Metrics

**When are results "good"?**

| Metric | Excellent | Good      | Acceptable | Poor   |
| ------ | --------- | --------- | ---------- | ------ |
| MAE    | < 0.01    | 0.01-0.02 | 0.02-0.05  | > 0.05 |
| MAPE   | < 2%      | 2-5%      | 5-10%      | > 10%  |
| CORR   | > 0.95    | 0.90-0.95 | 0.80-0.90  | < 0.80 |

_Note: These are rough guidelines for normalized exchange rate data. Actual thresholds depend on your application._

**Comparing Models:**

To compare different configurations:

1. Train baseline: `adaptive=False, multi_scale=False`
2. Train variant: `adaptive=True, multi_scale=False`
3. Compare metrics from `predict.py` output
4. Improvement = `(Baseline_MSE - Variant_MSE) / Baseline_MSE × 100%`

**Statistical Significance:**

With 5 ensemble members, you have 5 predictions per sample. To test significance:

```python
from scipy import stats
# Compare two model configurations
t_stat, p_value = stats.ttest_rel(baseline_errors, improved_errors)
if p_value < 0.05:
    print("Improvement is statistically significant!")
```

### Feature Attribution Interpretation

**Example Attribution Output:**

```
   Feature  Integrated Gradients  Permutation Importance
0       OT              0.089234                0.023456
1     Rate1             0.045678                0.012345
2     Rate2             0.012345                0.003456
```

**Interpretation:**

- **OT** is the most important feature (target itself has autoregressive signal)
- **Rate1** is moderately important (likely correlated with OT)
- **Rate2** is least important (weak relationship)

**Actionable Insights:**

- Features with high permutation importance are critical → ensure data quality
- Features with low importance → may be candidates for removal
- Negative integrated gradients → inverse relationship with target

---

## Customization Guide

### Using Your Own Dataset

**1. Prepare Your Data**

Your CSV should have this structure:

```csv
date,feature1,feature2,feature3,...
2020-01-01 00:00:00,1.234,5.678,9.012,...
2020-01-01 01:00:00,1.235,5.679,9.013,...
...
```

Requirements:

- First column: `date` (timestamp)
- Remaining columns: Numeric features
- Chronologically sorted
- No missing values (or handle them first)

**2. Update Configuration**

```python
# Edit config.py
self.data_path = 'your_data.csv'
self.enc_in = 10  # Number of features (excluding date)
self.target = 'your_target_feature'
self.freq = 'h'  # 'h': hourly, 'd': daily, etc.
```

**3. Adjust Sequence Lengths**

For different forecasting horizons:

```python
# Daily data, predict 30 days ahead
self.seq_len = 365   # 1 year of history
self.pred_len = 30   # 1 month prediction

# Hourly data, predict 24 hours ahead
self.seq_len = 168   # 1 week of history
self.pred_len = 24   # 1 day prediction
```

**4. Run Training**

```bash
python main.py
python predict.py
```

### Modifying Model Architecture

**1. Change Decomposition Kernel Size**

```python
# In src/models/DLinear.py, find:
kernel_size = 25  # Default

# Change to:
kernel_size = 49  # Larger (smoother trends)
kernel_size = 13  # Smaller (more responsive)
```

**2. Customize Multi-Scale Kernel Sizes**

```python
# In src/models/DLinear.py, find:
self.decompsition = MultiScaleDecomposition(
    kernel_sizes=[9, 25, 49],  # Default
    ...
)

# Change to your preferred scales:
self.decompsition = MultiScaleDecomposition(
    kernel_sizes=[7, 21, 63],  # Custom scales
    ...
)
```

**3. Add More Ensemble Members**

```python
# Edit config.py
self.n_ensemble = 10  # More diversity, better uncertainty estimates
```

### Hyperparameter Tuning

**Quick Tuning Guide:**

| To improve                             | Try adjusting                                 |
| -------------------------------------- | --------------------------------------------- |
| Underfitting (high train/val loss)     | Increase `seq_len`, decrease kernel size      |
| Overfitting (low train, high val loss) | Decrease `learning_rate`, increase `patience` |
| Slow training                          | Increase `batch_size`, enable GPU             |
| Unstable training                      | Decrease `learning_rate`                      |
| Poor long-term forecasts               | Increase `seq_len`, enable `multi_scale`      |

**Example Tuning Workflow:**

```python
# Start with defaults
args.learning_rate = 0.0005
args.train_epochs = 10
args.batch_size = 8

# If validation loss plateaus early:
args.learning_rate = 0.0001  # Lower learning rate
args.train_epochs = 20       # More epochs

# If training is too slow:
args.batch_size = 16         # Larger batches
args.use_gpu = True          # Enable GPU
```

---

## Troubleshooting

### Common Issues and Solutions

**Issue 1: Import Errors**

```
ImportError: No module named 'captum'
```

**Solution:**

```bash
pip install captum
# Or install all dependencies:
pip install -r requirements.txt
```

---

**Issue 2: CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# Option 1: Reduce batch size
self.batch_size = 4  # Instead of 8

# Option 2: Use CPU
self.use_gpu = False

# Option 3: Train one model at a time
self.n_ensemble = 1
```

---

**Issue 3: File Not Found**

```
FileNotFoundError: [Errno 2] No such file or directory: '../data/exchange_rate.csv'
```

**Solution:**

```bash
# Check you're in the src/ directory
cd src
python main.py

# Or update the path in config.py
self.root_path = '/absolute/path/to/data/'
```

---

**Issue 4: Poor Performance / High Loss**

**Diagnosis:**

- Check if data is properly normalized
- Verify train/val/test split makes sense
- Ensure no data leakage

**Solutions:**

```python
# 1. Check data statistics
import pandas as pd
df = pd.read_csv('../data/exchange_rate.csv')
print(df.describe())

# 2. Try different model configurations
self.adaptive = True
self.multi_scale = True

# 3. Adjust sequence lengths
self.seq_len = 512  # More history
self.pred_len = 48  # Shorter horizon
```

---

**Issue 5: Training Takes Too Long**

**Solutions:**

```python
# 1. Enable GPU (if available)
self.use_gpu = True

# 2. Reduce ensemble size
self.n_ensemble = 3

# 3. Increase batch size
self.batch_size = 16

# 4. Reduce epochs (for quick experiments)
self.train_epochs = 5
```

---

**Issue 6: Predictions Are All the Same**

**Possible Causes:**

- Model hasn't learned (check training loss)
- Data not properly normalized
- All ensemble members using same seed

**Solutions:**

```python
# 1. Verify training loss is decreasing
# Check console output during training

# 2. Ensure different seeds for ensemble
# This should happen automatically in main.py

# 3. Try longer training
self.train_epochs = 20
self.patience = 5
```

---

**Issue 7: Feature Attribution Fails**

```
Warning: captum is not installed. Feature attribution will not be available.
```

**Solution:**

```bash
pip install captum
```

---

### Getting Help

If you encounter issues not covered here:

1. **Check Documentation**: Review `methodology.md` and `research_proposal.md`
2. **Search Issues**: Check GitHub issues for similar problems
3. **Create Issue**: Open a new issue with:
   - Error message (full traceback)
   - Configuration used
   - Steps to reproduce
   - System information (OS, Python version)

---

## Advanced Usage

### Custom Loss Functions

Want to optimize for a different metric? Modify `src/train.py`:

```python
# Current: MSE loss
criterion = nn.MSELoss()

# Change to MAE (L1) loss
criterion = nn.L1Loss()

# Custom weighted loss
class WeightedMSE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, true):
        return torch.mean(self.weights * (pred - true) ** 2)

criterion = WeightedMSE(weights=torch.tensor([1.0, 1.5, 2.0, ...]))
```

### Batch Prediction

Predict on multiple samples efficiently:

```python
# In predict.py, modify to save all predictions
all_predictions = mean_preds_inv  # Shape: [n_samples, pred_len, n_features]
all_truths = trues_inv

# Save to file
np.save('all_predictions.npy', all_predictions)
np.save('all_truths.npy', all_truths)

# Or save to CSV
import pandas as pd
for i in range(n_samples):
    df = pd.DataFrame({
        'time_step': range(pred_len),
        'true': all_truths[i, :, target_feature_idx],
        'pred': all_predictions[i, :, target_feature_idx]
    })
    df.to_csv(f'prediction_sample_{i}.csv', index=False)
```

### Analyzing Learned Parameters

Visualize what the model learned:

```python
import torch
import matplotlib.pyplot as plt
from models.DLinear import Model
from config import Config

# Load model
args = Config()
args.adaptive = True
args.multi_scale = True
model = Model(args)
checkpoint = torch.load('checkpoints/Exchange_336_96_sl336_pl96_ensemble_0/checkpoint.pth')
model.load_state_dict(checkpoint)

# Extract and visualize adaptive kernel weights
if args.adaptive and not args.multi_scale:
    kernel_weights = model.decompsition.moving_avg.weights.detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(kernel_weights)
    plt.title('Learned Adaptive Kernel Weights')
    plt.xlabel('Position in Kernel')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.savefig('adaptive_kernel_weights.png')
    plt.show()

# Extract multi-scale weights
if args.multi_scale:
    scale_weights = model.decompsition.scale_weights.detach().numpy()
    import torch.nn.functional as F
    scale_weights_normalized = F.softmax(torch.tensor(scale_weights), dim=0).numpy()

    plt.figure(figsize=(8, 6))
    plt.bar(['Short (k=9)', 'Medium (k=25)', 'Long (k=49)'], scale_weights_normalized)
    plt.title('Learned Multi-Scale Weights')
    plt.ylabel('Weight')
    plt.grid(True, axis='y')
    plt.savefig('multi_scale_weights.png')
    plt.show()
```

### Implementing Walk-Forward Validation

For more realistic evaluation:

```python
# Create a new script: walk_forward.py
import numpy as np
from config import Config
from train import Trainer

args = Config()
n_splits = 5  # Number of walk-forward windows
test_size = len(test_data) // n_splits

all_predictions = []
all_truths = []

for i in range(n_splits):
    print(f"Walk-forward split {i+1}/{n_splits}")

    # Train on data up to this point
    # (Requires modifying data_loader to accept custom splits)

    # Evaluate on next window
    # ...

# Aggregate results
overall_mae = np.mean([...])
```

### Exporting for Production

Convert model to production-ready format:

```python
# Export to TorchScript for deployment
import torch
from models.DLinear import Model
from config import Config

args = Config()
model = Model(args)
model.load_state_dict(torch.load('checkpoints/.../checkpoint.pth'))
model.eval()

# Create example input
example_input = torch.randn(1, args.seq_len, args.enc_in)

# Trace and save
traced_model = torch.jit.trace(model, example_input)
traced_model.save('dlinear_improved_production.pt')

# Later, in production:
# loaded_model = torch.jit.load('dlinear_improved_production.pt')
# prediction = loaded_model(new_data)
```

---

## Best Practices

### For Research

1. **Version Control**: Commit after each experiment
2. **Reproducibility**: Always set random seeds
3. **Documentation**: Document all configuration changes
4. **Ablation**: Test one change at a time
5. **Multiple Runs**: Run experiments 3-5 times with different seeds
6. **Statistical Tests**: Use significance tests for comparisons

### For Development

1. **Start Simple**: Begin with baseline, add features incrementally
2. **Small Experiments**: Test on subset of data first
3. **Monitor Training**: Watch for overfitting signs
4. **Save Often**: Save checkpoints frequently
5. **Profile Code**: Identify bottlenecks before optimizing

### For Production

1. **Validate Data**: Check for distribution shift
2. **Monitor Performance**: Track prediction errors over time
3. **Version Models**: Keep track of model versions
4. **Error Handling**: Add robust error checking
5. **Logging**: Implement comprehensive logging
6. **Testing**: Write unit tests for critical components

---

## Performance Optimization Tips

### Speed Up Training

```python
# 1. Use GPU
self.use_gpu = True

# 2. Increase batch size (if memory allows)
self.batch_size = 32

# 3. Reduce ensemble size for experiments
self.n_ensemble = 3

# 4. Use fewer workers for data loading
self.num_workers = 4  # Adjust based on CPU cores

# 5. Mixed precision training (requires modification)
# See PyTorch AMP documentation
```

### Reduce Memory Usage

```python
# 1. Smaller batch size
self.batch_size = 4

# 2. Gradient accumulation (requires code modification)
# Accumulate gradients over multiple batches

# 3. Clear cache periodically
import torch
torch.cuda.empty_cache()  # If using GPU

# 4. Use CPU instead of GPU
self.use_gpu = False
```

### Improve Model Quality

```python
# 1. More training epochs
self.train_epochs = 20

# 2. Larger ensemble
self.n_ensemble = 10

# 3. Longer input sequences
self.seq_len = 512

# 4. Enable both enhancements
self.adaptive = True
self.multi_scale = True

# 5. Lower learning rate for fine-tuning
self.learning_rate = 0.0001
```

---

## Frequently Asked Questions

**Q: How long does training take?**
A: On CPU, about 1-2 hours for 5 ensemble members. On GPU, 30-50 minutes.

**Q: Can I use this for univariate forecasting?**
A: Yes! Set `features='S'` and specify your target feature in config.

**Q: How do I know if adaptive/multi-scale helps?**
A: Run ablation studies comparing all configurations and check if improvements are statistically significant.

**Q: What if I have daily data instead of hourly?**
A: Adjust `freq='d'` and scale `seq_len`/`pred_len` accordingly (e.g., seq_len=365 for 1 year, pred_len=30 for 1 month).

**Q: Can I use this for classification?**
A: No, this model is designed for regression (continuous value prediction).

**Q: How do I handle missing values?**
A: Preprocess your data to fill missing values (forward fill, interpolation, or removal) before training.

**Q: Is this suitable for real-time forecasting?**
A: Yes, inference is fast (~milliseconds). For real-time, deploy using TorchScript export.

**Q: How many features can the model handle?**
A: Tested with 8 features, but should scale to hundreds with appropriate memory.

---

## Additional Resources

### Documentation

- **Literature Review**: `docs/literature_review.md` - Background research
- **Methodology**: `docs/methodology.md` - Detailed technical approach
- **Research Proposal**: `docs/research_proposal.md` - Project overview

### Code Examples

- **Training**: `src/main.py`
- **Prediction**: `src/predict.py`
- **Model**: `src/models/DLinear.py`

### External Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Captum Documentation](https://captum.ai/)
- [Original DLinear Paper](https://arxiv.org/abs/2205.13504)

---

## Contact and Support

**Student:** 210515V  
**Repository:** https://github.com/hasanga1/DLinear-Improved

For questions or issues:

1. Check this documentation
2. Review the methodology document
3. Open an issue on GitHub
4. Contact your supervisor

---

**Last Updated:** October 20, 2025  
**Version:** 1.0

Happy forecasting! 🚀📈
