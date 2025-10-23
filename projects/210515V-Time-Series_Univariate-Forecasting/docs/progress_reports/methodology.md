# Methodology: Time Series Univariate Forecasting

**Student:** 210515V
**Research Area:** Time Series Univariate Forecasting
**Date:** 2025-09-01
**Last Updated:** 2025-10-20

## 1. Overview

This document outlines the methodology for the DLinear-Improved project, which aims to enhance the DLinear time series forecasting model through adaptive decomposition, multi-scale temporal analysis, and feature attribution mechanisms. The research follows an empirical approach, implementing incremental improvements to the baseline DLinear architecture and rigorously evaluating their impact on forecasting accuracy and interpretability.

The methodology encompasses: (1) data collection and preprocessing, (2) model architecture design with adaptive and multi-scale components, (3) ensemble learning for improved robustness, (4) feature attribution for interpretability, and (5) comprehensive experimental evaluation against baseline models.

## 2. Research Design

### 2.1 Research Approach

This research adopts an **empirical, iterative experimental design** with the following components:

1. **Baseline Implementation**: First establish a working implementation of the original DLinear model to serve as a performance benchmark.

2. **Incremental Enhancement**: Systematically add improvements one at a time:

   - Adaptive moving averages with learnable kernels
   - Multi-scale decomposition with multiple kernel sizes
   - Learnable scale weighting mechanisms
   - Feature attribution methods for interpretability

3. **Ablation Studies**: Conduct ablation experiments to isolate the contribution of each component, determining which enhancements provide genuine value.

4. **Ensemble Learning**: Implement ensemble methods with multiple model instances to improve robustness and reduce prediction variance.

5. **Comparative Evaluation**: Compare the improved model against the baseline and potentially other time series forecasting methods.

### 2.2 Research Questions

1. **RQ1**: Does adaptive moving average decomposition with learnable kernels outperform fixed-kernel decomposition?

2. **RQ2**: Can multi-scale decomposition capture temporal patterns more effectively than single-scale approaches?

3. **RQ3**: How do learnable scale weights distribute across different temporal scales, and what does this reveal about the data?

4. **RQ4**: Which input features and time steps contribute most to predictions, as revealed by feature attribution methods?

5. **RQ5**: Does ensemble learning with multiple DLinear-Improved instances provide better performance than a single model?

### 2.3 Hypotheses

- **H1**: Adaptive moving averages will achieve lower forecasting error than fixed kernels by learning data-specific smoothing patterns.

- **H2**: Multi-scale decomposition will capture both short-term fluctuations and long-term trends more effectively than single-scale decomposition.

- **H3**: Feature attribution methods will reveal interpretable patterns in which features and time windows are most predictive.

- **H4**: Ensemble models will reduce prediction variance and improve overall accuracy compared to individual models.

## 3. Data Collection

### 3.1 Data Sources

**Primary Dataset**: Exchange Rate Dataset

- Source: Publicly available time series forecasting benchmark
- File: `exchange_rate.csv`
- Location: `../data/exchange_rate.csv`

This dataset is widely used in time series forecasting research and allows for direct comparison with published results.

### 3.2 Data Description

**Exchange Rate Dataset Characteristics:**

- **Temporal Coverage**: Historical exchange rate data with hourly frequency
- **Features**: 8 different exchange rate series (multivariate)
  - Target feature: 'OT' (primary currency pair)
  - Additional features: 7 related exchange rate series
- **Total Records**: Multiple years of hourly observations
- **Frequency**: Hourly (`freq='h'`)
- **Missing Values**: Handled during preprocessing (if any)

**Data Statistics:**

- The dataset exhibits typical financial time series characteristics:
  - Non-stationary trends
  - Periodic patterns (daily, weekly cycles)
  - Volatility clustering
  - Occasional regime changes

### 3.3 Data Preprocessing

**3.3.1 Train-Validation-Test Split**

The dataset is split into three sets using temporal ordering (no shuffling):

```
Training Set:   70% (earliest data)
Validation Set: 10% (middle portion)
Test Set:       20% (most recent data)
```

This temporal split ensures no data leakage and realistic evaluation on future predictions.

**3.3.2 Normalization**

- **Method**: StandardScaler (z-score normalization)
- **Fit**: Scaler is fit only on training data
- **Transform**: Applied to all three sets (train, validation, test)
- **Formula**: $x_{normalized} = \frac{x - \mu_{train}}{\sigma_{train}}$

This prevents information leakage from validation/test sets and handles different scales across features.

**3.3.3 Sequence Generation**

Time series data is converted to supervised learning format:

- **Input Sequence Length** (`seq_len`): 336 time steps (2 weeks of hourly data)
- **Label Length** (`label_len`): 48 time steps (overlap for decoder initialization)
- **Prediction Length** (`pred_len`): 96 time steps (4 days ahead)

For each sample:

- Input: `X[t-336:t]` → 336 historical observations
- Output: `Y[t:t+96]` → 96 future values to predict

**3.3.4 Feature Engineering**

Time features are optionally extracted from timestamps:

- Month, Day, Weekday, Hour (for `timeenc=0`)
- Frequency-based embeddings (for `timeenc=1`)

Currently using raw features without additional time encoding (`features='M'` for multivariate).

## 4. Model Architecture

### 4.1 DLinear-Improved Architecture

The DLinear-Improved model enhances the baseline DLinear through several key components:

#### 4.1.1 Decomposition Module

**Baseline Decomposition** (Standard DLinear):

```
moving_avg(kernel_size=25, stride=1)
→ Trend = MA(X)
→ Seasonal = X - Trend
```

**Adaptive Decomposition** (DLinear-Improved with `adaptive=True`):

```python
AdaptiveMovingAvg(kernel_size, learnable=True)
- Learnable kernel weights: W ∈ ℝ^k (initialized to 1/k)
- Softmax normalization: ensures ∑W_i = 1
- Replicate padding: preserves boundary information
→ Trend = Conv1D(X, weights=W)
→ Seasonal = X - Trend
```

**Multi-Scale Decomposition** (DLinear-Improved with `multi_scale=True`):

```python
MultiScaleDecomposition(kernel_sizes=[9, 25, 49])
- Three parallel moving averages at different scales:
  * Scale 1 (k=9):  Short-term smoothing
  * Scale 2 (k=25): Medium-term trends
  * Scale 3 (k=49): Long-term trends
- Learnable scale weights: α ∈ ℝ^3 (softmax normalized)
- Combined trend: Trend = ∑ α_s · MA_s(X)
- Seasonal: Seasonal = X - Trend
```

#### 4.1.2 Linear Projection Module

After decomposition, separate linear projections forecast trend and seasonal components:

```python
For each component (Seasonal, Trend):
  Input shape:  [Batch, Seq_len=336, Channels=8]
  Projection:   Linear(336 → 96)
  Output shape: [Batch, Pred_len=96, Channels=8]

Final prediction: Y = Y_seasonal + Y_trend
```

**Individual vs. Shared Mode**:

- `individual=True`: Separate linear layers for each of 8 channels (more parameters, channel-specific patterns)
- `individual=False`: Shared linear layer across channels (fewer parameters, shared patterns)

Currently using `individual=False` for efficiency.

#### 4.1.3 Feature Attribution Module

Two complementary attribution methods are implemented:

**1. Integrated Gradients** (Gradient-based):

```python
IntegratedGradients(model)
- Baseline: Zero tensor
- Integration steps: 50 (default)
- Attributes prediction to input features via path integral
- Output: Feature importance scores for each input feature
```

**2. Permutation Importance** (Perturbation-based):

```python
For each feature:
  1. Compute baseline loss on original data
  2. Shuffle feature values across samples
  3. Compute loss on permuted data
  4. Importance = Loss_permuted - Loss_baseline
- Captures feature contribution to overall performance
```

### 4.2 Ensemble Architecture

To improve robustness, the system implements ensemble learning:

```
Ensemble Size: n_ensemble = 5
Strategy: Train 5 independent models with different random initializations
Prediction: Ensemble_pred = (1/5) ∑ Model_i(X)
```

Benefits:

- Reduces prediction variance
- Averages out random initialization effects
- More robust to outliers and noise

### 4.3 Mathematical Formulation

**Complete DLinear-Improved Forward Pass:**

Given input $X \in \mathbb{R}^{B \times L_{in} \times C}$ where:

- $B$ = batch size
- $L_{in}$ = input sequence length (336)
- $C$ = number of channels (8)

**Step 1: Multi-Scale Decomposition**
$$T = \sum_{s=1}^{3} \text{softmax}(\alpha)_s \cdot \text{Conv1D}(X, W_s)$$
$$S = X - T$$

where $W_s$ are learnable kernel weights for scale $s$.

**Step 2: Linear Projections**
$$\hat{Y}_T = W_T \cdot T + b_T$$
$$\hat{Y}_S = W_S \cdot S + b_S$$

where $W_T, W_S \in \mathbb{R}^{L_{out} \times L_{in}}$, $b_T, b_S \in \mathbb{R}^{L_{out}}$.

**Step 3: Combination**
$$\hat{Y} = \hat{Y}_T + \hat{Y}_S \in \mathbb{R}^{B \times L_{out} \times C}$$

where $L_{out}$ = prediction length (96).

## 5. Experimental Setup

### 5.1 Evaluation Metrics

The following metrics are computed to comprehensively evaluate model performance:

1. **Mean Squared Error (MSE)**:
   $$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

   - Primary metric for optimization
   - Heavily penalizes large errors

2. **Mean Absolute Error (MAE)**:
   $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

   - Robust to outliers
   - Interpretable in original units

3. **Root Mean Squared Error (RMSE)**:
   $$RMSE = \sqrt{MSE}$$

   - In same units as target variable
   - Standard metric for comparison

4. **Mean Absolute Percentage Error (MAPE)**:
   $$MAPE = \frac{100\%}{N} \sum_{i=1}^{N} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

   - Scale-independent
   - Interpretable as percentage error

5. **Root Relative Squared Error (RSE)**:
   $$RSE = \frac{\sqrt{\sum(y - \hat{y})^2}}{\sqrt{\sum(y - \bar{y})^2}}$$

   - Relative to naive mean prediction
   - Values < 1 indicate improvement over baseline

6. **Correlation (CORR)**:
   - Measures linear relationship between predictions and ground truth
   - Values closer to 1 indicate better alignment

**Primary Metrics**: MSE and MAE are used as primary metrics for model selection and comparison.

### 5.2 Baseline Models

**Internal Baselines** (Ablation Study):

1. **DLinear (Original)**: Fixed kernel (k=25), no adaptive components
2. **DLinear-Adaptive**: Adaptive moving average, single scale
3. **DLinear-MultiScale**: Multi-scale decomposition, non-adaptive
4. **DLinear-Improved**: Full model with adaptive + multi-scale

**External Baselines** (for context, not necessarily implemented):

- Simple moving average
- ARIMA (classical statistical method)
- NLinear (no decomposition variant of DLinear)
- Transformer-based models (Informer, Autoformer)

### 5.3 Hyperparameters

**Model Configuration:**

```python
seq_len = 336          # Input sequence length
pred_len = 96          # Prediction horizon
label_len = 48         # Overlap for decoder
enc_in = 8             # Number of input features
individual = False     # Shared vs. individual linear layers
multi_scale = False/True   # Single vs. multi-scale decomposition
adaptive = True/False      # Fixed vs. adaptive moving average
```

**Training Configuration:**

```python
train_epochs = 10      # Maximum training epochs
batch_size = 8         # Samples per batch
learning_rate = 0.0005 # Adam optimizer learning rate
patience = 3           # Early stopping patience
loss = 'mse'           # Loss function
lradj = 'type1'        # Learning rate adjustment strategy
n_ensemble = 5         # Number of ensemble members
```

**Kernel Sizes:**

```python
# Single-scale: kernel_size = 25
# Multi-scale: kernel_sizes = [9, 25, 49]
```

### 5.4 Hardware/Software Requirements

**Hardware:**

- CPU: Any modern multi-core processor (CPU mode used by default)
- RAM: Minimum 8 GB (16 GB recommended)
- GPU: Optional (CUDA-compatible GPU for acceleration)
  - Currently running with `use_gpu=False`
  - Can enable GPU with `use_gpu=True, gpu=0`

**Software Environment:**

```
Python: 3.8+
PyTorch: 2.0.1
NumPy: 1.25.1
Pandas: 2.0.3
Scikit-learn: 1.7.2
Captum: 0.8.0 (for feature attribution)
Matplotlib: 3.7.2 (for visualization)
Seaborn: 0.13.2 (for advanced plotting)
```

**Development Tools:**

- Version Control: Git
- IDE: VS Code, PyCharm, or Jupyter Notebook
- Experiment Tracking: Manual logging to files/console

### 5.5 Training Procedure

**1. Data Loading:**

```python
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```

**2. Optimization:**

```python
optimizer = Adam(model.parameters(), lr=0.0005)
criterion = MSELoss()
```

**3. Training Loop:**

```python
For each epoch:
  For each batch in train_loader:
    1. Forward pass: pred = model(batch_x)
    2. Compute loss: loss = MSE(pred, batch_y)
    3. Backward pass: loss.backward()
    4. Update weights: optimizer.step()

  Validation:
    1. Compute validation loss
    2. Check early stopping criterion
    3. Save best model checkpoint
```

**4. Early Stopping:**

- Monitor validation loss
- Stop if no improvement for 3 consecutive epochs
- Restore best checkpoint for final evaluation

**5. Ensemble Training:**

- Train 5 models independently with different random seeds
- Save each model checkpoint separately
- Average predictions during inference

## 6. Implementation Plan

| Phase       | Tasks                                      | Duration | Deliverables                            | Status         |
| ----------- | ------------------------------------------ | -------- | --------------------------------------- | -------------- |
| **Phase 1** | Literature review, baseline implementation | 2 weeks  | Literature review doc, DLinear baseline | ✅ Complete    |
| **Phase 2** | Adaptive moving average implementation     | 1 week   | AdaptiveMovingAvg module                | ✅ Complete    |
| **Phase 3** | Multi-scale decomposition                  | 1 week   | MultiScaleDecomposition module          | ✅ Complete    |
| **Phase 4** | Feature attribution integration            | 1 week   | Attribution methods in model            | ✅ Complete    |
| **Phase 5** | Ensemble implementation                    | 1 week   | Ensemble training pipeline              | ✅ Complete    |
| **Phase 6** | Experiments and ablation studies           | 2 weeks  | Results for all configurations          | 🔄 In Progress |
| **Phase 7** | Analysis and visualization                 | 1 week   | Plots, tables, interpretation           | 📅 Planned     |
| **Phase 8** | Documentation and final report             | 1 week   | Complete research report                | 📅 Planned     |

**Total Duration**: ~10 weeks

### 6.1 Detailed Phase Breakdown

**Phase 6: Experiments (Current Focus)**

- [ ] Train baseline DLinear (non-adaptive, single-scale)
- [ ] Train DLinear with adaptive decomposition only
- [ ] Train DLinear with multi-scale decomposition only
- [ ] Train DLinear-Improved (adaptive + multi-scale)
- [ ] Train ensemble of 5 DLinear-Improved models
- [ ] Evaluate all models on test set
- [ ] Compute feature attributions
- [ ] Record training curves and convergence behavior

**Phase 7: Analysis**

- [ ] Compare quantitative results (MSE, MAE, RMSE, etc.)
- [ ] Statistical significance testing
- [ ] Visualize predictions vs. ground truth
- [ ] Analyze learned kernel weights (adaptive mode)
- [ ] Analyze scale weights (multi-scale mode)
- [ ] Visualize feature attribution results
- [ ] Identify failure cases and error patterns

**Phase 8: Documentation**

- [ ] Complete methodology documentation
- [ ] Write results section with tables and figures
- [ ] Discussion of findings and implications
- [ ] Limitations and future work
- [ ] Final report compilation

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk                                                   | Likelihood | Impact | Mitigation Strategy                                                       |
| ------------------------------------------------------ | ---------- | ------ | ------------------------------------------------------------------------- |
| **Adaptive kernels fail to learn meaningful patterns** | Medium     | High   | Compare with fixed kernels; add regularization; visualize learned weights |
| **Multi-scale decomposition overfits**                 | Medium     | Medium | Use validation set for early stopping; reduce model complexity if needed  |
| **Ensemble doesn't improve over single model**         | Low        | Low    | Expected behavior in some cases; document findings regardless             |
| **Feature attribution is uninformative**               | Medium     | Low    | Try multiple attribution methods; analyze different samples               |
| **Computational resources insufficient**               | Low        | Medium | Use CPU mode; reduce batch size; train one model at a time                |

### 7.2 Data Risks

| Risk                               | Likelihood | Impact   | Mitigation Strategy                                                 |
| ---------------------------------- | ---------- | -------- | ------------------------------------------------------------------- |
| **Data preprocessing errors**      | Low        | High     | Validate preprocessing pipeline; check data statistics before/after |
| **Train/val/test contamination**   | Low        | Critical | Strict temporal split; automated validation of splits               |
| **Non-stationarity affects model** | Medium     | Medium   | Document as limitation; consider differencing or detrending         |

### 7.3 Methodological Risks

| Risk                              | Likelihood | Impact   | Mitigation Strategy                                              |
| --------------------------------- | ---------- | -------- | ---------------------------------------------------------------- |
| **Insufficient ablation studies** | Low        | Medium   | Systematically test each component independently                 |
| **Unfair baseline comparison**    | Low        | High     | Use identical data, preprocessing, and evaluation for all models |
| **Overfitting to validation set** | Medium     | High     | Use early stopping conservatively; report test set performance   |
| **Cherry-picking results**        | Low        | Critical | Report all experiments; document failures and successes equally  |

### 7.4 Timeline Risks

| Risk                                          | Likelihood | Impact | Mitigation Strategy                                       |
| --------------------------------------------- | ---------- | ------ | --------------------------------------------------------- |
| **Implementation takes longer than expected** | Medium     | Medium | Prioritize core features; defer nice-to-have enhancements |
| **Debugging consumes excessive time**         | Medium     | High   | Write unit tests; validate components incrementally       |
| **Experiments require more iterations**       | High       | Medium | Start experiments early; parallelize where possible       |

## 8. Expected Outcomes

### 8.1 Primary Outcomes

1. **Improved Forecasting Accuracy**:

   - Target: 5-15% reduction in MSE/MAE compared to baseline DLinear
   - Specifically expect improvements on test set performance
   - Better handling of both short-term fluctuations and long-term trends

2. **Adaptive Component Validation**:

   - Demonstration that adaptive moving averages outperform fixed kernels
   - Visualization of learned kernel weights showing data-specific patterns
   - Evidence that adaptation helps across different data characteristics

3. **Multi-Scale Insights**:

   - Analysis of how scale weights distribute across temporal resolutions
   - Understanding which scales are most important for this dataset
   - Potential discovery of dataset-specific temporal hierarchies

4. **Interpretability Enhancement**:

   - Feature importance rankings from attribution methods
   - Identification of key predictive features and time windows
   - Validation that attributions align with domain knowledge

5. **Ensemble Benefits**:
   - Quantification of variance reduction through ensembling
   - Comparison of ensemble vs. single model performance
   - Confidence intervals for predictions

### 8.2 Research Contributions

**Methodological Contributions**:

- Novel combination of adaptive decomposition with multi-scale analysis
- Integration of interpretability into simple linear forecasting models
- Systematic ablation study identifying valuable components

**Practical Contributions**:

- Improved forecasting tool for exchange rate prediction
- Reusable code for adaptive time series decomposition
- Guidelines for when to use adaptive vs. fixed decomposition

**Scientific Contributions**:

- Evidence supporting or refuting the value of adaptive components
- Insights into multi-scale temporal patterns in financial time series
- Contribution to the ongoing discussion about model complexity vs. performance

### 8.3 Deliverables

**Code Artifacts**:

- ✅ Modular, well-documented Python implementation
- ✅ Reusable components (AdaptiveMovingAvg, MultiScaleDecomposition)
- ✅ Training and evaluation scripts
- 📅 Jupyter notebooks with analysis and visualizations

**Documentation**:

- ✅ Literature review
- ✅ Methodology document (this document)
- 📅 Results report with tables and figures
- 📅 Final research report

**Experimental Results**:

- 📅 Quantitative comparison tables
- 📅 Training curves and convergence plots
- 📅 Prediction visualizations
- 📅 Feature attribution visualizations
- 📅 Learned parameter analysis (kernel weights, scale weights)

**Research Report**:

- 📅 Complete manuscript following academic format
- 📅 Abstract, introduction, related work, methodology, results, discussion, conclusion
- 📅 Ready for submission or presentation

### 8.4 Success Criteria

The project will be considered successful if:

1. **Implementation Complete**: All proposed components are implemented and functional
2. **Fair Evaluation**: Rigorous ablation studies isolate each component's contribution
3. **Interpretable Results**: Attribution methods provide actionable insights
4. **Documented Learning**: Even if improvements are modest, understanding of what works and why is achieved
5. **Reproducible**: Code and documentation allow others to replicate experiments

**Note**: Even if adaptive/multi-scale components don't significantly outperform the baseline, documenting these findings contributes valuable knowledge about what doesn't work and why.

## 9. Evaluation Protocol

### 9.1 Cross-Validation Strategy

Due to temporal dependencies in time series data:

- **No k-fold cross-validation** (would violate temporal ordering)
- **Single train/val/test split** with temporal ordering preserved
- **Walk-forward validation** (optional extension): Retrain on expanding window

### 9.2 Statistical Testing

To ensure results are not due to random variation:

**1. Multiple Runs**:

- Train 5 ensemble members with different random seeds
- Report mean and standard deviation of metrics

**2. Significance Tests** (if comparing models):

- Paired t-test on test set predictions
- Diebold-Mariano test for forecast comparison

**3. Confidence Intervals**:

- Bootstrap confidence intervals for metric estimates

### 9.3 Visualization and Analysis

**Performance Visualizations**:

- Line plots: Predictions vs. ground truth
- Error distributions: Histogram of prediction errors
- Temporal error analysis: Error over time
- Training curves: Loss vs. epoch for train/validation

**Model Analysis**:

- Kernel weight visualization: Heatmap of learned weights
- Scale weight evolution: How weights change during training
- Feature attribution plots: Bar charts of feature importance
- Attention maps: Time step importance (if applicable)

**Comparative Analysis**:

- Metric comparison tables: All models, all metrics
- Performance by horizon: Accuracy at different prediction steps (1-96)
- Ablation study results: Component-wise contribution

## 10. Reproducibility

### 10.1 Random Seed Control

All random processes are seeded for reproducibility:

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

### 10.2 Environment Documentation

- Dependencies listed in `requirements.txt`
- Python version specified (3.8+)
- PyTorch version specified (2.0.1)

### 10.3 Code Organization

```
src/
  ├── config.py           # All hyperparameters
  ├── main.py             # Entry point
  ├── train.py            # Training logic
  ├── predict.py          # Inference logic
  ├── data_loader.py      # Data preprocessing
  ├── models/
  │   └── DLinear.py      # Model implementation
  └── utils/
      ├── metrics.py      # Evaluation metrics
      ├── tools.py        # Helper functions
      └── timefeatures.py # Time feature extraction
```

### 10.4 Checkpoint Management

Models saved at:

- `src/checkpoints/{model_id}_ensemble_{i}/checkpoint.pth`
- Contains model state dict, optimizer state, training metrics

---

## Appendix A: Configuration Reference

Complete configuration for reproducing main experiments:

```python
# Main Experiment: DLinear-Improved with Adaptive + Multi-Scale
config = {
    'model': 'DLinear',
    'seq_len': 336,
    'pred_len': 96,
    'enc_in': 8,
    'individual': False,
    'adaptive': True,
    'multi_scale': True,
    'kernel_sizes': [9, 25, 49],
    'n_ensemble': 5,
    'train_epochs': 10,
    'batch_size': 8,
    'learning_rate': 0.0005,
    'patience': 3,
    'data_path': 'exchange_rate.csv',
    'features': 'M',
}
```

## Appendix B: Metric Formulas Reference

See Section 5.1 for detailed metric definitions.

---

**Note:** This methodology document should be updated as experiments progress and insights emerge. Document any deviations from the planned methodology and justify the reasons.
