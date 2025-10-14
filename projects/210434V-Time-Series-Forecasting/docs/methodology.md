# Methodology: Enhanced TS2Vec with Hybrid Ensemble Architecture for Time Series Forecasting

**Student:** 210434V  
**Research Area:** Time Series Forecasting  
**Date:** 2025-10-13

## 1. Overview

This research presents an enhanced version of TS2Vec that significantly improves forecasting accuracy through a novel hybrid ensemble architecture. Our approach combines the strengths of representation learning with explicit temporal feature engineering and adaptive ensemble weighting. The methodology addresses limitations in existing time series forecasting models by integrating learned representations with domain knowledge through intelligent ensemble strategies.

**Key Innovation:** Dual-model ensemble that combines pure TS2Vec embeddings with TS2Vec+temporal features, optimized through validation-based weight selection for each dataset and prediction horizon.

## 2. Research Design

### 2.1 Research Approach
- **Type:** Empirical research with algorithmic innovation
- **Methodology:** Comparative experimental analysis
- **Baseline:** Original TS2Vec model
- **Enhancement Strategy:** Hybrid ensemble with adaptive weighting

### 2.2 Research Questions
1. Can explicit temporal features complement learned TS2Vec representations?
2. How should ensemble weights be optimized for different forecasting horizons?
3. What is the optimal balance between learned and engineered features across datasets?

### 2.3 Hypothesis
*"A hybrid ensemble combining TS2Vec representations with explicit temporal features, using validation-optimized weights, will consistently outperform the baseline TS2Vec model across multiple forecasting horizons and datasets."*

## 3. Data Collection

### 3.1 Data Sources
- **ETT Datasets:** Energy Transformer Temperature data (ETTh1, ETTh2, ETTm1)
  - Source: Kaggle ETT Small dataset collection (`/kaggle/input/ettsmall/`)
  - Temporal Resolution: Hourly (ETTh) and 15-minute (ETTm)
  - Experimental Environment: Kaggle Notebooks with GPU acceleration


### 3.2 Data Description

| Dataset | Variables | Length | Frequency | Domain | Experiments Conducted |
|---------|-----------|---------|-----------|---------|----------------------|
| ETTh1   | 7 (6+target) | 17,420 | Hourly | Energy | Univariate + Multivariate |
| ETTh2   | 7 (6+target) | 17,420 | Hourly | Energy | Univariate + Multivariate |
| ETTm1   | 7 (6+target) | 69,680 | 15-min | Energy | Univariate + Multivariate |

**Prediction Horizons Tested:**
- All ETT datasets: {24, 48, 168, 336, 720} timesteps
- Covers short-term (1-2 days) to long-term (30 days) forecasting
- **Experimental Design:** Both univariate and multivariate forecasting evaluated for each dataset

### 3.3 Data Preprocessing

#### 3.3.1 Normalization Strategy
```python
# StandardScaler applied per dataset
scaler = StandardScaler().fit(data[train_slice])
normalized_data = scaler.transform(data)
```

#### 3.3.2 Train/Validation/Test Splits
- **Training:** 60% (for TS2Vec pre-training and Ridge training)
- **Validation:** 20% (for hyperparameter tuning and ensemble weight optimization)
- **Test:** 20% (for final evaluation only)

#### 3.3.3 Feature Engineering
- **Target Selection:** Remove covariate columns, focus on time series variables
- **Sequence Processing:** Maximum training length of 3000 timesteps
- **Missing Values:** Handled by TS2Vec architecture (no explicit imputation)

## 4. Model Architecture

### 4.1 Enhanced TS2Vec Architecture

```
Input Time Series X ∈ ℝ^(N×T×D)
       ↓
   TS2Vec Encoder (Pre-trained, Frozen)
   • Hidden dims: 64
   • Depth: 10 residual blocks  
   • Output dims: 320
       ↓
   Representations Z ∈ ℝ^(N×T×320)
       ↓
    ┌─────────────────────────────┐
    ↓                             ↓
Model A: Pure TS2Vec          Model B: TS2Vec + Time Features
Features: Z ∈ ℝ^320           Features: [Z||T] ∈ ℝ^322
    ↓                             ↓
Ridge Regression              Ridge Regression
(α optimized)                 (α optimized)
    ↓                             ↓
Predictions Ŷ_A               Predictions Ŷ_B
    ↓                             ↓
    └─────────────────────────────┘
                   ↓
         Adaptive Ensemble
         Ŷ = w₁Ŷ_A + w₂Ŷ_B
         (weights optimized per horizon)
                   ↓
           Final Predictions
```

### 4.2 Key Components

#### 4.2.1 TS2Vec Base Model
- **Architecture:** Dilated convolutions with hierarchical contrastive learning
- **Hyperparameters:**
  - Learning rate: 0.001
  - Batch size: 8
  - Representation dimensions: 320
  - Max training length: 3000

#### 4.2.2 Temporal Feature Engineering
```python
# Daily cycle encoding (24-hour period)
T(t) = [sin(2πt/24), cos(2πt/24)]
```
- **Rationale:** Captures diurnal patterns in energy and consumption data
- **Dimension:** 2 features per timestep
- **Mathematical Properties:** Periodic, bounded [-1,1], differentiable

#### 4.2.3 Ridge Regression Protocol
- **Regularization Search:** α ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
- **Selection Criterion:** Minimum validation RMSE + MAE
- **Efficiency:** Subsample to 100K samples if dataset larger

#### 4.2.4 Ensemble Weight Optimization
```python
# Weight candidates (17 combinations)
W_candidates = {[0.9,0.1], [0.85,0.15], ..., [0.1,0.9]}

# Optimization objective
w* = argmin_w √MSE(w₁Ŷ_A + w₂Ŷ_B, Y_val) + MAE(w₁Ŷ_A + w₂Ŷ_B, Y_val)
```

## 5. Experimental Setup

### 5.1 Evaluation Metrics

#### 5.1.1 Primary Metrics
- **MSE (Mean Squared Error):** Emphasizes large prediction errors
- **MAE (Mean Absolute Error):** Robust to outliers

#### 5.1.2 Evaluation Scales
- **Normalized:** On standardized data (for model comparison)
- **Raw Scale:** Inverse-transformed (for practical interpretation)

#### 5.1.3 Statistical Analysis
```python
# Experimental Configuration (as implemented)
- Fixed random seed: 42 (for reproducibility)
- Evaluation approach: Single-run deterministic results
- Command structure: python -u train.py [dataset] [task] --loader [type] --repr-dims 320 --max-threads 8 --seed 42 --eval
- Tasks evaluated: Both univariate (forecast_univar) and multivariate (forecast_multivar) forecasting
```

### 5.2 Baseline Models

#### 5.2.1 Primary Baseline
- **TS2Vec (Original):** Pure learned representations with Ridge regression
- **Implementation:** Identical hyperparameters, same evaluation protocol

#### 5.2.2 Ablation Studies
1. **Time Features Only:** Ridge regression on temporal features alone
2. **Fixed Weights:** Ensemble with predetermined weights [0.8, 0.2]
3. **Equal Weights:** Simple average ensemble [0.5, 0.5]

#### 5.2.3 Comparative Baselines (Literature)
- **Linear Models:** ARIMA, Linear Regression
- **Deep Learning:** LSTM, GRU, Transformer variants
- **Recent SOTA:** Informer, Autoformer, PatchTST (where available)

### 5.3 Hardware/Software Requirements

#### 5.3.1 Computational Environment
- **Platform:** Kaggle Notebooks with GPU acceleration
- **GPU:** NVIDIA Tesla P100 or T4 (provided by Kaggle)
- **Memory:** 13GB RAM (Kaggle standard allocation)
- **Storage:** 20GB working directory + 5GB temp storage

#### 5.3.2 Software Stack
```python
# Dependencies installed via pip in Kaggle environment
bottleneck                   # Fast NumPy array functions
statsmodels                  # Statistical modeling
scikit-learn                 # Ridge regression, metrics
torch                        # Deep learning framework (pre-installed)
numpy                        # Numerical computation (pre-installed)
pandas                       # Data manipulation (pre-installed)
```

#### 5.3.3 Experimental Commands Executed
```bash
# Univariate Forecasting
python -u train.py ETTm1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ETTh1 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ETTh2 forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval

# Multivariate Forecasting  
python -u train.py ETTm1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ETTh1 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval
python -u train.py ETTh2 forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval
```

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| **Phase 1** | Literature review & baseline implementation | 2 weeks | Working TS2Vec baseline |
| **Phase 2** | Ensemble architecture development | 3 weeks | Hybrid ensemble model |
| **Phase 3** | Experimental evaluation | 3 weeks | Results on all datasets |
| **Phase 4** | Statistical analysis & ablation studies | 2 weeks | Complete evaluation |
| **Phase 5** | Documentation & paper writing | 2 weeks | Research paper |

### 6.1 Milestone Details

#### Phase 1: Foundation (Weeks 1-2)
- [ ] Reproduce original TS2Vec results
- [ ] Implement data loading pipeline
- [ ] Validate baseline performance

#### Phase 2: Development (Weeks 3-5)
- [ ] Implement temporal feature engineering
- [ ] Develop dual Ridge regression training
- [ ] Create ensemble weight optimization
- [ ] Add evaluation protocols

#### Phase 3: Experimentation (Weeks 6-8)
- [ ] Run experiments on all datasets
- [ ] Collect performance metrics
- [ ] Generate horizon-wise analysis

#### Phase 4: Analysis (Weeks 9-10)
- [ ] Statistical significance testing
- [ ] Ablation studies
- [ ] Failure case analysis
- [ ] Weight visualization

#### Phase 5: Documentation (Weeks 11-12)
- [ ] Write research paper
- [ ] Create reproducibility documentation
- [ ] Prepare code submission

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| Memory overflow on large datasets | Medium | High | Batch processing, data subsampling |
| Ensemble doesn't improve performance | Low | Medium | Fallback to stronger single models |
| Hyperparameter sensitivity | Medium | Medium | Extensive grid search, robust defaults |
| Reproducibility issues | Low | High | Fixed seeds, detailed documentation |

### 7.2 Methodological Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| Overfitting to validation set | Medium | High | Separate test set, statistical testing |
| Dataset-specific improvements | Medium | Medium | Multi-dataset evaluation |
| Baseline implementation errors | Low | High | Code review, literature verification |

### 7.3 Timeline Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| Computational delays | Medium | Medium | Parallel experiments, cloud resources |
| Implementation complexity | Low | Medium | Incremental development, testing |
| Results analysis complexity | Low | Low | Automated analysis scripts |

## 8. Expected Outcomes

### 8.1 Performance Improvements

#### 8.1.1 Quantitative Targets
- **Short-term forecasting (H≤48):** 5-10% MSE reduction
- **Medium-term forecasting (H=168):** 8-15% MSE reduction  
- **Long-term forecasting (H≥336):** 10-20% MSE reduction

#### 8.1.2 Consistency Goals
- **ETT Dataset Coverage:** Improvements across ETTh1, ETTh2, and ETTm1
- **Forecasting Task Coverage:** Consistent performance in both univariate and multivariate settings
- **Horizon Robustness:** Effective performance across all tested prediction horizons (24-720 timesteps)
- **Practical Relevance:** Raw-scale improvements meaningful for energy system applications

### 8.2 Scientific Contributions

#### 8.2.1 Algorithmic Innovation
1. **Hybrid Ensemble Architecture:** Novel combination of learned and engineered features
2. **Adaptive Weight Optimization:** Validation-based ensemble tuning per horizon
3. **Temporal Feature Integration:** Strategic use of cyclical patterns in TS2Vec context

#### 8.2.2 Empirical Insights
1. **Feature Complementarity:** Quantify benefits of explicit vs. learned temporal features in ETT datasets
2. **Horizon-Specific Patterns:** Characterize optimal ensemble strategies by forecast length for energy data
3. **Univariate vs. Multivariate:** Compare ensemble effectiveness between single-variable and multi-variable forecasting
4. **Energy Domain Patterns:** Identify temporal patterns specific to energy transformer data

#### 8.2.3 Reproducibility Contribution
1. **Complete Technical Specification:** All hyperparameters and implementation details documented
2. **Kaggle Notebook Implementation:** Fully reproducible experimental environment
3. **Deterministic Results:** Fixed random seed (42) ensures consistent reproduction

### 8.3 Potential Impact

#### 8.3.1 Academic Impact
- **Method Advancement:** Push state-of-the-art in representation learning for time series forecasting
- **Ensemble Innovation:** Demonstrate effective hybrid learning strategies for energy data
- **ETT Benchmark Enhancement:** Contribute improved baselines for standard ETT dataset evaluation

#### 8.3.2 Practical Applications
- **Energy Transformer Monitoring:** Improved temperature prediction for power grid equipment
- **Smart Grid Operations:** Better energy demand and supply forecasting
- **Renewable Energy Integration:** Enhanced prediction for variable energy sources
- **Industrial Energy Management:** Optimized energy consumption planning

---

## 10. Experimental Workflow (As Implemented)

### 10.1 Environment Setup
```python
# Kaggle Environment Configuration
1. Clone repository: git clone https://github.com/Niroshan2001/ts2vec.git
2. Switch to enhanced branch: git checkout optimized-ensemble
3. Install dependencies: pip install bottleneck statsmodels scikit-learn
4. Copy ETT datasets from Kaggle input directory to datasets/ folder
```

### 10.2 Experiment Execution Order
```python
# Phase 1: Univariate Forecasting
- ETTm1 univariate: train.py ETTm1 forecast_univar --loader forecast_csv_univar
- ETTh1 univariate: train.py ETTh1 forecast_univar --loader forecast_csv_univar  
- ETTh2 univariate: train.py ETTh2 forecast_univar --loader forecast_csv_univar

# Phase 2: Multivariate Forecasting
- ETTm1 multivariate: train.py ETTm1 forecast_multivar --loader forecast_csv
- ETTh1 multivariate: train.py ETTh1 forecast_multivar --loader forecast_csv
- ETTh2 multivariate: train.py ETTh2 forecast_multivar --loader forecast_csv
```

### 10.3 Data Processing Pipeline
```python
# Automated data handling in notebook
1. Copy CSV files from /kaggle/input/ettsmall/ to datasets/
2. Automatic train/validation/test splitting by datautils.load_forecast_csv()
3. StandardScaler normalization applied per dataset
4. Both univariate and multivariate target preparation
```

### 10.4 Results Collection
- **Training Time:** Automatically logged during TS2Vec training phase
- **Evaluation Time:** Measured during Ridge regression and ensemble evaluation
- **Performance Metrics:** MSE and MAE reported for each prediction horizon
- **Model Outputs:** Saved to training/{dataset}__{timestamp}/ directories

---

**Note:** This methodology document reflects the actual implementation and experimental design of the Enhanced TS2Vec research conducted on ETT datasets in Kaggle environment.

**Implementation Status:** ✅ Complete  
**Experimental Status:** ✅ Complete (All ETT datasets evaluated)  
**Documentation Status:** ✅ Complete
