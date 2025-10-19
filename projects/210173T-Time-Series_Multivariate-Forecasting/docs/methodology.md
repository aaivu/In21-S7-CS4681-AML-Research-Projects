# Methodology: Time Series:Multivariate Forecasting

**Student:** 210173T
**Research Area:** Time Series:Multivariate Forecasting
**Date:** 2025-09-01

## 1. Overview

This research introduces a Selective Multi-Scale PatchTST (MS-PatchTST) model to enhance long-term multivariate time series forecasting. The methodology integrates architectural innovation, systematic experimentation, and comparative benchmarking.

The approach builds upon the Patch Time Series Transformer (PatchTST) by incorporating multi-scale temporal pattern learning, enabling simultaneous extraction of short-, medium-, and long-term dependencies. The study involves dataset preparation, model design, implementation, and rigorous evaluation across multiple benchmarks (Weather, Electricity, and National Illness datasets).


## 2. Research Design

The research follows a quantitative, experimental design structured in three stages:

1. **Baseline Verification:** Reproduce the original PatchTST results to establish consistent evaluation conditions.
2. **Model Enhancement:** Develop a multi-scale Transformer architecture that processes input sequences using multiple patch sizes in parallel, followed by a learnable fusion layer.
3. **Evaluation & Comparison:** Conduct experiments using standardized datasets and metrics, followed by ablation studies to analyze the contribution of each scale.

This design ensures both empirical validity and reproducibility, enabling objective performance comparisons against existing state-of-the-art models.


## 3. Data Collection

### 3.1 Data Sources

Publicly available benchmark datasets are used to ensure comparability with prior work:
- **Weather Dataset:** 21 meteorological variables (temperature, humidity, pressure).
- **Electricity Dataset:** Hourly consumption records of 321 clients.
- **National Illness (ILI) Dataset:** Weekly influenza-like illness rates across multiple regions.

All datasets are publicly accessible through standard repositories and prior research benchmarks (e.g., PatchTST GitHub and UCI repositories).

### 3.2 Data Description

Each dataset represents multivariate temporal sequences with distinct periodicities:

| Dataset     | Variables | Frequency        | Forecast Horizon (T) | Look-back Window (L) |
|-------------|-----------|------------------|----------------------|----------------------|
| Weather     | 21        | 10-min intervals | 96                   | 336                  |
| Electricity | 321       | Hourly           | 96                   | 336                  |
| ILI         | 7         | Weekly           | 24                   | 104                  |

This diversity allows evaluation of both short-term seasonal and long-term trend learning capabilities.

### 3.3 Data Preprocessing

- **Normalization:** Apply instance normalization (zero mean, unit variance) to each channel to mitigate distributional shifts between training and testing.
- **Patching:** Convert continuous sequences into overlapping subseries (patches) using variable patch lengths (P) and strides (S).
- **Splitting:** Divide each dataset into training (70%), validation (10%), and testing (20%) partitions.
- **Feature Engineering:** Convert timestamps into sinusoidal positional embeddings when required for temporal order preservation.


## 4. Model Architecture

The proposed MS-PatchTST extends the PatchTST by integrating multi-scale feature extraction and learnable fusion:

1. **Parallel Multi-Scale Backbones:**
   - Three Transformer backbones, each processing patches of different lengths (e.g., 8, 16, 32), capture fine-grained, medium, and coarse temporal dynamics.
   - Each backbone uses independent patching, embedding, and self-attention mechanisms to prevent scale interference.

2. **Fusion Layer:**
   - Outputs from each backbone are concatenated along the feature axis and passed through a fully connected fusion layer.
   - This layer learns optimal weighting for each scale via backpropagation, enabling dynamic prioritization of relevant temporal features.

3. **Forecasting Head:**
   - Flattened multi-scale features are linearly mapped to the forecast horizon T.
   - The final prediction represents a scale-adaptive forecast capturing both short-term fluctuations and long-term trends.

**Architectural Summary:**

```
Input Sequence → Multi-scale Patching → Parallel Transformers → Fusion Layer → Forecast Output
```


## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Mean Squared Error (MSE):** Measures overall forecasting accuracy.
- **Mean Absolute Error (MAE):** Captures deviation robustness.
- **Relative Squared Error (RSE):** Compares error against data variance.

Lower values for all metrics indicate superior performance.

### 5.2 Baseline Models

Comparisons will be made against the following models:
- **PatchTST** (Nie et al., 2023): Base model with fixed patch size.
- **Informer** (Zhou et al., 2021): Employs ProbSparse attention for efficiency.
- **FEDformer** (Zhou et al., 2022): Frequency-domain linear complexity model.
- **Autoformer** (Wu et al., 2021): Introduces decomposition and auto-correlation.

These baselines represent major paradigms in time series Transformer design—data representation, attention efficiency, and frequency modeling.

### 5.3 Hardware/Software Requirements

- **Framework:** PyTorch (v2.1+), Python 3.10
- **Libraries:** NumPy, Pandas, scikit-learn, Matplotlib, tqdm
- **Hardware:** NVIDIA GPU (≥ 8 GB VRAM), 16 GB RAM
- **Environment:** Ubuntu 22.04 / Google Colab / Kaggle GPU instances
- **Version Control:** GitHub repository (branch-based for milestones)


## 6. Implementation Plan

| Phase   | Tasks                                                | Duration | Deliverables                            |
|---------|------------------------------------------------------|----------|-----------------------------------------|
| Phase 1 | Data preprocessing and normalization                 | 2 weeks  | Clean, standardized datasets            |
| Phase 2 | Implement PatchTST and develop multi-scale module    | 3 weeks  | Verified single & multi-scale models    |
| Phase 3 | Conduct experiments across datasets                  | 2 weeks  | Evaluation metrics and visualizations   |
| Phase 4 | Analyze performance, ablation studies, documentation | 1 week   | Final report, charts, and model code    |


## 7. Risk Analysis

| Risk                          | Impact                  | Mitigation Strategy                                      |
|-------------------------------|-------------------------|----------------------------------------------------------|
| High computational cost       | Delays in training      | Optimize batch sizes, use mixed-precision training       |
| Dataset imbalance or noise    | Reduced accuracy        | Apply normalization and outlier filtering                |
| Overfitting on small datasets | Poor generalization     | Use dropout, early stopping, and cross-validation        |
| Fusion layer instability      | Training divergence     | Initialize with small weights, apply gradient clipping   |

The project will also maintain a reproducibility checklist to ensure consistency across experiments.


## 8. Expected Outcomes

- A validated multi-scale Transformer architecture (MS-PatchTST) capable of handling variable temporal patterns.
- Demonstrated improvement of 5–15% in MSE over the baseline PatchTST across most datasets.
- A public open-source implementation supporting configurable patch scales and fusion modes.
- Comprehensive evaluation insights on scale selection, interpretability, and trade-offs between accuracy and efficiency.

The outcomes aim to advance long-horizon forecasting by providing a scalable, interpretable, and empirically validated Transformer framework.
