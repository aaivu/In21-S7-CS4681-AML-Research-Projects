# Methodology: Time Series Univariate Forecasting

**Student:** 210172N
**Research Area:** Time Series Univariate Forecasting
**Date:** 2025-10-05

## 1. Overview

This methodology outlines the systematic approach for adapting the PatchTST (Patch Time Series Transformer) model for real-time, multi-horizon forecasting on the M4 Competition benchmark. The research bridges the gap between modern Long-Term Series Forecasting (LTSF) models and classical univariate forecasting benchmarks while optimizing for practical deployment through quantization and ONNX runtime optimization.

### Research Objectives

1.  **Baseline Establishment:** Implement and evaluate vanilla PatchTST on M4 benchmark
2.  **Structural Adaptation:** Optimize model architecture for M4's diverse series characteristics
3.  **Real-Time Enhancement:** Apply quantization and ONNX conversion for low-latency inference
4.  **Rigorous Benchmarking:** Compare accuracy (OWA) vs. latency tradeoffs against N-BEATS baseline

## 2. Research Design

### 2.1 Overall Approach

The methodology follows a three-stage pipeline:

**Stage 1: Baseline Implementation & Evaluation**

- Implement PatchTST architecture for univariate forecasting
- Establish baseline performance on M4 monthly subset (48,000 series)
- Identify architectural bottlenecks and adaptation opportunities

**Stage 2: Structural Optimization**

- Systematic hyperparameter tuning (look-back windows, patch lengths)
- Model simplification experiments (layer reduction, hidden dimension optimization)
- Adaptive patching strategies for variable-length series

**Stage 3: Deployment Optimization**

- Post-training quantization (FP32 → INT8)
- ONNX conversion and runtime optimization
- Latency-accuracy tradeoff analysis

### 2.2 Experimental Framework

All experiments will use:

- **Hardware:** Google Colab T4 GPU (16GB VRAM) for training, CPU for inference benchmarking
- **Framework:** PyTorch 2.0+ with ONNX Runtime for deployment
- **Version Control:** Git with regular commits and experiment tracking
- **Reproducibility:** Fixed random seeds (42), documented hyperparameters

## 3. Data Collection

### 3.1 Data Sources

#### Primary Dataset: M4 Competition

**Source:** https://github.com/Mcompetitions/M4-methods

- **Total Series:** 100,000 univariate time series
- **Frequencies:** Yearly (23,000), Quarterly (24,000), Monthly (48,000), Weekly (359), Daily (4,227), Hourly (414)
- **Format:** CSV files with train/test splits predefined
- **Evaluation:** Official M4 metrics (sMAPE, MASE, OWA)

**Frequency Distribution:**

| Frequency   | Count      | Min Length | Forecast Horizon | Focus        |
| ----------- | ---------- | ---------- | ---------------- | ------------ |
| Yearly      | 23,000     | 13         | 6                | Optional     |
| Quarterly   | 24,000     | 16         | 8                | Secondary    |
| **Monthly** | **48,000** | **42**     | **18**           | **Primary**  |
| Weekly      | 359        | 80         | 13               | Out of scope |
| Daily       | 4,227      | 93         | 14               | Out of scope |
| Hourly      | 414        | 700        | 48               | Out of scope |

**Rationale for Monthly Focus:**

- Largest subset (48,000 series) provides strongest training signal
- Reasonable series length (typically 100-200 points) suitable for patching
- 18-step forecast horizon allows meaningful evaluation
- Feasible within 12-week timeline constraints

#### Secondary Datasets: PatchTST Benchmarks

**Source:** https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy (Autoformer collection)

These datasets enable comparison with published PatchTST results:

1.  **ETTh1 & ETTh2** (Electricity Transformer Temperature - Hourly)
2.  **ETTm1 & ETTm2** (Electricity Transformer Temperature - 15min)
3.  **Weather Dataset**

**Usage:** These datasets will be used for:

- Initial model validation before M4 experiments
- Architecture debugging and hyperparameter search
- Transfer learning experiments (optional, if time permits)

### 3.2 Data Description

#### M4 Monthly Series Characteristics

**Typical Series Structure:**

- **Length:** 42-300 observations (median ~120 months, ~10 years)
- **Domains:** Economics, finance, demographics, industry, micro-level data
- **Patterns:** Trend, seasonality, noise, structural breaks
- **Challenge:** High diversity in patterns and scales

**Train/Test Split:**

- **Training:** Historical data excluding last 18 months
- **Testing:** Last 18 months (forecast horizon = 18)
- **Validation:** Last 18 months of training data (for hyperparameter tuning)

### 3.3 Data Preprocessing

#### Standard Preprocessing Pipeline

1.  **Instance Normalization (RevIN)**
2.  **Patching Strategy**
3.  **Look-back Window Selection**
4.  **Data Augmentation (Training Only)**
5.  **Batch Formation**

## 4. Model Architecture

### 4.1 Baseline PatchTST Architecture

**Core Components:**

1.  **Patching Module**
2.  **Patch Embedding**
3.  **Transformer Encoder**
4.  **Forecasting Head**
5.  **Channel Independence**

**Default Hyperparameters (from PatchTST paper):**

- `d_model = 128`
- `n_layers = 3`
- `n_heads = 16`
- `d_ff = 256`
- `patch_length = 16`
- `stride = 8`
- `dropout = 0.2`
- `activation = GELU`

### 4.2 M4-Specific Adaptations

**Adaptation 1: Simplified Architecture**

_Hypothesis:_ M4's shorter series and simpler patterns may not require deep architectures

**Experimental Configurations:**

- **Tiny:** $n_{layers}=2, d_{model}=64, n_{heads}=8$ (~250K parameters)
- **Small:** $n_{layers}=3, d_{model}=128, n_{heads}=16$ (~1M parameters)
- **Medium:** $n_{layers}=4, d_{model}=256, n_{heads}=16$ (~4M parameters)
- **Default:** $n_{layers}=3, d_{model}=128, n_{heads}=16$ (baseline)

**Evaluation:** Compare OWA scores versus model size and training time.

**Adaptation 2: Adaptive Patching**

_Rationale:_ This strategy aims to maintain a consistent number of patches across M4's variable-length series, stabilizing the input dimension for the Transformer encoder.

_Implementation:_ The patching parameters (patch length $P$ and stride $S$) are determined dynamically based on the input series length $L$. A target number of patches is established. If $L$ is short ($<40$), $P$ is calculated as $L/10$ (clamped at a minimum of 2). For longer series, $P$ is computed to achieve the target number of patches, then clamped between 4 and 24. The stride $S$ is consistently set as the floor of $P/2$.

**Adaptation 3: Look-back Window Strategy**

_Experiments:_ The optimal look-back window ($\text{seq\_len}$) for training PatchTST is determined by experimenting with:

1.  **Fixed Window:** Using a constant look-back (e.g., 54 months).
2.  **Proportional Window:** Setting $\text{seq\_len}$ as a multiple ($\alpha$) of the forecast horizon ($H=18$), where $\alpha \in \{2, 3, 4\}$.
3.  **Data-Constrained Window:** Choosing $\text{seq\_len}$ as the minimum of the desired length and the total available data minus the forecast horizon.

**Grid Search:** The specific look-back values tested for Monthly data are $\{36, 54, 72, 96, 120\}$.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

#### M4 Competition Metrics (Primary)

1.  **sMAPE (Symmetric Mean Absolute Percentage Error):** The average symmetric percentage error.
2.  **MASE (Mean Absolute Scaled Error):** The mean absolute error scaled by the in-sample seasonal naive error.
3.  **OWA (Overall Weighted Average):** The primary M4 ranking metric, calculated as the equally weighted average of relative sMAPE and relative MASE against the Naive2 benchmark. This is the main indicator of forecasting accuracy.

#### Real-Time Performance Metrics (Secondary)

1.  **Inference Latency**
    - **Measurement:** Average wall-clock time (ms) for a single forecast on a batch size of 1.
    - **Protocol:** A 10-iteration warm-up phase is performed, followed by averaging the time over 100 subsequent inference calls.
    - **Hardware:** Standardized CPU environment (e.g., Google Colab CPU runtime), as typical production deployment often relies on CPU-only inference for latency-critical tasks.
2.  **Model Size**
    - **Measurement:** On-disk storage size (MB) of the final model file (.onnx).

### 5.2 Baseline Models

**Primary Baseline: N-BEATS**

- **Source:** Published results from the official N-BEATS paper on the M4 dataset.
- **Rationale:** This provides the state-of-the-art deep learning benchmark on the M4 competition against which our model's accuracy (OWA) is compared.

**Secondary Baselines:**

- **Naive2:** The seasonal naive forecast (M4 reference benchmark). Achieving OWA $< 1.0$ is the minimum acceptable performance.
- **PyTorch PatchTST:** The unoptimized, FP32 implementation serves as the direct baseline for measuring the acceleration and compression gains.

### 5.3 Hardware/Software Requirements

**Execution Environment:**

- **Platform:** Google Colab (Pro recommended for stable sessions).
- **GPU:** NVIDIA T4 (16GB VRAM) for training.
- **CPU:** Standard Google Colab CPU for inference benchmarking.
- **Storage:** 50GB minimum.

**Software Stack:**
A standardized software environment is used:

- Python (latest stable version)
- PyTorch (latest stable version)
- ONNX (latest stable version)
- ONNX Runtime (latest stable version, configured for CPU execution)
- Essential data science libraries (NumPy, Pandas, Scikit-learn).

## 6. Implementation Plan

### Implementation Structure

The project is organized into a modular, production-ready codebase located in the `src/` directory:

**Core Modules:**
- `config/`: Configuration management system (M4Config, StandardConfig)
- `models/`: PatchTST architecture with ONNX-compatible implementation
- `data/`: Data loaders for M4 and standard datasets
- `training/`: Training pipeline with early stopping and checkpointing
- `evaluation/`: Metrics (sMAPE, MASE, OWA) and evaluator
- `optimization/`: ONNX export and INT8 quantization
- `inference/`: Unified predictor for PyTorch/ONNX models
- `utils/`: Helper functions, logging, and checkpoint management

**Example Usage:**
Ready-to-use example scripts are provided in `src/examples/` for both M4 and standard datasets.

### Phase 1: Foundation & Baseline Evaluation

1.  **Environment and Data Setup:** Configure the environment and implement the M4 data loader (implemented in `src/data/m4_dataset.py`), including RevIN and a batching strategy to handle variable-length series.
2.  **Baseline Model Implementation:** Implement the PatchTST model in PyTorch (implemented in `src/models/patchtst.py`), applying M4-specific adaptations.
3.  **Training and Benchmarking:** Train the adapted model on the full M4 Monthly dataset using the production training pipeline (`src/training/trainer.py`). Establish the baseline OWA score and measure the initial latency and size of the PyTorch model.

### Phase 2: Model Adaptation and Optimization

1.  **Structural Adaptation:** Execute grid searches for look-back window, patch configurations, and model size variations using the configuration system (`src/config/m4_config.py`). Select the best performing, most efficient configuration based on validation OWA.
2.  **ONNX Conversion:** Export the final, trained PyTorch model to the ONNX format using `src/optimization/onnx_export.py`. This is critical for platform-agnostic deployment and enables graph-level optimizations.
3.  **Post-Training Quantization:** Apply post-training dynamic quantization to the ONNX model using `src/optimization/quantization.py`, converting the FP32 weights to INT8. This is expected to achieve significant model compression and inference speedup.
4.  **Benchmarking:** Measure the OWA, latency (on CPU), and size (MB) for both the ONNX FP32 and ONNX INT8 models using the unified evaluator (`src/evaluation/evaluator.py`) and predictor (`src/inference/predictor.py`).

### Phase 3: Final Analysis and Reporting

1.  **Comprehensive Evaluation:** Evaluate the final optimized model on the M4 test set to obtain the final OWA score.
2.  **Latency-Accuracy Analysis:** Consolidate all performance metrics (OWA, latency, size) across all three model variants (PyTorch, ONNX FP32, ONNX INT8).
3.  **Reporting:** Calculate and report the final speedup and compression ratios. Generate a clear comparison, including a Pareto frontier plot (OWA vs. latency), to visualize the efficacy of the optimization pipeline.

## 7. Risk Analysis

| Risk                                                      | Likelihood | Impact | Mitigation Strategies                                                                                                 | Contingency Plan                                                                                                                      |
| :-------------------------------------------------------- | :--------- | :----- | :-------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Insufficient Training Time on Full M4 Dataset**         | Medium     | High   | Use mixed-precision training (FP16); implement gradient accumulation; save checkpoints frequently.                    | Train on a 10,000-series subset if the full 48,000-series dataset is infeasible within time constraints.                              |
| **Quantization Degrades OWA Beyond Acceptable Threshold** | Low        | Medium | Utilize dynamic quantization first; use a calibration dataset for static quantization if necessary.                   | Accept minor degradation (e.g., $<2\%$ relative OWA increase) if the latency gains are substantial, documenting the efficiency focus. |
| **Model Underperforms N-BEATS Baseline**                  | Medium     | High   | Conduct extensive tuning of the M4-specific adaptations (RevIN, adaptive patching); ensure architectural correctness. | Focus the narrative on superior deployment efficiency (latency, size) compared to N-BEATS, even if absolute OWA is slightly lower.    |
| **Inconsistent Performance Benchmarks**                   | Medium     | Medium | Standardize the CPU hardware for all latency tests; implement warm-up and average over 100 timed iterations.          | Increase the number of runs and report average and standard deviation if results are volatile.                                        |

## 8. Expected Outcomes

### Quantitative Goals

**Accuracy (M4 Monthly):**

- **Minimum Acceptable:** OWA $< 1.0$ (beat Naive2 benchmark).
- **Target:** OWA $< 0.85$ (competitive with top statistical and ML methods).

**Latency (Single Forecast, CPU):**

- **Target:** Achieve a $>4\times$ speedup with the final Quantized+ONNX model compared to the baseline PyTorch model.
- **Stretch Goal:** Achieve sub-10ms latency per forecast.

**Model Size:**

- **Target:** Achieve an approximate $4\times$ reduction in model size with the INT8 quantized model compared to the FP32 baseline.

### Qualitative Outcomes

1.  **Implementation Artifacts:** A production-ready, modular PatchTST codebase (`src/`) adapted for the M4 benchmark, including:
    - Fully documented API with type hints and comprehensive docstrings
    - Ready-to-use example scripts (`src/examples/`)
    - Complete optimization pipeline (PyTorch → ONNX FP32 → INT8)
    - Unified configuration system for reproducible experiments
2.  **Research Insights:** A clear quantification of the efficiency gains achievable when deploying modern Transformer-based forecasting models, specifically tailored for the characteristics of the M4 competition.
3.  **Reproducible Research:** A complete code repository with:
    - Documented configurations (`src/config/`)
    - Checkpoint management system (`src/utils/checkpoint.py`)
    - Comprehensive usage instructions (`docs/usage_instructions.md`)
    - Example end-to-end training scripts

## 9. Validation & Quality Assurance

### Model Validation

1.  **Sanity Checks:** Verify that the model can overfit small training samples and that the initial baseline performance significantly outperforms the Naive2 reference.
2.  **Cross-Validation:** Strictly adhere to the official M4 train/test split. Use a held-out portion of the training data for validation.
3.  **Ablation Studies:** Systematically analyze the impact of key architectural and preprocessing choices (Adaptive Patching, RevIN, Quantization) on the final OWA score.

### Code Quality

1.  **Testing:** Implement unit tests for critical components (M4 data loader, RevIN) and an integration test for the end-to-end training pipeline.
2.  **Documentation:** Provide clear docstrings for all functions and classes, and maintain a comprehensive README file.
3.  **Version Control:** Use Git for version control, tagging important milestones (baseline model, final optimized model).

## 10. Ethical Considerations

### Data Usage

- All datasets are public and released for research. No private or sensitive data is involved. Proper attribution will be given in all final reports.

### Reproducibility

- The research is committed to full transparency. Fixed random seeds and complete documentation of hyperparameters ensure the reproducibility of results.

### Environmental Impact

- The research leverages cloud resources efficiently (early stopping, checkpointing). A key positive outcome is the creation of highly energy-efficient models for inference, reducing the computational footprint of deploying advanced forecasting models.
