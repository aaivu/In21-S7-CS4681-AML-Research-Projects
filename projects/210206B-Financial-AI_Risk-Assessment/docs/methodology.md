# Methodology: Enhanced Neural Oblivious Decision Ensembles for Tabular Data

**Student:** 210206B  
**Research Area:** Deep Learning Optimization for Tabular Data 
**Date:** 2025-09-01  

---

## 1. Overview

This research proposes an enhanced training framework for **Neural Oblivious Decision Ensembles (NODE)**, focusing on improving efficiency, stability, and scalability under memory-constrained environments. Unlike prior works that emphasize architectural changes, this study optimizes the training process itself leveraging cyclical learning rate scheduling, loss shaping via focal loss and label smoothing, and mixed-precision computation. The methodology integrates both theoretical and empirical investigations to establish reproducible, resource-aware training practices for deep tabular models.

---

## 2. Research Design

The study follows an **experimental research design**, combining quantitative evaluations and ablation studies. The process involves:
1. Benchmarking baseline NODE models under standard configurations.
2. Introducing optimization techniques to improve convergence and memory efficiency.
3. Conducting controlled experiments to assess their combined effects.
4. Comparing results with state-of-the-art tabular models (e.g., XGBoost, TabNet, FT-Transformer).

The research emphasizes **comparative experimentation**, where variations in optimizers, loss functions, and learning rate schedules are systematically evaluated.

---

## 3. Data Collection

### 3.1 Data Sources
Two widely used public tabular datasets are selected for evaluation:
- **A9A (Adult Dataset)** – UCI Repository (binary classification: income prediction)
- **CLICK Dataset** – Criteo click-through rate prediction benchmark

These datasets represent both balanced and imbalanced classification scenarios, allowing robust evaluation of optimization strategies.

### 3.2 Data Description
| Dataset | Samples | Features | Type | Class Balance |
|----------|----------|----------|------|----------------|
| A9A | 48,842 | 123 (binary encoded) | Binary classification | Balanced |
| CLICK | 1,200,000 | 50 (mixed categorical + numerical) | Binary classification | Imbalanced (~5% positive) |

### 3.3 Data Preprocessing
1. **Normalization:** Numerical features quantile-normalized to \( N(0,1) \).  
2. **Encoding:** Categorical variables mean-target encoded.  
3. **Noise Injection:** Gaussian noise (sigma = 10^{-3}) added for regularization.  
4. **Splitting:** 70–15–15 split for train, validation, and test sets.  
5. **Batching:** Stratified sampling to preserve class ratios during mini-batch construction.

---

## 4. Model Architecture

The base model is the **Neural Oblivious Decision Ensemble (NODE)**:
- Composed of multiple **oblivious decision trees**, where each layer shares identical split conditions.
- Differentiable via **entmoid15** activation, a smooth approximation of binary gating.
- Aggregates leaf outputs with learned weights for prediction.

### Architectural Parameters:
- Trees: 16–32  
- Depth: 6–8  
- Hidden dimension: 128  
- Activation: entmoid15  
- Regularization: Dropout (0.1–0.2)

Enhancements are applied at the training level rather than modifying the model’s structure:
- Optimizer: **QHAdam** with β₁ = 0.9, β₂ = 0.999, ν = (0.7, 1.0)  
- Scheduler: **Cosine annealing with warmup**  
- Loss: **Focal loss + Label smoothing (ε = 0.05–0.1)**  
- Precision: **Automatic Mixed Precision (AMP)**  
- Stability: **Gradient accumulation** and **checkpoint averaging**

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **AUC (Area Under Curve)** – measures classification performance.
- **Log Loss** – evaluates calibration quality.
- **Training Time (epochs)** – efficiency measure.
- **GPU Memory Usage** – hardware efficiency indicator.
- **Convergence Steps** – training stability proxy.

### 5.2 Baseline Models
- **XGBoost** (Chen & Guestrin, 2016) — a gradient boosting framework based on decision trees.
- **LightGBM** (Ke et al., 2017) — a fast, efficient gradient boosting model for large-scale data.
- **CatBoost** (Dorogush et al., 2018) — a categorical feature–aware boosting method.
- **TabNet** (Arik & Pfister, 2021) — a deep learning model using sequential attention for tabular data.
- **FT-Transformer** (Gorishniy et al., 2021) — a transformer-based model optimized for tabular datasets.

These baselines represent both traditional and deep-learning-based tabular models for fair benchmarking.

### 5.3 Hardware/Software Requirements
- **Hardware:** NVIDIA GTX 1650 GPU (4 GB VRAM), 16 GB RAM, Intel i7 CPU.  
- **Software:**  
  - PyTorch 2.2, CUDA 12.2  
  - Python 3.10  
  - Scikit-learn, Pandas, Numpy  
  - Weights & Biases for experiment tracking

---

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| **Phase 1** | Data preprocessing, normalization, encoding | 2 weeks | Clean and structured dataset |
| **Phase 2** | NODE baseline implementation | 2 weeks | Reproducible baseline results |
| **Phase 3** | Integrate optimization techniques (QHAdam, label smoothing, AMP) | 3 weeks | Enhanced training pipeline |
| **Phase 4** | Experiments and ablation studies | 2 weeks | Comparative performance metrics |
| **Phase 5** | Analysis and visualization | 1 week | Analytical report and plots |
| **Phase 6** | Final documentation and reproducibility package | 1 week | Research paper + GitHub repo |

---

## 7. Risk Analysis

| Risk | Impact | Mitigation Strategy |
|------|---------|--------------------|
| GPU memory overflow | Medium | Use AMP and gradient accumulation |
| Overfitting on small datasets | High | Employ early stopping, dropout, label smoothing |
| Hyperparameter instability | Medium | Use learning rate warmup and scheduler |
| Reproducibility issues | Low | Fix random seeds, record all configurations |
| Time constraints | Medium | Parallelize experiments and automate logging |

---

## 8. Expected Outcomes

- A **stable and efficient NODE training framework** suitable for low-memory environments.
- Demonstrated improvements in **AUC** and **training stability** compared to standard NODE.
- Quantified impact of optimization techniques (focal loss, label smoothing, cosine scheduling).
- Public release of reproducible code and experiment logs for future benchmarking.
- Foundation for **resource-aware deep tabular learning research**, applicable to domains like healthcare analytics, credit scoring, and CTR prediction.
