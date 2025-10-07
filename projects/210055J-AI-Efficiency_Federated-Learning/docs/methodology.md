# Methodology: AI Efficiency – Federated Learning

**Student:** 210055J  
**Research Area:** AI Efficiency: Federated Learning  
**Date:** 2025-09-01  

---

## 1. Overview
This methodology outlines the design and implementation plan for enhancing the Federated Averaging (FedAvg) algorithm using a lightweight client-side local–global knowledge regularization mechanism. The goal is to improve stability, convergence speed, and fairness of federated learning under heterogeneous (non-IID) conditions while preserving the simplicity and communication efficiency of FedAvg.  

---

## 2. Research Design
The research adopts an **experimental comparative design**:  
- Implement the baseline FedAvg algorithm.  
- Introduce the proposed modification (local–global knowledge distillation).  
- Conduct controlled experiments across multiple datasets under IID and non-IID partitions.  
- Compare performance against standard FedAvg and selected state-of-the-art baseline methods (e.g., FedProx, SCAFFOLD).  

---

## 3. Data Collection

### 3.1 Data Sources
- **MNIST** – Handwritten digit dataset (10 classes).  
- **CIFAR-10** – Natural image classification dataset (10 classes).  
- **Shakespeare** – Character-level text dataset (client = speaking role).  

### 3.2 Data Description
- MNIST: 70,000 grayscale images of digits (28×28 pixels).  
- CIFAR-10: 60,000 RGB images (32×32 pixels).  
- Shakespeare: Text corpus partitioned into client-specific dialogues.  

### 3.3 Data Preprocessing
- Normalize images to [0,1].  
- Partition datasets into IID and non-IID splits (label-skew shards, client-based splits).  
- Tokenize and preprocess text for the Shakespeare dataset.  
- Ensure balanced train/test splits across experiments.  

---

## 4. Model Architecture
- **Baseline:** Standard convolutional neural networks (CNNs) for MNIST and CIFAR-10, LSTM-based models for Shakespeare.  
- **Proposed Enhancement:**  
  - Treat the received global model as a **teacher** (frozen).  
  - Train a duplicate **student** model using a combined loss:  
    - Cross-entropy with local labels.  
    - Knowledge distillation loss between teacher and student predictions, weighted by λ.  
  - Apply a confidence threshold (τ) and optional warm-up schedule for λ.  

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **Test Accuracy over Rounds** – measures convergence trajectory.  
- **Final Accuracy** – accuracy after fixed communication rounds.  
- **Rounds-to-Target Accuracy** – efficiency in reaching predefined thresholds.  
- **Stability & Fairness** – variance across clients and oscillations in accuracy.  

### 5.2 Baseline Models
- FedAvg (McMahan et al., 2017).  
- FedProx (Li et al., 2020).  
- SCAFFOLD (Karimireddy et al., 2020).  
- Optional: FedMD, FedDF (for KD-based comparisons).  

### 5.3 Hardware/Software Requirements
- **Hardware:** GPU-enabled workstation or cloud environment (NVIDIA GPU with ≥12GB memory).  
- **Software:**  
  - Python 3.9+  
  - PyTorch / TensorFlow for deep learning.  
  - Federated learning libraries (e.g., Flower, FedML).  
  - Jupyter/Colab for experimentation.  

---

## 6. Implementation Plan

| Phase   | Tasks                          | Duration | Deliverables        |
|---------|--------------------------------|----------|---------------------|
| Phase 1 | Data preprocessing             | 2 weeks  | Clean dataset splits|
| Phase 2 | Model implementation           | 3 weeks  | Baseline + enhanced models |
| Phase 3 | Experiments                    | 2 weeks  | Results (metrics & plots) |
| Phase 4 | Analysis                       | 1 week   | Final report & discussion |

---

## 7. Risk Analysis
- **Risk:** Non-IID partitions may cause unstable results.  
  - *Mitigation:* Use multiple seeds and average results.  
- **Risk:** Hyperparameter sensitivity (λ, T, τ).  
  - *Mitigation:* Perform systematic grid search.  
- **Risk:** Hardware/GPU limitations.  
  - *Mitigation:* Optimize batch sizes, use smaller models, or cloud resources.  
- **Risk:** Overfitting to small datasets.  
  - *Mitigation:* Include diverse benchmarks (vision + text).  

---

## 8. Expected Outcomes
- Modest but consistent improvements in accuracy (1–2%) over FedAvg under non-IID conditions.  
- Faster convergence, with ~5–10% reduction in rounds-to-target accuracy.  
- Reduced client drift and variance in per-client performance, improving fairness.  
- Demonstration that client-side regularization is a practical, lightweight solution without server-side modifications.  

---

**Note:** This methodology will be updated iteratively as experiments progress and new findings emerge.  
