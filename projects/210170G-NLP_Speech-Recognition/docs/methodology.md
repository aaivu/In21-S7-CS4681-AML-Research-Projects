# Methodology: NLP: Speech Recognition

**Student:** 210170G  
**Research Area:** NLP: Speech Recognition  
**Date:** 2025-09-01  

---

## 1. Overview

This research focuses on improving the **efficiency and adaptability of Wav2Vec2** in **low-resource speech recognition** tasks. The methodology integrates two complementary techniques:  
1. **Inter-Codebook Similarity Loss (ICSL)** — enhances pretraining efficiency by reducing redundancy between codebooks.  
2. **Residual Vector Quantization (RVQ)** — improves fine-tuning adaptability with limited data.  

The study is experimental, using a **quantitative design** to evaluate performance improvements in convergence speed, loss reduction, and word error rate (WER) across low-resource scenarios.

---

## 2. Research Design

The research follows an **experimental-comparative design** consisting of two main stages:  
1. **Pretraining Enhancement:** Introduce ICSL during Wav2Vec2 pretraining to enforce diversity among quantized embeddings.  
2. **Fine-Tuning Enhancement:** Integrate RVQ between the embedding layer and classification head during fine-tuning to refine latent speech representations.  

Each stage includes controlled experiments comparing baseline and modified configurations under identical low-resource conditions.

---

## 3. Data Collection

### 3.1 Data Sources

- **LibriSpeech Dataset** (Panayotov et al., 2015)  
  - Publicly available corpus of English read speech.  
  - Originally 960 hours of labeled audio.  

### 3.2 Data Description

To simulate **low-resource conditions**, a **10-hour subset** was extracted from LibriSpeech’s “clean” portion:  
- **Training set:** 2,850 utterances (≥5 seconds).  
- **Validation set:** 2,703 utterances (≥5 seconds).  

This subset preserves data quality while emphasizing computational constraints representative of real-world low-resource environments.

### 3.3 Data Preprocessing

1. Filtered out utterances shorter than 5 seconds.  
2. Normalized sampling rate to 16 kHz.  
3. Converted all audio to mono channel format.  
4. Applied masking and feature extraction per Wav2Vec2 protocol.  

---

## 4. Model Architecture

The core model is based on **Facebook’s Wav2Vec2.0 Base architecture**, enhanced with two proposed modifications:

1. **Inter-Codebook Similarity Loss (ICSL):**  
   - Applied during pretraining.  
   - Penalizes cosine similarity across multiple codebooks to ensure each learns distinct features.  
   - Improves feature diversity and stabilizes training.

2. **Residual Vector Quantization (RVQ):**  
   - Integrated during fine-tuning.  
   - Sequentially encodes residual errors through multiple quantizers.  
   - Produces refined latent representations that enable faster convergence and better adaptation with limited data.

The **final fine-tuning layer** employs **Connectionist Temporal Classification (CTC) loss** for sequence-to-text mapping without frame-level alignment.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Contrastive Loss:** For pretraining performance assessment.  
- **CTC Loss:** For fine-tuning optimization and alignment quality.  
- **Word Error Rate (WER):** For end-to-end speech recognition accuracy.  
- **Convergence Speed:** Number of steps required to reach stable loss.

### 5.2 Baseline Models

1. **Baseline Wav2Vec2.0 without pretrained weights** — Use as pretraining experiment baseline.
1. **Baseline Wav2Vec2.0 pretrained model** — Use as fintuning experiment basline

### 5.3 Hardware/Software Requirements

- **Hardware:** NVIDIA P100 GPU (16 GB VRAM)  
- **Frameworks:** PyTorch, Hugging Face Transformers  
- **Environment:** Kaggle GPU runtime  
- **Libraries:** NumPy, Librosa, Transformers, TorchAudio  
- **Training Configuration:**
  - Learning Rate: `1e-4`  
  - Batch Size: 8–16  
  - Optimizer: AdamW  
  - Scheduler: Linear decay with warm-up  

---

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| **Phase 1** | Data preprocessing (filtering, normalization, feature extraction) | 2 weeks | Clean and preprocessed dataset |
| **Phase 2** | Model implementation (ICSL and RVQ integration) | 3 weeks | Functional Wav2Vec2-based model |
| **Phase 3** | Experiments (training, fine-tuning, evaluation) | 2 weeks | Training and validation results |
| **Phase 4** | Analysis and documentation | 1 week | Final report with performance comparison |

---

## 7. Risk Analysis

| Risk | Impact | Mitigation Strategy |
|------|---------|----------------------|
| GPU memory limitation | High | Use gradient accumulation and mixed-precision training |
| Overfitting due to limited data | Medium | Apply regularization and early stopping |
| Long training times | Medium | Reduce codebook count and batch size where feasible |
| Convergence instability | Low | Use warm-up scheduling and adaptive learning rates |

---

## 8. Expected Outcomes

- Improved **pretraining efficiency** via ICSL through diverse codebook learning.  
- Enhanced **fine-tuning adaptability** with RVQ under data-scarce conditions.  
- Demonstrated reduction in **training time** and **contrastive/CTC loss**.  
- Achieved competitive or superior **WER performance** compared to baseline models.  
- Provided practical insights into **efficient low-resource ASR design** using SSL frameworks.

