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
5. Split dataset into training and validation folds for consistency.

---

## 4. Model Architecture

The core model is based on **Facebook’s Wav2Vec2.0 Base architecture**, enhanced with two proposed modifications:

1. **Inter-Codebook Similarity Loss (ICSL):**  
   - Applied during pretraining.  
   - Penalizes cosine similarity across multiple codebooks to ensure each learns distinct features.  
   - Improves feature diversity and stabilizes training.

2. **Residual Vector Quantization (RVQ):**  
   - Integrated during fine-tuning.  
   - Seq
