# Methodology: NLP – Machine Translation

**Student:** 210352R  
**Research Area:** NLP: Machine Translation  
**Date:** 2025-10-01

---

## 1. Overview

This study introduces a **denoising pretraining framework** to enhance the **mT5** multilingual model for **machine translation (MT)**, particularly in low-resource and domain-specific contexts.  
The methodology involves a two-stage process:

1. **Intermediate denoising pretraining** on the OPUS-100 dataset using two corruption strategies—monolingual and bilingual.
2. **Instruction fine-tuning with Low-Rank Adaptation (LoRA)** for efficient and lightweight adaptation to translation tasks.

This approach aims to strengthen mT5’s language representation capabilities, improve translation accuracy, and reduce computational requirements.

---

## 2. Research Design

The research follows an **experimental and comparative design** with multiple configurations of the mT5-small model.  
The study is structured as follows:

- **Baseline:** Instruction fine-tuned mT5-small without denoising pretraining.
- **Approach 1:** Monolingual denoising pretraining (EN noisy → EN) followed by instruction fine-tuning.
- **Approach 2:** Bilingual denoising pretraining (EN noisy → FR and FR noisy → EN) followed by instruction fine-tuning.

Each configuration is evaluated across **high- and low-resource language pairs** to determine relative performance improvements in BLEU, chrF, and TER metrics.

---

## 3. Data Collection

### 3.1 Data Sources

- **Primary Dataset:** [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) multilingual parallel corpus.
- **Additional Resources:** Publicly available English-centric translation pairs across 15 language directions (e.g., EN–FR, EN–ES, EN–SI, EN–TA, EN–HI).

### 3.2 Data Description

The OPUS-100 dataset includes **100 language pairs** with **1 million parallel sentences per pair**, covering diverse domains like subtitles, legal, and technical text.  
This study focuses on **English ↔ Other language pairs**, with a subset of **100,000 sentences per direction** for efficient training under GPU constraints.

### 3.3 Data Preprocessing

- Unicode normalization and whitespace cleanup.
- Removal of empty or extremely short sentences.
- Tokenization using the **mT5 SentencePiece tokenizer**.
- Filtering sequences to **5–512 tokens**.
- Creation of **monolingual corpora** for denoising pretraining.
- Application of **Naïve Noise Injection**:
  - Word Deletion (10%)
  - Character Deletion (10%)
  - Word Swapping (5%)

---

## 4. Model Architecture

The experiments use **mT5-small**, a multilingual encoder–decoder Transformer model with approximately **300M parameters**.  
Key architectural details:

| Component              | Description |
| ---------------------- | ----------- |
| Encoder Layers         | 8           |
| Decoder Layers         | 8           |
| Attention Heads        | 6           |
| Model Dimension        | 512         |
| Feed-forward Dimension | 1024        |

Two pretraining objectives are applied:

1. **Monolingual Denoising:** Predict clean English from noisy English (EN noisy → EN).
2. **Bilingual Denoising:** Predict target language from noisy source (EN noisy → FR, FR noisy → EN).

After pretraining, models undergo **Instruction Fine-Tuning** with **LoRA**:

- Injects low-rank trainable matrices into query (`q`) and value (`v`) projections.
- Freezes all base parameters—only ~0.7% remain trainable.
- LoRA settings: Rank = 8, Alpha = 32, Dropout = 0.1.

---

## 5. Experimental Setup

Experiments were conducted on **NVIDIA T4 GPUs (16GB)** using PyTorch and Hugging Face Transformers.

### 5.1 Evaluation Metrics

- **BLEU** – measures n-gram precision and overall translation adequacy.
- **chrF** – character-level F-score capturing morphological correctness.
- **TER (Translation Edit Rate)** – measures minimal edits needed to match the reference.

### 5.2 Baseline Models

| Model Configuration | Description                                          |
| ------------------- | ---------------------------------------------------- |
| **Baseline**        | mT5-small + Instruction Fine-Tuning (no pretraining) |
| **Approach 1**      | Monolingual Denoising + Instruction Fine-Tuning      |
| **Approach 2**      | Bilingual Denoising + Instruction Fine-Tuning        |

Each model was trained for **3 epochs** using the **AdamW optimizer (lr = 2e-4)** with batch-size scaling via gradient accumulation.

### 5.3 Hardware/Software Requirements

- **Hardware:** Dual NVIDIA T4 GPUs (16GB each)
- **Software Stack:**
  - Python 3.10
  - PyTorch ≥ 2.0
  - Hugging Face Transformers ≥ 4.30
  - PEFT library for LoRA
  - SentencePiece tokenizer
  - SacreBLEU, NLTK, and chrF++ for evaluation

---

## 6. Implementation Plan

| Phase       | Tasks                                           | Duration | Deliverables                        |
| ----------- | ----------------------------------------------- | -------- | ----------------------------------- |
| **Phase 1** | Dataset acquisition, cleaning, and tokenization | 2 weeks  | Preprocessed multilingual datasets  |
| **Phase 2** | Denoising pretraining (Approach 1 & 2)          | 3 weeks  | Pretrained mT5 checkpoints          |
| **Phase 3** | Instruction fine-tuning with LoRA               | 2 weeks  | Fine-tuned translation models       |
| **Phase 4** | Evaluation and analysis                         | 2 weeks  | BLEU, chrF, TER results             |
| **Phase 5** | Documentation and model release                 | 1 week   | Report + Hugging Face model uploads |

---

## 7. Risk Analysis

| Risk                             | Description                                 | Mitigation Strategy                                     |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| **GPU memory limitation**        | mT5 training requires high memory           | Use `mt5-small`, gradient accumulation, mixed precision |
| **Data imbalance**               | OPUS-100 pairs have varied data quality     | Uniform sampling and filtering                          |
| **Overfitting**                  | Fine-tuning may overfit on small subsets    | Apply early stopping and validation loss tracking       |
| **Noise corruption instability** | Excessive corruption may degrade learning   | Tune corruption ratios and validate outputs             |
| **Metric bias**                  | BLEU may not fully capture semantic quality | Complement with chrF and TER metrics                    |

---

## 8. Expected Outcomes

- Demonstrated improvement of **translation quality** via denoising pretraining.
- Establishment of a **resource-efficient adaptation pipeline** using **LoRA**.
- Empirical proof that **monolingual denoising** enhances target-language fluency.
- Public release of **pretrained multilingual models** for 15 English-centric pairs on Hugging Face.
- Insights contributing to future **low-resource machine translation research**.

---
