# Student Usage Instructions

**Project Title:** Denoising Pretraining of mT5 on OPUS-100 for Domain Adaptation in Machine Translation  
**Student:** 210352R – Eshan Maduranga  
**Supervisor:** Dr. Uthayasanker Thayasivam  
**Department:** Department of Computer Science and Engineering, University of Moratuwa  
**Research Area:** NLP: Machine Translation

---

## 1. Project Overview

This project investigates how **denoising pretraining** can improve the **mT5 multilingual model** for **machine translation**, especially in **low-resource and domain-specific** settings.  
The work introduces two intermediate pretraining strategies—**monolingual denoising** and **bilingual denoising**—followed by **instruction fine-tuning with LoRA (Low-Rank Adaptation)** for parameter-efficient optimization.

All experiments use the **OPUS-100 multilingual dataset** and evaluate results using BLEU, chrF, and TER metrics across 15 English-centric language pairs.

---

## 2. Objective of This Document

This guide provides **step-by-step usage instructions** for students or researchers who wish to:

- Reproduce the original experiments
- Extend the methodology to new language pairs
- Adapt the denoising and fine-tuning pipeline to other domains
- Analyze translation performance using standard metrics

---

## 3. Prerequisites

### 3.1 Hardware

- GPU: **NVIDIA T4 (16GB)** or higher (A100 recommended for large-scale training)
- Minimum 12GB RAM
- At least 50GB of free disk space

### 3.2 Software

| Tool                      | Version  | Purpose                                |
| ------------------------- | -------- | -------------------------------------- |
| Python                    | ≥ 3.10   | Core scripting language                |
| PyTorch                   | ≥ 2.0    | Deep learning backend                  |
| Hugging Face Transformers | ≥ 4.30   | Model and tokenizer API                |
| PEFT                      | ≥ 0.6    | Parameter-efficient fine-tuning        |
| SentencePiece             | ≥ 0.1.99 | Tokenization                           |
| SacreBLEU                 | ≥ 2.3    | Evaluation metric                      |
| NLTK / chrF++             | Latest   | Auxiliary metrics and corpus utilities |

Install dependencies:

```bash
pip install torch transformers peft sentencepiece sacrebleu nltk
```
