# Research Proposal: NLP – Machine Translation

**Student:** 210352R  
**Research Area:** NLP: Machine Translation  
**Date:** 2025-09-01

---

## Abstract

Large multilingual language models such as **mT5** have achieved remarkable success in natural language understanding and translation. However, their performance often deteriorates for **low-resource languages** and **domain-specific translation tasks** due to data scarcity and domain mismatch.  
This research investigates **denoising pretraining** as an intermediate adaptation step for the mT5 model using the **OPUS-100 dataset**. Two strategies are proposed:

1. **Monolingual Denoising (EN noisy → EN)** – reconstructing clean English sentences from corrupted input.
2. **Bilingual Denoising (EN noisy → FR)** – combining denoising and translation objectives.

Both are followed by **instruction fine-tuning** with **Low-Rank Adaptation (LoRA)** for efficient optimization. Experiments across multiple language pairs, including English–French, English–Spanish, and several low-resource pairs (EN–SI, EN–HI, EN–TA), demonstrate that denoising pretraining substantially improves translation performance compared to baseline instruction-tuned models.  
The **monolingual denoising approach** yields the highest improvements, particularly for translations into English, confirming the effectiveness of language-targeted pretraining for domain and task adaptation.

---

## 1. Introduction

Neural Machine Translation (NMT) has become the foundation of multilingual text processing due to **Transformer-based architectures**. Multilingual models such as **mT5**, trained on large-scale corpora like **mC4**, enable cross-lingual transfer and zero-shot translation.  
However, these models suffer from **domain mismatch** and **inefficiency** when applied to resource-limited translation tasks.  
This research aims to **bridge this gap** by enhancing mT5 through an intermediate **denoising pretraining phase**, enabling the model to generalize better and perform efficiently with limited computational resources.

---

## 2. Problem Statement

While mT5 exhibits strong multilingual capabilities, its effectiveness diminishes when translating domain-specific or low-resource language pairs.  
This degradation primarily arises due to:

- Pretraining on **generic web data** that lacks domain alignment.
- Absence of robust cross-lingual noise-handling mechanisms.
- High computational cost of full fine-tuning.

Hence, the research problem is:

> _How can denoising pretraining be leveraged as an intermediate adaptation technique to enhance mT5’s translation performance in resource-constrained and domain-specific settings?_

---

## 3. Literature Review Summary

Recent multilingual models like **mBERT**, **XLM-R**, and **mT5** have advanced cross-lingual representation learning. However:

- **Domain Adaptation** methods such as **DAPT** (Don’t Stop Pretraining) show that continued pretraining on specific domains can significantly improve downstream performance.
- **Denoising Objectives** (BART, T5) effectively build robust linguistic representations by reconstructing clean text from corrupted sequences.
- **Parameter-Efficient Fine-Tuning (PEFT)** techniques like **LoRA** [9] drastically reduce computational overhead by updating a small subset of model parameters.  
  Despite these advancements, few works systematically evaluate **multilingual denoising pretraining** as a mid-stage adaptation technique for **low-resource translation tasks**, leaving a research gap this study addresses.

---

## 4. Research Objectives

### Primary Objective

To evaluate the effectiveness of **denoising pretraining** as an intermediate adaptation step for **mT5** in improving translation quality under resource constraints.

### Secondary Objectives

- Develop and compare **monolingual** and **bilingual denoising** strategies.
- Integrate **instruction fine-tuning** with **LoRA** for efficient adaptation.
- Evaluate performance across **high- and low-resource** language pairs.
- Analyze metric-based improvements (BLEU, chrF, TER) and determine best-performing configurations.

---

## 5. Methodology

### Model and Dataset

- Base Model: `google/mt5-small` (∼300M parameters).
- Dataset: **OPUS-100**, containing parallel corpora for 100 languages.
- Languages: 15 English-centric pairs (e.g., EN–FR, EN–ES, EN–SI, EN–TA).

### Pretraining Strategies

1. **Approach 1 – Monolingual Denoising (EN noisy → EN):**  
   Strengthens target-language fluency and syntactic understanding.
2. **Approach 2 – Bilingual Denoising (EN noisy → FR, FR noisy → EN):**  
   Encourages bilingual alignment under noisy input conditions.

Noise Injection Methods:

- Word Deletion (10%), Character Deletion (10%), Word Swapping (5%).

### Fine-Tuning Stage

- **Instruction Fine-Tuning:** Using natural language prompts (e.g., “Translate English to French: ...”).
- **Parameter-Efficient Fine-Tuning (LoRA):** Injects low-rank trainable matrices into attention layers, reducing trainable parameters to ~0.7%.

### Evaluation

- Metrics: **BLEU**, **chrF**, and **TER**.
- Comparison between:
  - Baseline mT5 (instruction-tuned only).
  - Denoising-pretrained models (Approach 1 & 2).
- Experiments conducted on **NVIDIA T4 GPUs (16GB)** for 3 epochs per phase.

---

## 6. Expected Outcomes

- Demonstrate measurable performance gains from denoising pretraining.
- Establish a **computationally efficient pipeline** combining denoising and PEFT.
- Achieve **higher BLEU and lower TER** across multiple language pairs.
- Show that **monolingual denoising** particularly improves target-language generation quality.
- Release all fine-tuned models on **Hugging Face** for reproducibility.

---

## 7. Timeline

| Week  | Task                                                      |
| ----- | --------------------------------------------------------- |
| 1–2   | Literature review on multilingual MT and denoising models |
| 3–4   | Dataset acquisition and preprocessing (OPUS-100)          |
| 5–7   | Denoising pretraining (Approach 1 & 2)                    |
| 8–10  | Instruction fine-tuning with LoRA                         |
| 11–13 | Evaluation and metric analysis                            |
| 14–15 | Comparative analysis on low-resource pairs                |
| 16    | Documentation and final submission                        |

---

## 8. Resources Required

- **Hardware:** GPU access (Kaggle T4 or equivalent).
- **Libraries:** Hugging Face Transformers, PEFT, SentencePiece, SacreBLEU.
- **Datasets:** OPUS-100 multilingual parallel corpus.
- **Version Control:** Git & GitHub for code and experiment tracking.
- **Evaluation Tools:** BLEU, chrF++, and TER scoring scripts.

---

## References

1. Raffel et al. (2020). _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer._
2. Vaswani et al. (2017). _Attention Is All You Need._
3. Xue et al. (2021). _mT5: A Massively Multilingual Pre-Trained Text-to-Text Transformer._
4. Tiedemann (2012). _Parallel Data, Tools and Interfaces in OPUS._
5. Gururangan et al. (2020). _Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks._
6. Lewis et al. (2019). _BART: Denoising Sequence-to-Sequence Pre-training for NLP._
7. Liu et al. (2020). _Multilingual Denoising Pre-training for NMT._
8. Reid & Artetxe (2022). _PARADISE: Exploiting Parallel Data for Multilingual Seq2Seq Pretraining._
9. Hu et al. (2021). _LoRA: Low-Rank Adaptation of Large Language Models._
10. Papineni et al. (2002). _BLEU: A Method for Automatic Evaluation of MT._
11. Popović (2015). _chrF: Character n-gram F-score for MT Evaluation._
12. Snover et al. (2006). _Translation Edit Rate with Targeted Human Annotation._

---

> _“Enhancing multilingual translation through denoising — bridging the gap between low-resource data and global understanding.”_
