# Results and Conclusion: NLP – Machine Translation

**Student:** 210352R – Eshan Maduranga  
**Research Area:** NLP: Machine Translation  
**Date:** 2025-09-01

---

## 1. Overview

This section presents the **quantitative results and analysis** obtained from the proposed denoising pretraining framework for the **mT5-small** model.  
We evaluate performance across **English–French (EN↔FR)**, **English–Spanish (EN↔ES)**, and **other OPUS-100 language pairs**, followed by an overall summary, discussion, and conclusion.

All experiments were performed using:

- **Base model:** `google/mt5-small` (~300M parameters)
- **Pretraining strategies:**
  - **Approach 1:** Monolingual Denoising (EN noisy → EN)
  - **Approach 2:** Bilingual Denoising (EN noisy → FR)
- **Fine-tuning:** Instruction-based fine-tuning with **LoRA (Low-Rank Adaptation)**
- **Metrics:** BLEU ↑, chrF ↑, TER ↓
- **Hardware:** 2× NVIDIA T4 GPUs (16 GB each)

---

## 2. English → French (EN→FR)

| Model                               | BLEU ↑    | chrF ↑    | TER ↓     |
| ----------------------------------- | --------- | --------- | --------- |
| Original mT5-small                  | 0.10      | 2.46      | 100.0     |
| mT5-small + Instruction Fine-tuning | 10.60     | 28.01     | 94.14     |
| **Approach 1 + Instruct-FT**        | **15.84** | 36.97     | **84.48** |
| **Approach 2 + Instruct-FT**        | 15.37     | **38.41** | 84.77     |

### Observations

- The **baseline (LoRA instruction tuning)** provides a major gain over the raw model.
- Both denoising strategies outperform the baseline by a wide margin.
- **Monolingual denoising (Approach 1)** achieves the best BLEU (+49% relative improvement), while **Bilingual denoising (Approach 2)** yields slightly higher chrF, indicating stronger morphological handling.

> ✅ **Approach 1 demonstrates the strongest improvement in syntactic and contextual reconstruction.**

---

## 3. French → English (FR→EN)

| Model                               | BLEU ↑    | chrF ↑    | TER ↓     |
| ----------------------------------- | --------- | --------- | --------- |
| Original mT5-small                  | 0.10      | 2.46      | 100.0     |
| mT5-small + Instruction Fine-tuning | 14.73     | 35.50     | 83.87     |
| **Approach 1 + Instruct-FT**        | **21.01** | **42.84** | **71.12** |
| **Approach 2 + Instruct-FT**        | 19.51     | 41.01     | 74.50     |

### Observations

- The **FR→EN** direction shows even stronger gains than EN→FR.
- **Approach 1 (Monolingual Denoising)** achieves the highest BLEU (21.01), a **42.6% relative improvement** over baseline.
- The lower TER indicates fewer edit operations, confirming enhanced fluency and accuracy in English generation.

> 💡 **Key Insight:** Strengthening the model’s understanding of English (target language) during pretraining leads to better translation generation.

---

## 4. English → Spanish (EN→ES)

| Model                               | BLEU ↑    | chrF ↑    | TER ↓     |
| ----------------------------------- | --------- | --------- | --------- |
| Original mT5-small                  | 0.08      | 3.42      | 100.0     |
| mT5-small + Instruction Fine-tuning | 16.51     | 33.52     | 81.59     |
| **Approach 1 + Instruct-FT**        | **21.59** | **38.05** | **75.99** |
| **Approach 2 + Instruct-FT**        | 18.27     | 33.56     | 79.95     |

### Observations

- Fine-tuning significantly boosts performance.
- **Approach 1** again yields the highest BLEU and lowest TER, confirming its generalizability beyond French.
- The consistent upward trend across metrics shows that denoising pretraining effectively scales to related Romance languages.

---

## 5. Spanish → English (ES→EN)

| Model                               | BLEU ↑    | chrF ↑    | TER ↓     |
| ----------------------------------- | --------- | --------- | --------- |
| Original mT5-small                  | 0.11      | 3.12      | 100.0     |
| mT5-small + Instruction Fine-tuning | 22.53     | 37.86     | 75.51     |
| **Approach 1 + Instruct-FT**        | **24.17** | **40.18** | **73.71** |
| **Approach 2 + Instruct-FT**        | 23.66     | 37.72     | 76.18     |

### Observations

- The results closely mirror FR→EN trends.
- **Approach 1** continues to outperform other methods with consistent gains across all metrics.
- Improvements in TER highlight enhanced syntactic fluency and reduced translation errors.

---

## 6. Average Performance Across 15 OPUS-100 Language Pairs

| Model Configuration          | Avg. Relative BLEU Gain (X→EN) |
| ---------------------------- | ------------------------------ |
| **Approach 1 + Instruct-FT** | **+38.5%**                     |
| **Approach 2 + Instruct-FT** | **+29.1%**                     |

### Observations

- **Approach 1** consistently provides superior translation quality across high- and low-resource pairs.
- Reinforcing the **target-language generative capacity** improves cross-lingual robustness in translation tasks.
- The consistency of performance gains demonstrates that **denoising pretraining generalizes well across diverse linguistic families**.

---

## 7. Low-Resource Language Evaluation

Evaluated pairs:  
**EN–HI (Hindi), EN–TA (Tamil), EN–SI (Sinhala), EN–UK (Ukrainian)**

### Summary of Findings

- Both denoising approaches outperform the instruction-tuned baseline.
- **Approach 1** achieves more stable and higher improvements in BLEU and chrF, especially for **translations into English**.
- This shows that reinforcing the target language (English) via monolingual denoising benefits even under limited parallel data.
- **Approach 2** also yields improvements, though the complexity of dual objectives makes it less consistent in extremely low-resource scenarios.

> 🌍 **Conclusion:** Denoising pretraining is a highly effective and resource-efficient method for boosting translation performance in low-resource environments.

---

## 8. Discussion

### (a) Why Monolingual Denoising Excels

Monolingual denoising (EN noisy → EN) strengthens the **target language’s generative capabilities**, improving grammatical and contextual coherence.  
When later fine-tuned for translation, the model can focus on mapping from the source language, leveraging its already strong target language model.

### (b) Limitations of Bilingual Denoising

Bilingual denoising combines noise handling with translation. However, this dual learning burden makes training less efficient and slightly less fluent in target-language generation, especially when translating into English.

### (c) Impact for Low-Resource Adaptation

This study provides a **computationally feasible path** for translation improvement using:

- Small models (mT5-small)
- Short pretraining cycles (3 epochs)
- Minimal hardware (T4 GPU)

This two-step process—**Monolingual Denoising → LoRA Fine-tuning**—offers a low-cost yet high-impact pipeline for real-world applications.

### (d) Limitations

- Only **mT5-small** was evaluated; results may scale further with mT5-base or large.
- The **noise injection** function used is simple (deletion, swapping) — future work can explore **span corruption**.
- Training was conducted on a limited subset (100k samples per pair) for feasibility.
- Only LoRA was used for PEFT; adapters and prefix tuning could be tested in future work.

---

## 9. Conclusion

This research successfully demonstrated that **denoising pretraining** significantly enhances **multilingual translation performance** when applied as an **intermediate adaptation step** for mT5.  
Key findings include:

- **Approach 1 (Monolingual Denoising)** outperformed all baselines and alternatives, especially in **FR→EN** translation, where BLEU improved from 14.73 → **21.01** (+42.6%).
- Similar improvements were observed across **EN–ES** and **15 total language pairs**, averaging **+38.5% BLEU gain**.
- The results validate denoising pretraining as an **effective, scalable, and resource-efficient** enhancement for multilingual models.
- Combining **denoising pretraining** with **LoRA-based instruction fine-tuning** achieves high-quality translation on modest hardware, bridging the performance gap between high- and low-resource languages.

---

## 10. Future Directions

- Integrate **advanced corruption functions** like span masking (as in T5).
- Extend monolingual denoising to **non-English target languages** (e.g., FR noisy → FR).
- Explore **adapter-based PEFT** methods for comparative efficiency.
- Conduct large-scale evaluation on full OPUS-100 corpus for cross-domain generalization.

---

## 11. Publicly Released Models

| Pair  | Model Link                                                                                                                        | Description                              |
| ----- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| EN–FR | [mt5-small-denoising-en-fr](https://huggingface.co/Eshan210352R/mt5-small-denoising-en-fr-correct-deonoise-lora-instruct-ft-enfr) | Monolingual Denoising + LoRA Fine-tuning |
| EN–ES | [mt5-small-denoising-en-es-final](https://huggingface.co/Eshan210352R/mt5-small-denoising-en-es-final)                            | Monolingual Denoising + LoRA Fine-tuning |
| EN–IT | [mt5-span-denoising-en-it-final](https://huggingface.co/Eshan210352R/mt5-span-denoising-en-it-final)                              | Span-Denoising Variant                   |
| EN–SI | [mt5-small-denoising-en-si-final](https://huggingface.co/Eshan210352R/mt5-small-denoising-en-si-final)                            | Low-Resource Adaptation                  |
| EN–TA | [mt5-small-denoising-en-ta-final](https://huggingface.co/Eshan210352R/mt5-small-denoising-en-ta-final)                            | Dravidian Language Pair                  |
| EN–ZH | [mt5-small-denoising-en-zh-final](https://huggingface.co/Eshan210352R/mt5-small-denoising-en-zh-final)                            | Chinese Translation Model                |

Additional pretrained models for all 15 OPUS-100 pairs are available on [Hugging Face – Eshan210352R](https://huggingface.co/Eshan210352R).

---

> **Final Remark:**  
> _This work demonstrates that strengthening the target language through denoising pretraining is a powerful catalyst for improving multilingual translation accuracy — making high-quality, low-resource MT both feasible and accessible._
