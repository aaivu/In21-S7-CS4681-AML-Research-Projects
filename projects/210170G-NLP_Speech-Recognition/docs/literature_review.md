# Literature Review: NLP: Speech Recognition

**Student:** 210170G  
**Research Area:** NLP: Speech Recognition  
**Date:** 2025-09-01  

---

## Abstract

Pretrained speech representation models like **Wav2Vec2** have revolutionized automatic speech recognition (ASR) through self-supervised learning. However, their performance drops in **low-resource** or **domain-shifted** scenarios such as minority languages, child speech, and animal sounds. This review examines key developments in self-supervised speech models, efficiency techniques (e.g., quantization and adapters), and domain adaptation challenges. The literature reveals ongoing gaps in efficient adaptation and low-resource optimization, motivating the exploration of **Inter-Codebook Similarity Loss (ICSL)** and **Residual Vector Quantization (RVQ)** for enhancing training efficiency and adaptability.

---

## 1. Introduction

Speech recognition has advanced rapidly with **self-supervised learning (SSL)** frameworks that leverage vast unlabeled audio datasets. **Wav2Vec2.0**, introduced by Baevski et al. (2020), serves as a milestone model, achieving state-of-the-art results in high-resource languages.  
However, applications such as **low-resource language recognition**, **child voice recognition**, and **animal vocalization detection** remain challenging due to data scarcity and domain differences. Existing approaches like fine-tuning and transfer learning struggle to generalize effectively under these conditions. This literature review surveys prior research addressing these challenges and identifies potential strategies for enhancing efficiency and adaptability in SSL-based speech models.

---

## 2. Search Methodology

### Search Terms Used
- “Speech recognition,” “self-supervised learning,” “low-resource ASR,” “Wav2Vec2,”  
- “Residual vector quantization,” “speech domain adaptation,” “ICSL,”  
- “HuBERT,” “Data2Vec,” “WavLM,” “efficient fine-tuning,” “quantized neural networks”

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2018–2024 (focusing on recent developments in SSL-based speech models)

---

## 3. Key Areas of Research

### 3.1 Self-Supervised Speech Representation Learning

Self-supervised learning has enabled speech models to pretrain on unlabeled audio using contrastive or masked prediction objectives.

**Key Papers:**
- **Baevski et al. (2020)** – Introduced *Wav2Vec2.0*, combining contrastive prediction and vector quantization for rich contextual representations.  
- **Hsu et al. (2021)** – Proposed *HuBERT*, leveraging iterative pseudo-labeling to improve phonetic structure learning.  
- **Baevski et al. (2022)** – Presented *Data2Vec*, unifying SSL across speech, vision, and text.  
- **Chen et al. (2022)** – Developed *WavLM*, incorporating denoising and speaker-aware modeling for robust representations.

These models significantly improved downstream ASR performance but remain limited in adapting to unseen or low-resource domains.

---

### 3.2 Low-Resource and Domain Adaptation Challenges

Fine-tuning large SSL models on small datasets often leads to overfitting and poor adaptation due to acoustic mismatches.

**Key Papers:**
- **Yu et al. (2020)** – Reviewed deep learning methods for low-resource ASR, highlighting data scarcity and transfer inefficiencies.  
- **Babu et al. (2021)** – Proposed *XLS-R*, a cross-lingual Wav2Vec2 trained on 128 languages, showing promise but requiring massive resources.  
- **Reitmaier et al. (2022)** – Discussed accessibility and fairness issues in low-resource ASR technologies.

Despite progress, efficient domain adaptation for low-resource scenarios remains an open problem.

---

### 3.3 Efficient Model Training Techniques

Recent research emphasizes reducing computational cost while maintaining performance.

**Key Papers:**
- **Hinton et al. (2015)** – Introduced *knowledge distillation*, enabling smaller models to mimic large teacher networks.  
- **Chang et al. (2022)** – *DistilHuBERT*: distilled HuBERT representations for compact speech models.  
- **Thomas et al. (2022)** – *Speech Adapter*: parameter-efficient fine-tuning with adapters.  
- **Zaken et al. (2021)** – *BitFit*: fine-tuning only bias parameters for minimal training overhead.  
- **Guo (2018)** – Surveyed quantized neural networks, motivating quantization-based efficiency strategies.

---

### 3.4 Quantization and Residual Vector Quantization (RVQ)

**Residual Vector Quantization (RVQ)** enhances standard quantization by sequentially refining encoded residuals through multiple codebooks, enabling high-fidelity, compact representations.

**Key Papers:**
- **Zeghidour et al. (2021)** – *SoundStream*: introduced hierarchical RVQ for neural audio compression.  
- **Lugo & Vielzeuf (2021)** – Explored efficient SSL pretraining for speech models under constrained data.

RVQ can capture fine-grained details, making it well-suited for adaptation in low-resource ASR contexts.

---

## 4. Research Gaps and Opportunities

### Gap 1: Inefficient Adaptation of SSL Models in Low-Resource Domains
**Why it matters:** Fine-tuning large SSL models often fails to generalize in small or domain-shifted datasets.  
**How your project addresses it:** Proposes **Residual Vector Quantization (RVQ)** to improve adaptability and accelerate convergence with limited data.

### Gap 2: Redundant Representations in Multi-Codebook SSL Models
**Why it matters:** Codebook redundancy in quantized models reduces representational diversity.  
**How your project addresses it:** Introduces **Inter-Codebook Similarity Loss (ICSL)** to penalize redundant embeddings and promote richer feature learning.

---

## 5. Theoretical Framework

The research builds on the **self-supervised learning (SSL)** framework using **contrastive predictive coding** and **vector quantization**.  
The **ICSL** regularizer extends the diversity loss in Wav2Vec2 by minimizing cosine similarity across multiple codebooks, encouraging distinct feature learning.  
**RVQ**, grounded in hierarchical quantization theory, enhances latent representation granularity during fine-tuning.

---

## 6. Methodology Insights

Common methodologies in recent literature include:
- **Contrastive learning** (e.g., Wav2Vec2, CPC)
- **Masked prediction** (e.g., HuBERT, WavLM)
- **Fine-tuning with CTC loss**
- **Parameter-efficient adaptation** (Adapters, BitFit)
- **Quantization and compression** (SoundStream, RVQ)

For this research, **ICSL** was applied during pretraining, and **RVQ** was integrated during fine-tuning on a **10-hour LibriSpeech subset**, evaluated via **CTC Loss** and **Word Error Rate (WER)**.

---

## 7. Conclusion

Literature in speech recognition shows strong progress in SSL, yet challenges remain in **efficient adaptation** and **low-resource generalization**.  
This study contributes by integrating **ICSL** to enhance pretraining diversity and **RVQ** to improve fine-tuning efficiency. Together, they represent a practical step toward making **ASR more accessible, data-efficient, and robust** across diverse domains.

---

## References

1. Baevski, A. et al. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations.*  
2. Reitmaier, T. et al. (2022). *Opportunities and challenges of ASR for low-resource language speakers.*  
3. Lugo, L., & Vielzeuf, V. (2021). *Towards efficient self-supervised learning in speech processing.*  
4. Schneider, S. et al. (2019). *Wav2Vec: Unsupervised pre-training for speech recognition.*  
5. van den Oord, A. et al. (2018). *Contrastive Predictive Coding.*  
6. Hsu, W.-N. et al. (2021). *HuBERT: Self-supervised speech representation learning by masked prediction.*  
7. Baevski, A. et al. (2022). *Data2Vec: A general framework for SSL in speech, vision, and language.*  
8. Chen, S. et al. (2022). *WavLM: Large-scale pre-training for speech processing.*  
9. Yu, C. et al. (2020). *Deep learning for low-resource speech recognition: An overview.*  
10. Babu, A. et al. (2021). *XLS-R: Cross-lingual speech representation learning at scale.*  
11. Hinton, G. et al. (2015). *Distilling the knowledge in a neural network.*  
12. Chang, H.-J. et al. (2022). *DistilHuBERT: Speech representation learning by layer-wise distillation.*  
13. Wang, R. et al. (2022). *LightHuBERT: Lightweight and configurable SSL for speech.*  
14. Thomas, B. et al. (2022). *Efficient adapter transfer of SSL speech models.*  
15. Zaken, E. et al. (2021). *BitFit: Simple parameter-efficient fine-tuning for transformers.*  
16. Guo, Y. (2018). *A survey on methods and theories of quantized neural networks.*  
17. Zeghidour, N. et al. (2021). *SoundStream: An end-to-end neural audio codec.*  
18. Panayotov, V. et al. (2015). *LibriSpeech: An ASR corpus based on audiobooks.*

---
