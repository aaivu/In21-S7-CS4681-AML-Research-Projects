# Research Proposal: Recommendation Systems: Collaborative Filtering

**Student:** 210401T  
**Research Area:** Recommendation Systems: Collaborative Filtering  
**Date:** 2025-10-21  

---

## Abstract

This research proposal focuses on improving the performance, scalability, and interpretability of **sequential recommendation systems**, specifically extending the **Self-Attentive Sequential Recommendation (SASRec)** model. Traditional collaborative filtering and matrix factorization approaches capture only static user preferences and fail to model temporal dynamics. SASRec introduced a breakthrough by leveraging self-attention mechanisms to capture both short- and long-term dependencies efficiently. However, limitations remain in terms of context-awareness, computational cost, and robustness.  
To address these challenges, this project proposes **SASRec++**, an enhanced version that integrates time-interval and session-aware contextual embeddings, introduces a sparse attention mechanism for scalability, and employs advanced regularization for improved generalization. The enhanced model will be evaluated on benchmark datasets such as MovieLens-1M and Amazon Beauty using metrics like HR@K and NDCG@K. The expected outcome is a context-aware, computationally efficient, and interpretable sequential recommender that outperforms the SASRec baseline.

---

## 1. Introduction

Recommender systems are fundamental to modern digital platforms, providing personalized experiences by predicting user preferences. Among various approaches, **sequential recommendation** stands out as it captures the temporal order of user interactions rather than treating preferences as static.  
Traditional models like Matrix Factorization and collaborative filtering fail to handle evolving user behavior effectively. Sequential models such as **Markov Chains** and **Recurrent Neural Networks (RNNs)** improve this by modeling transitions between items but struggle with long-term dependencies and training inefficiencies.  
The **Transformer architecture** revolutionized sequence modeling through **self-attention**, enabling the **SASRec** model to efficiently capture both short- and long-term patterns. This proposal extends that line of work, aiming to further enhance SASRec’s contextual awareness and scalability.

---

## 2. Problem Statement

While SASRec achieves strong performance, it has notable limitations:
- It lacks explicit modeling of **temporal gaps** and **session context**, reducing its ability to understand contextual user behavior.
- The **quadratic complexity** of self-attention limits scalability for long sequences.
- Its robustness and interpretability can be further improved through advanced regularization and visualization.  

This research aims to address these limitations by developing a **context-aware, sparse-attention variant** of SASRec that maintains accuracy while improving computational efficiency and explainability.

---

## 3. Literature Review Summary

Sequential recommendation research evolved from **Markov Chain** and **RNN-based** models to attention-based architectures.  
- **FPMC** modeled short-term dependencies but struggled with scalability.  
- **GRU4Rec** leveraged recurrent units but faced parallelization challenges.  
- **SASRec** [Kang & McAuley, 2018] introduced self-attention for sequential recommendation, outperforming RNNs by modeling global and local dependencies simultaneously.  
Recent works such as **TiSASRec** (Li et al., 2020) and **Informer** (Zhou et al., 2021) incorporated temporal embeddings and sparse attention to improve performance.  
However, there remains a research gap in combining **context-awareness (time/session)** with **sparse attention mechanisms** while preserving interpretability — a gap this research seeks to fill.

---

## 4. Research Objectives

### Primary Objective
To develop an enhanced **SASRec++** model that integrates contextual embeddings and sparse attention for improved efficiency, accuracy, and interpretability in sequential recommendation.

### Secondary Objectives
- Implement the baseline SASRec model for reference and benchmarking.  
- Introduce **time-interval** and **session-based embeddings** to improve context-awareness.  
- Employ **sparse attention mechanisms** to reduce computational complexity.  
- Integrate **label smoothing** and **dropout** regularization for improved robustness.  
- Conduct comprehensive experiments on benchmark datasets using HR@K and NDCG@K metrics.  
- Analyze attention heatmaps for interpretability insights.

---

## 5. Methodology

The methodology consists of two main stages — **baseline replication** and **enhancement implementation**.

### 5.1 Baseline Implementation
The original SASRec architecture will be reproduced with:
- Two Transformer blocks, embedding size of 50, and two attention heads.  
- Adam optimizer (learning rate = 0.001), dropout regularization, and positional encodings.  
- Datasets: MovieLens-1M (dense) and Amazon Beauty (sparse).  
This ensures a validated and comparable baseline.

### 5.2 Enhancement Implementation
The enhanced model, SASRec++, introduces:
- **Context-Aware Embeddings:** Integrating time-interval and session information into the input representation to capture temporal and contextual patterns.  
- **Sparse Attention Mechanism:** Reducing the quadratic attention cost by attending only to local windows and global positions, improving scalability.  
- **Regularization Improvements:** Label smoothing and dropout in attention/FFN layers to prevent overfitting.  
- **Interpretability:** Visualizing attention weights as heatmaps to analyze which past interactions influence next-item predictions.

### 5.3 Experimental Design
- **Evaluation Metrics:** HR@K and NDCG@K for K = 10.  
- **Efficiency Metrics:** Training time per epoch and GPU memory usage.  
- **Statistical Validation:** Paired t-tests (p < 0.05) to confirm significance of improvements.  
All experiments will be conducted using GPU-accelerated environments such as Kaggle.

---

## 6. Expected Outcomes

- A **context-aware sequential recommendation system** (SASRec++) that effectively models temporal and session dynamics.  
- **Reduced computational complexity** through sparse attention mechanisms.  
- **Improved generalization** via label smoothing and dropout regularization.  
- **Enhanced interpretability** through attention heatmap visualization.  
Overall, the project aims to advance the scalability, accuracy, and explainability of self-attention-based recommendation models.

---

## 7. Timeline

| Week | Task |
|------|------|
| 1–2  | Literature Review |
| 3–4  | Baseline SASRec Implementation |
| 5–8  | SASRec++ Model Development |
| 9–12 | Experiments and Evaluation |
| 13–15| Result Analysis and Documentation |
| 16   | Final Submission |

---

## 8. Resources Required

- **Datasets:** MovieLens-1M, Amazon Beauty  
- **Hardware:** GPU-enabled environment (e.g., Kaggle or local RTX GPU)  
- **Software:** Python 3.10, PyTorch ≥ 2.0, CUDA 12+, NumPy, Pandas, Matplotlib  
- **Tools:** Git, Jupyter Notebook, and visualization utilities for attention analysis  

---

## References

1. W.-C. Kang and J. McAuley, *Self-Attentive Sequential Recommendation*, arXiv:1808.09781, 2018.  
2. F. Ricci et al., *Recommender Systems Handbook*, Springer US, 2011.  
3. B. Hidasi et al., *Session-Based Recommendations with Recurrent Neural Networks*, ICLR, 2016.  
4. A. Vaswani et al., *Attention Is All You Need*, NeurIPS, 2017.  
5. Y. Koren, *Collaborative Filtering with Temporal Dynamics*, Communications of the ACM, 2010.  
6. S. Rendle et al., *Factorizing Personalized Markov Chains for Next-Basket Recommendation*, WWW, 2010.  
7. D. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR, 2015.  
8. Y. Li et al., *Lightweight Self-Attentive Sequential Recommendation*, arXiv:2108.11333, 2021.  
9. H. Chen et al., *Denoising Self-Attentive Sequential Recommendation*, arXiv:2212.04120, 2022.  
10. P. Zhou et al., *Attention Calibration for Transformer-Based Sequential Recommendation (AC-TSR)*, arXiv:2308.09419, 2023.  
11. C. Wu et al., *Hyperbolic Self-Attention for Sequential Recommendation*, arXiv, 2024.  
12. J. Chiang, *Self-Attention on Recommendation System — Self-Attentive Sequential Recommendation Review*, Medium, 2024.  

---

**Submission Instructions:**  
1. Complete all sections above.  
2. Commit your changes to the repository.  
3. Create an issue with labels **"milestone"** and **"research-proposal"**.  
4. Tag your supervisors in the issue for review.  
