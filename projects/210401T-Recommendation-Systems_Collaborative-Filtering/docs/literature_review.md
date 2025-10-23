# Literature Review: Recommendation Systems – Collaborative Filtering

**Student:** 210401T  
**Research Area:** Recommendation Systems: Collaborative Filtering  
**Date:** 2025-10-21

## Abstract

This literature review explores the evolution of recommendation systems with a focus on collaborative filtering and its extension to sequential recommendation models. Traditional collaborative filtering and matrix factorization methods effectively capture static user preferences but fail to consider temporal dynamics. Recent advances, particularly self-attention-based architectures like SASRec, have enabled modeling of both short- and long-term user behavior. The review examines the development from Markov-based and recurrent models to transformer-based architectures, emphasizing methodological innovations, interpretability, and computational efficiency. Identified gaps include scalability in long sequences, cold-start handling, and hybridization with contextual data, which form the foundation for future research directions.

---

## 1. Introduction

Recommendation systems aim to predict user preferences by leveraging past behavior, playing a central role in e-commerce, streaming, and social platforms. Among various approaches, **collaborative filtering (CF)** remains a cornerstone, exploiting user–item interaction patterns. However, traditional CF and matrix factorization models capture only *static* preferences, overlooking sequential dynamics inherent in user behavior.  
Sequential recommendation systems emerged to model *temporal order* and *evolving interests*, using models like Markov Chains, RNNs, and most recently, Transformers. This review focuses on the transition from conventional CF to self-attention-based sequential recommendation models, emphasizing SASRec as a representative framework.

---

## 2. Search Methodology

### Search Terms Used
- “Collaborative filtering”
- “Sequential recommendation systems”
- “Attention-based recommendation”
- “Transformer recommender models”
- “SASRec model”
- “Self-attention recommender”
- Synonyms: “user-item prediction”, “temporal recommendation”, “next-item prediction”

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2018–2024 (with inclusion of seminal works prior to 2018)

---

## 3. Key Areas of Research

### 3.1 Evolution of Sequential Recommendation Models

Early sequential models were based on **Markov Chains**, modeling item-to-item transition probabilities [6]. While effective for short sequences, they struggled to represent long-term dependencies.  
**Recurrent Neural Networks (RNNs)**, particularly GRU4Rec [3], introduced hidden states to retain longer contextual information. However, their sequential computation caused training inefficiencies and limited parallelism.  
The introduction of **Transformer architectures** [4] revolutionized sequence modeling by replacing recurrence with *self-attention mechanisms*, enabling global dependency modeling and parallel computation. This paved the way for attention-based sequential recommender systems such as SASRec [1].

**Key Papers:**
- [1] Kang & McAuley (2018) – Introduced SASRec, the first self-attention-based sequential recommender.
- [3] Hidasi et al. (2016) – Proposed GRU4Rec, applying RNNs for session-based recommendations.
- [4] Vaswani et al. (2017) – Introduced the Transformer architecture.
- [6] Rendle et al. (2010) – Proposed FPMC, integrating Markov Chains with matrix factorization.

---

### 3.2 SASRec Model and Methodology

SASRec applies **self-attention mechanisms** to model sequential dependencies in user-item interactions.  
Each user’s interaction sequence is embedded using an **item embedding matrix** combined with **positional embeddings** to preserve temporal order. Zero-padding and masking handle variable-length sequences [12].

The model architecture includes **stacked self-attention blocks** with multi-head attention, causal masking, residual connections, and feed-forward networks. Attention layers dynamically capture local and global dependencies, offering interpretability through attention visualization [12].

**Output and Prediction:**  
The representation of the last non-padded position is projected to predict the next likely item using a softmax layer. Training employs **binary cross-entropy loss** with negative sampling and **Adam optimization** [1].

**Computational Efficiency:**  
Unlike RNNs, SASRec performs all operations in parallel, yielding faster training despite quadratic attention complexity. Truncation strategies mitigate performance trade-offs in long sequences [12].

**Key Papers:**
- [1] Kang & McAuley (2018) – SASRec: Self-Attentive Sequential Recommendation.
- [4] Vaswani et al. (2017) – Transformer architecture fundamentals.
- [12] Medium Review (2019) – Interpretability analysis of SASRec attention mechanisms.

---

## 4. Research Gaps and Opportunities

### Gap 1: Scalability in Long Sequences
**Why it matters:** SASRec’s attention complexity grows quadratically with sequence length, limiting scalability in real-time environments.  
**How your project addresses it:** Investigate truncated or sparse attention variants to improve efficiency without losing predictive power.

### Gap 2: Cold-Start and Sparse Data Limitations
**Why it matters:** Attention-based models rely on interaction history, struggling with new users/items.  
**How your project addresses it:** Integrate contextual and content-based features to support hybrid recommendation in sparse data settings.

### Gap 3: Interpretability and Evaluation Metrics
**Why it matters:** While attention visualizations improve interpretability, consistent quantitative evaluation is lacking.  
**How your project addresses it:** Employ explainability metrics and visualization tools for interpretable recommendation outputs.

---

## 5. Theoretical Framework

The theoretical foundation draws from **collaborative filtering** (user-item interaction modeling), **sequence modeling** (temporal order via deep learning), and **attention mechanisms** (context weighting).  
SASRec’s framework builds on **representation learning theory**, where item embeddings and positional encodings jointly capture the semantics of user behavior sequences.

---

## 6. Methodology Insights

Common methodologies include:
- **Embedding-based representation learning**
- **Self-attention for dependency modeling**
- **Negative sampling with binary cross-entropy loss**
- **Evaluation using HR@K and NDCG@K metrics**

Promising directions involve hybrid models combining *self-attention* with *graph neural networks* or *reinforcement learning* to further enhance adaptability and personalization.

---

## 7. Conclusion

Sequential recommendation research has evolved from simple transition models to sophisticated attention-based architectures. SASRec represents a key milestone, balancing interpretability, accuracy, and computational efficiency. However, challenges remain in scalability, cold-start handling, and multi-modal integration. Addressing these gaps will be crucial for developing next-generation recommender systems that can adapt to dynamic user behavior and diverse data modalities.

---

## References

1. Kang, W., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. IEEE ICDM.  
2. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. Computer, 42(8).  
3. Hidasi, B., et al. (2016). *Session-based Recommendations with Recurrent Neural Networks*. ICLR.  
4. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
5. He, X., et al. (2017). *Neural Collaborative Filtering*. WWW.  
6. Rendle, S., et al. (2010). *Factorizing Personalized Markov Chains for Next-Basket Recommendation*. WWW.  
7. Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.  
8. Chen, J., et al. (2022). *A Survey on Sequential Recommendation: Models, Methods, and Metrics*. ACM Computing Surveys.  
9. Tang, J., et al. (2019). *Deep Sequential Recommendation: A Survey*. IEEE TKDE.  
10. Wu, C., et al. (2020). *Self-Attention-based Sequential Recommendation: An Empirical Study*. arXiv preprint arXiv:2007.14235.  
11. Zhang, S., et al. (2019). *Deep Learning based Recommender System: A Survey and New Perspectives*. ACM Computing Surveys.  
12. Medium Review (2019). *SASRec: Self-Attentive Sequential Recommendation Explained*. Medium.

---
