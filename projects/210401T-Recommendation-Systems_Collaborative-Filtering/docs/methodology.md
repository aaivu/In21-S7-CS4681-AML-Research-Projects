# Methodology: Recommendation Systems: Collaborative Filtering

**Student:** 210401T  
**Research Area:** Recommendation Systems: Collaborative Filtering  
**Date:** 2025-10-20  

---

## 1. Overview

We propose **SASRec++**, a *context-aware, sparse-attention* extension of the SASRec model. This architecture enhances traditional sequential recommendation by incorporating **time-interval bucket embeddings** and **session embeddings**, along with a **local–global sparse attention mechanism** that improves efficiency and interpretability.  

The model follows a **causal next-item prediction objective**, trained with **weight tying** and **label smoothing**. Evaluation is conducted under both *full-catalog* and *sampled-negative* ranking protocols. Our design builds on established work in self-attention (Vaswani et al., 2017; Kang & McAuley, 2018), time-aware self-attention (Li et al., 2020), and structured sparsity (Beltagy et al., 2020; Zaheer et al., 2020; Zhou et al., 2021).

---

## 2. Research Design

The overall design focuses on **enhancing sequential recommendation** through contextual and efficiency-driven modifications to the Transformer-based SASRec model. We aim to:  

- Capture **temporal dynamics** via time-interval bucketing.  
- Incorporate **session context** into sequence modeling.  
- Reduce **computational complexity** using sparse local–global attention patterns.  
- Maintain interpretability through **attention visualization**.  

---

## 3. Data Collection

### 3.1 Data Sources
- Public benchmark datasets for sequential recommendation (e.g., Amazon Reviews, MovieLens, or YooChoose).  
- Each dataset provides **user–item interaction histories**, **timestamps**, and **session identifiers** where available.

### 3.2 Data Description
Each dataset is represented as a set of user sequences  
\[
S_u = (i_1, i_2, \dots, i_L)
\]  
where each \( i_t \) is an interacted item. Datasets vary in sparsity and average sequence length. When available, **timestamps** enable time-interval computation, and **session IDs** support session-level context modeling.

### 3.3 Data Preprocessing
- Convert raw logs into ordered user–item sequences.  
- Quantize time intervals into *log-scaled buckets* (see Eq. below).  
- Apply padding and masking for variable-length sequences.  
- Split data into train/validation/test sets following the *leave-one-out* strategy.  
- Convert item IDs to continuous embeddings; apply 0-based indexing for model alignment.

---

## 4. Model Architecture

### 4.1 Input Encoding
For a user sequence \( \mathbf{s} = (i_1, \ldots, i_L) \), each token \( i_t \) is represented as:
\[
\mathbf{x}_t = \mathbf{e}_{i_t} + \mathbf{p}_t + \mathbf{g}_{b_t} + \mathbf{h}_{c_t}
\]
where:
- \( \mathbf{e}_{i_t} \): item embedding  
- \( \mathbf{p}_t \): positional embedding  
- \( \mathbf{g}_{b_t} \): time-interval bucket embedding  
- \( \mathbf{h}_{c_t} \): session embedding  

If time or session information is unavailable, those components are omitted.

### 4.2 Local–Global Sparse Self-Attention
To address the quadratic cost of dense attention, we use a **causal sparse pattern** combining a local sliding window and global pivots:
\[
\mathcal{A}(i) = \{ j \mid i - w \le j \le i \} \cup (\{1,\ldots,i\} \cap \mathcal{P}_{\text{global}})
\]
where \( w \) is the local window size and global positions occur at a stride \( s \).  
This pattern ensures efficient modeling of both **short-term** and **long-term** dependencies with a reduced complexity of \( \mathcal{O}(Lw) \).

Each self-attention block includes:
- Multi-head attention with causal masking  
- Residual connections  
- Layer normalization  
- Position-wise feed-forward networks  

Dropout is applied to embeddings, attention probabilities, and FFN layers.

### 4.3 Next-Item Prediction with Weight Tying
The final hidden state \( \mathbf{h}_{t^*} \) of the last non-padded position is projected into the item space using **weight tying**:
\[
\boldsymbol{\ell} = \mathbf{h}_{t^*} E^\top
\]
where \( E \) is the shared item embedding matrix. This ensures parameter efficiency and consistent representation learning between input and output layers.

### 4.4 Objective and Label Smoothing
We optimize the cross-entropy loss with mild label smoothing:
\[
q_k =
\begin{cases}
1 - \varepsilon, & k = y \\
\varepsilon / (|\mathcal{I}| - 1), & k \neq y
\end{cases}
\]
\[
\mathcal{L}_{CE} = -\sum_k q_k \log p_\theta(k \mid \mathbf{s}_{\le t^*})
\]
Optimization is performed using **Adam** with appropriate learning rate scheduling and regularization.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **Hit Rate (HR@K)**
- **Normalized Discounted Cumulative Gain (NDCG@K)**  
Metrics are computed under both:
- **Full-catalog evaluation:** ranking against all items.  
- **Sampled-negative evaluation:** ranking against 1 positive + N random negatives.

### 5.2 Baseline Models
We compare SASRec++ against:
- **SASRec** (Transformer-based baseline)  
- **TiSASRec** (Time-aware SASRec)  
- **GRU4Rec** (RNN-based sequential recommender)  
- **FPMC** (Markov-based baseline)  

### 5.3 Hardware/Software Requirements
- **Hardware:** NVIDIA GPU with ≥ 12 GB VRAM (e.g., RTX 3060 or higher)  
- **Software:**  
  - Python 3.10  
  - PyTorch ≥ 2.0  
  - CUDA 12+  
  - NumPy, Pandas, Matplotlib for preprocessing and visualization

---

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Cleaned and preprocessed dataset |
| Phase 2 | Model implementation | 3 weeks | SASRec++ model with training pipeline |
| Phase 3 | Experiments | 2 weeks | Evaluation results (HR@K, NDCG@K) |
| Phase 4 | Analysis | 1 week | Interpretability analysis and final report |

---

## 7. Risk Analysis

| Risk | Description | Mitigation |
|------|--------------|-------------|
| Data sparsity | Some datasets have very few interactions per user | Use sequence truncation and augmentation |
| Computational overhead | Attention layers can be memory-intensive | Employ sparse attention and smaller window sizes |
| Overfitting | Deep models may memorize frequent sequences | Apply dropout, early stopping, and label smoothing |
| Evaluation bias | Sampled metrics may misestimate rank quality | Use both full-catalog and sampled evaluations |

---

## 8. Expected Outcomes

- **Enhanced recommendation accuracy** by integrating temporal and session contexts.  
- **Improved scalability** through sparse attention mechanisms.  
- **Greater interpretability** via attention heatmaps showing item and session dependencies.  
- **Benchmark-level performance** surpassing baseline models such as SASRec and GRU4Rec.  

The final deliverable is a **context-aware sequential recommender system** that efficiently models long-term dependencies while remaining computationally feasible and interpretable.

---

### References (optional)
- Vaswani et al., *Attention is All You Need*, 2017  
- Kang & McAuley, *Self-Attentive Sequential Recommendation*, 2018  
- Li et al., *Time Interval Aware Self-Attention for Sequential Recommendation*, 2020  
- Beltagy et al., *Longformer: The Long-Document Transformer*, 2020  
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, 2020  
- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, 2021  
