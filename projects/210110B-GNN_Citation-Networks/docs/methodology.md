# Methodology: GNN:Citation Networks

**Student:** 210110B  
**Research Area:** GNN:Citation Networks  
**Date:** 2025-09-10  

---

## 1. Overview  

This methodology outlines the design and implementation of an enhanced heterogeneous graph neural network (H-UniMP++) for citation network classification. The model extends UniMP and R-UniMP by incorporating relation-aware propagation, uncertainty-gated label injection, and curriculum masking. The goal is to improve robustness and accuracy in heterogeneous citation networks under realistic constraints (limited compute and noisy labels).  

---

## 2. Research Design  

The research follows an **iterative experimental design**:  
1. **Baseline Replication:** Implement a UniMP-lite variant for homogeneous graphs.  
2. **Heterogeneous Extension:** Extend to R-UniMP-style relation-aware propagation.  
3. **Proposed Enhancement (H-UniMP++):** Add uncertainty-gated label injection and curriculum masking.  
4. **Evaluation:** Compare with baselines on citation network datasets (MAG240M-LSC, DBLP, ogbn-arxiv).  

This design allows clear attribution of performance gains at each stage.  

---

## 3. Data Collection  

### 3.1 Data Sources  
- **MAG240M-LSC (KDD Cup 2021)** – Large-scale heterogeneous academic graph.  
- **DBLP-V12** – Citation + authorship dataset.  
- **OGB Benchmarks** – ogbn-arxiv, ogbn-products, ogbn-mag.  

### 3.2 Data Description  
- **Nodes:** Papers, authors, venues.  
- **Edges:** Citation (paper → paper), authorship (author ↔ paper).  
- **Features:** Title/abstract embeddings (768-dim), metapath2vec embeddings, handcrafted features (year, author count).  
- **Labels:** Paper venue/conference (classification task).  

### 3.3 Data Preprocessing  
- Parse raw text datasets (titles, authors, years, citations).  
- Build heterogeneous graph structures (paper-paper, author-paper, paper-author).  
- Extract features (pretrained embeddings + handcrafted).  
- Split into train/validation/test (70/15/15).  
- Normalize features and encode labels.  

---

## 4. Model Architecture  

The **H-UniMP++ model** consists of:  
1. **Input Layer:**  
   - Paper features (768-dim).  
   - Metapath2vec embeddings projected to feature dimension.  
   - Label embeddings (with masking).  

2. **Relation-Aware Propagation:**  
   - Separate linear projections for each edge type.  
   - Relation-wise BatchNorm and dropout.  
   - Aggregation across relation types.  

3. **Uncertainty-Gated Label Injection (UGLI):**  
   - Gate `g ∈ [0,1]` modulates contribution of injected labels.  
   - Learned jointly to reduce noise sensitivity.  

4. **Curriculum Masking:**  
   - Masking rate increases with epochs (easy → hard).  
   - Encourages gradual learning of robust features.  

5. **Prediction Head:**  
   - Residual MLP layers.  
   - Softmax over venue classes.  

---

## 5. Experimental Setup  

### 5.1 Evaluation Metrics  
- Accuracy (top-1)  
- Macro-F1, Micro-F1  
- Training stability (loss curves)  
- Robustness to label noise (noisy label ablations)  

### 5.2 Baseline Models  
- GCN (Kipf & Welling, 2017)  
- GAT (Velickovic et al., 2018)  
- R-GCN (Schlichtkrull et al., 2018)  
- UniMP (Shi et al., 2020)  
- R-UniMP (Shi et al., 2021)  

### 5.3 Hardware/Software Requirements  
- **Hardware:** MacBook M2 (CPU-only), Kaggle T4 (GPU, 16GB)  
- **Software:**  
  - Python 3.10+  
  - PaddlePaddle 2.6+  
  - PGL (Paddle Graph Learning)  
  - Numpy, Scipy, Scikit-learn  
  - TensorBoardX for logging  

---

## 6. Implementation Plan  

| Phase   | Tasks                                | Duration | Deliverables              |
|---------|--------------------------------------|----------|---------------------------|
| Phase 1 | Data preprocessing                   | 2 weeks  | Clean datasets, graphs    |
| Phase 2 | Model implementation (UniMP-lite, H-UniMP, H-UniMP++) | 3 weeks  | Working models, configs   |
| Phase 3 | Experiments (train, validate, compare) | 2 weeks  | Accuracy/F1 scores, logs  |
| Phase 4 | Analysis & reporting                 | 1 week   | Final report, paper draft |

---

## 7. Risk Analysis  

- **Risk:** Memory/compute limitations on large datasets.  
  **Mitigation:** Use subsampled DBLP/OGB datasets; minimal model variants.  
- **Risk:** Label noise leading to unstable training.  
  **Mitigation:** Introduce uncertainty gating and curriculum masking.  
- **Risk:** Segmentation faults on macOS Paddle backend.  
  **Mitigation:** Implement CPU-safe minimal training scripts.  

---

## 8. Expected Outcomes  

- A robust **H-UniMP++ model** that improves citation network classification by ~2–3% over R-UniMP.  
- Demonstration of effectiveness of **uncertainty-gated label injection**.  
- Practical lightweight implementation suitable for CPU/GPU-constrained environments.  
- Contribution of methodology and results to ongoing research in graph neural networks for heterogeneous academic graphs.  

---

**Note:** Update this document as methodology evolves during implementation.  
