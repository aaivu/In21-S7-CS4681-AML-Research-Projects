# Methodology: GNN:Knowledge Graphs

**Student:** 210553J  
**Research Area:** GNN:Knowledge Graphs  
**Date:** 2025-09-01  

## 1. Overview

This research introduces **Temporal Neural Bellman–Ford Networks (T-NBFNet)**, an extension of the Neural Bellman–Ford Network (NBFNet), to model dynamic knowledge graphs. While NBFNet provides interpretable path-based reasoning for static link prediction, it fails to account for temporal dependencies. T-NBFNet integrates sinusoidal time encodings, temporal decay weighting, causal masking, and memory updates to capture evolving relationships while maintaining interpretability.

## 2. Research Design

We adopt a **path-based temporal reasoning approach** that combines the interpretability of NBFNet with temporal graph learning techniques. The model is evaluated through comparative experiments with static and temporal baselines on benchmark datasets (ICEWS, WIKI, YAGO, GDELT). The design emphasizes both **predictive performance** and **explainability**.

## 3. Data Collection

### 3.1 Data Sources
- ICEWS14, ICEWS18 (Integrated Crisis Early Warning System events)  
- WIKI (temporal facts from Wikipedia history)  
- YAGO (yearly-granularity temporal knowledge graph)  
- GDELT (global event data)  

### 3.2 Data Description
Each dataset contains quadruples of the form **(head, relation, tail, timestamp)**.  
- Entities: 10K–23K  
- Relations: 10–260  
- Events: 160K–500K  
- Temporal granularity: daily or yearly  

### 3.3 Data Preprocessing
- Chronological split into train, validation, and test sets to prevent temporal leakage.  
- Causal filtering ensures only past events are visible during prediction.  
- Negative sampling for training by corrupting head or tail entities.  

## 4. Model Architecture

The **T-NBFNet** extends NBFNet with:  
- **Temporal sinusoidal encodings** for timestamps.  
- **Temporal decay weighting & causal masking** to enforce recency and causality.  
- **GRU-based memory updates** for long-term node state evolution.  
- **Rotation-based relational transformations** (e.g., RotatE) for message passing.  
- **MLP scoring function** with residual connections, normalization, and dropout.  

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Mean Rank (MR)  
- Mean Reciprocal Rank (MRR)  
- Hits@K (K = 1, 3, 10)  

### 5.2 Baseline Models
- **Static:** DistMult, ComplEx, ConvE, RotatE, R-GCN, NBFNet  
- **Temporal:** Know-Evolve, HyTE, RE-NET, CyGNet, TGAT, TGN  

### 5.3 Hardware/Software Requirements
- NVIDIA GeForce GTX TITAN X (16 GB)  
- PyTorch with GPU acceleration  
- Adam optimizer (lr=5e-4, weight decay=1e-5)  
- Batch sizes: 3–16 depending on GPU memory  

## 6. Implementation Plan

| Phase   | Tasks                  | Duration | Deliverables     |
|---------|------------------------|----------|-----------------|
| Phase 1 | Data preprocessing     | 2 weeks  | Clean datasets   |
| Phase 2 | Model implementation   | 3 weeks  | T-NBFNet code    |
| Phase 3 | Experiments            | 2 weeks  | Evaluation logs  |
| Phase 4 | Analysis               | 1 week   | Final report     |

## 7. Risk Analysis

- **Overfitting** on small datasets → mitigate with dropout & early stopping.  
- **Temporal leakage** → ensure strict chronological splits.  
- **Scalability issues** on large datasets → optimize batching and memory updates.  
- **Trade-off between accuracy & interpretability** → emphasize explainability in analysis.  

## 8. Expected Outcomes

- A novel interpretable temporal GNN for link prediction.  
- Improved performance over static methods (e.g., DistMult, RotatE).  
- Competitive results against temporal baselines (e.g., RE-NET, CyGNet).  
- Contributions toward **transparent and temporally aware reasoning** in dynamic knowledge graphs.  

---

**Note:** Update this document as your methodology evolves during implementation.
