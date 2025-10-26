# Methodology: GNN:Graph Classification

**Student:** 210296X
**Research Area:** GNN:Graph Classification
**Date:** 2025-09-01

## 1. Overview

This research adopts a two-stage learning methodology combining self-supervised pretraining and few-shot fine-tuning for graph classification. The approach begins by training a Graph Masked Autoencoder (GraphMAE) on large-scale unlabeled graph datasets to learn generalizable structural representations. Subsequently, the pretrained model is fine-tuned on small labeled datasets using few-shot learning protocols (N-way, K-shot). This design enables effective adaptation to new tasks with minimal labeled data while maintaining strong generalization across graph domains such as bioinformatics and social networks.

## 2. Research Design

The research follows a quantitative experimental design with a focus on empirical performance evaluation.

- Stage 1: Self-Supervised Pretraining — Train a GraphMAE with a Graph Isomorphism Network (GIN) encoder to reconstruct masked node features.

- Stage 2: Few-Shot Fine-Tuning — Adapt the pretrained model for few-shot classification through episodic meta-learning, where each episode simulates a small classification task.

- Stage 3: Evaluation and Analysis — Measure accuracy, F1 score, and transfer performance across different datasets and class granularities (binary vs. multi-class).

This design helps evaluate how pretrained representations transfer under data scarcity and how few-shot fine-tuning influences model adaptability.

## 3. Data Collection

### 3.1 Data Sources
- TUDataset
### 3.2 Data Description

- ENZYMES: 600 protein graphs labeled into 6 enzyme classes, with node labels and continuous attributes.
- PROTEINS: 1,113 protein graphs labeled by whether they are enzymes.
- DD: 1,178 protein structures with binary classification labels.
- MSRC-9 and MSRC-21: Image-segmentation datasets represented as region adjacency graphs with 9 and 21 classes respectively.

## 4. Model Architecture

The proposed model integrates Graph Masked Autoencoder (GraphMAE) with a Graph Isomorphism Network (GIN) backbone.

- Encoder: Two-layer GIN for learning node and graph-level embeddings.

- Decoder: GIN-based reconstruction module to reconstruct masked node features.

- Pretraining Objective: Scaled Cosine Error (SCE) loss between reconstructed and original node features.

- Fine-Tuning Phase: A linear classification layer is appended for N-way K-shot episodic tasks.

- Pooling Mechanism: Mean pooling to obtain graph-level embeddings.

- Optimization: Adam optimizer (learning rate = 1e-4), dropout = 0.5.

This design leverages GraphMAE’s generative pretraining for transferable representations and GIN’s strong discriminative power for small-graph datasets.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Accuracy — Proportion of correctly classified graphs.
- Micro-F1 Score — Balances precision and recall across all classes.
- Macro-F1 Score — Evaluates class-wise performance equally.
- Standard Deviation — Measures consistency across random few-shot tasks.
### 5.2 Baseline Models
- GraphMAE (pretrained only) — Self-supervised baseline without fine-tuning.
- GraphCL — Contrastive learning-based self-supervised method.
- Meta-GNN — Optimization-based few-shot baseline.
- Prototypical Networks — Metric-based few-shot learning baseline.
### 5.3 Hardware/Software Requirements
Hardware: NVIDIA T4 GPU (preferred), 16GB RAM, 4 CPU cores. 
Software:
- Python 3.10
- PyTorch 2.1+
- DGL 1.1+
- scikit-learn, NumPy, pandas
- CUDA 11.8
## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Model implementation | 2 weeks | Working model |
| Phase 1 | Model improvement | 3 weeks | Improved model |
| Phase 3 | Experiments | 2 weeks | Results |
| Phase 4 | Analysis | 1 week | Final report |

## 7. Risk Analysis

| Risk	| Impact | Mitigation Strategy |
|-------|--------|---------------------|
| Limited GPU resources	| High |	Use smaller batch sizes and reduced masking rates |
| Model overfitting in fine-tuning	| Medium |	Apply dropout, early stopping, and data augmentation |
| Dataset imbalance | 	Medium |	Use class-weighted loss or balanced sampling |
| Poor transfer performance |	Medium |	Experiment with different masking strategies and partial fine-tuning |
## 8. Expected Outcomes

- A pretrained and fine-tuned GraphMAE–GIN model capable of accurate graph classification with few labeled examples.
- Demonstrated performance improvements over self-supervised and meta-learning baselines.
- Analytical insights on how label-space complexity influences few-shot learning performance.
- Contribution of a reproducible pipeline for few-shot graph classification using masked autoencoders.
---

