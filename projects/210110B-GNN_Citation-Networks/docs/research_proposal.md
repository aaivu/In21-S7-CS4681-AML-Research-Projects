# Research Proposal: GNN:Citation Networks

**Student:** 210110B  
**Research Area:** GNN:Citation Networks  
**Date:** 2025-09-27  

---

## Abstract

Graph Neural Networks (GNNs) have emerged as powerful tools for semi-supervised node classification in large-scale graphs, including citation networks. However, most existing models are designed for homogeneous graphs and fail to fully use the heterogeneity of real-world citation networks, which involve multiple node and relation types (e.g., authors, papers, venues; cites, writes, published in). The Unified Message Passing (UniMP) model has achieved state-of-the-art results by unifying feature and label propagation through masked label prediction. Yet, it is limited to homogeneous graphs. This proposal aims to develop **Heterogeneity-Aware UniMP (H-UniMP)**, an extension of UniMP that introduces relation-aware attention to model semantic differences between edge types. By applying H-UniMP to the DBLP-Citation-Network V12 dataset, the research seeks to achieve more accurate and robust node classification while addressing the challenges of label sparsity via systematic tuning of masked label prediction strategies. The project will include comparative benchmarking with UniMP, R-GCN, and HAN, along with ablation studies and sensitivity analyses to evaluate the impact of heterogeneity modeling. Ultimately, this work contributes to advancing GNN methodologies for real-world heterogeneous networks.

---

## 1. Introduction

Graphs provide a natural way to represent relationships in domains such as citation networks, social media, and biology. In citation networks, predicting the topic or field of a paper based on limited labeled data is a critical task. While UniMP has unified feature and label propagation with significant improvements over traditional GNNs, it has only been explored in homogeneous settings. Real-world citation graphs, however, are inherently heterogeneous. This research aims to bridge this gap by extending UniMP to handle heterogeneity effectively, thereby enabling more semantically informed message passing.

---

## 2. Problem Statement

Existing GNN models, including UniMP, treat all nodes and edges as identical, ignoring the heterogeneity of citation networks. This leads to information loss when different node types (authors, papers, venues) and relation types (cites, writes, published in) are collapsed into a homogeneous structure. The problem is to design a unified message passing model that respects heterogeneity while preserving UniMP’s strength of combining feature and label propagation.

---

## 3. Literature Review Summary

Early **Graph Neural Network (GNN)** models demonstrated the strength of message passing for citation networks. The **Graph Convolutional Network (GCN)** [1] introduced efficient spectral convolutions, while the **Graph Attention Network (GAT)** [2] improved this by learning attention coefficients to weight neighbors differently. Both achieved strong performance on citation benchmarks but assumed homogeneous graphs with a single node and edge type.  

To better capture multi-typed entities, **heterogeneous GNNs** emerged. **R-GCN** [3] introduced relation-specific weight matrices for multi-relational message passing, though it struggled with parameter explosion. Embedding methods like **metapath2vec** [4] leveraged random walks guided by semantic paths (e.g., author–paper–author), capturing relational structure and inspiring later GNN enhancements.  

A major leap came with **UniMP** [5], which unified feature and label propagation through **masked label prediction**. By embedding labels into the feature space and masking a portion of them during training, UniMP achieved state-of-the-art performance on Open Graph Benchmark datasets. Building on this, **R-UniMP** [6] extended UniMP with relation-aware propagation, normalization, and metapath embeddings, achieving top accuracy in the KDD Cup 2021 MAG240M-LSC competition.  

Despite these advances, UniMP and R-UniMP are sensitive to hyperparameters (especially masking rates) and require substantial computational resources. Robustness-focused methods such as **Correct & Smooth (C&S)** [7] showed that combining simple predictions with label propagation can outperform heavier GNNs under some conditions. The **Open Graph Benchmark (OGB)** [8] further standardized large-scale datasets and evaluation, exposing challenges of scalability and reproducibility.  

In summary, while GNNs like UniMP and R-UniMP push the state-of-the-art by unifying features and labels, practical challenges remain around heterogeneity, hyperparameter robustness, and resource efficiency. These gaps motivate the proposed **H-UniMP**, which integrates relation-aware attention with systematic hyperparameter tuning to improve training stability and performance on heterogeneous citation networks.  

**Gap Identified:** Lack of models that unify feature and label propagation while explicitly modeling heterogeneous relations in citation networks.

---

## 4. Research Objectives

### Primary Objective
To develop a **Heterogeneity-Aware UniMP (H-UniMP)** that extends UniMP with relation-aware attention for effective node classification in heterogeneous citation networks.

### Secondary Objectives
- Extend UniMP to handle multiple node and edge types.  
- Introduce relation-aware attention to distinguish between semantic relations (e.g., cites vs. writes).  
- Optimize masked label prediction strategies via systematic hyperparameter tuning.  
- Benchmark H-UniMP against UniMP, R-GCN, and HAN on DBLP-Citation-Network V12.  
- Conduct ablation and sensitivity analyses to evaluate enhancements.  

---

## 5. Methodology

1. **Baseline Reproduction**: Implement and verify UniMP on homogeneous datasets (e.g., OGBN-Arxiv).  
2. **Dataset Preparation**: Preprocess DBLP-Citation-Network V12 into a heterogeneous graph with papers, authors, and venues, and construct relation-specific adjacency lists.  
3. **Model Design (H-UniMP)**: Extend UniMP with relation-aware graph transformer layers to apply distinct attention for different relation types.  
4. **Training Strategy**: Adopt masked label prediction with varying masking rates (20–60%) to mitigate label leakage.  
5. **Evaluation**: Compare H-UniMP against UniMP, R-GCN, and HAN using accuracy, macro-F1, and micro-F1.  
6. **Ablation Studies**: Measure contributions of relation-aware attention and sensitivity to masking rates.  

---

## 6. Expected Outcomes

- A novel **H-UniMP framework** for heterogeneous citation networks.  
- Improved classification accuracy and robustness compared to UniMP and existing baselines.  
- Insights from ablation studies on the role of heterogeneity modeling and label masking.  
- A reproducible repository with implementation, datasets, and evaluation scripts.  

---

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Baseline Reproduction & Methodology Development |
| 5-7 | H-UniMP Implementation |
| 8-9 | Experimentation on Citation Network V1 |
| 10-12| Analysis, Ablation Studies, and Writing |
| 12   | Final Submission |

---

## 8. Resources Required

- **Datasets:** Citation Network V1, DBLP-Citation-Network V12, OGBN-Arxiv (for baseline).  
- **Tools & Frameworks:** PyTorch Geometric / DGL, Python.  
- **Compute:** CPU only implementation.  
- **Libraries:** PyTorch, Scikit-learn, Pandas, NumPy.  

---

## References

1. Kipf, T. N., & Welling, M. (2016). *Semi-Supervised Classification with Graph Convolutional Networks.* arXiv:1609.02907.  
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). *Graph Attention Networks.* arXiv:1710.10903.  
3. Schlichtkrull, M., Kipf, T. N., Bloem, R., Van Den Berg, R., Titov, I., & Welling, M. (2018). *Modeling Relational Data with Graph Convolutional Networks.* arXiv:1703.06103.  
4. Dong, Y., Chawla, N., & Swami, A. (2017). *metapath2vec: Scalable Representation Learning for Heterogeneous Networks.* KDD.  
5. Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., & Sun, Y. (2020). *Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification.* arXiv:2009.03509.  
6. Shi, Y., Huang, Z., Li, W., Su, W., & Feng, S. (2021). *R-UniMP: Solution for KDD Cup 2021 MAG240M-LSC.*  
7. Huang, Q., He, H., Singh, A., Lim, S., & Benson, A. R. (2020). *Combining Label Propagation and Simple Models Outperforms Graph Neural Networks.* arXiv:2010.13993.  
8. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs.* arXiv:2005.00687.  

