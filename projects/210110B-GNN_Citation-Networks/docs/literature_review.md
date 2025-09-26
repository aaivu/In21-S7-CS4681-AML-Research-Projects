# Literature Review: GNN:Citation Networks

**Student:** 210110B  
**Research Area:** GNN:Citation Networks  
**Date:** 2025-09-05  

---

## Abstract  
This literature review surveys recent developments in graph neural networks (GNNs) applied to citation networks, with a focus on methods that integrate structural, semantic, and label information for improved node classification. The review traces the evolution from early homogeneous GNNs (e.g., GCN, GAT) to heterogeneous and relation-aware models (e.g., R-GAT, R-GCN), before highlighting the innovation of UniMP, which unified label and feature propagation. Extensions such as R-UniMP adapted these ideas to heterogeneous graphs, achieving state-of-the-art performance on large benchmarks like MAG240M-LSC. Despite these advances, challenges remain around robustness to noisy labels, adaptive masking strategies, and lightweight training for practical deployments. This review identifies these gaps and positions the proposed H-UniMP++ approach—relation-aware UniMP with uncertainty-gated label injection and curriculum masking—as a novel contribution that builds on and extends prior work.  

---

## 1. Introduction  
Citation networks have long served as canonical benchmarks for semi-supervised node classification. Traditional approaches treated them as homogeneous graphs, limiting their ability to capture the heterogeneous relations among papers, authors, and venues. The advent of GNNs, particularly Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), introduced end-to-end message passing mechanisms that effectively combined node features and local structure. However, performance remained bounded by limited use of label information and inadequate treatment of heterogeneity.  

Recent breakthroughs such as UniMP (Shi et al., 2020) proposed masked label prediction to unify feature and label propagation, significantly advancing accuracy on semi-supervised tasks. R-UniMP (Shi et al., 2021) extended this to heterogeneous citation graphs, as demonstrated in the KDD Cup 2021 MAG240M-LSC challenge, setting new benchmarks for large-scale graph learning. Yet, these approaches also revealed new challenges: sensitivity to masking rates, dependence on noisy labels, and computational demands.  

This review explores these developments systematically and motivates H-UniMP++ as an adaptive, robust extension to UniMP tailored for heterogeneous citation networks.  

---

## 2. Search Methodology  

### Search Terms Used  
- “Graph Neural Networks” + “citation networks”  
- “Semi-supervised node classification” + “label propagation”  
- “Heterogeneous GNN” / “relational GNN”  
- “UniMP” / “R-UniMP” / “Masked label prediction”  
- Synonyms: “graph representation learning,” “hetero-GNN,” “graph transformers”  

### Databases Searched  
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [x] Other: Open Graph Benchmark (OGB-LSC resources)  

### Time Period  
2018–2025, emphasizing recent post-UniMP advances.  

---

## 3. Key Areas of Research  

### 3.1 Early GNN Models for Citation Networks  
Initial research relied on homogeneous GNNs:  
- **Kipf & Welling (2017)** – Introduced GCN, using spectral convolutions for semi-supervised classification on citation networks.  
- **Velickovic et al. (2018)** – Proposed GAT, incorporating attention to weigh neighbors adaptively.  

These methods achieved strong baselines but ignored node and edge heterogeneity and did not incorporate label information effectively.  

### 3.2 Heterogeneous and Relation-Aware GNNs  
- **Schlichtkrull et al. (2018)** – Introduced R-GCN for multi-relational knowledge graphs.  
- **Hu et al. (2020)** – Benchmarked large-scale graph learning with OGB datasets.  
- **R-GAT (2020)** – Applied relation-wise attention and normalization to citation networks.  

These works recognized the need for relation-specific transformations, but label utilization remained underexplored.  

### 3.3 UniMP and Masked Label Prediction  
- **Shi et al. (2020)** – UniMP unified feature and label propagation via masked label prediction, directly incorporating labels into the input space.  
- Key insight: labels act as strong supervision signals, but must be masked to avoid trivial propagation.  
- Demonstrated state-of-the-art results across OGB citation tasks.  

### 3.4 R-UniMP and Large-Scale Extensions  
- **Shi et al. (2021, Baidu PGL)** – R-UniMP extended UniMP to heterogeneous graphs (MAG240M-LSC, KDD Cup 2021).  
- Contributions: relation-wise neighborhood sampling, relation-wise BatchNorm, masked label prediction, random label inputs, and relation-wise attention.  
- Achieved 73.71% single-model validation accuracy and 77.73% ensemble accuracy.  

### 3.5 Beyond UniMP: Robustness and Efficiency  
- **Huang et al. (2020)** – Correct & Smooth post-processing to refine predictions.  
- **Dong et al. (2017)** – Metapath2Vec for heterogeneous embeddings, later integrated into R-UniMP.  
- Recent works highlight challenges in scaling, noisy label sensitivity, and robustness—key motivations for H-UniMP++.  

---

## 4. Research Gaps and Opportunities  

### Gap 1: Label Noise and Over-Reliance  
**Why it matters:** UniMP and R-UniMP inject labels directly; noisy or imbalanced labels reduce generalization.  
**How your project addresses it:** H-UniMP++ introduces **uncertainty-gated label injection** to down-weight unreliable labels dynamically.  

### Gap 2: Fixed Masking Strategies  
**Why it matters:** UniMP relies on fixed random masking; performance varies with label coverage.  
**How your project addresses it:** H-UniMP++ implements a **curriculum masking schedule**, gradually increasing difficulty during training.  

### Gap 3: Lightweight Deployment  
**Why it matters:** R-UniMP was trained on massive GPUs; real-world setups often lack such resources.  
**How your project addresses it:** H-UniMP++ simplifies relation-aware layers and supports CPU-friendly training, enabling deployment on constrained environments (e.g., MacBook M2).  

---

## 5. Theoretical Framework  
The work builds on the **message passing neural network (MPNN)** paradigm, where each node updates its representation by aggregating neighbors’ features. UniMP redefined this by extending the feature space to include **label embeddings**, while R-UniMP added **relation-aware projections**. H-UniMP++ adds **gated fusion mechanisms** grounded in uncertainty estimation, aligning with Bayesian principles for robust label utilization.  

---

## 6. Methodology Insights  
- **Common methodologies:** Neighbor sampling (GraphSAGE, R-GAT), attention mechanisms, residual connections, and batch normalization.  
- **Promising directions:** Masked label prediction (UniMP), relation-aware attention (R-UniMP), and uncertainty modeling (H-UniMP++).  
- **Evaluation protocols:** Use of OGB datasets (ogbn-arxiv, ogbn-products, MAG240M) with accuracy and F1 as benchmarks.  

---

## 7. Conclusion  
Literature shows a clear progression: GCN/GAT established GNN baselines; R-GCN and R-GAT introduced heterogeneity; UniMP integrated label propagation; R-UniMP scaled these ideas for heterogeneous citation graphs. Yet, issues of label noise, masking sensitivity, and efficiency remain. H-UniMP++ addresses these gaps with uncertainty-gated label injection and curriculum masking, aiming to combine the strengths of UniMP with robustness and adaptability for real-world citation networks.  

---

## References  

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.  
2. Velickovic, P., et al. (2018). Graph Attention Networks. *ICLR*.  
3. Schlichtkrull, M., et al. (2018). Modeling Relational Data with Graph Convolutional Networks. *ESWC*.  
4. Shi, Y., et al. (2020). Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. *arXiv:2009.03509*.  
5. Shi, Y., et al. (2021). R-UniMP: Solution for KDD Cup 2021 MAG240M-LSC. *arXiv preprint*.  
6. Hu, W., et al. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. *NeurIPS*.  
7. Huang, Q., et al. (2020). Combining Label Propagation and Simple Models Outperforms GNNs. *ICLR*.  
8. Dong, Y., Chawla, N., & Swami, A. (2017). metapath2vec: Scalable Representation Learning for Heterogeneous Networks. *KDD*.  
9. Li, G., et al. (2019). DeepGCNs: Can GCNs Go as Deep as CNNs? *ICCV*.  
10. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.  
