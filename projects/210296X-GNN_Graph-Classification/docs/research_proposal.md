# Research Proposal: GNN:Graph Classification

**Student:** 210296X
**Research Area:** GNN:Graph Classification
**Date:** 2025-09-01

## Abstract

Graph Neural Networks (GNNs) have achieved significant progress in graph-based learning tasks, but their dependence on large labeled datasets limits their effectiveness in label-scarce domains such as bioinformatics and drug discovery. This research explores a two-stage learning paradigm that combines self-supervised pretraining with Graph Masked Autoencoders (GraphMAE) and few-shot fine-tuning for graph classification. In the pretraining stage, a GIN-based GraphMAE learns structure-aware graph representations by reconstructing masked node attributes. The pretrained model is then fine-tuned under few-shot learning settings to adapt rapidly to new classes with minimal labeled data. The study aims to investigate the transferability of pretrained graph embeddings, the effect of episodic sampling, and the influence of masking strategies on adaptation performance. Expected results include improved classification accuracy on multi-class datasets and insights into how label-space granularity affects fine-tuning performance.

## 1. Introduction

Graphs are a natural way to represent structured data in domains such as molecular biology, social networks, and chemistry. Graph classification—assigning labels to entire graphs—is essential for tasks like protein function prediction and molecule property identification. While GNNs like Graph Convolutional Networks (GCNs) and Graph Isomorphism Networks (GINs) have excelled in graph-based learning, they rely heavily on large labeled datasets. In many real-world scenarios, labeled data is costly or impractical to obtain. Few-shot learning offers a promising solution by enabling models to generalize from limited labeled examples. This research leverages Graph Masked Autoencoders (GraphMAE) for self-supervised pretraining and applies few-shot fine-tuning to achieve robust graph classification in low-label regimes.

## 2. Problem Statement

Despite the success of self-supervised GNN pretraining, adapting these models to new graph classification tasks with very few labeled examples remains challenging. The problem is to determine how pretrained GraphMAE encoders can be effectively fine-tuned in few-shot scenarios to achieve high classification performance, particularly across datasets with varying label-space complexities.

## 3. Literature Review Summary

Recent advances in graph learning highlight self-supervised pretraining and few-shot learning as key strategies to overcome label scarcity. GraphMAE pretrains models by reconstructing masked node features, achieving strong transfer performance on benchmark datasets. Few-shot graph learning methods—such as Meta-GNN and Prototypical Networks—apply meta-learning to rapidly adapt to new tasks. However, integrating masking-based pretraining with episodic fine-tuning for few-shot graph classification remains underexplored. Prior work identifies key challenges in balancing pretraining objectives with adaptation efficiency and in handling graph heterogeneity and multi-class complexity. This study bridges that gap by combining GraphMAE pretraining with few-shot fine-tuning and analyzing their synergy.

## 4. Research Objectives

### Primary Objective

To investigate how self-supervised pretrained Graph Masked Autoencoders can be fine-tuned effectively for few-shot graph classification.

### Secondary Objectives
- Evaluate the transferability of pretrained GraphMAE representations across different graph datasets.
- Analyze the impact of masking strategies and episodic sampling on few-shot adaptation.
- Compare performance across datasets with varying class granularities (binary vs. multi-class).

## 5. Methodology

The research follows a two-stage framework:

1. Self-supervised Pretraining:

  - Train a GraphMAE model with a GIN encoder and decoder on unlabeled graphs using a masked-node reconstruction objective (scaled cosine error loss).
  - Capture node- and structure-level representations without supervision.

2. Few-shot Fine-tuning:

  - Initialize the pretrained encoder for N-way K-shot classification tasks.
  - Use episodic meta-learning to train on a small support set (K samples per class) and evaluate on a query set.
  - Fine-tune with a cross-entropy objective while optionally applying an auxiliary reconstruction loss to retain pretraining knowledge.
  - Report performance as mean ± standard deviation of accuracy and F1 across multiple tasks.

This approach will be tested primarily on the ENZYMES dataset from TUDataset, along with other benchmarks like PROTEINS, DD, and MSRC.

## 6. Expected Outcomes

- A fine-tuned GraphMAE–GIN model capable of strong performance in few-shot graph classification.
- Insights into how class granularity and dataset complexity influence adaptation.
- Evidence that combining self-supervised pretraining and few-shot fine-tuning yields robust graph representations under limited supervision.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-8  | Implementation |
| 9-12 | Experimentation |
| 13-15| Analysis and Writing |
| 16   | Final Submission |

## 8. Resources Required

- Datasets: ENZYMES, DD, PROTEINS, MSRC-9, MSRC-21 (TUDataset)
- Tools: PyTorch, DGL, scikit-learn
- Hardware: GPU-enabled environment (preferably NVIDIA T4)

## References

 [1] J. Chen et al., “Zero-Shot and Few-Shot Learning With
 Knowledge Graphs: A Comprehensive Survey,” Proceedings
 of the IEEE, vol. 111, no. 6, pp. 653–685, Jun. 2023, doi:
 https://doi.org/10.1109/jproc.2023.3279374.  
 
 [2] S. Wang, C. Chen, and J. Li, “Graph Few-shot Learning with Task
specific Structures,” arXiv.org, 2022. https://arxiv.org/abs/2210.12130  
 
 [3] C. Zhang et al., “Few-Shot Learning on Graphs,” arXiv.org, 2022.
 https://arxiv.org/abs/2203.09308  
 
[4] Z. Guo et al., “Few-Shot Graph Learning for Molecular Prop
erty Prediction,” arXiv (Cornell University), Apr. 2021, doi:
 https://doi.org/10.1145/3442381.3450112.  
 
 [5] D. Crisostomi, S. Antonelli, V. Maiorca, L. Moschella, R. Marin, and
 E. Rodol`a, “Metric Based Few-Shot Graph Classification,” arXiv.org,
 2022. https://arxiv.org/abs/2206.03695  
 
 [6] X. Yu et al., “A Survey of Few-Shot Learning on Graphs: from
 Meta-Learning to Pre-Training and Prompt Learning,” arXiv.org, 2024.
 https://arxiv.org/abs/2402.01440  
 
 [7] Z. Hou et al., “GraphMAE: Self-Supervised Masked Graph Autoen
coders,” arXiv.org, 2022. https://arxiv.org/abs/2205.10803  

 [8] H. Yao et al., “Graph Few-shot Learning via Knowledge Transfer,”
 arXiv.org, 2019. https://arxiv.org/abs/1910.03053  
 
 [9] “Datasets,” TUDataset, May 20, 2023. https://chrsmrrs.github.io/datasets/docs/datasets/  
 
 ---
