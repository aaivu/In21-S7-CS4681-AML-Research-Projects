# Literature Review: GNN:Graph Classification

**Student:** 210296X
**Research Area:** GNN:Graph Classification
**Date:** 2025-09-01

## Abstract

This literature review explores the development of Graph Neural Networks (GNNs) for graph classification, focusing on self-supervised pretraining and few-shot learning approaches. It synthesizes research from 2018 to 2025, including foundational models like GCN, GIN, and Graph Attention Networks (GAT), as well as recent innovations such as Graph Masked Autoencoders (GraphMAE) and meta-learning frameworks for few-shot graph classification. Key findings reveal that while self-supervised pretraining significantly improves graph representation quality, adapting pretrained models to novel tasks with minimal labels remains challenging. Hybrid approaches combining masked pretraining and episodic fine-tuning show promise in achieving robust performance under label-scarce conditions. Identified research gaps include the need for unified benchmarks, scalable few-shot evaluation protocols, and better understanding of label-space granularity effects.  

## 1. Introduction

Graph classification aims to predict the label of an entire graph, such as identifying molecular properties or social network communities. GNNs have become the dominant paradigm due to their ability to capture complex structural dependencies. However, conventional supervised training demands large labeled datasets, limiting real-world applicability in domains like bioinformatics and chemistry. To address data scarcity, recent research has shifted toward self-supervised pretraining and few-shot learning on graphs. Self-supervised methods such as GraphMAE learn generalized representations from unlabeled data, while few-shot learning enables rapid adaptation with limited supervision. This review surveys progress in these directions, highlighting key models, methodologies, and open research challenges.  

## 2. Search Methodology

### Search Terms Used
- “Graph Neural Networks (GNNs)”
- “Graph Classification”
- “Few-shot Graph Learning”
- “Self-supervised Graph Learning”
- “Graph Masked Autoencoder (GraphMAE)”
- “Meta-learning on Graphs”

### Databases Searched
- [✅] Google Scholar
- [✅] ArXiv

### Time Period

2018–2025, focusing on recent developments in self-supervised graph learning and few-shot adaptation.

## 3. Key Areas of Research

### 3.1 Few-Shot Learning on Graphs (FSLG)
This area addresses the challenge of generalizing graph classification models from limited labeled examples (a handful per class) to new classes or tasks, which is critical in domains like drug discovery and bioinformatics where label acquisition is costly. FSLG methods aim to transfer knowledge from base classes to new ones through meta-learning or transfer learning  

The main research directions in FSLG include:
- Metric-based approaches (e.g., Prototypical Networks) that compare query embeddings to class prototypes.
- Optimization-based models (e.g., MAML, Meta-GNN) that learn initialization parameters for rapid gradient-based adaptation.
- Hybrid methods that combine self-supervised pre-training (like GraphMAE) with meta-fine-tuning for fast adaptation.
- Task-adaptive modules such as conditional embeddings and MixUp-based augmentations to enrich representations.

**Key Papers:**
- Crisostomi et al. (2022) - Demonstrated that episodic fine-tuning with GIN-based graph encoders outperforms vanilla transfer learning for few-shot graph classification.
- Yu et al. (2024) - Provided an IEEE survey highlighting meta-learning (MAML, Prototypical Networks), pre-training (autoencoding, contrastive learning), and prompt-based techniques as orthogonal strategies for label and structure scarcity.

### 3.2 Graph Masked Autoencoders and Self-Supervised Pre-training (GraphMAE/SSL)  

This area focuses on learning transferable graph representations without explicit supervision using self-supervised learning (SSL). Specifically, Graph Masked Autoencoders (GraphMAE) pre-train Graph Neural Networks (GNNs) by randomly masking a portion of node features and then training an encoder/decoder to reconstruct the original features.

Key aspects and contributions:
- Architecture: GraphMAE often uses a GIN encoder and a GIN-based decoder.
- Objective: It utilizes a scaled cosine error (SCE) loss for reconstruction, which down-weights easy examples, unlike contrastive methods.
- Role in FSL: The pre-trained GraphMAE representations can be effectively adapted to downstream tasks like few-shot graph classification through episodic fine-tuning.
- Hybridization: Combining the masked reconstruction loss with discriminative or contrastive objectives is a trend to improve few-shot generalization.

Key Papers:
- Hou et al. (2022) - GraphMAE: Self-Supervised Masked Graph Autoencoders  - Demonstrated strong transfer learning results in graph classification, outperforming previous self-supervised baselines using GIN backbones on datasets like MUTAG, PROTEINS, and COLLAB.

## 4. Research Gaps and Opportunities

### Gap 1: Lack of unified few-shot benchmarks for graph classification
**Why it matters:** The absence of standardized evaluation protocols makes it difficult to compare models fairly across datasets.
**How your project addresses it:** Implements few-shot protocols on standard TUDatasets (e.g., ENZYMES, PROTEINS) with reproducible configurations.

### Gap 2: Limited understanding of label-space granularity effects
**Why it matters:** Fine-tuning effectiveness varies significantly between binary and multi-class tasks.
How your project addresses it: Analyzes how class granularity influences few-shot adaptation performance and proposes adjustments to masking and sampling strategies.

### Gap 1: Inefficient parameter adaptation in pretrained models
**Why it matters:** Full fine-tuning is computationally expensive and may lead to overfitting on small datasets.
**How your project addresses it:** Experiments with partial fine-tuning and reconstruction regularization to preserve pretrained knowledge.

## 5. Theoretical Framework

This study is grounded in representation learning and meta-learning theory. It assumes that graph-level embeddings learned via self-supervised pretraining capture transferable features that can be adapted to new tasks. The theoretical base draws from the inductive bias of GIN for graph structure representation and meta-learning’s task-distribution optimization principle, enabling rapid adaptation under few-shot settings.

## 6. Methodology Insights

Most reviewed studies use Graph Neural Networks (GCN, GAT, GIN) as backbones. Common training strategies include contrastive learning (GraphCL), masked autoencoding (GraphMAE), and episodic meta-training (Prototypical Networks, Meta-GNN). Hybrid approaches—combining SSL pretraining with meta-fine-tuning—have shown the best generalization in label-scarce settings. For this research, the GraphMAE-GIN architecture with scaled cosine loss and N-way K-shot fine-tuning offers a promising direction.

## 7. Conclusion

The literature reveals a growing convergence between self-supervised and few-shot graph learning. Masked graph modeling has emerged as a powerful technique for pretraining, while episodic meta-learning provides a mechanism for efficient adaptation. However, key challenges remain—such as benchmark consistency, label granularity, and parameter efficiency. This review informs the current research by identifying these gaps and positioning the proposed work as a step toward robust, sample-efficient graph classification through GraphMAE-based few-shot fine-tuning.

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
 
