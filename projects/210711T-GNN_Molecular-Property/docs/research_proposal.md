# Research Proposal: GNN:Molecular Property

**Student:** 210711T
**Research Area:** Graph Neural Networks (GNNs) for Molecular Property Prediction
**Date:** 2025-09-01

## Abstract

This research focuses on improving molecular property prediction using Graph Neural Networks (GNNs), specifically targeting HIV inhibition classification from the MoleculeNet benchmark. Molecules are represented as graphs, where atoms and bonds serve as nodes and edges, respectively. While recent works such as GraphMVP have shown that multi-view pretraining across 2D and 3D molecular representations can enhance generalization, many real-world deployments lack sufficient data or computational resources for large-scale pretraining.
This project explores an alternative  improving fine-tuning stability and accuracy without pretraining by addressing class imbalance and optimization challenges. We extend the GraphMVP framework to incorporate focal loss, class-balanced sampling, automatic mixed precision (AMP), and adaptive learning rate scheduling. Preliminary experiments show consistent improvement in ROC-AUC and PR-AUC scores on the HIV dataset. The outcome of this study will be a lightweight, reproducible fine-tuning protocol adaptable to both 2D and future 3D-aware molecular encoders.

## 1. Introduction

Predicting molecular properties is a central task in computational chemistry and drug discovery. Traditional descriptor-based machine learning models fail to generalize across diverse molecular structures. Graph Neural Networks (GNNs) have emerged as a powerful alternative, representing molecular graphs in a way that preserves chemical and topological information.
Recent frameworks such as GraphMVP and SchNet demonstrated the power of multi-view learning using both 2D molecular graphs and 3D conformations. However, these approaches depend on extensive pretraining and 3D data availability.
This research aims to improve the fine-tuning stage for HIV inhibition prediction using a lightweight 2D GNN model — making it computationally efficient while maintaining strong predictive performance even under severe class imbalance.

## 2. Problem Statement

The MoleculeNet HIV dataset is highly imbalanced, with only a small fraction of active compounds compared to inactives. Standard training using Binary Cross-Entropy (BCE) loss often leads to overfitting on easy negatives and poor recall on active compounds.
Moreover, fine-tuning stability can degrade due to uncalibrated learning rates and limited validation feedback. Hence, the problem is:

How can we improve fine-tuning stability and classification performance of GNNs for imbalanced molecular datasets like MoleculeNet HIV, without relying on pretraining?

## 3. Literature Review Summary

Wu et al. (2018) introduced MoleculeNet, providing benchmark datasets and standardized scaffold splits for molecular property prediction. Xu et al. (2019) proposed the Graph Isomorphism Network (GIN), a strong 2D encoder architecture with theoretical guarantees for expressiveness. Schütt et al. (2018) presented SchNet, which uses continuous-filter convolutions to handle 3D molecular data. Hou et al. (2022) further improved performance through GraphMVP, aligning 2D and 3D representations in a contrastive pretraining framework.
However, prior works emphasize pretraining rather than fine-tuning strategies. Few studies systematically address class imbalance and optimization dynamics in molecular GNNs. This project fills that gap by enhancing fine-tuning for improved performance under constrained settings.

## 4. Research Objectives

### Primary Objective

To enhance GNN fine-tuning performance for molecular property prediction on the MoleculeNet HIV dataset by incorporating imbalance-aware training and optimization strategies.

### Secondary Objectives

- mplement focal loss to reduce dominance of easy negative samples.
- Apply class-balanced mini-batch sampling for better representation of active compounds.
- Integrate automatic mixed precision (AMP) for efficient GPU utilization.
- Introduce adaptive learning rate scheduling and early stopping to stabilize convergence.
- Evaluate and compare the improved pipeline against the baseline BCE-trained GIN model using ROC-AUC, PR-AUC, and F1-score.

## 5. Methodology

The research will be conducted using the GraphMVP codebase as the foundation.

1. Dataset: MoleculeNet HIV (41,127 molecules), using scaffold split (80/10/10).
2. Model: 5-layer GIN with 300-dimensional hidden units, mean pooling, and a linear prediction head.
3. Training Improvements:
   - Focal loss (γ = 1.5) for imbalance handling.
   - Balanced DataLoader for equal sampling of positives and negatives.
   - Mixed precision (AMP) for computational efficiency.
   - Learning-rate scheduler and early stopping to optimize training duration.
4. Evaluation Metrics: ROC-AUC (primary), PR-AUC, and validation-based F1-score.
5. Implementation Framework: PyTorch + PyTorch Geometric + RDKit (for molecule parsing).

## 6. Expected Outcomes

- A reproducible fine-tuning pipeline demonstrating improved ROC-AUC, PR-AUC, and F1 performance compared to the baseline.
- Reduced overfitting and more stable convergence during training.
- A well-documented training procedure adaptable to other molecular benchmarks (e.g., BBBP, Tox21).
- A solid foundation for future extensions into 3D-aware and multi-view GNN models such as GraphMVP and SchNet.

## 7. Timeline

| Week  | Task                    |
| ----- | ----------------------- |
| 1-2   | Literature Review       |
| 3-4   | Methodology Development |
| 5-8   | Implementation          |
| 9-12  | Experimentation         |
| 13-15 | Analysis and Writing    |
| 16    | Final Submission        |

## 8. Resources Required

| Resource                | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| **Hardware**            | NVIDIA GPU (≥ 8GB VRAM recommended)                           |
| **Software**            | Python 3.7+, PyTorch, PyTorch Geometric, RDKit                |
| **Dataset**             | MoleculeNet HIV from GraphMVP repository                      |
| **Tools**               | VS Code, Git, Conda environment, Matplotlib for visualization |
| **Compute Environment** | Local GPU or University cluster access                        |

## References

1. Z. Wu, B. Ramsundar, E. N. Feinberg et al., “MoleculeNet: A benchmark for molecular machine learning,” Chemical Science, vol. 9, no. 2, pp. 513–530, 2018.

2. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How Powerful are Graph Neural Networks?” Proc. ICLR, 2019.

3. K. T. Schütt, P.-J. Kindermans, H. E. Sauceda et al., “SchNet: A continuous-filter convolutional neural network for modeling quantum interactions,” J. Chem. Theory Comput., vol. 14, no. 11, pp. 6633–6642, 2018.

4. B. Hou, S. Zhang, M. Qiao et al., “GraphMVP: Multi-View Prototype Learning for Molecular Property Prediction,” Proc. ICLR, 2022.

5. T.-Y. Lin, P. Goyal, R. Girshick et al., “Focal Loss for Dense Object Detection,” Proc. ICCV, pp. 2980–2988, 2017.

---

**Submission Instructions:**

1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review
