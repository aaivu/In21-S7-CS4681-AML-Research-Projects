# Methodology: GNN:Molecular Property

**Student:** 210711T
**Research Area:** Graph Neural Networks (GNNs) for Molecular Property Prediction
**Date:** 2025-09-01

## 1. Overview

This project investigates fine-tuning strategies for molecular property prediction using Graph Neural Networks (GNNs), focusing on the HIV inhibition dataset from MoleculeNet. The objective is to improve model performance under severe class imbalance without pretraining.
We build upon the GraphMVP framework which integrates 2D/3D molecular representations but simplify the pipeline to analyze imbalance-aware fine-tuning using a 2D encoder (GIN). The study evaluates the effect of focal loss, balanced sampling, and learning-rate scheduling on model stability and accuracy.

## 2. Research Design

The research follows an experimental design with the following phases:

1. Baseline reproduction: Implement the default GIN classifier with BCE loss on the HIV dataset.

2. Imbalance-aware fine-tuning: Introduce focal loss, balanced sampling, mixed-precision (AMP), and adaptive learning-rate scheduling.

3. omparative evaluation: Analyze performance differences between baseline and improved setups using ROC-AUC, PR-AUC, and F1 metrics.

4. Result interpretation: Identify how imbalance handling and optimization tuning affect downstream molecular classification.

This design ensures controlled comparison under identical conditions, isolating the effects of training strategies.

## 3. Data Collection

### 3.1 Data Sources

- MoleculeNet HIV dataset — a benchmark molecular dataset curated for binary HIV inhibition prediction.

- GraphMVP framework repository — for preprocessing and experimental scripts.

- OGB MoleculeNet API — to automatically download and prepare the dataset.

### 3.2 Data Description

- Samples: 41,127 molecules

- Features: Atom type, chirality, bond type, 2D molecular structure

- Labels: Binary — HIV inhibitor (active = 1, inactive = 0)

- Split: Scaffold split (80% train, 10% validation, 10% test) to ensure different molecular scaffolds across sets.

### 3.3 Data Preprocessing

- Molecules were converted from SMILES strings into graph representations.

- Nodes represent atoms (with embeddings for atom type and chirality).

- Edges represent bonds (with embeddings for bond type and direction).

- The dataset was processed using RDKit and Torch Geometric for graph construction.

- No data augmentation or 3D coordinates were used (2D-only baseline).

## 4. Model Architecture

The proposed model is based on the Graph Isomorphism Network (GIN):

- Input: Atom and bond embeddings
- Architecture: 5 GIN layers (hidden dimension = 300), ReLU + BatchNorm after each layer
- Readout: Mean pooling over all nodes
- Classifier: Single-layer MLP head for binary prediction

Imbalance-aware modifications:

- Focal Loss (γ = 1.5) replaces BCE to focus on hard-to-classify examples.
- Balanced DataLoader ensures rare actives appear in each mini-batch.
- AMP (Automatic Mixed Precision) speeds up and stabilizes training.
- Adaptive LR Scheduling & Early Stopping improve convergence stability.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- ROC-AUC: Primary metric for classification performance.
- PR-AUC: More informative under class imbalance.
- F1-score: Computed at validation-selected thresholds.
- Loss: Binary cross-entropy (baseline) vs. focal loss (improved).

### 5.2 Baseline Models

| Model               | Description                                       | Loss            | Notes                    |
| ------------------- | ------------------------------------------------- | --------------- | ------------------------ |
| Baseline GIN        | Default BCE fine-tuning on MoleculeNet HIV        | BCE             | From GraphMVP defaults   |
| Improved GIN (Ours) | Focal Loss + Balanced Sampling + AMP + Scheduling | Focal (γ = 1.5) | Handles imbalance better |

### 5.3 Hardware/Software Requirements

| Category   | Specification                    |
| ---------- | -------------------------------- |
| Hardware   | NVIDIA RTX 3060 (12 GB VRAM)     |
| OS         | Windows 10 / Ubuntu 22.04        |
| Frameworks | PyTorch 1.9.1, PyTorch Geometric |
| Libraries  | RDKit, NumPy, Pandas             |
| CUDA       | 11.1                             |
| Python     | 3.7 (Conda env)                  |

## 6. Implementation Plan

| Phase   | Tasks                                             | Duration | Deliverables                |
| ------- | ------------------------------------------------- | -------- | --------------------------- |
| Phase 1 | Dataset download & preprocessing (SMILES → graph) | 2 weeks  | Ready-to-train dataset      |
| Phase 2 | Baseline GIN implementation (BCE loss)            | 2 weeks  | Verified baseline accuracy  |
| Phase 3 | Improved training (focal loss, AMP, scheduling)   | 3 weeks  | Stable fine-tuning pipeline |
| Phase 4 | Experiments & metrics logging                     | 2 weeks  | Comparative results         |
| Phase 5 | Results analysis & short paper prep               | 1 week   | Final report + plots        |

## 7. Risk Analysis

| Risk                 | Impact | Mitigation Strategy                    |
| -------------------- | ------ | -------------------------------------- |
| GPU resource limits  | Medium | Use smaller batches and AMP            |
| Data imbalance       | High   | Apply focal loss and balanced sampling |
| Training instability | Medium | Early stopping + LR scheduling         |
| Implementation bugs  | Low    | Cross-verify with GraphMVP baseline    |

## 8. Expected Outcomes

- A robust fine-tuning pipeline for molecular property prediction using GNNs.
- Demonstrated improvements in ROC-AUC, PR-AUC, and training stability.
- A reproducible framework that does not rely on expensive pretraining.
- A strong foundation for future integration with 3D encoders (e.g., GraphMVP or SchNet).

---
