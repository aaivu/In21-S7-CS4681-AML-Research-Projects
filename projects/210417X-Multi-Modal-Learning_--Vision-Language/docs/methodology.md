# Methodology: Multi-Modal Learning\- Vision-Language

**Student:** 210417X
**Research Area:** Multi-Modal Learning-Vision-Language
**Date:** 2025-10-20

## 1. Overview
This methodology outlines the development and evaluation of FLAMINGO-VQA, a modular enhancement to the Flamingo pipeline for few-shot Visual Question Answering (VQA). It details the research design, data collection, preprocessing, model architecture, experimental setup, implementation plan, risk analysis, and expected outcomes, reflecting the completed work as documented in the final research paper.

## 2. Research Design
The research adopts a non-trainable, modular enhancement approach, building on the frozen Flamingo architecture. It employs a combination of feature pre-selection, semantic exemplar selection, and voting mechanisms, validated through empirical evaluation on the VQA v2.0 benchmark to assess accuracy, robustness, and generalization.

## 3. Data Collection

### 3.1 Data Sources
- VQA v2.0 dataset, a widely recognized benchmark for VQA tasks.

### 3.2 Data Description
The VQA v2.0 dataset comprises over 200,000 images with balanced question types (yes/no, number, other), including complex object relationships and scene semantics, designed to mitigate language priors and support few-shot evaluation.

### 3.3 Data Preprocessing
Data preprocessing involved extracting visual features using a frozen CLIP Vision Transformer (ViT) and textual features from a frozen CLIP text encoder. Irrelevant patches were filtered via Question-Guided Feature Pre-Selection (QGFP), and a curated set of semantically similar exemplars was selected using Semantic Few-Shot Selection (SFS) based on CLIP embeddings.

## 4. Model Architecture
FLAMINGO-VQA enhances the Flamingo model by integrating three non-trainable modules: 
- **QGFP**: Filters noisy visual patches using CLIP text encoder alignments.
- **SFS**: Retrieves relevant exemplars via CLIP image embeddings.
- **SCV**: Aggregates multiple predictions using a probabilistic consensus mechanism. The architecture leverages Flamingo’s gated cross-attention and Perceiver Resampler, requiring no retraining.
![alt text](<flamingo_modified (1).png>)

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Accuracy (percentage of correct answers).
- Robustness (performance under noisy inputs, assessed via per-type accuracies).
- Generalization (consistency across question types).

### 5.2 Baseline Models
- Baseline configurations without QGFP, SFS, or SCV (e.g., 31.0% accuracy).

### 5.3 Hardware/Software Requirements
- Hardware: GPU (e.g., T4 x2) with 24GB RAM.
- Software: Python 3.9, PyTorch, CLIP library, Flamingo implementation, VQA v2.0 toolkit.

## 6. Implementation Plan

| Phase       | Tasks                          | Duration | Deliverables         |
|-------------|--------------------------------|----------|----------------------|
| Phase 1     | Data preprocessing            | 1 week | Clean dataset        |
| Phase 2     | Model implementation (QGFP, SFS, SCV) | 1 weeks  | Working model        |
| Phase 3     | Experiments (ablation, threshold sweep) | 0.5 weeks  | Results (41.0% accuracy) |
| Phase 4     | Analysis (per-type, robustness) | 0.5 weeks   | Final report         |

## 7. Risk Analysis
- **Risk**: Inaccurate feature filtering by QGFP due to CLIP misalignment.
  - **Mitigation**: Conducted threshold sweeps (\(\tau = 0.3, 0.5, 0.7\)) to optimize alignment.
- **Risk**: Limited exemplar relevance in SFS.
  - **Mitigation**: Used CLIP embeddings for semantic similarity, validated by per-type gains.
- **Risk**: Computational overhead from SCV.
  - **Mitigation**: Ensured lightweight design with no retraining, tested on standard hardware.

## 8. Expected Outcomes
The methodology yielded a 41.0% accuracy on VQA v2.0, an 10.0 percentage point improvement over the 31.0% baseline, demonstrating enhanced accuracy, robustness to noise, and generalization across question types. This contributes a scalable, resource-efficient framework for few-shot VQA, applicable to real-world, constrained environments.
