# Methodology: CV: Semantic Segmentation

**Student:** 210372D  
**Research Area:** CV: Semantic Segmentation  
**Date:** 2025-10-20

---

## 1. Overview

This study investigates semantic segmentation using the SegFormer-B0 architecture on the ADE20K dataset. The methodology focuses on evaluating baseline performance and systematically introducing architectural, regularization, and training schedule modifications to improve pixel-level scene understanding. Key goals include improving feature representation, spatial continuity, and generalization.

---

## 2. Research Design

The approach is experimental and quantitative. The study involves:

- Implementing the SegFormer-B0 baseline.
- Modifying training schedules, dropout configurations, and decoder architecture.
- Training all variants under identical conditions.
- Comparing performance using **mean Intersection-over-Union (mIoU)** across 150 ADE20K classes.

---

## 3. Data Collection

### 3.1 Data Sources
**ADE20K Dataset** – Over 20,000 images labeled into 150 semantic categories, sourced from publicly available Kaggle repositories.

### 3.2 Data Description
Paired RGB images and pixel-level semantic annotations with diverse indoor and outdoor scenes.

### 3.3 Data Preprocessing
- Image resizing to 512×512.
- Normalization using ImageNet statistics.
- Label remapping for consistent class indices.
- Dataset split: 70% training, 15% validation, 15% testing.

---

## 4. Model Architecture

- **Baseline:** SegFormer-B0 with standard MiT backbone and MLP decoder.
- **Modifications:**
  - **Squeeze-and-Excitation (SE) layer** for channel-wise feature recalibration.
  - **Modified convolution layers** for local spatial refinement.
  - **Extra convolution layer** after linear fusion for enhanced post-fusion processing.
  - **Partial unfreeze of SE layers** to test selective attention adaptation.
- **Training Loss:** Cross-entropy loss with ignore_index for background class.

---

## 5. Experimental Setup

### 5.1 Evaluation Metric
- **Mean Intersection-over-Union (mIoU)**

### 5.2 Baseline Models
- SegFormer-B0 pretrained on ADE20K.

### 5.3 Hardware/Software Requirements
- **Hardware:** NVIDIA P100 GPU, 16–24GB VRAM.
- **Software:** Python 3.10, PyTorch, Hugging Face SegFormer implementation, CUDA, KaggleHub.

---

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Dataset preprocessing | 1 week | Standardized dataset |
| Phase 2 | Baseline SegFormer-B0 training | 1 week | Baseline model & mIoU metrics |
| Phase 3 | Training schedule experiments | 1 week | mIoU performance tables |
| Phase 4 | Dropout modification | 1 week | Validation mIoU progression |
| Phase 5 | Architectural enhancements | 1 weeks | SE layers, conv modifications, extra conv |
| Phase 6 | Comparative analysis | 1 week | Summary of improvements |

---

## 7. Risk Analysis

| Risk | Description | Mitigation |
|------|-------------|------------|
| Overfitting | Small dataset variability | Dropout removal/addition, regularization |
| Divergence | High learning rate | Monitor mIoU; use conservative schedules |
| Computational constraints | GPU memory and training time | Batch-size adjustment, gradient checkpointing |
| Convergence instability | Transformer-based models sensitive to hyperparameters | Gradual warm-up learning rate and adaptive optimizers |
| Evaluation bias | Single dataset | Use validation set, cross-comparison of epochs |

---

## 8. Expected Outcomes

- Quantitative assessment of architectural and regularization modifications via **mIoU**.
- Identification of effective decoder enhancements for SegFormer.
- Performance benchmarking for mIoU improvements on ADE20K.
- Insights into training schedule impact on transformer-based segmentation.
- Reproducible pipeline with documented code and experiments.

---
