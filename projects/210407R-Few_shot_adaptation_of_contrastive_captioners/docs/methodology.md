# Methodology: Few-Shot Adaptation of Contrastive Captioners

**Student:** 210407R  
**Research Area:** Multimodal Foundation Models for Few-Shot Learning  
**Date:** 2025-10-20

## 1. Overview

This study presents a comprehensive empirical investigation on few-shot adaptation of the Contrastive Captioners (CoCa) model for image classification. We systematically evaluate a hierarchy of adaptation methods ranging from parameter-free approaches to parameter-efficient fine-tuning techniques. The research addresses the challenge of adapting large-scale multimodal models to downstream tasks with sparse labeled data, avoiding computational costs and overfitting associated with full fine-tuning.

## 2. Research Design

We employ a comparative methodology evaluating three progressively complex adaptation strategies:

1. **Parameter-free Hybrid Prototype Method** - Leverages CoCa's multimodal nature without any training
2. **Linear Probing** - Trains only a classification head with frozen visual encoder
3. **LoRA Fine-Tuning** - Parameter-efficient adaptation of the visual encoder with hybrid loss functions

The study evaluates performance across varying data regimes (1, 3, 5, 10, and 20 shots) to understand data-dependent trade-offs between different adaptation approaches.

## 3. Data Collection

### 3.1 Data Sources

Mini-ImageNet dataset, a standard few-shot learning benchmark containing 100 classes with 600 images each.

### 3.2 Data Description

Using official dataset splits, we construct balanced few-shot datasets:
- **Training split:** First 20 images per class (2,000 images total)
- **Validation split:** First 20 images per class (2,000 images total)
- **Test split:** Next 20 images per class (2,000 images total)
- **Total:** 6,000 images formatted in Image Folder structure

### 3.3 Data Preprocessing

All images are preprocessed using the Mini-ImageNet standard pipeline. For linear probing experiments, data augmentation strategies are applied including:
- Random Resized Crop
- Horizontal Flip
- Color Jitter
- Random Grayscale

Strong augmentation is applied in low-shot scenarios to add feature diversity and improve generalization.

## 4. Model Architecture

### 4.1 Base Model

Pre-trained CoCa model (ViT-L/14) with weights from mscoco_finetuned_laion2B-s13B-b90k. The model combines contrastive and generative pre-training paradigms, featuring:
- Image encoder (ViT-based)
- Unimodal text decoder
- Multimodal decoder for captioning

### 4.2 Adaptation Strategy 1: Hybrid Prototype Classification

A training-free method that fuses visual and textual embeddings:

**Visual Prototype:** Mean of normalized image embeddings
$$P_{image}(c) = \text{normalize}\left(\frac{1}{N}\sum_{i=1}^{N} f_{img}(I_i)\right)$$

**Textual Prototype:** Mean of text embeddings with prompt ensembling (e.g., "an image of [class]")
$$P_{text}(c) = \text{normalize}\left(\frac{1}{M}\sum_{j=1}^{M} f_{txt}(T_j)\right)$$

**Hybrid Fusion:** Weighted combination with hyperparameter α
$$P_{hybrid}(c) = \text{normalize}((1-\alpha)P_{image}(c) + \alpha P_{text}(c))$$

Classification via cosine similarity between query embedding and hybrid prototypes.

### 4.3 Adaptation Strategy 2: Linear Probing

A linear classification head is attached to the frozen CoCa image encoder. Only the head parameters are trained using cross-entropy loss with label smoothing. The frozen encoder preserves pre-trained multimodal knowledge while enabling efficient adaptation.

### 4.4 Adaptation Strategy 3: LoRA Fine-Tuning

Low-Rank Adaptation constrains weight updates through low-rank decomposition:
$$h = W_0 x + \Delta W x = W_0 x + BAx$$

where $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{k \times r}$, and $r \ll \min(d, k)$.

**Adaptive LoRA Configuration:**
- **1-2 shots:** Rank r = 4, applied to attn.out_proj only
- **3-10 shots:** Rank r = 8, applied to attn.out_proj
- **>10 shots:** Rank r = 16, applied to attn.out_proj, mlp.c_fc, and mlp.c_proj

**Hybrid Loss Function:**
$$L_{total} = L_{metric} + L_{CE}$$

Three loss functions are investigated:
- **Cross-Entropy Loss:** Standard classification loss
- **Prototypical Loss:** Metric learning loss minimizing Euclidean distance between queries and class prototypes
- **Supervised Contrastive Loss:** Metric learning loss structuring embedding space by pulling same-class samples together

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Accuracy (%):** Mean classification accuracy across test set
- **Number of trainable parameters:** For efficiency comparison
- **Performance variance:** Standard deviation across 3 random seeds

### 5.2 Experimental Protocol

- **Shot settings:** N ∈ {1, 3, 5, 10, 20}
- **Test set:** Fixed 2,000 images (20 per class across 100 classes)
- **Repetitions:** Each experiment repeated 3 times with different random seeds
- **Hyperparameter search:** Manual tuning for each scenario to maximize performance
- **Hybrid prototype:** Optimal text weight α searched for each shot setting

### 5.3 Baseline Models

The three adaptation strategies serve as baselines for comparison:
1. Hybrid prototype (parameter-free) establishes the baseline for multimodal fusion
2. Linear probing (minimal training) provides efficiency baseline
3. LoRA with cross-entropy (standard fine-tuning) enables loss function comparison

### 5.4 Hardware/Software Requirements

- **Framework:** PyTorch with Hugging Face Transformers library
- **Datasets:** Hugging Face datasets library
- **GPU:** NVIDIA GPU (16GB+ recommended)
- **Model weights:** Pre-trained CoCa-ViT-L/14 from Hugging Face Model Hub

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preparation and preprocessing | 1 week | Mini-ImageNet few-shot splits prepared |
| Phase 2 | Hybrid prototype implementation | 1 week | Parameter-free baseline results |
| Phase 3 | Linear probing setup and experiments | 2 weeks | Augmentation and hyperparameter analysis |
| Phase 4 | LoRA implementation and loss function study | 2 weeks | Results across all loss functions |
| Phase 5 | Analysis and comparison | 1 week | Comprehensive performance analysis |
| Phase 6 | Documentation and reporting | 1 week | Final research paper and results |

## 7. Risk Analysis

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Computational resource constraints | Medium | Use gradient checkpointing and lower batch sizes if needed |
| Hyperparameter sensitivity | High | Systematic grid search and multiple random seeds |
| Limited data overfitting | Medium | Strong regularization, augmentation, and validation monitoring |
| Memory limitations for large models | Medium | Use lower-rank LoRA configurations; implement gradient accumulation |

## 8. Expected Outcomes

**Research Contributions:**
- Empirical evaluation of CoCa's effectiveness in few-shot learning scenarios
- Identification of optimal adaptation strategy based on data availability
- Demonstration of data-dependent trade-offs between metric-based and cross-entropy losses
- Practical guidelines for efficient adaptation of multimodal foundation models

**Key Findings Expected:**
- Hybrid prototype approach achieves strong performance in extremely low-shot settings (1-5 shots)
- Metric-based losses (Prototypical, SupCon) outperform cross-entropy in low-data regimes
- Cross-entropy becomes competitive with increased samples (20+ shots)
- Adaptive LoRA configuration provides consistent improvements across shot settings
- CoCa's pre-trained multimodal representations are highly effective for few-shot learning without complex adaptation