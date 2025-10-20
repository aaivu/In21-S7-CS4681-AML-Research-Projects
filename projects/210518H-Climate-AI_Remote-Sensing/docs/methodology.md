# Methodology: Climate AI:Remote Sensing

**Student:** 210518H
**Research Area:** Climate AI:Remote Sensing
**Date:** 2025-10-15

## 1. Overview

The methodology involves fine-tuning pre-trained SatMAE models with architectural refinements for temporal and multi-spectral satellite imagery classification, under computational constraints using transfer learning.

## 2. Research Design

Overall approach: Transfer learning from SatMAE pre-trained weights (ViT-Large for temporal, ViT-Base for multi-spectral). Focus on fine-tuning enhancements: adaptive sequences, hybrid temporal encodings, spectral fusion attention. Evaluation via classification on benchmark datasets.

## 3. Data Collection

### 3.1 Data Sources
- Functional Map of the World (fMoW) dataset for RGB temporal sequences.
- fMoW-Sentinel, cross-referenced with Sentinel-2 for multi-spectral bands.

### 3.2 Data Description
- fMoW-RGB: High-res RGB images, 62 land use classes, temporal sequences; splits: 712,874 train / 84,939 val / 84,966 test.
- fMoW-Sentinel: 10-band multi-spectral (10-20m res, excluding 60m bands), same classes/splits.

### 3.3 Data Preprocessing
- Resize to 224x224 pixels.
- For temporal: Select up to T frames; random duplication if fewer; min-size cropping for consistency.
- For multi-spectral: Band grouping (Group1: B2-B4,B8; Group2: B5-B7,B8A; Group3: B11,B12).

## 4. Model Architecture

SatMAE-AdaFuse: Encoder-decoder ViT with MAE masking (75%). Temporal: Concatenate patch embeddings across T frames, add hybrid encodings (relative years + absolute month/hour). Multi-spectral: Group embeddings, ViT process, then multi-head attention (128 heads) on pooled group features for fusion, averaged to classifier.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Top-1 Accuracy (%)
- Train/Test Loss

### 5.2 Baseline Models
- Original SatMAE (fixed 3-frame temporal, basic spectral grouping)

### 5.3 Hardware/Software Requirements
- Single NVIDIA RTX 4070 Ti Super (16GB)
- PyTorch/Transformers; AdamW optimizer

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data collection and preprocessing (fMoW/Sentinel, sequence handling) | 2 weeks | Clean datasets with temporal/multi-spectral prep |
| Phase 2 | Implement refinements (adaptive seq, hybrid enc, fusion attention) on SatMAE weights | 3 weeks | Working SatMAE-AdaFuse model |
| Phase 3 | Fine-tuning experiments (5 epochs, ablations on seq len, dynamics) | 2 weeks | Accuracy/loss results, tables/figures |
| Phase 4 | Efficiency analysis and ablations | 1 week | Final report with validations |

## 7. Risk Analysis

Risk: Memory overflow with long sequences – Mitigation: Adjust batch size (e.g., 1 for len. 6+).  
Risk: Modest gains due to fine-tuning only – Mitigation: Validate via ablations; suggest full pre-training in future.  
Risk: Dataset inconsistencies (e.g., varying resolutions) – Mitigation: Min-cropping and duplication.

## 8. Expected Outcomes

Improved Top-1 accuracy (e.g., +1-2% over SatMAE) with minimal overhead, validating adaptations for SSL in RS. Contributions: Configurable framework for temporal/spectral handling in constrained environments.