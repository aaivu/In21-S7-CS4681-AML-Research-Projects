# Research Proposal: Climate AI:Remote Sensing

**Student:** 210518H
**Research Area:** Climate AI:Remote Sensing
**Date:** 2025-10-15

## Abstract

Satellite imagery is vital for climate monitoring, but labeled data scarcity hinders progress. This proposal introduces SatMAE-AdaFuse, refining SatMAE for SSL on temporal/multi-spectral data under compute constraints. Key innovations: adaptive temporal sequences (configurable lengths), hybrid encodings (relative years + absolute cyclic features), and spectral fusion attention. Fine-tuning on fMoW datasets yields up to 76.57% Top-1 accuracy (+1.52%) for temporal and 69.77% (+0.41%) for multi-spectral, validating gains. This advances efficient SSL for tasks like land cover and change detection, accessible for resource-limited research.

## 1. Introduction

In Climate AI for Remote Sensing, satellite data enables tracking environmental changes, but annotation costs limit supervised methods. SSL leverages unlabeled data; SatMAE adapts MAE for satellite specifics, but fixed sequences and basic fusion limit performance. This research proposes enhancements for better fine-tuning under constraints, significant for global Earth observation.

## 2. Problem Statement

SatMAE's fixed 3-frame temporal limits long-term dependency capture in irregular SITS; basic spectral grouping misses inter-band correlations. Under compute limits, full pre-training is infeasible, necessitating efficient fine-tuning refinements.

## 3. Literature Review Summary

SSL in RS [3] shifts to generative [5,8] over contrastive [11-15]. SITS needs temporal modeling [1,16-25]; multi-spectral requires fusion [2,26-28]. Gaps: Non-adaptive sequences [19], simple band handling [26] – this project addresses via configurable extensions.

## 4. Research Objectives

### Primary Objective
Enhance SatMAE fine-tuning for improved accuracy in temporal/multi-spectral satellite classification under constraints.

### Secondary Objectives
- Implement adaptive temporal sequencing for variable lengths.
- Develop hybrid temporal encodings for better seasonal/diurnal capture.
- Integrate spectral fusion attention for inter-band dependencies.

## 5. Methodology

Transfer learning: Fine-tune SatMAE weights with adaptations. Data: fMoW-RGB/Sentinel, preprocessed (resize, crop). Model: ViT-based with hybrid encodings and attention fusion. Train: 5 epochs, AdamW, ablations. Eval: Top-1 Acc on test splits.

## 6. Expected Outcomes

Higher accuracy (e.g., +1-2% over baseline), validated framework for RS SSL, implications for change detection/monitoring. Open-source code for adoption.

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

- Datasets: fMoW, Sentinel-2 (public).
- Tools: PyTorch, SatMAE codebase.
- Hardware: Single NVIDIA RTX 4070 Ti Super GPU.

## References

L. Miller et al., “Deep learning for satellite image time series analysis: A review,” Remote Sens., vol. 15, no. 11, p. 2844, 2023.  
D. Phiri et al., “Sentinel-2 data for land cover/use mapping: A review,” Remote Sens., vol. 12, no. 14, p. 2291, 2020.  
Y. Wang et al., “Self-supervised learning in remote sensing: A review,” IEEE Geosci. Remote Sens. Mag., vol. 10, no. 4, pp. 213–247, 2022.  
A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv:2010.11929, 2020.  
K. He et al., “Masked autoencoders are scalable vision learners,” in Proc. CVPR, 2022, pp. 16000–16009.  
Y. Cong et al., “SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Proc. NeurIPS, vol. 35, 2022, pp. 197–211.  
G. Christie et al., “Functional map of the world,” in Proc. CVPR, 2018.  
A. Jaiswal et al., “A survey on contrastive self-supervised learning,” Technologies, vol. 9, no. 1, p. 2, 2021.  
L. Khiali et al., “Application of deep learning architectures for satellite image time series prediction: A review,” Remote Sens., vol. 13, no. 23, p. 4822, 2021.  
D. Hong et al., “SpectralFormer: Rethinking hyperspectral image classification with transformers,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–15, 2022.