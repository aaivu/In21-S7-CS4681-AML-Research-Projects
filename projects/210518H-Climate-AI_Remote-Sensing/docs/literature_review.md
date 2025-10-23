# Literature Review: Climate AI:Remote Sensing

**Student:** 210518H
**Research Area:** Climate AI:Remote Sensing
**Date:** 2025-10-15

## Abstract

This literature review covers self-supervised learning (SSL) for satellite imagery, focusing on transformer-based methods like Vision Transformers (ViT) and Masked Autoencoders (MAE) adapted for temporal and multi-spectral data. Key findings include the shift from contrastive to generative SSL for robustness in remote sensing, the importance of temporal modeling in Satellite Image Time Series (SITS) for capturing dynamics, and spectral fusion techniques for inter-band dependencies. Gaps identified: fixed temporal sequences limiting long-term dependencies and basic spectral groupings underutilizing correlations, addressed in recent enhancements.

## 1. Introduction

The research area of Climate AI in Remote Sensing involves using AI to analyze satellite data for Earth observation tasks like land cover classification, change detection, and environmental monitoring. This review scopes recent advancements in SSL for satellite imagery, emphasizing transformer architectures and adaptations for temporal/multi-spectral characteristics, drawing from 2018-2025 literature to inform improvements to SatMAE for resource-constrained fine-tuning.

## 2. Search Methodology

### Search Terms Used
- "self-supervised learning satellite imagery"
- "masked autoencoders remote sensing"
- "temporal satellite image time series"
- "multi-spectral transformer fusion"
- Synonyms and variations: "SSL SITS", "MAE ViT satellite", "contrastive learning RS", "spectral attention hyperspectral"

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv

### Time Period
2018-2025, focusing on recent developments in transformers and SSL, with seminal papers from earlier (e.g., 2010 for trend detection).

## 3. Key Areas of Research

### 3.1 Self-Supervised Learning in Computer Vision
SSL has shifted from contrastive methods requiring positive/negative pairs to generative approaches like MAE for efficiency. Transformers enable long-range dependencies.

**Key Papers:**
- Dosovitskiy et al., 2020 [4] - Introduces ViT, treating images as patch sequences for scalable recognition, outperforming CNNs with large data.
- He et al., 2022 [5] - Proposes MAE with asymmetric encoder-decoder for masked patch reconstruction, achieving SOTA on ImageNet/COCO without labels.
- Vaswani et al., 2017 [10] - Defines transformer with self-attention, foundational for ViT/MAE.
- Jaiswal et al., 2021 [11] - Surveys contrastive SSL (e.g., MoCo, SimCLR), noting challenges in augmentations.

### 3.2 Self-Supervised Learning for Satellite Imagery
Adapts SSL to satellite specifics: irregular timing, multi-spectral bands. Generative > contrastive for robustness.

**Key Papers:**
- Wang et al., 2022 [3] - Reviews SSL in RS, emphasizing unlabeled data leverage for representations.
- Cong et al., 2022 [8] - SatMAE adapts MAE with temporal/spectral encodings, improving classification by 7-14%.
- Manas et al., 2021 [14] - SeCo uses seasonal contrasts for positive pairs in RS pre-training.
- Ayush et al., 2021 [15] - GASSL incorporates geography for contrastive pairs, but relies on curation.

### 3.3 Satellite Image Time Series (SITS) Analysis
Focuses on temporal dynamics for change/trend detection.

**Key Papers:**
- Miller et al., 2023 [1] - Reviews DL for SITS, highlighting architectures for phenology/forecasting.
- Pelletier et al., 2019 [17] - TempCNN for SITS classification, outperforming RF/recurrent.
- Garnot and Landrieu, 2021 [18] - Temporal attention for panoptic segmentation in SITS.
- Khiali et al., 2021 [19] - Reviews DL for SITS prediction, stressing LSTMs/attention.
- Verbesselt et al., 2010 [20] - Detects trends/seasonal changes in time series.
- He et al., 2021 [21] - Conditional pixel synthesis for spatial-temporal super-resolution.
- Vincent et al., 2024 [22] - Analyzes semantic change detection in SITS with domain shifts.
- Interdonato et al., 2024 [23] - Studies augmentations for SITS classification/cross-year adaptation.
- Zhang et al., 2024 [24] - Inception-enhanced temporal attention for fine-grained SITS classification.
- Zhang et al., 2025 [25] - 3D spatiotemporal convolutions for complex sequence prediction.

### 3.4 Multi-Spectral and Hyperspectral Processing
Models band dependencies beyond RGB.

**Key Papers:**
- Phiri et al., 2020 [2] - Reviews Sentinel-2 for land cover mapping.
- Hong et al., 2022 [26] - SpectralFormer uses transformers for hyperspectral classification via spectral attention.
- Cai et al., 2022 [27] - MST++ for multi-stage spectral reconstruction from RGB.
- Xia et al., 2024 [28] - ViT-CoMer fuses CNN-transformer for dense predictions.

## 4. Research Gaps and Opportunities

### Gap 1: Fixed temporal sequences in existing models limit capturing long-term dependencies in irregular SITS.
**Why it matters:** Critical for applications like urban tracking or deforestation [1, 21], but SatMAE's fixed 3-frames underperforms on varying data [19, 23].
**How your project addresses it:** Introduces adaptive sequences configurable at runtime, extending to 7-8 frames with random duplication/min-cropping.

### Gap 2: Basic spectral grouping fails to exploit inter-band correlations in multi-spectral data.
**Why it matters:** Vegetation/land indices require adaptive weighting [2, 26]; simple approaches miss dependencies [27, 28].
**How your project addresses it:** Adds multi-head attention for fusion post-grouping, learning informative combinations.

## 5. Theoretical Framework

Built on ViT [4] for patch-based transformers, MAE [5] for masked reconstruction, and SatMAE [8] for satellite adaptations. Positional encodings [10] extended for temporal/spectral. Theoretical basis: Self-attention captures long-range dependencies [10]; generative SSL robust to RS noise [3, 11].

## 6. Methodology Insights

Common: Pre-training with masking/contrastive [11-15], fine-tuning on classification [8]. Promising for this work: Generative MAE [5] over contrastive [12-15] for satellite irregularities; temporal attention [18, 24] and spectral transformers [26-28] for fusion. Ablations and efficiency analysis [Table IV] key under constraints.

## 7. Conclusion

SSL advancements enable effective use of unlabeled satellite data, with transformers excelling in temporal/spectral modeling. Gaps in adaptability inform this project's refinements to SatMAE, promising better performance in resource-limited settings for climate/environmental tasks.

## References

1. L. Miller, C. Pelletier, and G. I. Webb, “Deep learning for satellite image time series analysis: A review,” Remote Sens., vol. 15, no. 11, p. 2844, 2023.
2. D. Phiri et al., “Sentinel-2 data for land cover/use mapping: A review,” Remote Sens., vol. 12, no. 14, p. 2291, 2020.
3. Y. Wang et al., “Self-supervised learning in remote sensing: A review,” IEEE Geosci. Remote Sens. Mag., vol. 10, no. 4, pp. 213–247, 2022.
4. A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv:2010.11929, 2020.
5. K. He et al., “Masked autoencoders are scalable vision learners,” in Proc. CVPR, 2022, pp. 16000–16009.
6. J. Deng et al., “ImageNet: A large-scale hierarchical image database,” in Proc. CVPR, 2009, pp. 248–255.
7. T.-Y. Lin et al., “Microsoft COCO: Common objects in context,” in Proc. ECCV, 2014, pp. 740–755.
8. Y. Cong et al., “SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Proc. NeurIPS, vol. 35, 2022, pp. 197–211.
9. G. Christie et al., “Functional map of the world,” in Proc. CVPR, 2018.
10. A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017, pp. 5998–6008.
11. A. Jaiswal et al., “A survey on contrastive self-supervised learning,” Technologies, vol. 9, no. 1, p. 2, 2021.
12. K. He et al., “Momentum contrast for unsupervised visual representation learning,” in Proc. CVPR, 2020, pp. 9729–9738.
13. T. Chen et al., “A simple framework for contrastive learning of visual representations,” in Proc. ICML, 2020, pp. 1597–1607.
14. O. Mañas et al., “Seasonal contrast: Unsupervised pre-training from uncurated remote sensing data,” in Proc. ICCV, 2021, pp. 9414–9423.
15. K. Ayush et al., “Geography-aware self-supervised learning,” in Proc. ICCV, 2021, pp. 10181–10190.
16. M. Rußwurm and M. Körner, “Convolutional LSTMs for cloud-robust segmentation of remote sensing imagery,” arXiv:1811.02471, 2018.
17. C. Pelletier et al., “Temporal convolutional neural network for the classification of satellite image time series,” Remote Sens., vol. 11, no. 5, p. 523, 2019.
18. V. Sainte Fare Garnot and L. Landrieu, “Panoptic segmentation of satellite image time series with convolutional temporal attention networks,” in Proc. ICCV, 2021, pp. 4872–4881.
19. L. Khiali et al., “Application of deep learning architectures for satellite image time series prediction: A review,” Remote Sens., vol. 13, no. 23, p. 4822, 2021.
20. J. Verbesselt et al., “Detecting trend and seasonal changes in satellite image time series,” Remote Sens. Environ., vol. 114, no. 1, pp. 106–115, 2010.
21. Y. He et al., “Spatial-temporal super-resolution of satellite imagery via conditional pixel synthesis,” in Proc. NeurIPS, vol. 34, 2021, pp. 27903–27915.
22. E. Vincent et al., “Satellite image time series semantic change detection: Novel architecture and analysis of domain shift,” arXiv:2407.07616, 2024.
23. T. Interdonato et al., “An empirical study on data augmentation for pixelwise satellite image time-series classification and cross-year adaptation,” IEEE Geosci. Remote Sens. Lett., 2024.
24. Z. Zhang et al., “Satellite image time-series classification with inception-enhanced temporal attention encoder,” Remote Sens., vol. 16, no. 23, p. 4579, 2024.
25. Z. Zhang et al., “3D long time spatiotemporal convolution for complex sequence prediction,” Sci. Rep., vol. 15, no. 1, p. 13828, 2025.
26. D. Hong et al., “SpectralFormer: Rethinking hyperspectral image classification with transformers,” IEEE Trans. Geosci. Remote Sens., vol. 60, pp. 1–15, 2022.
27. Y. Cai et al., “MST++: Multi-stage spectral-wise transformer for efficient spectral reconstruction,” in Proc. CVPRW, 2022, pp. 745–755.
28. C. Xia et al., “ViT-CoMer: Vision transformer with convolutional multi-scale feature interaction for dense predictions,” arXiv:2403.07392, 2024.