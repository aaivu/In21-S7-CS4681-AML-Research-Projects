# Research Proposal: Healthcare AI:Medical Imaging

**Student:** 210471F

**Research Area:** Healthcare AI:Medical Imaging

**Date:** 2025-09-15

## Abstract

This proposal focuses on developing an advanced deep learning architecture for medical image analysis, specifically for detecting subtle pathological features in CT scans. The project aims to address the challenges of early diagnosis, where conventional CNNs often fail to capture fine-grained patterns due to limited contrast, noise, and small datasets. To overcome these, a novel multi-branch ConvNeXt architecture is proposed, combining global average pooling, global max pooling, and attention-weighted pooling to capture both holistic and localized features. The research employs a rigorous pipeline consisting of data preprocessing, contrast enhancement, region-of-interest extraction, and augmentation for class balancing. A two-phase training strategy leveraging transfer learning is used for efficiency and robustness. Initial experiments on combined COVID-19 CT datasets demonstrated promising results, achieving ROC-AUC of 0.9937 and F1-score of 0.9825, outperforming many state-of-the-art models. This research has strong potential for broader applications in clinical diagnostics and serves as a step towards building AI systems capable of reliable, scalable, and interpretable medical imaging analysis.

## 1. Introduction

Medical imaging is a cornerstone of modern diagnostics, but its interpretation requires time and specialized expertise. During the COVID-19 pandemic, the limitations of RT-PCR testing highlighted CT scans as a crucial diagnostic tool.[1] However, analyzing CT scans is labor-intensive and subject to human variability. Artificial Intelligence (AI) methods, especially deep learning, have emerged as powerful tools to automate such tasks, but many models fall short in clinical robustness due to dataset limitations and inability to detect subtle features. This research seeks to bridge this gap by proposing a novel ConvNeXt-based multi-branch architecture to enhance diagnostic reliability.

## 2. Problem Statement

Existing deep learning models for CT scan analysis are limited in their ability to capture fine-grained, subtle pathological features.[2] They also struggle with dataset imbalance, noise, and generalization across sources. There is a need for an AI model that can accurately identify early-stage or subtle pathologies from CT scans with high sensitivity and specificity, making it clinically useful and robust.

## 3. Literature Review Summary

Early approaches relied on conventional CNNs and transfer learning from natural image datasets, achieving moderate accuracy but limited robustness. Later works introduced DenseNet, ResNet, EfficientNet, GAN-based augmentation, and hybrid models combining CNNs with Vision Transformers or Graph Neural Networks. While performance improved (95–96% accuracy in some cases), many models still faced challenges in detecting subtle features and generalizing across datasets. This gap highlights the need for architectures that integrate multi-scale feature extraction and domain-specific preprocessing to achieve state-of-the-art performance in clinical tasks.[3]

## 4. Research Objectives

### Primary Objective
To develop and evaluate a novel deep learning architecture capable of identifying subtle pathological features in medical CT scans, with the aim of improving diagnostic reliability and clinical applicability.

### Secondary Objectives
- To develop a preprocessing pipeline for enhancing contrast and isolating lung regions.

- To implement data augmentation techniques for class balancing and dataset expansion.

- To integrate and evaluate attention-weighted pooling alongside traditional pooling methods.

- To benchmark the proposed model against state-of-the-art approaches in medical imaging.

## 5. Methodology

The proposed research will adopt a systematic approach to design and evaluate a novel deep learning model for medical imaging:

- **Data Acquisition**: Publicly available CT scan datasets will be identified and combined where appropriate to ensure sufficient data diversity and robustness.

- **Preprocessing**: A preprocessing pipeline will be developed to standardize CT scans. This may include normalization, resizing, contrast enhancement, and region-of-interest extraction to highlight diagnostically relevant areas.

- **Data Augmentation**: To address limited dataset size and class imbalance, augmentation techniques (such as rotations, flips, intensity adjustments, and noise injection) will be applied to expand the training data and improve generalization.

- **Model Architecture**: A new deep learning architecture will be designed, drawing inspiration from recent advances in convolutional and attention-based models. The architecture will aim to capture both global and fine-grained pathological features, with the flexibility to incorporate multiple feature extraction strategies.

- **Training Strategy**: Transfer learning and staged fine-tuning will be considered to accelerate convergence and adapt pre-trained models to the medical imaging domain. Regularization techniques and training callbacks will be employed to prevent overfitting.

- **Evaluation**: The model will be assessed using clinically meaningful metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Its performance will be benchmarked against state-of-the-art methods reported in the literature.

## 6. Expected Outcomes

The research is expected to result in a deep learning model that can effectively detect subtle pathological features in CT scans, demonstrating improved diagnostic reliability compared to conventional approaches. The study also aims to provide:

- A validated preprocessing and augmentation pipeline tailored for medical imaging.

- A novel architectural framework capable of capturing both global and fine-grained features.

- Benchmark comparisons that highlight the potential of the proposed approach relative to existing state-of-the-art methods.

- Insights into how advanced AI techniques can be generalized to support a wide range of medical diagnostic applications beyond the initial case study.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-6  | Implementation |
| 7-8 | Experimentation |
| 9 | Analysis and Writing |
| 10   | Final Submission |

## 8. Resources Required

- Datasets: COVID-19 CT Lung and Infection Segmentation Dataset[4], MedSeg Covid Dataset 2[5].

- Hardware: GPU-enabled workstation (NVIDIA RTX 3080 or higher).

- Software: Python, TensorFlow/Keras, NumPy, OpenCV, Matplotlib.

- Libraries: Scikit-learn, Albumentations for augmentation, Grad-CAM for interpretability.

## References

<small>

[1] Deep learning approach for classifying CT images of COVID-19: A Systematic Review, accessed on August 22, 2025, https://www.researchgate.net/publication/364545817 Deep learning approach for classifying CT images of COVID-19 A Systematic Review

[2]  Z. Ye, Y. Zhang, Y. Wang, Z. Huang, and B. Song, “Chest CT manifestations of new coronavirus disease 2019 (COVID-19): a pictorial review,” Eur Radiol, vol. 30, no. 8, pp. 4381–4389, Aug. 2020, doi:10.1007/s00330-020-06801-0.

[3]  X. (Freddie) Liu, G. Karagoz, and N. Meratnia, “Analyzing the Impact of Data Augmentation on the Explainability of Deep Learning-Based Medical Image Classification,” Machine Learning and Knowledge Extraction, vol. 7, no. 1, p. 1, Mar. 2025, doi: 10.3390/make7010001.

[4] M. Jun et al., “COVID-19 CT Lung and Infection Segmentation Dataset.” Zenodo, Apr. 20, 2020. Accessed: Sept. 04, 2025. [Online]. Available: https://zenodo.org/records/3757476

[5] “MedSeg Covid Dataset 2.” figshare, Jan. 05, 2021. doi: 10.6084/m9.figshare.13521509.v2.
</small>
