# Research Proposal: Neurotechnology:Brain-Computer Interface

**Student:** 210146N
**Research Area:** Neurotechnology:Brain-Computer Interface
**Date:** 2025-09-01

## Abstract

This research proposes improving EEG-based classification performance on standard cognitive paradigms using the EEG-ExPy benchmark datasets. Focusing primarily on N170 and P300 event-related potentials, with optional inclusion of SSVEP, the study aims to enhance single-trial and cross-subject decoding using advanced machine learning and deep learning approaches. Baseline methods such as logistic regression, linear discriminant analysis, and pyRiemann/TRCA will be compared against multilayer perceptrons, CNNs, hybrid LR-CNN architectures, and transformer models. The project will evaluate classification accuracy (ACC) and area under the ROC curve (AUC) as primary metrics, with cross-subject generalization and information transfer rates as secondary measures. Results are expected to demonstrate improved accuracy, robustness, and generalization over existing baselines, providing a reproducible methodology for EEG-ExPy users and contributing to the advancement of reliable EEG-based brain-computer interfaces.

## 1. Introduction

Brain-Computer Interfaces (BCIs) allow direct communication between neural activity and external devices, with EEG being the most widely used non-invasive modality. Accurate single-trial classification of event-related potentials is critical for cognitive neuroscience research and practical BCI applications. EEG-ExPy provides openly accessible datasets and baseline models for N170, P300, and SSVEP paradigms, offering a foundation for benchmarking improvements in decoding performance. This study focuses on leveraging advanced neural network architectures to enhance classification accuracy while maintaining interpretability and generalization across participants.

## 2. Problem Statement

While EEG-ExPy provides baseline classification using logistic regression, LDA, and TRCA, these approaches achieve only moderate performance, particularly in low signal-to-noise conditions or cross-subject scenarios. There is a need for models that can improve single-trial decoding, reduce calibration requirements, and generalize reliably across different subjects. This research addresses the problem of enhancing EEG classification for N170 and P300 paradigms by applying advanced machine learning techniques to the existing EEG-ExPy datasets.

## 3. Literature Review Summary

Recent studies have shown that multilayer perceptrons and CNN-based architectures outperform traditional linear models for single-trial ERP decoding, achieving higher accuracy and better cross-subject generalization. Hybrid models, such as LR-CNN, and transformer-based networks further capture temporal and spatial dependencies in EEG data, improving robustness across subjects. However, these improvements have not yet been systematically applied as drop-in replacements for EEG-ExPy baselines, leaving a clear opportunity for evaluation and benchmarking.

## 4. Research Objectives

### Primary Objective
- To improve classification accuracy and AUC for N170 and P300 paradigms using EEG-ExPy datasets through advanced machine learning and deep learning models.

### Secondary Objectives
- Evaluate cross-subject generalization and robustness of the proposed models.
- Compare hybrid and transformer architectures with baseline models.
- Provide a reproducible methodology for researchers using EEG-ExPy datasets.

## 5. Methodology

The study will use EEG-ExPy datasets for N170 and P300 experiments, with optional inclusion of SSVEP. Preprocessing steps include bandpass filtering, epoch extraction, baseline correction, artifact rejection, and normalization. Baseline models (logistic regression, LDA, pyRiemann/TRCA) will be implemented to establish reference performance. Advanced models—multilayer perceptrons, CNNs, hybrid LR-CNN, and transformer architectures—will be trained on the preprocessed data. Evaluation will be performed using stratified cross-validation for within-subject and cross-subject experiments, measuring ACC and AUC primarily, with secondary metrics including cross-subject generalization and information transfer rate for SSVEP.

## 6. Expected Outcomes

- Improved ACC and AUC compared with EEG-ExPy baseline classifiers.
- Demonstration of model robustness and cross-subject generalization.
- Recommendations for drop-in advanced models to enhance EEG-ExPy experiments.
- A reproducible framework for EEG classification research that can be extended to other ERP paradigms.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5  | Implementation |
| 6-7 | Experimentation |
| 8-9| Analysis and Writing |
| 10   | Final Submission |

## 8. Resources Required

- EEG-ExPy datasets (N170, P300, optional SSVEP)
- Python 3.10+ with PyTorch/TensorFlow and Scikit-learn
- GPU-equipped workstation for deep learning experiments
- Libraries for EEG preprocessing (MNE, NumPy, SciPy)

## References

[1] Yi Liu, Zach Quince, Steven Goh, Shoryu Teragawa, and Tobias Low. A lightweight deep learning model for eeg classification across visual stimuli. University of Southern Queensland preprint, 2023.
[2] G. Zhang et al. Assessing the effectiveness of spatial pca on svm-based decoding of eeg data. NeuroImage, 2023.
[3] R. Afrah, Z. Amini, and R. Kafieh. An unsupervised feature extraction method based on clstm-ae for accurate p300 classification. Journal of Biomedical Physics & Engineering, 2024.
[4] D. Borra et al. Ms-eegnet: A lightweight multi-scale convolutional neural network for p300 decoding. Journal of Neural Engineering, 2021.
[5] J. Chen, Y. Zhang, P. Peng, et al. A transformer-based deep neural network model for ssvep classification (ssvepformer). arXiv preprint, 2022.
[6] Y. Dai, Z. Chen, et al. A time-frequency feature fusion-based deep learning network for ssvep frequency recognition (ssvep-tffnet). Frontiers in Neuroscience, 2025.