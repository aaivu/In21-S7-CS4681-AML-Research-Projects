# Methodology: Neurotechnology:Brain-Computer Interface

**Student:** 210146N
**Research Area:** Neurotechnology:Brain-Computer Interface
**Date:** 2025-09-01

## 1. Overview

This study aims to improve EEG-based classification performance on standard cognitive paradigms using publicly available EEG-ExPy datasets. The focus is on N170 and P300 ERP decoding, with optional inclusion of SSVEP. The methodology combines classical and advanced deep learning approaches, including CNNs, MLPNNs, hybrid architectures, and transformers, to optimize single-trial and cross-subject performance.

## 2. Research Design

The research follows an experimental design using open EEG datasets. Baseline comparisons will replicate EEG-ExPy models (logistic regression, LDA, pyRiemann/TRCA). Advanced models will then be implemented and evaluated against these baselines. Experiments include both within-subject and cross-subject evaluations to assess generalization.

## 3. Data Collection

### 3.1 Data Sources

- EEG-ExPy example datasets:
- N170: Face vs. non-face visual stimuli (ERP CORE dataset)
- P300: Target vs. non-target visual stimuli
- SSVEP (optional): Frequency-based visual stimuli

### 3.2 Data Description

- Multi-channel EEG recordings (typically 32–64 channels)
- Sampling rates: 250–512 Hz
- Event markers indicating stimulus onset
- Balanced classes for target/non-target or stimulus categories

### 3.3 Data Preprocessing

- Bandpass filtering (e.g., 1-30 Hz) to remove drift and high-frequency noise
- Epoch extraction around stimulus onset (e.g., −200 ms to +800 ms)
- Baseline correction
- Optional artifact rejection using ICA or automated methods
- Normalization or standardization of channel data

## 4. Model Architecture

**Baseline models**: Logistic Regression, LDA, pyRiemann MDM (for comparison)
**Advanced models**:
- Convolutional Neural Networks (CNN) for temporal-spatial feature extraction
- Multilayer Perceptrons (MLPNN) for single-trial ERP decoding
- Hybrid LR-CNN or CNN-RNN architectures for combining global and local features
- Transformers for sequence modeling across EEG channels/time points

**Input**: preprocessed multi-channel EEG epochs
**Output**: class probabilities (e.g., face vs. house, target vs. non-target)

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Primary**: Accuracy (ACC), Area Under ROC Curve (AUC)
- **Optional**: F1-score, precision, recall, Information Transfer Rate (ITR) for SSVEP

### 5.2 Baseline Models

- Logistic Regression
- Linear Discriminant Analysis (LDA)
- pyRiemann MDM / TRCA (SSVEP)

### 5.3 Hardware/Software Requirements

- Python 3.10+
- PyTorch or TensorFlow for deep learning models
- Scikit-learn for baseline classifiers
- GPU recommended for CNN/transformer training

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean dataset |
| Phase 2 | Model implementation | 3 weeks | Working model |
| Phase 3 | Experiments | 2 weeks | Results |
| Phase 4 | Analysis | 1 week | Final report |

## 7. Risk Analysis

- **Overfitting**: Mitigate with cross-validation, dropout, early stopping
- **Low SNR in EEG**: Apply artifact rejection and data augmentation
- **Cross-subject variability**: Use domain adaptation or transfer learning
- **Computational constraints**: Optimize model size; consider lightweight CNNs for faster training

## 8. Expected Outcomes

- Improved classification accuracy and AUC for N170 and P300 paradigms compared to EEG-ExPy baselines
- Demonstration of generalizable models for cross-subject decoding
- Insights into the effectiveness of hybrid and transformer architectures for EEG classification
- A reusable methodology for researchers using EEG-ExPy datasets