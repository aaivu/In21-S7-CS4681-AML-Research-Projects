# Research Proposal: Cybersecurity AI:Threat Detection

**Student:** 210144G
**Research Area:** Cybersecurity AI:Threat Detection
**Date:** 2025-09-01

## Abstract

This research proposes enhancing CNN–LSTM intrusion detection systems with real-time response capabilities to address the growing sophistication of cyber threats. The project focuses on improving detection accuracy for minority attack classes (R2L, U2R) while maintaining operational efficiency. Using the NSL-KDD dataset, we will implement and enhance a hybrid CNN–LSTM baseline with focal loss for class imbalance, SMOTE for data augmentation, attention mechanisms for feature selection, and automated hyperparameter optimization. The system will be evaluated through streaming simulation to assess real-time performance. Expected outcomes include improved macro-F1 scores on imbalanced data, reduced false negative rates for critical attacks, and demonstrated feasibility for operational deployment with sub-second response times.

## 1. Introduction

The growing reliance on interconnected digital infrastructures has made organizations increasingly vulnerable to cyberattacks. These attacks exploit weaknesses in networks, applications, and user practices, often resulting in severe financial, reputational, and operational damage. Cyber threats have evolved from isolated incidents into sophisticated campaigns, employing advanced tactics to bypass security mechanisms and remain undetected for extended periods. Traditional signature-based defense mechanisms struggle to adapt to evolving attack patterns, necessitating the development of intelligent, behavior-based intrusion detection systems that can identify novel threats and respond in real-time.

## 2. Problem Statement

Current intrusion detection systems face several critical challenges: (1) severe class imbalance in network data where minority attack classes (R2L, U2R) are often missed despite their high impact; (2) the presence of vast amounts of heterogeneous and potentially irrelevant data that can obscure subtle attack patterns; (3) the need for real-time response capabilities that can keep up with network traffic while maintaining high detection accuracy; and (4) limited adaptability to evolving attack vectors that bypass traditional rule-based systems. This research addresses these challenges by developing an enhanced CNN–LSTM intrusion detection system with improved minority class detection and real-time response capabilities.

## 3. Literature Review Summary

Recent literature demonstrates the effectiveness of deep learning approaches for intrusion detection, particularly hybrid CNN–LSTM architectures that combine spatial feature extraction with temporal pattern recognition. However, significant gaps remain: (1) most studies focus on overall accuracy rather than minority class performance, leading to poor detection of high-impact attacks like U2R and R2L; (2) limited attention to real-time operational constraints and streaming data processing; (3) insufficient exploration of imbalance-aware training techniques like focal loss in the IDS context; and (4) lack of comprehensive evaluation frameworks that consider both detection performance and operational feasibility. This research builds on established CNN–LSTM foundations while addressing these critical gaps through targeted enhancements and comprehensive evaluation.

## 4. Research Objectives

### Primary Objective
Develop and evaluate an enhanced CNN–LSTM intrusion detection system that achieves improved macro-F1 performance on the NSL-KDD dataset, with particular focus on minority class detection (R2L, U2R) and real-time response capabilities.

### Secondary Objectives
- Implement focal loss and SMOTE techniques to address severe class imbalance in network intrusion data
- Integrate attention mechanisms to improve feature selection and model interpretability
- Develop automated hyperparameter optimization using Optuna for systematic performance tuning
- Evaluate system performance under streaming simulation conditions to assess real-time operational feasibility
- Establish comprehensive evaluation metrics that balance detection accuracy with operational constraints

## 5. Methodology

### 5.1 Dataset and Preprocessing
- Utilize NSL-KDD dataset with standardized train/test splits for fair comparison across studies
- Implement reproducible preprocessing pipeline: attack type mapping, one-hot encoding of categorical features, standardization of numeric features
- Use KDDTrain+ for training/validation and KDDTest+ for final evaluation to avoid data leakage
- Export preprocessing as versioned scikit-learn Pipeline for consistency across offline and streaming modes

### 5.2 Baseline Implementation
- Reproduce CNN–LSTM baseline architecture with CNN feature extraction followed by LSTM temporal aggregation
- Implement in TensorFlow with documented random seeds and hyperparameters for reproducibility
- Establish performance benchmarks on NSL-KDD for comparison with enhanced models

### 5.3 Model Enhancements
- **Focal Loss**: Replace cross-entropy with focal loss (FL(pt) = −α(1−pt)^γ log(pt)) to address class imbalance
- **SMOTE**: Implement Synthetic Minority Oversampling Technique for balanced training data
- **Attention Mechanisms**: Integrate attention layers for improved feature selection and interpretability
- **Hyperparameter Optimization**: Use Optuna for automated tuning with macro-F1 as primary objective

### 5.4 Real-Time Evaluation
- Implement streaming simulation to evaluate system performance under operational conditions
- Measure latency, throughput, and detection accuracy in near real-time scenarios
- Assess scalability and resource requirements for deployment feasibility

## 6. Expected Outcomes

- **Improved Detection Performance**: Enhanced macro-F1 scores, particularly for minority classes (R2L, U2R), compared to baseline CNN–LSTM models
- **Operational Feasibility**: Demonstrated real-time processing capabilities with sub-second response times suitable for network deployment
- **Reproducible Framework**: Well-documented, version-controlled implementation that enables replication and extension of results
- **Comprehensive Evaluation**: Detailed analysis of trade-offs between detection accuracy, computational efficiency, and operational constraints
- **Research Contributions**: Novel insights into the application of imbalance-aware techniques and attention mechanisms for network intrusion detection

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review and Dataset Preparation |
| 3-4  | Baseline CNN–LSTM Implementation and Reproduction |
| 5-6  | Implementation |
| 7-8  | Experimentation |
| 9    | Analysis and Writing |
| 10   | Final Submission |

## 8. Resources Required

### Datasets
- NSL-KDD dataset (publicly available) for training and evaluation
- Preprocessed and versioned data splits for reproducibility

### Software and Tools
- Python 3.8+ with TensorFlow/Keras for deep learning implementation
- Scikit-learn for preprocessing and traditional ML components
- Pandas, NumPy for data manipulation and analysis
- Matplotlib, Seaborn for visualization and results presentation

### Hardware
- GPU-enabled computing environment for efficient model training (NVIDIA GPU with CUDA support preferred)
- Sufficient memory (16GB+ RAM) for handling dataset preprocessing and model training
- Storage capacity for model checkpoints, experimental results, and dataset versions

### Development Environment
- Version control system (Git) for code management and reproducibility
- Jupyter notebooks for exploratory analysis and result visualization
- Cloud computing resources (if local hardware insufficient) for intensive training phases

## References

[1] S. S. Bamber, A. V. R. Katkuri, S. Sharma, and M. Angurala, "A hybrid CNN–LSTM approach for intelligent cyber intrusion detection system," Computers & Security, vol. 148, p. 104146, 2025.

[2] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," in 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications, pp. 1–6, 2009.

[3] Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, "Kitsune: An ensemble of autoencoders for online network intrusion detection," in Proc. NDSS, 2018.

[4] M. Mulyanto, M. Faisal, S. W. Prakosa, and J.-S. Leu, "Effectiveness of focal loss for minority classification in network intrusion detection systems," Symmetry, vol. 13, no. 1, p. 4, 2020.

[5] T. B. Shana, N. Kumari, M. Agarwal, S. Mondal, and U. Rathnayake, "Anomaly-based intrusion detection system based on SMOTE-IPF, whale optimization algorithm, and ensemble learning," Intelligent Systems with Applications, vol. 27, p. 200543, 2025.
