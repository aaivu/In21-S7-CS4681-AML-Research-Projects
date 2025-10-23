# Literature Review: Neurotechnology:Brain-Computer Interface

**Student:** 210146N
**Research Area:** Neurotechnology:Brain-Computer Interface
**Date:** 2025-09-01

## Abstract

This literature review examines recent developments in EEG-based Brain-Computer Interfaces (BCIs), focusing on the use of standardized benchmarks for classification of event-related potentials (ERPs) and steady-state visual evoked potentials (SSVEPs). The EEG-ExPy framework provides example datasets and baseline classifiers for N170, P300, and SSVEP paradigms. Recent advances show that machine learning and deep learning approaches, including multilayer perceptrons, convolutional neural networks, and hybrid architectures, can significantly improve classification accuracy, cross-subject generalization, and information transfer rates compared with traditional linear models.

## 1. Introduction

Brain-Computer Interfaces (BCIs) translate neural activity into commands for external devices. Non-invasive EEG is widely used due to its high temporal resolution and ease of use. Accurate EEG classification is crucial for real-time BCI applications, including cognitive assessment, communication, and assistive technologies. The EEG-ExPy framework provides open-source Python notebooks with example EEG datasets and baseline classifiers such as logistic regression, linear discriminant analysis, and TRCA, serving as a foundation for evaluating novel classification methods.

## 2. Search Methodology

### Search Terms Used
- EEG classification
- Brain-Computer Interface (BCI)
- EEG-ExPy, EEG-Notebooks
- N170, P300, SSVEP
- Deep learning, CNN, ResNet, MLP

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [ ] Other: ___________

### Time Period
2018–2025, focusing on recent deep learning and hybrid-model approaches for EEG classification.

## 3. Key Areas of Research

### 3.1 N170-related EEG Experiments
N170 ERPs are typically elicited in response to faces and other visual stimuli. Classification of N170 components is challenging due to low signal-to-noise ratios. Approaches using multilayer perceptrons and lightweight convolutional neural networks have demonstrated substantial improvements over traditional linear classifiers. These methods leverage raw EEG signals and learn spatial-temporal patterns directly, achieving high single-trial classification accuracy even under low-amplitude conditions.

**Key Papers:**
- Yi Liu, Zach Quince, Steven Goh, Shoryu Teragawa, and Tobias Low. A lightweight deep learning model for eeg classification across visual stimuli. University of Southern Queensland preprint, 2023.
- G. Zhang et al. Assessing the effectiveness of spatial pca on svm-based decoding of eeg data. NeuroImage, 2023.

### 3.2 P300-related EEG Experiments
P300 detection relies on distinguishing target from non-target stimuli. Traditional classifiers perform well within subjects but often struggle with cross-subject generalization. Hybrid approaches that combine logistic regression with convolutional neural networks capture both global and subject-specific features, enabling robust performance across individuals. Multi-scale convolutional architectures also provide improved feature representation, reducing calibration time and enhancing classification reliability.

**Key Papers:**
- R. Afrah, Z. Amini, and R. Kafieh. An unsupervised feature extraction method based on clstm-ae for accurate p300 classification. Journal of Biomedical Physics & Engineering, 2024.
- D. Borra et al. Ms-eegnet: A lightweight multi-scale convolutional neural network for p300 decoding. Journal of Neural Engineering, 2021.

### 3.3 SSVEP-related EEG Experiments
SSVEP paradigms require decoding periodic brain responses to visual flickers. Canonical correlation analysis and TRCA provide moderate accuracy but can be enhanced by combining spatial filtering with deep neural networks. Hybrid models that integrate sub-band CNNs with task-related component analysis improve both classification accuracy and information transfer rate, particularly for short data segments, by capturing complementary temporal and spatial features.

**Key Papers:**
- J. Chen, Y. Zhang, P. Peng, et al. A transformer-based deep neural network model for ssvep classification (ssvepformer). arXiv preprint, 2022.
- Y. Dai, Z. Chen, et al. A time-frequency feature fusion-based deep learning network for ssvep frequency recognition (ssvep-tffnet). Frontiers in Neuroscience, 2025.

## 4. Research Gaps and Opportunities

### Gap 1: Cross-Paradigm Generalization
**Why it matters**: Models trained on a single paradigm may not generalize to others, limiting BCI flexibility.
**How the project addresses it**: Investigate multi-paradigm architectures and transformer-based models to enable consistent performance across N170, P300, and SSVEP datasets.

### Gap 2: Cross-Subject Robustness
**Why it matters**: Inter-subject variability can degrade classifier performance.
**How the project addresses it**: Apply subject-adaptive learning and domain adaptation techniques to improve generalization across different participants.

### Gap 3: Real-Time Deployment
**Why it matters**: High computational complexity of deep models can hinder real-time BCI applications.
**How the project addresses it**: Focus on lightweight models such as ResNet-18 or compact sub-band CNNs for efficient inference without sacrificing accuracy.

## 5. Theoretical Framework

The review builds on the neurophysiological basis of ERP components and SSVEP responses. Deep learning models are used to extract spatial-temporal patterns from EEG signals, while hybrid methods combine interpretable linear classifiers with neural network representations to balance accuracy, robustness, and interpretability.

## 6. Methodology Insights

Common methodologies include:

- Convolutional Neural Networks for temporal-spatial feature extraction
- Multilayer perceptrons for single-trial ERP detection
- Hybrid architectures combining linear models with CNNs for cross-subject generalization
- Sub-band CNNs integrated with task-related component analysis for SSVEP decoding

Lightweight and hybrid approaches provide the best combination of accuracy, generalization, and computational efficiency for EEG-ExPy datasets.

## 7. Conclusion

The EEG-ExPy benchmark provides open datasets and baseline models for N170, P300, and SSVEP paradigms. Recent studies show that advanced machine learning and deep learning methods dramatically improve classification accuracy, generalization, and information transfer rates compared to traditional linear methods. These improvements enable more reliable and efficient EEG-based BCIs, offering practical strategies for researchers seeking to enhance EEG classification using publicly available data.

## References

[1] R. Afrah, Z. Amini, and R. Kafieh. An unsupervised feature extraction method based on clstm-ae for accurate p300 classification. Journal of Biomedical Physics & Engineering, 2024.

[2] Bruno Aristimunha, Raphael Y. de Camargo, Walter H. Lopez Pinaya, Sylvain Chevallier, Alexandre Gramfort, and Cedric Rommel. Evaluating the structure of cognitive tasks with transfer learning. arXiv preprint, 2023.

[3] D. Borra et al. Ms-eegnet: A lightweight multi-scale convolutional neural network for p300 decoding. Journal of Neural Engineering, 2021.

[4] J. Chen, Y. Zhang, P. Peng, et al. A transformer-based deep neural network model for ssvep classification (ssvepformer). arXiv preprint, 2022.

[5] Y. Dai, Z. Chen, et al. A time-frequency feature fusion-based deep learning network for ssvep frequency recognition (ssvep-tffnet). Frontiers in Neuroscience, 2025.

[6] W. Dong, C. Xu, et al. Enhanced ssvep bionic spelling via xlstm-based deep learning and spatial attention. Biomimetics, 2025.

[7] P. Du, P. Li, et al. Single-trial p300 classification algorithm based on centralized multi-person data fusion cnn. Frontiers in Neuroscience, 2023.

[8] Z. Ermaganbet, A. Mussabayeva, et al. Subject-independent p300 speller classification using time-frequency representation and double input cnn with feature concatenation. In IEEE DSP Conference, 2023.

[9] J. Hong, G. Mackellar, and S. Ghane. Spellerssl: Self-supervised learning with p300 aggregation for speller bcis. arXiv preprint, 2025.

[10] Roman Kessler, Alexander Enge, and Michael A. Skeide. How eeg preprocessing shapes decoding performance. Communications Biology,
2025.

[11] Yi Liu, Zach Quince, Steven Goh, Shoryu Teragawa, and Tobias Low. A lightweight deep learning model for eeg classification across visual stimuli. University of Southern Queensland preprint, 2023.

[12] V. Marochko et al. Integrated gradients for enhanced interpretation of p3b-erp classifiers trained with eeg-superlets in traditional and virtual environments. In CEUR Workshop, 2025.

[13] J. A. O’Reilly et al. Blind source separation of event-related potentials using a recurrent neural network. bioRxiv, 2024.

[14] J. A. O’Reilly, J. Wehrman, et al. Neural correlates of face perception modeled with a convolutional recurrent neural network. bioRxiv, 2023.

[15] Y. Ravipati, N. Pouratian, et al. Evaluating deep learning performance for p300 neural signal classification. In AMIA Annual Symposium
Proceedings, 2024.

[16] G. Zhang et al. Assessing the effectiveness of spatial pca on svm-based decoding of eeg data. NeuroImage, 2023.