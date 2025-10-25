# Adapting PatchTST for Real-Time, Multi-Horizon Forecasting on the M4 Competition Benchmark

## Progress Evaluation

**Project ID:** TS001
**Name:** Galappaththi A. S.
**Index Number:** 210172N
**Date:** 2025-10-05

---

## Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
   - 2.1 [The M4 Competition Landscape](#21-the-m4-competition-landscape)
   - 2.2 [The Deep Learning Breakthrough on M4](#22-the-deep-learning-breakthrough-on-m4)
   - 2.3 [The Evolution of Transformers and the PatchTST Model](#23-the-evolution-of-transformers-and-the-patchtst-model)
   - 2.4 [The Need for Real-Time Forecasting](#24-the-need-for-real-time-forecasting)
3. [Methodology](#3-methodology)
   - 3.1 [Baseline Model: PatchTST for Univariate Forecasting](#31-baseline-model-patchtst-for-univariate-forecasting)
   - 3.2 [Adaptation and Enhancement Strategy](#32-adaptation-and-enhancement-strategy)
   - 3.3 [Evaluation Protocol](#33-evaluation-protocol)
4. [Project Plan](#4-project-plan)
5. [Summary](#5-summary)
6. [References](#references)

---

## 1. Introduction

This report outlines the project plan for research conducted as part of the Advanced Machine Learning course. The project aims to evaluate and adapt a State-of-the-Art (SOTA) Deep Learning (DL) model for the Makridakis-4 (M4) Competition, a classic and highly competitive benchmark for univariate, multi-horizon time series forecasting.

The primary aims and objectives for this project are as follows:

- **Baseline Model Selection:** To utilize the PatchTST model [1], a SOTA model from the Long-Term Time Series Forecasting (LTSF) domain, as the baseline for this investigation.

- **Adaptation for the M4 Benchmark:** To systematically adapt and evaluate the PatchTST architecture for the M4 Competition, focusing on strategies for model simplification and look-back window selection to enhance its robustness and performance.

- **Enhancement for Real-Time Forecasting:** To investigate and implement model optimization techniques, such as quantization and Open Neural Network Exchange (ONNX) conversion, to assess and improve the model's suitability for real-time, low-latency forecasting scenarios.

- **Rigorous Benchmarking:** To benchmark the adapted models using official M4 competition metrics for accuracy and standard metrics for real-time performance, such as inference latency and model size.

---

## 2. Literature Review

The field of time series forecasting has seen a significant shift towards DL models based on the Transformer architecture [2]. This review establishes the research gap between modern LTSF models and classic, large-scale forecasting benchmarks like the M4 Competition, motivating the focus on real-time performance.

### 2.1 The M4 Competition Landscape

The M4 Competition represents a significant and long-standing challenge in the forecasting community. As detailed in [3], it comprises 100,000 diverse univariate time series from various domains and requires multi-horizon forecasting. A key finding from the competition was that the most accurate methods were predominantly combinations of statistical approaches or hybrid models. The six pure Machine Learning (ML) methods submitted performed poorly, with none being more accurate than the combination benchmark. This established a high bar for any pure DL model attempting to compete in this domain.

### 2.2 The Deep Learning Breakthrough on M4

The narrative that pure DL models were unsuitable for the M4 benchmark was challenged by the N-BEATS model [4]. It was the first pure DL model to outperform the M4 competition winner, a domain-adjusted hybrid model. N-BEATS demonstrated that a deep architecture based on backward and forward residual links could, by itself, effectively solve a wide range of forecasting problems without task-specific feature engineering. This proved the viability of pure DL in this domain and established a new, high-performance DL benchmark for this project to compare against.

### 2.3 The Evolution of Transformers and the PatchTST Model

In parallel, a separate line of research has focused on the LTSF problem. Models like Reformer [5] utilized Locality-Sensitive Hashing (LSH) to achieve near-linear complexity. Informer [6] introduced a ProbSparse attention mechanism, assuming that only a few key-query pairs are significant. However, the work of Zeng et al. [7] questioned the necessity of such complex models, showing that a simple linear model (LTSF-Linear) could outperform them on LTSF benchmarks.

The PatchTST model [1] provided a powerful response by introducing "patching" as an input representation technique, making a simple vanilla Transformer backbone highly effective again. This highlighted the importance of model simplicity and appropriate data representation. This philosophy of simplicity and effectiveness makes PatchTST a compelling candidate to adapt for the M4 benchmark.

### 2.4 The Need for Real-Time Forecasting

While academic benchmarks focus on accuracy, real-world applications often impose strict latency and memory constraints. The field of model compression and acceleration provides established techniques for deploying large models efficiently. As surveyed by Cheng et al. [8], methods like quantization (reducing the numerical precision of model weights) and deploying models on optimized inference engines like the ONNX Runtime are standard practice for reducing model size and improving inference speed. A significant gap exists in the literature regarding the application of these techniques to modern time series models, which is the enhancement opportunity this project will address.

---

## 3. Methodology

This section details the baseline model and the planned methodology for adapting and evaluating it for the M4 Competition, with a specific focus on enhancing its real-time forecasting capabilities.

### 3.1 Baseline Model: PatchTST for Univariate Forecasting

The PatchTST model [1] will serve as the baseline. For this univariate task, its architecture consists of two core components:

1. **Patching:** The input time series is segmented into patches of a fixed length, which are then projected and fed as a sequence of input tokens.

2. **Transformer Encoder:** A standard Transformer encoder stack processes the sequence of patches to produce a representation for forecasting.

### 3.2 Adaptation and Enhancement Strategy

The methodology focuses on adapting PatchTST for the M4 benchmark and then enhancing it for real-time performance.

#### 3.2.1 Structural Adaptation for the M4 Benchmark

The default configuration of PatchTST, designed for long series, is likely suboptimal for the M4 dataset. The first stage of adaptation will involve a systematic evaluation of two key architectural parameters:

- **Look-back Window Strategy:** A range of look-back window sizes will be tested to determine an optimal length that captures sufficient historical context for the diverse M4 series without being computationally prohibitive or prone to overfitting.

- **Model Simplification:** To enhance robustness, as suggested by the success of simpler models [7], we will experiment with reducing the number of Transformer encoder layers and hidden dimensions to find a minimal effective architecture that generalizes well across the entire dataset.

#### 3.2.2 Optimization for Real-Time Forecasting

The primary enhancement opportunity is to improve the model's suitability for real-time applications where low-latency inference is critical. This will be achieved through two industry-standard optimization techniques, as detailed in the survey by Cheng et al. [8]:

- **Model Quantization:** We will apply post-training quantization techniques available in PyTorch. This process reduces the model's precision from 32-bit floating-point numbers to 8-bit integers, leading to a significant reduction in model size and a substantial speed-up in inference, particularly on a Central Processing Unit (CPU).

- **ONNX Runtime Deployment:** The trained PyTorch model will be converted to the ONNX format. This allows the model to be run using the highly optimized ONNX Runtime, which is designed for fast, cross-platform inference and often provides significant latency improvements over native deep learning frameworks.

### 3.3 Evaluation Protocol

The evaluation will be two-fold, assessing both forecasting accuracy and real-time performance.

**Forecasting Accuracy:**
- Performance will be measured using the official M4 evaluation metrics, as defined in [3]: the Symmetric Mean Absolute Percentage Error (sMAPE) and the Mean Absolute Scaled Error (MASE). The primary ranking metric will be the Overall Weighted Average (OWA) of these two.

**Real-Time Performance:**
The enhanced models will be evaluated on:
- **Inference Latency:** Measured in milliseconds per forecast on a standardized CPU.
- **Model Size:** Measured in megabytes (MB).

**Benchmarking:**
The adapted PatchTST models will be compared against the N-BEATS model [4] and a baseline PatchTST implementation. The core analysis will focus on the trade-offs between accuracy (OWA) and inference latency for the quantized and ONNX-optimized versions.

---

## 4. Project Plan

The project is structured into four distinct phases over a 12-week timeline. The following table provides a detailed breakdown of all tasks, deliverables, and key milestones as required by the course guidelines.

### Project Timeline

| **Phase** | **Tasks** | **Weeks** | **Deliverables** |
|-----------|-----------|-----------|------------------|
| **Phase 1: Foundation & Planning** | | **1-4** | |
| | Literature Deep Dive | 1-2 | Comprehensive literature review |
| | Implement M4 Data Loader & Baseline | 2-3 | Working baseline implementation |
| | Methodology Refinement | 3 | Finalized methodology |
| | Initial Report Drafting | 3-4 | Progress report draft |
| | **Milestone:** Finalize & Submit Report | 4 | **Progress Evaluation Report** |
| **Phase 2: Progress & Implementation** | | **4-7** | |
| | Progress Evaluation | 4-5 | Feedback incorporation |
| | Implement Real-Time Enhancements | 5-7 | Quantized & ONNX models |
| **Phase 3: Evaluation & Analysis** | | **7-11** | |
| | Run M4 Benchmark Experiments | 7-9 | Experimental results |
| | Benchmark Accuracy & Latency | 9-10 | Performance metrics |
| | **Milestone:** Mid-Evaluation | 9 | **Methodology Implementation** |
| | Expand to Final Paper | 10-11 | Draft final report |
| **Phase 4: Finalization** | | **11-12** | |
| | Finalize Code & Paper | 11-12 | Complete codebase & report |
| | Prepare Submission Package | 12 | Final deliverables |
| | **Milestone:** Final Submission | 12 | **Final Report & Presentation** |

---

## 5. Summary

This report has outlined a clear and focused plan to investigate the applicability of the modern PatchTST model to the classic M4 Competition benchmark. The literature review identifies a distinct gap between models designed for LTSF and the requirements of large-scale, multi-horizon univariate forecasting. The proposed methodology addresses this gap by focusing on the systematic adaptation of PatchTST for accuracy and its optimization for real-time performance through quantization and ONNX deployment.

The next steps involve completing the implementation of these adaptations and executing the rigorous experimental plan to benchmark both the model's forecasting accuracy and its inference latency. The 12-week project timeline ensures adequate time for foundation building, implementation, evaluation, and finalization, with clear milestones aligned with course requirements.

---

## References

[1] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," arXiv preprint arXiv:2211.14730, 2023.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," arXiv preprint arXiv:1706.03762, 2017.

[3] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 competition: Results, findings, conclusion and way forward," International Journal of Forecasting, vol. 34, no. 4, pp. 802-808, 2018.

[4] B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting," arXiv preprint arXiv:1905.10437, 2020.

[5] N. Kitaev, Ł. Kaiser, and A. Levskaya, "Reformer: The efficient transformer," arXiv preprint arXiv:2001.04451, 2020.

[6] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, "Informer: Beyond efficient transformer for long sequence time-series forecasting," arXiv preprint arXiv:2012.07436, 2021.

[7] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?," arXiv preprint arXiv:2205.13504, 2022.

[8] Y. Cheng, D. Wang, P. Zhou, and T. Zhang, "A survey of model compression and acceleration for deep neural networks," IEEE Signal Processing Magazine, 2020.

---

**Status:** This progress report represents completion of Phase 1 (Foundation & Planning) and readiness to proceed to Phase 2 (Progress & Implementation).
