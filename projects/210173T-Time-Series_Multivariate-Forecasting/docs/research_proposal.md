# Research Proposal: Time Series:Multivariate Forecasting

**Student:** 210173T
**Research Area:** Time Series:Multivariate Forecasting
**Date:** 2025-09-01

## Abstract

Transformer-based architectures have revolutionized sequential modeling, achieving state-of-the-art results in long-term time series forecasting. The Patch Time Series Transformer (PatchTST) demonstrated that patch-based tokenization and channel independence can significantly enhance predictive accuracy while maintaining computational efficiency. However, its fixed patch size restricts the model’s ability to capture diverse temporal dynamics across multiple scales.

This research proposes Selective Multi-Scale PatchTST (MS-PatchTST), an extension that processes input time series across parallel patch granularities (small, medium, large) and fuses their outputs through a learnable aggregation layer. This multi-scale design enables simultaneous modeling of fine-grained, medium-term, and long-term temporal dependencies, improving robustness and reducing sensitivity to patch-size hyperparameters. Preliminary experiments show that MS-PatchTST achieves up to 5–15% forecasting accuracy improvement over the baseline while maintaining interpretability and efficiency. The project aims to generalize this framework across multiple public datasets, providing a scalable solution for real-world multivariate forecasting challenges.


## 1. Introduction

Time series forecasting plays a vital role in domains such as energy management, finance, and traffic prediction. While the Transformer architecture has achieved outstanding results in NLP and vision, its direct application to time series is hindered by high computational costs and limited local semantic awareness.

Recent models such as Informer, Autoformer, and FEDformer introduced efficient attention mechanisms to address these issues, whereas PatchTST redefined the problem by focusing on effective input representation through patching and channel independence. Despite its success, PatchTST’s single patch size restricts temporal diversity, preventing the model from capturing the full range of temporal patterns inherent in multiscale data.

The proposed research introduces a multi-scale adaptation of PatchTST, aiming to improve long-term forecasting accuracy by learning from multiple temporal resolutions simultaneously.


## 2. Problem Statement

Although PatchTST efficiently represents local time-series patterns, it relies on a fixed patch size, forcing the model to interpret temporal dynamics through a single resolution. This limitation reduces forecasting accuracy for complex datasets where both short-term fluctuations and long-term trends coexist. The research therefore addresses the following problem:

How can Transformer-based time series models be enhanced to effectively learn from multiple temporal scales without compromising computational efficiency?


## 3. Literature Review Summary

Research in Transformer-based forecasting has evolved from computational efficiency (Informer, FEDformer) toward representational redesign (PatchTST). Informer introduced ProbSparse attention to reduce complexity to O(L log L), while FEDformer used frequency-domain learning for linear scalability. PatchTST revolutionized input tokenization, representing time series as overlapping patches, which drastically reduced complexity while maintaining accuracy.

However, none of these models effectively capture multi-scale temporal dependencies. The proposed MS-PatchTST aims to fill this research gap by enabling simultaneous modeling at multiple patch resolutions and learning their optimal fusion dynamically.


## 4. Research Objectives

### Primary Objective

To develop and evaluate a multi-scale Transformer-based framework that enhances the representational capacity of PatchTST for multivariate long-term forecasting.

### Secondary Objectives

- To design a parallel multi-scale PatchTST architecture processing multiple patch sizes simultaneously.
- To implement a learnable fusion layer that adaptively combines predictions from different scales.
- To benchmark MS-PatchTST against baseline models (PatchTST, Informer, FEDformer) across public datasets.
- To analyze performance trade-offs between forecasting accuracy and computational cost.


## 5. Methodology

The study adopts a quantitative experimental methodology, implemented in PyTorch.

1. Baseline Analysis: Reproduce and verify the original PatchTST implementation for fair comparison.
2. Model Development:
	- Implement parallel Transformer backbones for small, medium, and large patch sizes.
	- Integrate a fusion network to learn optimal weightings for each temporal scale.
3. Datasets: Weather, Electricity, and National Illness (ILI) datasets, following standard long-term forecasting protocols (L = 336, T = 96).
4. Evaluation Metrics:
	- MSE and MAE for forecasting accuracy.
	- Training time and GPU memory for efficiency.
5. Ablation Studies: Assess the contribution of each temporal scale (Small, Medium, Large) and their combinations.
6. Comparison: Benchmark against baseline Transformer-based models and analyze scale sensitivity.


## 6. Expected Outcomes

- Development of an enhanced Transformer-based model capable of multi-scale temporal reasoning.
- 5–15% improvement in forecasting accuracy compared to the baseline PatchTST.
- Empirical insights into the optimal selection and fusion of temporal scales.
- Contribution of a publicly available PyTorch implementation for reproducibility and future research.


## 7. Timeline

| Week  | Task                                           |
|-------|------------------------------------------------|
| 1–2   | Literature review and problem refinement       |
| 3–5   | Baseline PatchTST replication                  |
| 6–8   | Multi-scale model design and implementation    |
| 9–10  | Experimental setup and dataset preparation     |
| 10–11 | Ablation studies and performance benchmarking  |
| 12–14 | Result analysis and paper writing              |
| 14    | Final submission and code release              |


## 8. Resources Required

- Hardware: NVIDIA GPU (≥ 8 GB VRAM)
- Software: PyTorch, NumPy, Pandas, Matplotlib
- Datasets: Weather, Electricity, ILI (publicly available)
- Version Control: GitHub repository for code and experiment tracking
- Libraries: scikit-learn for preprocessing and metrics


## References

1. Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
2. Nie, Y. et al. (2023). A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers. ICLR.
3. Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.
4. Zhou, T. et al. (2022). FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series Forecasting. ICML.
