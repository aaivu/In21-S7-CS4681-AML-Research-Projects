# Research Proposal: Climate AI:Climate Modeling

**Student:** 210491P
**Research Area:** Climate AI:Climate Modeling
**Date:** 2025-09-01

## Abstract

## Abstract

The research aims to enhance the **GraphCast** machine learning weather forecasting model by introducing an initial-condition ensemble method to enable probabilistic forecasting without retraining the base model. GraphCast, a graph neural network (GNN)-based model, generates accurate global forecasts in under one minute but remains limited by its deterministic nature. This work proposes adding small, unbiased perturbations (Gaussian noise) to model inputs to simulate forecast uncertainty and generate ensemble predictions. The outputs from multiple perturbed instances will be aggregated using ensemble-mean and probabilistic evaluation metrics such as RMSE and ACC. This ensemble-based post-processing approach is expected to improve the robustness and reliability of GraphCast forecasts for long-term and uncertainty-aware applications while maintaining computational efficiency.

## 1. Introduction

Accurate weather prediction is essential for decision-making in agriculture, transportation, and disaster management. While traditional numerical weather prediction (NWP) models have achieved high accuracy, they require significant computational resources. Machine learning models like **GraphCast** leverage historical reanalysis data to provide faster and high-resolution forecasts. However, GraphCast currently produces deterministic forecasts, limiting its ability to represent natural uncertainty in weather systems.  
This research focuses on extending GraphCast to support probabilistic forecasting through ensemble-based perturbation techniques, improving reliability and uncertainty estimation without retraining or fine-tuning.


## 2. Problem Statement

The **GraphCast** model provides only deterministic forecasts, which do not represent inherent uncertainty in atmospheric conditions. This limits its usefulness for long-term forecasting and risk-sensitive applications. Traditional ensemble systems address this by generating multiple perturbed forecasts but at high computational costs. The research problem is to develop a computationally efficient ensemble-based post-processing method for GraphCast that can produce probabilistic forecasts while preserving its speed and accuracy.


## 3. Literature Review Summary

Machine learning-based weather prediction models such as GraphCast outperform many traditional NWP methods in terms of accuracy and speed. However, they fail to represent uncertainty. Studies by Lam et al. (2023) demonstrate GraphCast’s superior skill in deterministic forecasting but highlight its lack of probabilistic outputs.  
Ensemble forecasting, as described by Leutbecher and Palmer (2008), provides a mechanism to quantify forecast uncertainty by introducing perturbations to initial conditions. Recent works show that lightweight perturbation schemes can emulate probabilistic behavior without retraining.  
**Research gap:** There is limited exploration of ensemble-based uncertainty quantification for pretrained ML weather models like GraphCast, presenting an opportunity for innovation.


## 4. Research Objectives

### Primary Objective
To develop an ensemble-based post-processing framework for the GraphCast model that generates probabilistic forecasts without retraining.

### Secondary Objectives
- To design and implement a perturbation mechanism using noise derived from historical forecast errors.  
- To evaluate ensemble mean and spread performance against deterministic GraphCast forecasts using RMSE and ACC.  
- To assess ensemble reliability through reliability diagrams and rank histograms.  

## 5. Methodology

The proposed methodology consists of five stages:

1. **Design & Setup:**  
   Develop an ensemble system using multiple GraphCast instances without retraining.  

2. **Perturbation Generation:**  
   Analyze prediction errors from historical data to design small, unbiased random noises representing initial condition uncertainty.

3. **Ensemble Forecasting:**  
   Run parallel inferences with perturbed inputs to obtain multiple forecasts, forming the ensemble.

4. **Post-Processing & Evaluation:**  
   Compute ensemble mean, spread, RMSE, and ACC; assess reliability and sharpness using rank histograms and reliability diagrams.

5. **Documentation:**  
   Compile results and produce a research paper summarizing findings and implications.


## 6. Expected Outcomes

- A functional ensemble pipeline for GraphCast producing probabilistic forecasts.  
- Improved forecast accuracy and uncertainty quantification compared to the deterministic baseline.  
- Demonstration of low-cost probabilistic forecasting without model retraining.  
- Evaluation metrics (RMSE, ACC, reliability diagrams) validating ensemble performance.

## 7. Timeline

| Week | Task |
|------|------|
| 6 | Understand the Model and design ensemble |
| 7  | Ensemble implementation |
| 8  | Dataset exploration |
| 9 | Noise Generation |
| 10 | Ensemble forecasting & post processing|
| 11 | Sensitivity analysis |
| 12 | Documentation |

## 8. Resources Required

- **Datasets:** ERA5 reanalysis dataset (1979–2021)  
- **Model:** Pretrained GraphCast model (Google DeepMind)  
- **Tools:** JAX, Haiku, NumPy, Xarray, Matplotlib, Cartopy  
- **Hardware:** Google Colab or equivalent GPU environment  

## References

[1]R. Lam et al., “GraphCast: Learning skillful medium-range global weather forecasting,” arXiv:2212.12794 [physics], Dec. 2022, Available: https://arxiv.org/abs/2212.12794  
[2] M. Leutbecher and T. N. Palmer, “Ensemble forecasting,” Journal of Computational Physics,
vol. 227, no. 7, pp. 3515–3539, Mar. 2008, doi: https://doi.org/10.1016/j.jcp.2007.02.014.  
