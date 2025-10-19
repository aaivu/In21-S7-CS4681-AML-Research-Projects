# Research Proposal: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## Abstract

This project aims to enhance the initial-condition ensemble generation process in the GenCast diffusion-based weather forecasting model. The current method—adding Gaussian-process (GP) noise to compensate for under-dispersed ERA5 Ensemble Data Assimilation (EDA) inputs—is acknowledged by the GenCast authors as “crude,” lacking physical and geographical structure. This study proposes a systematic exploration to determine which atmospheric variables should be perturbed, the appropriate magnitude of perturbations, and how geography-aware scaling can improve realism. The approach focuses on inference-time modifications, avoiding retraining or architectural changes to the model. Using evaluation metrics such as the Continuous Ranked Probability Score (CRPS), spread–skill ratio, and rank histograms, the study aims to quantify improvements in ensemble calibration and probabilistic forecast skill. The proposed work contributes a structured framework for generating more physically meaningful and adaptive perturbations that improve ensemble diversity while maintaining dynamical consistency.

## 1. Introduction

Weather forecasting models such as GenCast aim to capture both the most likely future atmospheric state and the range of plausible alternatives. This uncertainty representation is achieved through ensemble forecasting—running multiple simulations with slightly different initial conditions (ICs). However, GenCast relies on ERA5 EDA members, which are under-dispersed and have lower spatial resolution than ERA5 itself. To address this, the GenCast authors add ad-hoc GP noise to the inputs, yet this method lacks tuning, flow-dependence, and physical structure.  
This project focuses on enhancing GenCast’s initial-condition ensemble by designing improved perturbation mechanisms that reflect geographic variability, vertical correlations, and variable-specific sensitivity. The proposed modifications operate entirely at inference time, making the project computationally feasible while contributing to the understanding of ensemble calibration in AI-driven weather models.

## 2. Problem Statement

The current ensemble generation approach in GenCast introduces isotropic GP noise that is uniform across variables and regions. This results in ensembles that fail to capture realistic spatial and dynamical variability, particularly in regions such as the tropics where uncertainty is inherently higher. Furthermore, variable-specific sensitivity and vertical correlations are ignored, leading to under-dispersed and physically inconsistent ensembles. The core problem addressed in this research is how to systematically design and tune geographically and physically structured perturbations for GenCast’s initial conditions to improve ensemble realism and probabilistic forecast skill without retraining the model.

## 3. Literature Review Summary

Recent advances in machine learning weather prediction have introduced models like GraphCast and GenCast, which emulate atmospheric dynamics using deep neural networks trained on ERA5 reanalysis data. While these models achieve deterministic forecast accuracy comparable to numerical weather prediction (NWP) systems, ensemble generation remains a challenge. Traditional NWP ensembles use flow-dependent covariances and physically constrained perturbations, such as those in the ECMWF EDA.  
Prior research highlights that structured, flow-aware perturbations improve ensemble calibration and reduce bias. Studies also show that variable-specific scaling, regional adaptation, and vertical smoothing lead to more realistic uncertainty representations. However, no systematic investigation has been conducted for perturbation design in diffusion-based models like GenCast, motivating this work.

## 4. Research Objectives

### Primary Objective
To improve the initial-condition ensemble generation of the GenCast model through geographically and physically structured perturbation tuning.

### Secondary Objectives
- Identify which variables (e.g., wind, temperature, geopotential, 2m temperature) contribute most to ensemble spread and calibration.
- Determine optimal perturbation magnitudes for each variable through controlled experiments.
- Introduce geography-aware scaling, such as:
  - Stronger perturbations in the tropics.
  - Orography-dependent modulation based on terrain steepness.
  - Vertical smoothing across pressure levels to maintain physical coherence.

## 5. Methodology

The study modifies GenCast’s public evaluation code to incorporate configurable perturbations applied at inference time, before model sampling.  
Each perturbation will be generated using Gaussian-process noise on the spherical grid and modulated by variable-specific and geographic scaling factors.  
The experimental pipeline includes:
1. **Variable Selection:** Perturb subsets of input variables (e.g., winds, temperature, geopotential).
2. **Scaling Design:** Apply different noise magnitudes per variable and geographic region.
3. **Vertical Correlation:** Introduce smooth perturbations across pressure levels.
4. **Evaluation:** Run the modified model to generate ensembles and compute CRPS, spread–skill ratio, and rank histograms per variable and lead time (D+1, D+3, D+5, D+10).  
All experiments will be conducted using the GenCast Mini environment, ensuring compatibility with existing code and reproducibility.

## 6. Expected Outcomes

The project expects to produce:
- A reproducible perturbation module integrated with the GenCast framework.  
- Quantitative insights into the sensitivity of GenCast forecasts to variable-specific and regional noise.  
- Improved ensemble calibration metrics (CRPS, spread–skill ratio) compared to the default GP perturbation.  
- Recommendations for physically consistent perturbation design in future AI-based weather models.

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

- **Datasets:** ERA5 and ERA5 EDA reanalysis datasets (available via Copernicus CDS).  
- **Tools:** JAX, xarray, Haiku (for GenCast Mini model evaluation).  
- **Hardware:** GPU-enabled machine for inference (no TPU required).  
- **Software:** GenCast Mini notebook environment, Matplotlib, NumPy, and SciPy for analysis.  

## References

1. L. Keet et al., “GenCast: Diffusion-based ensemble forecasting for medium-range weather,” *arXiv preprint arXiv:2405.20804*, 2024.  
2. T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359–378, 2007.  
3. ECMWF, “Ensemble forecast verification,” *ECMWF Forecast User Guide*, 2022.  
4. J. Berner et al., “Stochastic parameterization: Toward a new view of weather and climate models,” *Bulletin of the American Meteorological Society*, vol. 98, no. 3, pp. 565–588, 2017.  
5. N. Christensen et al., “Flow-dependent perturbations for machine learning weather models,” *Proceedings of the 11th International Conference on Artificial Intelligence Applications and Innovations*, 2023.  

---

**Submission Instructions:**  
1. Complete all sections above  
2. Commit your changes to the repository  
3. Create an issue with the label "milestone" and "research-proposal"  
4. Tag your supervisors in the issue for review
