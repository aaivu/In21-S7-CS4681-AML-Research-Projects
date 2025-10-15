# Literature Review: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## Abstract

This literature review explores the emerging intersection between machine learning and numerical weather prediction (NWP), focusing on diffusion-based ensemble forecasting models such as GenCast. The review examines prior work on ensemble data assimilation, uncertainty quantification, and machine learning-based forecasting architectures, with particular attention to the challenge of generating realistic initial-condition perturbations. It also surveys ensemble calibration metrics, such as the Continuous Ranked Probability Score (CRPS), and identifies gaps in how current AI-based forecasting systems treat input uncertainty. The key finding is that while models like GenCast achieve unprecedented deterministic accuracy, their ensemble diversity remains limited by simplistic perturbation schemes. The review concludes by identifying opportunities for structured, geography-aware, and physically consistent perturbation strategies that could improve ensemble calibration in future AI-driven climate forecasting systems.

## 1. Introduction

Recent advances in machine learning have led to models that can emulate atmospheric dynamics with accuracy comparable to traditional NWP systems. Models such as GraphCast and GenCast, developed by DeepMind, leverage graph neural networks and diffusion-based sampling to generate medium-range global weather forecasts with lower computational cost.  
However, ensemble forecasting—an essential component of uncertainty estimation—remains a weak point in most AI-based systems. Traditional ensemble NWP systems rely on physically balanced perturbations derived from data assimilation, while ML models often inject ad-hoc Gaussian noise. This literature review focuses on ensemble generation methods for AI-based weather models, evaluation metrics for probabilistic forecasts, and recent work on physically constrained perturbations.

## 2. Search Methodology

### Search Terms Used
- "GenCast diffusion model"
- "GraphCast weather forecasting"
- "ensemble data assimilation ECMWF"
- "probabilistic forecast CRPS"
- "perturbation strategies in NWP"
- "flow-aware ensemble generation"
- "machine learning weather prediction"
- "diffusion probabilistic models for weather"

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [ ] Other: Copernicus Climate Data Store (for ERA5/EDA references)

### Time Period
2018–2024, focusing on the rise of AI-based weather prediction models and ensemble forecasting frameworks.

## 3. Key Areas of Research

### 3.1 Data-Driven Ensemble Forecasting

Machine learning weather models such as FourCastNet, GraphCast, and GenCast have redefined medium-range forecasting. These models are trained on ERA5 reanalysis data and use deep learning architectures to emulate atmospheric dynamics.  
**Key Papers:**
- **Kurth et al., 2022** – Introduced *FourCastNet*, a Fourier Neural Operator-based model achieving high-resolution forecasts.  
- **Lam et al., 2023** – Developed *GraphCast*, a graph neural network for global weather prediction.  
- **Keet et al., 2024** – Presented *GenCast*, a diffusion-based ensemble forecaster, outperforming ECMWF ENS in CRPS for 97.4% of targets.  
These studies demonstrate the growing capability of ML weather models to match or exceed traditional NWP performance, yet they rely on simple ensemble perturbations derived from Gaussian noise.

### 3.2 Ensemble Perturbation and Uncertainty Representation

Traditional NWP ensemble systems like ECMWF’s Ensemble of Data Assimilations (EDA) generate perturbations using flow-dependent covariances that maintain hydrostatic and geostrophic balance. In contrast, GenCast adds isotropic Gaussian-process (GP) noise to selected variables (winds, temperature, geopotential, and 2m temperature), which is described as “crude” and not flow-dependent.  
**Key Papers:**
- **Isaksen et al., 2010** – Defined the ECMWF EDA framework for flow-consistent ensemble generation.  
- **Berner et al., 2017** – Discussed stochastic parameterizations for representing model uncertainty in NWP.  
- **Hersbach et al., 2020** – Described the ERA5 and EDA datasets that underpin modern ML training.  

### 3.3 Evaluation Metrics for Ensemble Calibration

Probabilistic forecast performance is typically evaluated using:
- **CRPS (Continuous Ranked Probability Score)** – Measures overall probabilistic skill.  
- **Spread–Skill Ratio** – Assesses ensemble dispersion relative to forecast error.  
- **Rank Histograms** – Diagnose under- or overdispersion.  
**Key Papers:**
- **Gneiting and Raftery, 2007** – Defined CRPS as a proper scoring rule for probabilistic forecasts.  
- **Hamill, 2001** – Interpreted rank histograms in ensemble verification.  
These metrics are standard in both operational and ML-based ensemble evaluation.

## 4. Research Gaps and Opportunities

### Gap 1: Lack of Structured Perturbation Design in ML Weather Models  
**Why it matters:** Current ML models use uniform, isotropic Gaussian noise that ignores regional variability and physical constraints.  
**How your project addresses it:** Develops geography-aware perturbation schemes (latitude, terrain, vertical correlation) to represent realistic uncertainty.

### Gap 2: Absence of Flow-Dependent or Physics-Preserved Noise  
**Why it matters:** Unbalanced perturbations distort atmospheric fields and produce unrealistic ensemble spreads.  
**How your project addresses it:** Introduces physically coupled perturbations linking wind divergence, geopotential, and temperature using simplified balance equations.

### Gap 3: Lack of Systematic Evaluation Framework  
**Why it matters:** Prior work lacks reproducible benchmarks for perturbation performance under identical conditions.  
**How your project addresses it:** Integrates a modular perturbation pipeline into the open-source GenCast mini environment for controlled, reproducible testing.

## 5. Theoretical Framework

The project builds upon the principles of **probabilistic forecasting** and **data assimilation theory**.  
The diffusion-based GenCast model represents the atmospheric state as a conditional generative process:

```math
\mathbf{x}_{t+1} = \mathcal{D}(\mathbf{x}_t + \epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, \Sigma)
```

where ```math \mathcal{D} ``` is the learned diffusion operator, and \( \epsilon_t \) is the perturbation noise. The goal is to design \( \Sigma \) such that it reflects realistic atmospheric variability, informed by geographic and physical constraints.

## 6. Methodology Insights

Commonly used methods in the literature include:
- Gaussian-process noise on the sphere (GenCast, 2024).  
- Flow-dependent perturbations using ensemble Kalman filters (ECMWF EDA).  
- Spectral perturbations in spherical harmonics (Isaksen et al., 2010).  
For this project, a **hybrid approach** combining GP noise with geographic modulation and vertical smoothing is most promising, as it maintains dataset integrity while adding structured variability.

## 7. Conclusion

The reviewed literature demonstrates that while ML models such as GenCast have transformed medium-range forecasting, ensemble generation remains a critical challenge. Current perturbation methods are simplistic and lack physical awareness. This review identifies opportunities to integrate domain knowledge from traditional NWP—specifically, flow-dependent and balance-preserving noise generation—into AI-based ensemble forecasting. The proposed research directly targets this gap by designing reproducible, geography-aware perturbation strategies that can improve ensemble calibration in diffusion-based weather models.

## References

1. L. Keet et al., “GenCast: Diffusion-based ensemble forecasting for medium-range weather,” *DeepMind Technologies*, 2024.  
2. S. Lam et al., “GraphCast: Learning skillful medium-range global weather forecasting,” *Science*, 2023.  
3. T. Kurth et al., “FourCastNet: Accelerating global high-resolution weather forecasting using adaptive Fourier neural operators,” *arXiv preprint arXiv:2208.05419*, 2022.  
4. H. Hersbach et al., “The ERA5 Global Reanalysis,” *Quarterly Journal of the Royal Meteorological Society*, vol. 146, 2020.  
5. L. Isaksen et al., “Ensemble of Data Assimilations for the ECMWF Model,” *Quarterly Journal of the Royal Meteorological Society*, vol. 136, 2010.  
6. J. Berner et al., “Stochastic parameterization: Toward a new view of weather and climate models,” *Bulletin of the American Meteorological Society*, vol. 98, no. 3, 2017.  
7. T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” *JASA*, 2007.  
8. T. Hamill, “Interpretation of rank histograms for verifying ensemble forecasts,” *Monthly Weather Review*, 2001.  
9. ECMWF, “Ensemble Forecast Verification,” *ECMWF Forecast User Guide*, 2022.  

---

**Notes:**  
- Gaps identified form the foundation for the proposed GenCast perturbation enhancement study.
