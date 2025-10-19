# Methodology: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## 1. Overview

This study aims to improve the generation of initial-condition ensembles in the GenCast model—a diffusion-based weather forecasting system developed by DeepMind. The original GenCast ensemble uses simple Gaussian-process (GP) noise to increase ensemble spread, but the method is described by its authors as “crude” and physically unstructured.  
Our methodology introduces geography- and variable-aware perturbation schemes that modify the model’s input data at inference time, without altering the model’s training or architecture. We perform systematic experiments to evaluate the impact of these perturbations on probabilistic skill using metrics such as the Continuous Ranked Probability Score (CRPS) and spread–skill ratio.

## 2. Research Design

The research follows an **experimental, quantitative design**. The GenCast mini version is used for controlled experiments due to its lower computational requirements. We iteratively implement and test new perturbation schemes—starting from simple Gaussian noise to structured, physically-informed perturbations (v1–v4)—while keeping the forecasting model unchanged.  
Each variant is evaluated using a consistent evaluation framework, comparing forecast ensembles produced with and without added noise. This design isolates the effect of the perturbation structure on ensemble calibration and performance.

## 3. Data Collection

### 3.1 Data Sources
- **ERA5 Reanalysis Data** (ECMWF, 1979–2018): provides the atmospheric inputs used by GenCast.  
- **ERA5 Ensemble Data Assimilation (EDA)**: used to initialize ensembles; known to be under-dispersed.  
- **GenCast Mini Demo Dataset** (DeepMind public release): contains sample atmospheric states and targets.

### 3.2 Data Description
The dataset includes:
- 6 surface variables: 2m temperature, mean sea level pressure, 10m winds (U/V), sea surface temperature, and total precipitation.  
- 6 atmospheric variables across **13 pressure levels** (1000–50 hPa): temperature, geopotential, winds (U/V), vertical velocity, and humidity.  
Each sample consists of two input timesteps (t−12h, t) and one target (t+12h) with dimensions `(1, 2, 181, 360, 13)`.

### 3.3 Data Preprocessing
- Variables are normalized based on their mean and standard deviation.  
- Gaussian noise fields are generated using `xarray` and `numpy` with optional smoothing via `scipy.ndimage.gaussian_filter`.  
- Latitude masks and orography (if available) are used to scale noise amplitude geographically.  
- The preprocessing maintains shape and coordinate integrity, ensuring full model compatibility.

## 4. Model Architecture

The base model is **GenCast**, a conditional diffusion model with an encoder–processor–decoder structure:
- **Encoder:** Maps input atmospheric fields to an icosahedral mesh.  
- **Processor:** Graph transformer that performs message passing on the mesh.  
- **Decoder:** Maps back to the latitude–longitude grid to predict future atmospheric states.  
Our work does **not modify** this architecture; instead, it inserts a preprocessing module that perturbs initial inputs before inference.

**Perturbation Design (Enhancement Layer):**
Mathematically, the perturbed dataset \( X' \) is defined as:
\[
X' = X + \eta_v \cdot \sigma_v \cdot f(\phi)
\]
where:
- \( X \): original input field,  
- \( \eta_v \sim \mathcal{N}(0,1) \): Gaussian noise,  
- \( \sigma_v \): standard deviation of variable \( v \),  
- \( f(\phi) = 1 + \alpha \sin^2(\phi) \): latitude-dependent scaling, enhancing tropical variability.  
Later versions incorporated vertical smoothing and cross-variable coupling (e.g., wind divergence influencing geopotential perturbations).

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **CRPS (Continuous Ranked Probability Score)** – quantifies probabilistic forecast skill.  
- **Spread–Skill Ratio** – measures ensemble consistency (ideal value ≈ 1).  
- **Rank Histograms** – assess ensemble under- or overdispersion.  
- **RMSE and Mean Ensemble Spread** – used for diagnostic comparisons.

### 5.2 Baseline Models
- **Baseline:** Default GenCast Mini model without additional perturbations (EDA only).  
- **Experiment 1 (v1):** Uniform Gaussian noise added to all variables.  
- **Experiment 2 (v2):** Gaussian noise with spatial smoothing (σ=1).  
- **Experiment 3 (v3):** Physics-aware perturbations linking wind divergence, geopotential, and temperature.  
- **Experiment 4 (v4):** Relative, multiplicative perturbations excluding geopotential and MSLP.  

### 5.3 Hardware/Software Requirements
- **Hardware:** Single NVIDIA GPU (e.g., RTX A4000 / Colab T4).  
- **Software:** JAX, Xarray, NumPy, SciPy, Matplotlib.  
- **Environment:** Python 3.12; based on GenCast Mini Demo notebook provided by DeepMind.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Dataset familiarization and preprocessing module | 2 weeks | Clean ERA5-derived dataset |
| Phase 2 | Implement perturbation variants (v1–v4) | 3 weeks | Modular noise generation functions |
| Phase 3 | Ensemble experiments and evaluation (CRPS, spread–skill) | 2 weeks | Quantitative and qualitative results |
| Phase 4 | Analysis and reporting | 1 week | Final report and research paper submission |

## 7. Risk Analysis

| Risk | Description | Mitigation Strategy |
|------|--------------|---------------------|
| Computational limitations | Full GenCast requires TPU; limited to mini version | Use GenCast Mini and subset data |
| Model instability | Perturbations may produce unrealistic inputs | Gradual scaling and latitude-based masking |
| Poor performance outcomes | CRPS degradation under noise | Treat as diagnostic evidence and refine methods |
| Data mismatch | Dimension or coordinate alignment errors | Consistent use of `xarray.broadcast_like` and sanity checks |

## 8. Expected Outcomes

The expected outcomes include:
- A **reproducible perturbation pipeline** for inference-time ensemble diversification.  
- Quantitative assessment of how structured perturbations affect CRPS and spread–skill.  
- Identification of sensitivity factors in diffusion-based models under input perturbations.  
- A framework that can guide **future research on physics-consistent noise generation** for AI-based forecasting systems.  

Although preliminary experiments indicate deterioration in CRPS, the study contributes methodological insights into ensemble reliability, sensitivity, and the limits of naive noise injection in pretrained generative weather models.

---

**Note:** This methodology will be refined throughout experimentation as more results and error analyses become available.
