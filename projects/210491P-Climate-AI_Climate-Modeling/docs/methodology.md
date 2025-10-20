# Methodology: Climate AI:Climate Modeling

**Student:** 210491P
**Research Area:** Climate AI:Climate Modeling
**Date:** 2025-09-01

## 1. Overview

This research proposes an ensemble-based post-processing framework to improve the accuracy and robustness of **GraphCast**, a machine learning weather forecasting system developed by Google DeepMind. The study enhances deterministic forecasts by generating multiple perturbed inputs using Gaussian noise, aggregating them through ensemble mean, median, and trimmed mean techniques, and applying bias correction via linear regression. This methodology enables uncertainty representation and improved reliability without retraining the model.

## 2. Research Design

The overall research design adopts a quantitative experimental approach, where the **GraphCast small_model** is used as a base deterministic model. The framework introduces stochastic perturbations into the input space and evaluates ensemble aggregation effects through controlled experiments. The design focuses on two primary analyses:

1. **Ensemble Aggregation Analysis** – assessing statistical aggregation methods (mean, median, trimmed mean).  
2. **Sensitivity Analysis** – examining the impact of ensemble size and noise magnitude on performance.

The study emphasizes reproducibility, interpretability, and computational efficiency using standardized evaluation metrics and controlled environments.


## 3. Data Collection

### 3.1 Data Sources

The research utilizes the ERA5 reanalysis dataset produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5 provides comprehensive atmospheric, oceanic, and land variables at a global scale, making it a standard benchmark for climate research.

### 3.2 Data Description

A subset of ERA5 data was selected with:
- **Spatial resolution:** 1° × 1°  
- **Vertical levels:** 13 pressure levels  
- **Variables:**
  - 2m temperature  
  - 10m zonal wind component (u-wind)  
  - 10m meridional wind component (v-wind)  
  - Specific humidity  
  - Temperature (pressure levels)

These variables were used to evaluate near-surface and atmospheric dynamics under ensemble perturbations.

### 3.3 Data Preprocessing

Data preprocessing involved:
- Extracting relevant variables from ERA5 NetCDF files using Xarray.
- Normalizing input variables for stable model inference.
- Applying Gaussian perturbations to input fields to create ensemble diversity while preserving physical consistency.

## 4. Model Architecture

The model employed is **GraphCast small**, a downscaled version of DeepMind’s GraphCast, maintaining the **encode–process–decode** architecture with **graph neural network (GNN)** message-passing layers.  
Key features:
- Graph Neural Network 
- Encodes atmospheric states as graph nodes.  
- Uses message-passing to propagate spatial and temporal dependencies.  
- Decodes outputs into forecasted weather fields.
- Operates on a 1° grid for computational efficiency.  

The ensemble framework augments this architecture externally, leaving the pre-trained model weights unchanged.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

The **Root Mean Square Error (RMSE)** is the primary metric.It measures average deviation between forecasts and ground-truth values. Lower RMSE indicates higher accuracy and stability.

### 5.2 Baseline Models

- **Baseline:** Original deterministic GraphCast small output.  
- **Comparisons:** Ensemble-aggregated forecasts using mean, median, trimmed mean, and bias-corrected outputs.  
Each configuration was evaluated under identical data and runtime conditions for fair benchmarking.

### 5.3 Hardware/Software Requirements

- **Environment:** Google Colab (free-tier)  
- **Frameworks:** JAX, Haiku (model execution), Xarray, NumPy, Matplotlib, Cartopy  
- **Dataset Source:** Google DeepMind’s GraphCast public repository  
- **Reproducibility:** Fixed random seeds and consistent runtime settings

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Understanding Dataset and Model | 2 weeks | Literature review |
| Phase 2 | Ensemble Implementation | 3 weeks | Ensemble Pipeline |
| Phase 3 | Experiments with aggregation methods | 1 weeks | Aggregation RMSE scores |
| Phase 4 | Sensitivity Analysis | 1 weeks | Sensitivity Analysis Scores |
| Phase 5 | Documenting the results | 1 week | Final report |
## 7. Risk Analysis

| Risk | Impact | Mitigation |
|------|---------|------------|
| Limited computational resources | High | Use GraphCast small and Colab GPU runtime |
| Noise magnitude too large/small | Medium | Conduct sensitivity analysis to find optimal noise |
| Overfitting in bias correction | Medium | Apply regression selectively to sensitive variables |

## 8. Expected Outcomes

- Demonstration that ensemble post-processing improves deterministic forecasts.  
- Identification of the optimal configuration: Ensemble size, Noise magnitude, The stable aggregation method  
- Improved RMSE for key surface variables (2m temperature, 10m u/v wind).  
- Validation that ensemble post-processing and bias correction can enhance ML-based forecasts without retraining, offering a practical and efficient extension to models like GraphCast.

