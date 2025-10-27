# Experiments: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## 1. Overview

The experiments conducted in this study aimed to evaluate the effect of different perturbation strategies on the performance of the diffusion-based weather forecasting model, **GenCast Mini**. Each experiment introduced controlled modifications to the model’s input data during inference time, exploring how different forms of stochastic noise influence ensemble behavior, stability, and probabilistic skill.  
Rather than retraining or altering model parameters, all experiments were performed through modular, shape-preserving perturbations applied directly to the dataset before inference.  

The experiments were designed progressively — from simple additive noise to more physically and geographically informed perturbations — allowing systematic observation of how model performance evolves with increasing perturbation complexity.

---

## 2. Experimental Objectives

1. To assess how simple, unstructured perturbations affect the diffusion-based ensemble forecasts produced by GenCast.  
2. To introduce spatially smoothed and geographically aware perturbations that mimic natural atmospheric variability.  
3. To explore physically coupled perturbations maintaining partial balance between wind, geopotential, and temperature fields.  
4. To test scale-normalized (additive and multiplicative) perturbations designed to maintain physical realism and stable magnitudes.  
5. To evaluate whether excluding highly sensitive variables (geopotential, mean-sea-level pressure) from perturbation improves stability.  

---

## 3. Experimental Setup

- **Model:** GenCast Mini (public DeepMind release)  
- **Framework:** JAX / Haiku / Xarray (Python 3.12)  
- **Data:** ERA5 reanalysis data samples provided with GenCast Mini notebook  
- **Input shape:** `(batch=1, time=2, lat=181, lon=360, level=13)`  
- **Variables:** Winds (U/V), temperature, geopotential, humidity, surface and sea-surface temperatures, vertical velocity  
- **Perturbation application:** Inserted as a preprocessing module before ensemble inference  
- **Ensemble size:** 8 members per forecast  
- **Forecast step:** 12-hour lead time  
- **Evaluation metrics (computed but discussed separately):** CRPS, spread–skill ratio, RMSE  

All experiments were executed under identical random seeds and ensemble configurations to ensure comparability and reproducibility. The same GenCast sampler, checkpoint, and evaluation utilities were used across all runs.

---

## 4. Experiments Conducted

### 4.1 Experiment 1 – Additive Isotropic Noise (v1)

The first experiment introduced independent Gaussian noise to all input variables at each grid point. The noise amplitude was scaled by the variable’s standard deviation and adjusted by a latitude-dependent mask to simulate stronger variability in the tropics.  

Mathematically, the perturbation is expressed as:  
```math

X' = X + \sigma_v \cdot \eta, \quad \eta \sim \mathcal{N}(0, 1)
```  

where $\sigma_v$ is the standard deviation for each variable.  

**Rationale:**  
This served as a baseline to observe the model’s tolerance to unstructured random noise.  

**Implementation details:**  
- Implemented via `numpy.random.normal()` with per-variable scaling.  
- Perturbation applied uniformly across all variables and grid cells.  
- A latitude mask increased tropical perturbations by 50%.  

**Observation:**  
This approach revealed that while the model remained computationally stable, the forecasts displayed unrealistic spatial discontinuities, suggesting strong model sensitivity to unbalanced inputs.

---

### 4.2 Experiment 2 – Spatially Smoothed Perturbations (v2)

To reduce small-scale irregularities introduced in v1, the second experiment added a horizontal Gaussian filter to smooth noise along the latitude and longitude dimensions.  
```math

X' = X + \sigma_v \cdot (G_\sigma * \eta)

```
where $G_\sigma$  is a Gaussian kernel with standard deviation $\sigma = 1$ .


**Rationale:**  
Atmospheric variables exhibit spatial coherence, and uncorrelated noise violates physical smoothness. This step ensured spatially correlated perturbations mimicking mesoscale patterns.  

**Implementation details:**  
- Used `scipy.ndimage.gaussian_filter()` for 2D smoothing.  
- Perturbation scaling preserved through normalization.  
- Maintained the same amplitude modulation by latitude.  

**Observation:**  
The fields became smoother and visually realistic, but large-scale imbalances still persisted, motivating a move toward physically linked perturbations.

---

### 4.3 Experiment 3 – Physically Coupled Perturbations (v3)

The third experiment introduced **physics-aware relationships** between variables, capturing approximate dynamical coupling. Specifically, wind perturbations were used to derive changes in geopotential and temperature through simplified balance equations:
```math

\delta \Phi = \alpha \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right), \quad
\delta T = \beta \frac{\partial \Phi}{\partial p}

```

**Rationale:**  
To maintain geophysical consistency, perturbations in one variable should propagate physically across related fields (e.g., wind divergence influencing pressure).  

**Implementation details:**  
- Computed approximate divergence fields from noisy winds using finite differences.  
- Added geopotential perturbations proportional to divergence.  
- Coupled vertical temperature adjustments based on geopotential gradients.  
- Controlled strength using parameters $\alpha = 200$  and $\beta = 10^{-4}$.  

**Observation:**  
This approach produced the most physically interpretable perturbations with smoother structures, though still prone to amplification in sensitive variables during inference.

---

### 4.4 Experiment 4 – Normalized Hybrid Perturbations (v4)

The fourth experiment addressed scale heterogeneity by using **additive noise for small-scale variables** and **multiplicative noise for large-scale variables** (temperature, sea-surface temperature, etc.):
```math

X' =
\begin{cases}
X + \sigma_v \eta, & \text{(for winds, humidity, vertical velocity)} \\
X(1 + \lambda \eta), & \text{(for temperature-like variables)}
\end{cases}

```

**Rationale:**  
Additive noise may cause unrealistic magnitude jumps in variables with large natural scales. Multiplicative perturbations preserve proportional relationships and avoid sign inversions.  

**Implementation details:**  
- Relative perturbation limit: ±2%.  
- Separate handling of surface and atmospheric variables.  
- Applied small temporal decorrelation factors to introduce time variability.  

**Observation:**  
While structurally stable, diffusion sampling still exaggerated inconsistencies introduced by cross-variable coupling. This led to further refinement through selective variable exclusion.

---

### 4.5 Experiment 5 – Excluding Geopotential and MSLP

This final variant tested whether excluding geopotential and mean-sea-level pressure (the most sensitive variables) could prevent large-scale imbalance amplification.  
The perturbation operator was identical to v4 but applied only to winds, temperature, and surface variables.

**Rationale:**  
Geopotential and MSLP exhibited the highest instability, contributing disproportionately to ensemble divergence.  

**Implementation details:**  
- Excluded geopotential and MSLP from the perturbation loop.  
- Preserved same seeds and scaling parameters as v4 for comparability.  

**Observation:**  
Although minor improvements were observed in field stability, ensemble imbalance still propagated through wind–pressure coupling, revealing the tight dynamical dependencies encoded in GenCast’s diffusion process.

---

## 5. Comparative Summary

Each experiment contributed to understanding GenCast’s response to structured perturbations:
- v1 exposed sensitivity to unstructured noise.  
- v2 improved spatial smoothness but not balance.  
- v3 introduced physical realism via coupling.  
- v4 applied scale normalization for stability.  
- v4* (restricted) confirmed that perturbation sensitivity is systemic, not localized to specific variables.  

Together, these experiments form a coherent progression from naive perturbations toward physically aware inference-time ensemble generation.

---

## 6. Observations and Insights

- GenCast’s diffusion model enforces a strong implicit balance; even slight deviations from its training manifold trigger forecast instability.  
- Perturbations that ignore variable dependencies lead to large-scale artifacts, especially in geopotential and pressure fields.  
- Smoothness alone is insufficient—balance preservation across variables is essential.  
- The model exhibits non-linear amplification of initial inconsistencies, a hallmark of tightly coupled diffusion dynamics.

---

## 7. Challenges Encountered

- **Shape mismatches:** Early versions caused dimension conflicts due to variable-level differences (e.g., `(lat, lon, level)` arrays). Resolved using `xarray.broadcast_like()`.  
- **Geopotential instability:** Even well-structured perturbations caused runaway anomalies in geopotential and MSLP fields.  
- **CRPS computation errors:** Required alignment of prediction and target sample dimensions.  
- **Computational limits:** Restricted to GenCast Mini due to lack of TPU access, limiting ensemble size and temporal range.  

---

## 8. Summary of Findings

This experimental sequence demonstrated that inference-time perturbations, while conceptually simple, can profoundly affect the stability and realism of diffusion-based weather models.  
The experiments collectively highlight the necessity of **balance-aware and covariance-preserving perturbation schemes** rather than naive noise injection.  
While CRPS degradation confirmed the model’s sensitivity, the methodology developed here provides a reproducible framework for future testing and refinement of physically consistent ensemble generation strategies.

---

**Note:** Quantitative results, plots, and CRPS comparisons are documented separately in the `results/` folder.
