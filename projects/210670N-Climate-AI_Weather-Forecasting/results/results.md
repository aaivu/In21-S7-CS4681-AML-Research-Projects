# Results: Climate AI: Weather Forecasting

**Student:** 210670N  
**Research Area:** Climate AI: Weather Forecasting  
**Date:** 2025-09-01  

## 1. Overview

This section presents the quantitative and qualitative outcomes of the perturbation experiments performed on the **GenCast Mini** model. The focus was on assessing how different inference-time perturbation strategies influence ensemble calibration, forecast spread, and probabilistic skill. Results are presented in terms of CRPS (Continuous Ranked Probability Score), spread–skill ratio, and variable-wise performance analysis.

All experiments were executed under identical conditions: the same ensemble size (N=8), initialization seeds, and evaluation dataset, ensuring that observed differences stemmed solely from the applied perturbation design.

---

## 2. Quantitative Results Summary

| Variant | Description | Mean CRPS | Observation |
|----------|--------------|------------|--------------|
| Baseline | No added noise | **4.19** | Reference deterministic skill |
| v1 | Additive isotropic noise | **103.2** | Severe imbalance, noisy fields |
| v2 | Smoothed noise | **76.53** | Reduced grid noise but imbalance remains |
| v3 | Physics-coupled perturbations | **23.25** | Best performance; physically realistic but unstable |
| v4 | Normalized hybrid (additive + multiplicative) | **26.83** | Stable structure, moderate degradation |
| v4* | Excluding geopotential and MSLP | **25.02** | Slight improvement, imbalance persists |

The baseline achieved a CRPS of 4.19, establishing the model’s deterministic skill level. Introducing isotropic Gaussian noise (v1) led to an order-of-magnitude degradation in skill, while spatial smoothing (v2) offered only partial improvement. The physics-coupled approach (v3) produced the most coherent structures but still degraded performance relative to baseline. The normalized hybrid method (v4) achieved slightly better stability but similar CRPS levels, indicating persistent model sensitivity.

---

## 3. Variable-wise CRPS Analysis

| Variable | Baseline CRPS | Perturbed CRPS (v3) | Change |
|-----------|----------------|----------------------|--------|
| 2m Temperature | 0.29 | 0.57 | ↑ Moderate increase |
| MSLP | 24.32 | 122.18 | ↑ Large degradation |
| 10m Winds (U/V) | 0.40 | 1.17 | ↑ Mild increase |
| Temperature (3D) | 0.28 | 0.86 | ↑ Significant |
| Geopotential | 21.49 | 140.77 | ↑ Severe degradation |
| Vertical Velocity | 0.04 | 0.10 | ↑ Small |
| Specific Humidity | 0.00008 | 0.00018 | ↑ Minimal |

Geopotential and mean sea-level pressure (MSLP) showed the largest CRPS increases, confirming that imbalances in dynamical fields dominate model instability. Surface variables like temperature and winds exhibited smaller yet consistent degradation.

---

## 4. Spread–Skill Ratio Analysis

| Configuration | Mean Ensemble Spread | Mean RMSE | Spread–Skill Ratio |
|----------------|----------------------|------------|--------------------|
| Baseline | 6.64 | 8.49 | 0.78 |
| v3 Perturbed | 14.92 | 42.07 | 0.35 |
| v4 Perturbed | 15.98 | 47.34 | 0.34 |

The spread–skill ratio dropped from 0.78 (well-calibrated baseline) to approximately 0.34 in perturbed ensembles, indicating overdispersion — the ensembles became too spread out relative to their error. This demonstrates that inference-time noise injection disrupts the calibration of diffusion-based ensembles.

---

## 5. Spatial CRPS Difference Visualization

The CRPS difference maps (`ΔCRPS = CRPS_perturbed - CRPS_baseline`) visualize where degradation occurred.  

**Observations:**
- **Geopotential (500 hPa):** Large positive ΔCRPS over mid-latitude storm tracks and equatorial regions, indicating instability in geopotential balance.  
- **2m Temperature:** Smaller ΔCRPS, mostly confined to tropical zones where perturbations were strongest.  
- **Wind Fields:** Slight spread increase but minimal structural distortion.  

These maps confirm that diffusion sampling amplifies small inconsistencies in dynamically coupled variables, especially at mid-tropospheric levels.

---

## 6. Discussion

The experiments consistently show that GenCast’s diffusion manifold enforces a strong multivariate balance among variables. Any perturbation that violates this balance, even slightly, causes the denoising network to produce unphysical outputs.  

Key takeaways:
- **Smoothing improves visual coherence** but not dynamic consistency.  
- **Physics-aware coupling (v3)** introduces meaningful structure but cannot overcome the model’s inherent sensitivity.  
- **Multiplicative normalization (v4)** stabilizes magnitudes yet fails to maintain ensemble calibration.  

These findings confirm that inference-time perturbations must be drawn from the model’s learned covariance structure — something that cannot be approximated through independent Gaussian noise alone.

---

## 7. Interpretation

- The CRPS degradation across all perturbations suggests that GenCast’s ensemble sampling is tuned to balanced states close to its training distribution.  
- The diffusion model acts as a “balance enforcer,” amplifying inconsistencies rather than dampening them, unlike physical NWP models.  
- Improved ensemble diversity will likely require perturbations generated from **latent-space sampling** or **EOF/PCA-based covariance models**, not direct input-level noise.

---

## 8. Summary

Although none of the perturbation strategies improved probabilistic skill, the experiments provided crucial diagnostic insights:
- GenCast’s forecasts are highly sensitive to imbalance in geopotential and pressure fields.  
- Perturbations must preserve cross-variable relationships to avoid cascading instability.  
- The developed experimental pipeline enables systematic testing of new perturbation models and can be reused for future GenCast or GraphCast research.

---

**Note:**  
All figures, CRPS maps, and ensemble spread visualizations are located in the `results_figures/` subfolder.
