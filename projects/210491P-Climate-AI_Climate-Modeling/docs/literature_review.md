# Literature Review: Climate AI:Climate Modeling

**Student:** 210491P
**Research Area:** Climate AI:Climate Modeling
**Date:** 2025-09-01

## Abstract

This review surveys recent progress at the intersection of AI and climate/weather modeling, focusing on (i) global medium-range AI forecast models (GraphCast, Pangu-Weather, FourCastNet, AIFS), (ii) foundation models for weather/climate (ClimaX, Aurora), (iii) probabilistic/ensemble forecasting and post-processing (CRPS/fair-CRPS, EMOS, neural post-processing), (iv) high-resolution nowcasting and generative approaches (DGMR, NowcastNet), and (v) downscaling via diffusion/generative models (CorrDiff). I synthesize lessons on accuracy, reliability, computational cost, and evaluation benchmarks (WeatherBench 2), and position the project’s ensemble post-processing for GraphCast-small (Gaussian perturbations + trimmed-mean + bias correction) as a lightweight pathway to improve deterministic AI forecasts without retraining.

## 1. Introduction

AI weather models have rapidly advanced from promising prototypes to operational-grade systems that rival or surpass leading NWP baselines in medium-range skill while being orders of magnitude faster (e.g., GraphCast, Pangu-Weather, FourCastNet; ECMWF’s AIFS). These models learn from reanalyses (such as ERA5) and increasingly target probabilistic outputs and ensemble generation. The field is converging on hybrid workflows: 
- AI for fast, accurate base forecasts
- Ensembles/post-processing for calibrated uncertainty
- Diffusion/generative models for km-scale details.

## 2. Search Methodology

### Search Terms Used
- “GraphCast”, “FourCastNet”, “Pangu-Weather”, “ClimaX”, “AIFS”, “Aurora foundation model”
- “probabilistic weather forecasting”, “ensemble post-processing”, “EMOS”, “CRPS”, “fair-CRPS”
- “diffusion downscaling”, “CorrDiff”, “nowcasting DGMR/NowcastNet”
- “ERA5 reanalysis”, “WeatherBench 2 benchmark”

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] ECMWF

### Time Period
2019–2025 (specially focused on 2022–2025)

## 3. Key Areas of Research

### 3.1 Modern AI Models for Global Weather
GraphCast, Pangu-Weather, and FourCastNet demonstrate state-of-the-art medium-range skill at 0.25° resolution, often outperforming operational deterministic systems, with significant speed/efficiency gains. ECMWF’s AIFS extends this into an operational, data-driven pipeline and is exploring probabilistic training/ensembles. Foundation models (ClimaX, Aurora) aim for broad transfer across weather/climate tasks via pretraining on heterogeneous Earth-system data.

**Key Papers:**
- **Lam et al., 2023 – “GraphCast: Learning skillful medium-range weather forecasting”**
Introduced a graph-neural-network–based global forecasting model that produces 10-day, 0.25° forecasts in minutes. Demonstrated superior medium-range skill to ECMWF’s IFS while requiring far less computation, establishing GNNs as viable NWP replacements.

- **Bi et al., 2023 – “Accurate medium-range global weather forecasting with Pangu-Weather”**  
Proposed a 3-D Earth-specific Transformer trained on ERA5. Showed that deep spatiotemporal attention enables 7-day deterministic forecasts with state-of-the-art accuracy and dramatic speedups over traditional numerical models.

- **Pathak et al., 2022 – “FourCastNet: A Global Data-Driven High-Resolution Weather Model”**  
Developed a Fourier Neural Operator (AFNO) framework for global forecasting. Achieved NWP-comparable medium-range skill and improved computational efficiency, pioneering real-time machine-learning emulators for the atmosphere.

- **Lang et al., 2024 – “AIFS: ECMWF’s AI-Driven Forecasting System”**  
Presented the first operational-grade, data-driven system combining transformer and diffusion components. Demonstrated near-parity with the physical IFS and marked ECMWF’s entry into hybrid operational AI forecasting.

- **Nguyen et al., 2023 – “ClimaX: A Foundation Model for Weather and Climate”**  
Introduced a large foundation model pre-trained on heterogeneous Earth-system data. Unified multiple forecasting and downscaling tasks through flexible fine-tuning, showing strong cross-domain transfer.

- **Bodnar et al., 2024 – “Aurora: A Foundation Model for the Earth System”** 
Proposed a multimodal spatiotemporal transformer integrating physics constraints. Enabled scalable fine-tuning for atmosphere, ocean, and climate tasks, illustrating the potential of foundation models in Earth-system science.

### 3.2 Probabilistic Forecasting, Ensembles, and Post-processing

AI models are moving beyond deterministic outputs toward calibrated ensembles. Classical ensemble verification/post-processing tools remain central, and neural post-processing improves calibration/sharpness. ECMWF explores diffusion-based or CRPS-based training for stochastic AIFS ensembles. Your project adopts a post-processing ensemble strategy for GraphCast-small: input perturbations (Gaussian), aggregation (mean/median/trimmed-mean), and lightweight bias correction. 

**Key Papers:**

- **Gneiting et al., 2005 – “Calibrated Probabilistic Forecasting Using EMOS”**  
Established the Ensemble Model Output Statistics (EMOS) framework. Provided a theoretical basis for bias correction and spread calibration via the continuous ranked probability score (CRPS).

- **Leutbecher, 2019 – “Ensemble Size: How Suboptimal Is Less Than Infinity?”**  
Investigated ensemble-size impacts on forecast verification and introduced fair-CRPS normalization. Clarified statistical fairness in comparing probabilistic forecasts with differing member counts.

- **Rasp & Lerch, 2018 – “Neural Networks for Post-Processing Ensemble Forecasts”**  
Showed that neural architectures outperform EMOS in ensemble calibration. Improved probabilistic forecast sharpness and reliability across temperature and wind variables.



### 3.3 Real-time forecasting and High-Resolution Downscaling

Generative models (such as DGMR) and physics-informed learning approaches (like NowcastNet) have improved the ability to produce real-time, probabilistic forecasts of heavy rainfall. For finer spatial details, diffusion-based downscaling methods (such as CorrDiff) can efficiently convert coarse global data into high-resolution (around 2 km) outputs. These methods help preserve important features such as spatial patterns and extreme events, making them useful for local impact assessment and hazard prediction.


### 3.4 Data, Benchmarks, and Evaluation

ERA5 remains the primary dataset used for training and verification in weather prediction models. The WeatherBench 2 framework standardizes datasets, evaluation metrics, and leaderboards for comparing machine learning and numerical weather prediction (NWP) methods. This standardization improves reproducibility and enables fair and transparent model evaluation.


## 4. Research Gaps and Opportunities

### Gap 1: Gap 1: Reliable Uncertainty for Deterministic AI Models
- **Why it matters:** Most AI-based weather models such as GraphCast produce deterministic forecasts that provide only a single best estimate instead of a range of possible outcomes. Without calibrated uncertainty information, users cannot fully assess forecast reliability or make risk-aware decisions.  
- **How my project addresses it:** This study converts GraphCast-small into an ensemble system by adding controlled input perturbations. The use of trimmed-mean aggregation and bias correction improves both accuracy and spatial smoothness, offering an efficient way to represent forecast uncertainty.  

### Gap 2: Calibration and Extreme Events
- **Why it matters:** AI weather models often underestimate the probability and intensity of extreme events. Proper calibration methods such as Ensemble Model Output Statistics (EMOS), and newer generative approaches are important for capturing the tails of the forecast distribution accurately.  
- **How my project addresses it:** The proposed framework uses variable-specific bias correction and performs sensitivity tests on noise magnitude and ensemble size. This approach prevents over-adjustment and ensures that extreme events are represented more realistically for each variable.

### Gap 3: Regional Detail without High Computational Cost
- **Why it matters:** Operational forecasting and impact modeling often require kilometer-scale spatial detail, but achieving this through traditional numerical models is very expensive. Diffusion-based downscaling offers an efficient and scalable solution.  
- **How my project addresses it:** Future work will integrate diffusion-based downscaling techniques on top of the ensemble-enhanced forecasts. This will help transfer uncertainty information from the global scale to local-scale predictions while keeping computation low.


### Gap 4: Standardized and Reproducible Evaluation
- **Why it matters:** The absence of standardized benchmarks and metrics has made it difficult to compare AI forecasting systems consistently. The WeatherBench 2 framework provides uniform datasets and evaluation methods to ensure fair and reproducible model comparisons.
- **How my project addresses it:** The study consistently reports RMSE Aligning experiments with these benchmarks will enhance transparency and comparability across AI-based forecasting research.


## 5. Theoretical Framework

This review adopts the perspective of ensemble-based forecasting, where uncertainty is represented through multiple forecast members rather than a single deterministic output. Ensemble post-processing techniques such as bias correction, trimmed-mean, and median aggregation are used to improve accuracy and reduce systematic errors. These methods help balance forecast diversity and stability by addressing under or over dispersion in ensemble predictions. Stochastic perturbations applied to the model inputs increase variability, while aggregation helps minimize the effect of outlier predictions.

## 6. Methodology Insights

Most recent studies train AI-based weather models using the ERA5 reanalysis dataset and evaluate performance using metrics such as Root Mean Square Error (RMSE). Researchers typically explore different ensemble sizes and input noise magnitudes to analyze model sensitivity and stability. For resource-limited environments, ensemble post-processing using input perturbations and aggregation techniques provides an effective way to enhance forecast accuracy without retraining the model.

## 7. Conclusion

Artificial intelligence is redefining weather and climate prediction by achieving performance comparable to operational numerical weather models. Foundation models and diffusion-based approaches continue to expand the potential for multi-scale and multi-variable forecasting. At the same time, ensemble-based post-processing offers a simple and computationally efficient path to improve reliability and reduce bias in AI forecasts.

## References

[1]R. Lam et al., “GraphCast: Learning skillful medium-range global weather forecasting,” arXiv:2212.12794 [physics], Dec. 2022, Available: https://arxiv.org/abs/2212.12794  
[2]K. Bi, L. Xie, H. Zhang, X. Chen, X. Gu, and Q. Tian, “Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast,” arXiv:2211.02556 [physics], Nov. 2022, Available: https://arxiv.org/abs/2211.02556  
[3]J. Pathak et al., “FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators,” arXiv:2202.11214 [physics], Feb. 2022, Available: https://arxiv.org/abs/2202.11214  
[4]T. Kurth et al., “FourCastNet: Accelerating Global High-Resolution Weather Forecasting using Adaptive Fourier Neural Operators,” arXiv.org, 2022. https://arxiv.org/abs/2208.05419  
[5]S. Lang et al., “AIFS -- ECMWF’s data-driven forecasting system,” arXiv.org, 2024. https://arxiv.org/abs/2406.01465  
[6]T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, and A. Grover, “ClimaX: A foundation model for weather and climate,” arXiv:2301.10343 [cs], Feb. 2023, Available: https://arxiv.org/abs/2301.10343  
[7]C. Bodnar et al., “Aurora: A Foundation Model of the Atmosphere,” arXiv.org, 2024. https://arxiv.org/abs/2405.13063  
[8]H. Hersbach et al., “The ERA5 global reanalysis,” Quarterly Journal of the Royal Meteorological Society, vol. 146, no. 730, Jun. 2020, doi: https://doi.org/10.1002/qj.3803.  
[9]S. Rasp et al., “WeatherBench 2: A benchmark for the next generation of data-driven
  global weather models,” arXiv (Cornell University), Aug. 2023, doi: https://doi.org/10.48550/arxiv.2308.15560.  
[10]S. Rasp and S. Lerch, “Neural Networks for Postprocessing Ensemble Weather Forecasts,” Monthly Weather Review, vol. 146, no. 11, pp. 3885–3900, Oct. 2018, doi: https://doi.org/10.1175/mwr-d-18-0187.1.  
[11]T. Gneiting, A. E. Raftery, A. H. Westveld, and T. Goldman, “Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation,” Monthly Weather Review, vol. 133, no. 5, pp. 1098–1118, May 2005, doi: https://doi.org/10.1175/mwr2904.1.  
[12]M. Leutbecher, “Ensemble size: How suboptimal is less than infinity?,” Quarterly Journal of the Royal Meteorological Society, vol. 145, no. S1, pp. 107–128, Oct. 2018, doi: https://doi.org/10.1002/qj.3387.  
[13]S. Ravuri et al., “Skilful precipitation nowcasting using deep generative models of radar,” Nature, vol. 597, no. 7878, pp. 672–677, Sep. 2021, doi: https://doi.org/10.1038/s41586-021-03854-z.  
[14]Y. Zhang et al., “Skilful nowcasting of extreme precipitation with NowcastNet,” Nature, vol. 619, no. 7970, pp. 526–532, Jul. 2023, doi: https://doi.org/10.1038/s41586-023-06184-4.  
[15] Alexe, M., Lang, S., Clare, M., Leutbecher, M., Roberts, C., Magnusson, L., Chantry, M., Adewoyin, R., Prieto-Nemesio, A., Dramsch, J., Pinault, F., & Raoult, B., "Data-driven ensemble forecasting with the AIFS," ECMWF Newsletter, no. 181, pp. 32-37, Autumn 2024. Available online: https://www.ecmwf.int/sites/default/files/elibrary/102024/81620-data-driven-ensemble-forecasting-with-the-aifs.pdf
