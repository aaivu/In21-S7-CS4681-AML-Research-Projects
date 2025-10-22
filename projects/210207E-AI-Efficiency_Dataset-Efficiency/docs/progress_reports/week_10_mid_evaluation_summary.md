# Progress Report: Week 10 - Mid-Evaluation Summary

**Student Index Number:** 210207E

**Milestone:** Mid-Evaluation Submission (Preliminary Results)

## 1. Objectives for the Week

1.  Complete the Phase 1 preliminary experiment (150 iterations, 3 seeds).
2.  Analyze results to validate the project's direction and technical feasibility.
3.  Draft and submit the Mid-Evaluation Short Paper.

## 2. Preliminary Results Summary (150 Iterations, 3 Seeds)

The experiment successfully executed all 3-seed runs. The mean metrics collected are:

| Metric | Baseline (None) | LayerNorm (LN) | RMSNorm (RMS) |
| :--- | :--- | :--- | :--- |
| **Mean Val PPL (↓)** | 22.18 | **20.64** | 21.88 |
| **Std Dev PPL** | ±0.90 | ±0.81 | ±0.77 |
| **Mean Final Grad Norm (↓)** | 0.1762 | 0.1783 | **0.1706** |
| **Mean Embedding Norm (||. ||)** | 11.46 | 11.68 | 11.59 |

## 3. Key Observations & Validation

*   **Performance Success:** The **LayerNorm (LN) variant unexpectedly demonstrated a substantial performance gain** (approx. **7% PPL reduction** from 22.18 to 20.64) over the un-normalized Baseline. This result is particularly notable as it contradicts large-scale studies (Le Scao et al. [1]) and validates the project's core enhancement for this smaller GPT-style regime.
*   **Stability Diagnostic:** The RMSNorm variant achieved the lowest final mean gradient norm (0.1706), suggesting the smoothest convergence profile.
*   **Technical Validation:** The diagnostic Mean Embedding Norms successfully confirmed that the normalization layers are active.

## 4. Plan for Next Week (Final Submission Preparation)

The project direction is validated. The remaining work is to increase the rigor and fully characterize the trade-off.

*   **Extended Rigor:** Immediately begin training runs extended to **300 iterations** and expanded to **5 distinct random seeds** for all three variants.
*   **Analysis Preparation:** Prepare code for plotting loss/norm trends over time and begin implementing statistical analysis methods (e.g., paired bootstrap testing).
*   **Paper Draft:** Outline the "Experiments and Results" and "Discussion" sections for the Final Paper, focusing on the unexpected LN performance gain.

---
