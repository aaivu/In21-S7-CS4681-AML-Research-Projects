# Progress Report: Week 13 - Final Training and Data Collection

**Student Index Number:** 210207E

**Milestone:** Final Training Completion and Data Aggregation

## 1. Objectives for the Week

1.  Complete all remaining 5-seed, 300-iteration training runs.
2.  Collect all final metrics (PPL, Mock Accuracy, Grad Norm, Embedding Norm).
3.  Aggregate data into a final summary table and generate all required plots.

## 2. Progress Achieved

*   **Final Training:** All 15 training runs (3 variants $\times$ 5 seeds $\times$ 300 iterations) are complete, adhering strictly to the matched computational budget.
*   **Data Aggregation:** The final performance and stability metrics have been collected and aggregated into the `results/results_final_project_summary.csv` file.
*   **Metric Change:** The final analysis of the 5-seed, 300-iteration runs revealed that the **Baseline (No Norm) model is superior in final PPL**, achieving 11.31 compared to 11.92 for both normalized variants. This contrasts with the preliminary 150-iteration finding.

## 3. Key Finding: Robust Stability-Accuracy Trade-Off

The final, more rigorous runs conclusively established a trade-off:

*   **Accuracy Loss:** The LayerNorm and RMSNorm variants performed significantly worse on PPL ($\approx 5\%$ regression).
*   **Stability Gain:** Both normalized variants achieved significantly lower Final Total Gradient Norms (0.57 vs 1.10), confirming greater stability.
*   **Conclusion:** The techniques that stabilized training dynamics ultimately constrained the model's ability to achieve the best final performance.

## 4. Paper Finalization

*   **Analysis and Plotting:** Generated all final plots (Loss Curves, Gradient Norms, Embedding Norms) and integrated them into the paper draft.
*   **Discussion:** The discussion section is being finalized to focus on the robust **stability-accuracy trade-off** and why the normalization layers removed "crucial representational freedom" from the Baseline model.
*   **Final Plan:** The final week is dedicated to final statistical checks (paired bootstrap analysis, if time permits), code cleanup, documentation, and preparing the complete package for the final submission.

---
