# Week 7 Progress Report
**Focus:** Reproducing UniMP results

## Objectives
- Match reported UniMP performance on Citation-Network V1 using the baseline implementation.
- Establish reference metrics for subsequent iterations.

## Activities
- Ran multiple training seeds and tuned learning-rate/warmup schedules to stabilise convergence.
- Applied masking-rate schedules referenced in the UniMP paper to align with the published setup.
- Verified the evaluation script computes accuracy, Macro-F1, and Micro-F1 consistently.

## Outcomes
- Achieved 70.20% accuracy / 68.70% Macro-F1 / 69.40% Micro-F1, roughly aligning with the documented baseline.
- Captured hyperparameter settings and training logs for reproducibility.
- Updated the results table in `experiments/baseline_unimp` to reflect the replicated metrics.

## Next Steps
- Extend the architecture to the heterogeneous relation-aware variant (R-UniMP) in Week 8.
