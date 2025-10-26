# Week 11 Progress Report
**Focus:** Testing and evaluation

## Objectives
- Quantitatively evaluate H-UniMP++ against prior iterations using consistent metrics.
- Assess robustness to label noise and monitor training stability.

## Activities
- Ran the evaluation pipeline to compute accuracy, Macro-F1, and Micro-F1 for UniMP, R-UniMP, and H-UniMP++.
- Conducted a label-noise ablation with 15% random label injection to measure performance degradation.
- Reviewed loss and metric curves to ensure curriculum masking eliminated late-epoch oscillations.

## Outcomes
- H-UniMP++ achieved 73.92% accuracy / 73.30% Macro-F1 / 70.50% Micro-F1, outperforming earlier models.
- Label-noise sensitivity reduced by ~1.3 percentage points versus the R-UniMP benchmark.
- Compiled findings into the draft of `results/final-evaluation-results.md` for reporting.

## Next Steps
- Transition into short paper preparation in Week 12, incorporating evaluation results.
