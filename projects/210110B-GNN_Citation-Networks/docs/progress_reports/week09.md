# Week 9 Progress Report
**Focus:** H-UniMP++ design

## Objectives
- Define enhancements to R-UniMP that address label noise and training curriculum.
- Produce an implementation plan for uncertainty-gated label injection and the masking schedule.

## Activities
- Analysed failure cases from R-UniMP runs to identify instability sources (label noise, masking sensitivity).
- Designed the gating mechanism (UGLI) combining label embeddings with learned confidence scores.
- Drafted a curriculum masking schedule transitioning from low to high masking across epochs.
- Outlined architecture diagrams and configuration updates for the new components.

## Outcomes
- Completed a design specification stored in engineering notes to guide Week 10 implementation.
- Identified required updates to the training script (`src/h_unimp_train.py`) and model module (`src/models/hetero_unimp.py`).
- Established success criteria: improved Macro-F1, smoother validation curves, and robustness to injected noise.

## Next Steps
- Implement the H-UniMP++ model and launch training experiments in Week 10.
