# Week 6 Progress Report
**Focus:** UniMP baseline setup

## Objectives
- Implement a CPU-friendly UniMP-lite baseline for Citation-Network V1.
- Validate data loading and feature preprocessing pipeline.

## Activities
- Configured the PaddlePaddle environment with CPU safeguards (thread limits, allocator flags).
- Implemented the `baseline_unimp` experiment script using homogeneous aggregation and masked labels.
- Prepared dataset splits, feature normalisation, and metapath2vec embedding ingestion.

## Outcomes
- Verified the end-to-end training loop executes on the MacBook M2 without GPU requirements.
- Logged initial training curves and confirmed loss convergence within the reduced epoch budget.
- Stored experiment notes in `experiments/baseline_unimp/baseline_unimp_readme.md`.

## Next Steps
- Reproduce published UniMP accuracy on Citation-Network V1 during Week 7.
