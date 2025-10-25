# Week 8 Progress Report
**Focus:** R-UniMP setup and reproduction

## Objectives
- Integrate relation-aware projections and sampling to replicate the R-UniMP baseline.
- Validate heterogeneous data handling across citation, author→paper, and paper→author edges.

## Activities
- Implemented the `r_unimp` experiment with per-relation linear layers and typed aggregations.
- Adjusted the data loader to emit relation-specific adjacency and metapath2vec features.
- Ran controlled experiments comparing homogeneous versus heterogeneous propagation.

## Outcomes
- Reproduced Iter-2 R-UniMP metrics: 73.71% accuracy / 72.50% Macro-F1 / 70.20% Micro-F1.
- Documented architectural differences and configuration values in `experiments/r_unimp/r_unimp_readme.md`.
- Confirmed pipeline scalability while staying within CPU memory limits.

## Next Steps
- Begin designing the H-UniMP++ enhancements in Week 9.
