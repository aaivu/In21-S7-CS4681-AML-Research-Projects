# Ablation Study 5: Alternative Backbone Architectures

## Motivation

The Transformer has become a dominant architecture for sequence modeling, but recent alternatives like Mamba (a State Space Model) and advanced convolutional designs like SGP have emerged, offering potential benefits in computational efficiency and temporal modeling. This experiment is motivated by the desire to explore the viability of these modern architectures for the task of temporal action localization. By replacing the Transformer backbone in the ActionFormer baseline, we can directly compare its performance against these alternatives and understand their strengths and weaknesses in this specific domain.

## Replication Instructions

To replicate this study, start with the baseline ActionFormer model and replace its encoder with two different backbones. All models should be trained and evaluated on the THUMOS14 dataset. The original neck and prediction heads should remain unchanged to ensure a fair comparison of the backbones.

1.  **Hybrid Transformer-Mamba Backbone**:

    - Modify the encoder to use standard Transformer blocks for the initial, high-resolution feature layers (the "stem").
    - For the later, coarser-resolution layers (the "branch"), replace the Transformer blocks with Mamba blocks, which are based on a Selective State Space Model (SSSM).
    - The architecture is illustrated in the paper in `Mamba_archite.png`.

2.  **SGP-based Backbone**:
    - Replace all Transformer blocks in the encoder with SGP blocks.
    - Each SGP block consists of a Scalable-Granularity Perception (SGP) layer, Group Normalization, and a Feed-Forward Network. The SGP layer itself has an instant-level branch and a window-level branch to capture multi-granularity context.
    - The architecture is illustrated in the paper in `SGP.png`.

## Results

The results show that both alternative architectures are highly competitive, but do not surpass the performance of the highly optimized Transformer baseline.

- The **Hybrid Attn-Mamba** backbone achieved an average mAP of **65.8%**.
- The **SGP (Convolutional)** backbone achieved an average mAP of **66.2%**.
- The baseline **Transformer** backbone achieved **66.8%**.

This suggests that while Mamba and SGP are promising and efficient alternatives, the local self-attention mechanism in Transformers remains a very effective method for temporal feature extraction in TAL. These results open up valuable directions for future research into refining these architectures for video tasks.

| Backbone Type          | mAP@0.5 | mAP@0.7 | Avg. mAP |
| ---------------------- | :-----: | :-----: | :------: |
| Transformer (Baseline) |  71.0   |  43.9   |   66.8   |
| Hybrid Attn-Mamba      |  69.5   |  42.9   |   65.8   |
| SGP (Convolutional)    |  70.1   |  43.2   |   66.2   |
