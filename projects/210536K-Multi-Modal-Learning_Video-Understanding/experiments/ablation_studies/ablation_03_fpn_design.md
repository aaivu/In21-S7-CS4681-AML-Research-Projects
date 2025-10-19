# Ablation Study 3: Design of the Feature Pyramid

## Motivation

Temporal actions occur at vastly different scales, from a few seconds to several minutes. A multi-scale feature pyramid is a standard architectural component designed to handle this variation. This experiment aims to validate the importance and determine the optimal depth of the feature pyramid within the ActionFormer framework. The hypothesis is that a deeper pyramid will lead to better performance, as it provides the model with a richer set of representations for detecting both short and long actions.

## Replication Instructions

To replicate this study, use the baseline ActionFormer model configuration and train it on the THUMOS14 dataset. The experiment involves modifying the encoder architecture to produce feature pyramids of varying depths.

1.  Configure the model to use only **1 pyramid level** (i.e., no downsampling).
2.  Train and evaluate configurations that produce **3, 5, 6, and 7 pyramid levels**. This is achieved by adjusting the number of Transformer blocks that include a downsampling operation.
3.  For each configuration, train the model using the same protocol and compare the final performance metrics.

## Results

The results, visualized in the plot from the paper (referenced as `vis.png`), show a clear and strong correlation between the number of pyramid levels and model performance.

- A single-level model performs very poorly, achieving only **47.6%** average mAP.
- Performance consistently and significantly improves as more levels are added.
- The best performance is achieved with **6 pyramid levels**, reaching an average mAP of **66.8%**.
- Adding a 7th level results in a slight performance degradation, indicating that 6 levels provide the optimal trade-off between multi-scale representation and model complexity for this task.

This confirms that a deep, multi-scale feature pyramid is a critical component for effective temporal action localization.
