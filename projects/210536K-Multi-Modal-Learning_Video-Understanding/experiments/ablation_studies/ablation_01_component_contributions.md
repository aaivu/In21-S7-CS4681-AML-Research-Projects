# Ablation Study 1: Component-wise Contributions

## Motivation

This experiment is designed to validate the individual impact of each of the three core architectural enhancements proposed in TBT-Former. The primary motivation is to quantify how much the **Scaled Backbone**, the **Cross-Scale Feature Pyramid (CS-FPN)**, and the **Boundary Distribution Regression (BDR) Head** each contribute to the final performance gain over the ActionFormer baseline. The hypothesis is that each component provides a distinct and positive improvement, and their combination results in a synergistic performance boost.

## Replication Instructions

To replicate this study, conduct a series of experiments on the THUMOS14 dataset using the following model configurations. All other training parameters (optimizer, learning rate, epochs, etc.) should be kept identical to the baseline.

1.  **Baseline**: Train and evaluate the original ActionFormer model to establish a performance baseline.
2.  **+ Scaled Backbone**: Start with the baseline ActionFormer, but replace its standard Transformer encoder with our Scaled Transformer Backbone. This involves increasing the number of attention heads to 16 and expanding the MLP hidden dimension by a factor of 6x. The original neck and prediction heads are kept unchanged.
3.  **+ Cross-Scale FPN (CS-FPN)**: Start with the baseline ActionFormer, but replace its feed-forward feature pyramid with our CS-FPN, which includes a top-down pathway and lateral connections. The original backbone and prediction heads are kept unchanged.
4.  **+ Boundary Distribution Head (BDR)**: Start with the baseline ActionFormer, but replace its standard regression head with our BDR Head. The original backbone and neck are kept unchanged.
5.  **Full Model (TBT-Former)**: Combine all three modifications: the Scaled Backbone, the CS-FPN, and the BDR Head.

## Results

The results, summarized in the table below, confirm that each component contributes positively to the final performance. The BDR Head provides the most significant individual improvement (+0.8 mAP), validating the hypothesis that modeling boundary uncertainty is a highly effective strategy. When all components are combined, they work together to achieve a total improvement of +1.2 mAP over the baseline.

|  #  | Model Configuration                | Avg. mAP |  Δ   |
| :-: | ---------------------------------- | :------: | :--: |
|  1  | Baseline (ActionFormer)            |   66.8   |  -   |
|  2  | + Scaled Backbone                  |   67.2   | +0.4 |
|  3  | + Cross-Scale FPN (CS-FPN)         |   67.1   | +0.3 |
|  4  | + Boundary Distribution Head (BDR) |   67.6   | +0.8 |
|  5  | Full Model (TBT-Former)            |   68.0   | +1.2 |
