# Ablation Study 2: Effect of Local Attention Window Size

## Motivation

The original ActionFormer paper utilized a local attention window of size 19 for its experiments on THUMOS14. Our TBT-Former introduces a Scaled Transformer Backbone with significantly higher representational capacity (more attention heads and a wider MLP). The motivation for this experiment is to test the hypothesis that this enhanced backbone can effectively leverage a larger temporal context. By expanding the local attention window, the model can access more information from surrounding features, which should lead to improved temporal reasoning and localization performance.

## Replication Instructions

To replicate this study, use the full TBT-Former model configuration (including the Scaled Backbone, CS-FPN, and BDR Head) and train it on the THUMOS14 dataset. The experiment involves training the model multiple times, with the only change between runs being the size of the local attention window.

1.  Set up the full TBT-Former model.
2.  Train and evaluate the model with a local attention window size of **19** (the original ActionFormer setting).
3.  Repeat the training and evaluation process for window sizes of **25**, **30**, and **37**.
4.  Compare the performance across all runs to identify the optimal window size.

## Results

The results clearly demonstrate the benefit of a larger attention window for our enhanced architecture. Performance steadily increases as the window size is expanded from 19 to 30, where it peaks at an average mAP of 68.0%. A slight performance drop is observed with a window size of 37, suggesting that a window of 30 provides the optimal balance between capturing sufficient temporal context and maintaining model focus.

| Window Size        | mAP@0.5  | mAP@0.7  | Avg. mAP |
| ------------------ | :------: | :------: | :------: |
| 19 (Baseline size) |   71.5   |   44.2   |   67.1   |
| 25                 |   72.0   |   44.8   |   67.6   |
| **30**             | **72.4** | **45.3** | **68.0** |
| 37                 |   72.2   |   45.1   |   67.8   |
