# Ablation Study 4: Analysis of Regression Loss Weight ($\lambda_{reg}$)

## Motivation

The total loss function in TBT-Former is a weighted sum of a classification loss ($\mathcal{L}_{cls}$) and our novel regression loss ($\mathcal{L}_{DFL}$). The hyperparameter $\lambda_{reg}$ controls the balance between these two objectives. This experiment is motivated by the need to find the optimal weighting for this balance. A weight that is too low may cause the model to prioritize classification at the expense of localization accuracy, while a weight that is too high may lead to poor classification. This study analyzes the model's sensitivity to $\lambda_{reg}$ to identify the value that yields the best overall performance.

## Replication Instructions

To replicate this study, use a consistent model configuration (the paper uses the baseline ActionFormer with the BDR head) and train it on the THUMOS14 dataset. The experiment involves training the model multiple times, varying only the `lambda_reg` hyperparameter.

1.  Set up the model with the BDR head.
2.  Train and evaluate the model with $\lambda_{reg}$ set to **0.2**.
3.  Repeat the training and evaluation process for $\lambda_{reg}$ values of **0.5, 1.0, 2.0, and 5.0**.
4.  Compare the performance across all runs to determine the optimal weight.

## Results

The results indicate that the model is reasonably robust to the choice of $\lambda_{reg}$, but an optimal value exists. Performance is lowest at the extremes (0.2 and 5.0) and peaks when the weight is set to 1.0. This confirms that an equal balance between the classification and boundary distribution regression objectives provides the best trade-off for the model.

| $\lambda_{reg}$ | mAP@0.5  | mAP@0.7  | Avg. mAP |
| :-------------: | :------: | :------: | :------: |
|       0.2       |   70.1   |   39.8   |   65.0   |
|       0.5       |   71.4   |   41.7   |   66.4   |
|     **1.0**     | **71.0** | **43.6** | **66.9** |
|       2.0       |   69.7   |   43.1   |   66.3   |
|       5.0       |   68.8   |   42.5   |   65.1   |
