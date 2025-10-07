# EfficientNet-B2 (CLAHE) with Focal Loss — Random Seed Impact

This experiment evaluates the effect of different random seeds on the performance of the CLAHE-enhanced EfficientNet-B2 model trained with Focal Loss. All runs use the same architecture, preprocessing, and training protocol, varying only the random seed.

## Overall Metrics Comparison (per seed)

| Seed | Loss   | Avg AUROC | Avg F1  |
|------|--------|-----------|---------|
| 22   | 0.0422 | 0.8406    | 0.3843  |
| 32   | 0.0427 | 0.8391    | 0.3524  |
| 42   | 0.0423 | 0.8482    | 0.3676  |

## Per-Class AUROC Comparison (per seed)

| Class                | Seed 22 | Seed 32 | Seed 42 |
|----------------------|:-------:|:-------:|:-------:|
| Atelectasis          | 0.8063  | 0.8003  | 0.8168  |
| Cardiomegaly         | 0.9355  | 0.9248  | 0.9259  |
| Consolidation        | 0.7769  | 0.7691  | 0.7799  |
| Edema                | 0.8982  | 0.8887  | 0.9064  |
| Effusion             | 0.9019  | 0.9022  | 0.9032  |
| Emphysema            | 0.9539  | 0.9612  | 0.9612  |
| Fibrosis             | 0.7930  | 0.7895  | 0.8177  |
| Hernia               | 0.9743  | 0.9804  | 0.9733  |
| Infiltration         | 0.7094  | 0.7009  | 0.7093  |
| Mass                 | 0.8916  | 0.8773  | 0.8974  |
| Nodule               | 0.7765  | 0.7602  | 0.7848  |
| Pleural_Thickening   | 0.8021  | 0.7977  | 0.7845  |
| Pneumonia            | 0.6660  | 0.7176  | 0.7338  |
| Pneumothorax         | 0.8834  | 0.8781  | 0.8808  |

## Summary
- The CLAHE-enhanced EfficientNet-B2 with Focal Loss achieves high and stable AUROC across all random seeds, with only minor variation in per-class and overall metrics.
- Seed 42 achieves the highest overall AUROC, but all seeds perform strongly, demonstrating the robustness of this configuration.
- This model serves as a strong baseline for further experiments and comparisons with other loss functions or architectures.

## Best Seed per Class

| Class                | Best Seed | Highest AUROC |
|----------------------|:---------:|:-------------:|
| Atelectasis          |   42      |    0.8168     |
| Cardiomegaly         |   22      |    0.9355     |
| Consolidation        |   42      |    0.7799     |
| Edema                |   42      |    0.9064     |
| Effusion             |   42      |    0.9032     |
| Emphysema            |   32/42   |    0.9612     |
| Fibrosis             |   42      |    0.8177     |
| Hernia               |   32      |    0.9804     |
| Infiltration         |   22      |    0.7094     |
| Mass                 |   42      |    0.8974     |
| Nodule               |   42      |    0.7848     |
| Pleural_Thickening   |   22      |    0.8021     |
| Pneumonia            |   42      |    0.7338     |
| Pneumothorax         |   22      |    0.8834     |

**Summary:**
Seed 22 achieves the highest AUROC for Cardiomegaly, Infiltration, Pleural Thickening, and Pneumothorax. Seed 32 is best for Hernia and (tied with 42) for Emphysema. Seed 42 yields the top AUROC for Atelectasis, Consolidation, Edema, Effusion, Fibrosis, Mass, Nodule, and Pneumonia, and is tied for Emphysema. This highlights that while all seeds perform robustly, certain seeds may offer advantages for specific disease classes.
