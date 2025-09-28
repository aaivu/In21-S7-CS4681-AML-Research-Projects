# EfficientNet-B3 (CLAHE) with Focal Loss — Random Seed Impact

This experiment evaluates the effect of different random seeds on the performance of the CLAHE-enhanced EfficientNet-B3 model trained with Focal Loss. All runs use the same architecture, preprocessing, and training protocol, varying only the random seed.

## Overall Metrics Comparison (per seed)

| Seed | Loss   | Avg AUROC | Avg F1  |
|------|--------|-----------|---------|
| 22   | 0.0428 | 0.8411    | 0.3726  |
| 32   | 0.0433 | 0.8389    | 0.3741  |
| 42   | 0.0431 | 0.8382    | 0.3774  |

## Per-Class AUROC Comparison (per seed)

| Class                | Seed 22 | Seed 32 | Seed 42 |
|----------------------|:-------:|:-------:|:-------:|
| Atelectasis          | 0.8220  | 0.8105  | 0.8179  |
| Cardiomegaly         | 0.9335  | 0.9306  | 0.9358  |
| Consolidation        | 0.7674  | 0.7678  | 0.7693  |
| Edema                | 0.8947  | 0.8891  | 0.8931  |
| Effusion             | 0.9012  | 0.9017  | 0.9039  |
| Emphysema            | 0.9611  | 0.9561  | 0.9451  |
| Fibrosis             | 0.7762  | 0.7760  | 0.7671  |
| Hernia               | 0.9822  | 0.9876  | 0.9659  |
| Infiltration         | 0.6986  | 0.7035  | 0.6986  |
| Mass                 | 0.8946  | 0.9043  | 0.8975  |
| Nodule               | 0.7655  | 0.7680  | 0.7779  |
| Pleural_Thickening   | 0.7576  | 0.7488  | 0.7653  |
| Pneumonia            | 0.7442  | 0.7364  | 0.7234  |
| Pneumothorax         | 0.8763  | 0.8638  | 0.8736  |

## Best Seed per Class

| Class                | Best Seed | Highest AUROC |
|----------------------|:---------:|:-------------:|
| Atelectasis          |   22      |    0.8220     |
| Cardiomegaly         |   42      |    0.9358     |
| Consolidation        |   42      |    0.7693     |
| Edema                |   22      |    0.8947     |
| Effusion             |   42      |    0.9039     |
| Emphysema            |   22      |    0.9611     |
| Fibrosis             |   22      |    0.7762     |
| Hernia               |   32      |    0.9876     |
| Infiltration         |   32      |    0.7035     |
| Mass                 |   32      |    0.9043     |
| Nodule               |   42      |    0.7779     |
| Pleural_Thickening   |   42      |    0.7653     |
| Pneumonia            |   22      |    0.7442     |
| Pneumothorax         |   22      |    0.8763     |

**Summary:**
Seed 22 achieves the highest AUROC for Atelectasis, Edema, Emphysema, Fibrosis, Pneumonia, and Pneumothorax. Seed 32 is best for Hernia, Infiltration, and Mass. Seed 42 yields the top AUROC for Cardiomegaly, Consolidation, Effusion, Nodule, and Pleural Thickening. This highlights that while all seeds perform robustly, certain seeds may offer advantages for specific disease classes.

## Summary

- The CLAHE-enhanced EfficientNet-B3 with Focal Loss achieves high and stable AUROC across all random seeds, with only minor variation in per-class and overall metrics.
- Seed 22 and 42 achieve the highest AUROC for most classes, but all seeds perform strongly, demonstrating the robustness of this configuration.
- This model serves as a strong baseline for further experiments and comparisons with other loss functions or architectures.
