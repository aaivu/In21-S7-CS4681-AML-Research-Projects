# DenseNet-121 (CLAHE) with ZLPR Loss — Random Seed Impact

This experiment evaluates the effect of different random seeds on the performance of the CLAHE-enhanced DenseNet-121 model trained with ZLPR Loss. All runs use the same architecture, preprocessing, and training protocol, varying only the random seed.

## Overall Metrics Comparison (per seed)

| Seed | Loss   | Avg AUROC | Avg F1  |
|------|--------|-----------|---------|
| 12   | 1.6344 | 0.8483    | 0.3762  |
| 22   | 1.6276 | 0.8468    | 0.3758  |
| 32   | 1.6263 | 0.8479    | 0.3762  |
| 42   | 1.6268 | 0.8462    | 0.3621  |

## Per-Class AUROC Comparison (per seed)

| Class                | Seed 12 | Seed 22 | Seed 32 | Seed 42 |
|----------------------|:-------:|:-------:|:-------:|:-------:|
| Atelectasis          | 0.8095  | 0.8120  | 0.8153  | 0.8030  |
| Cardiomegaly         | 0.9276  | 0.9391  | 0.9244  | 0.9249  |
| Consolidation        | 0.7748  | 0.7742  | 0.7793  | 0.7764  |
| Edema                | 0.8779  | 0.8908  | 0.8928  | 0.8957  |
| Effusion             | 0.8986  | 0.8987  | 0.9024  | 0.8990  |
| Emphysema            | 0.9630  | 0.9668  | 0.9614  | 0.9655  |
| Fibrosis             | 0.8619  | 0.8416  | 0.8507  | 0.8564  |
| Hernia               | 0.9979  | 0.9896  | 0.9960  | 0.9830  |
| Infiltration         | 0.7049  | 0.6999  | 0.7070  | 0.6983  |
| Mass                 | 0.8989  | 0.9015  | 0.8969  | 0.9068  |
| Nodule               | 0.7814  | 0.7666  | 0.7705  | 0.7671  |
| Pleural_Thickening   | 0.8004  | 0.8012  | 0.8009  | 0.8015  |
| Pneumonia            | 0.7009  | 0.6845  | 0.6955  | 0.6817  |
| Pneumothorax         | 0.8787  | 0.8881  | 0.8776  | 0.8873  |

## Summary
- The CLAHE-enhanced DenseNet-121 with ZLPR Loss achieves high and stable AUROC across all random seeds, with only minor variation in per-class and overall metrics.
- Seed 12 achieves the highest overall AUROC, but all seeds perform strongly, demonstrating the robustness of this configuration.
- This model serves as a strong baseline for further experiments and comparisons with other loss functions or architectures.

## Best Seed per Class

| Class                | Best Seed | Highest AUROC |
|----------------------|:---------:|:-------------:|
| Atelectasis          |   32      |    0.8153     |
| Cardiomegaly         |   22      |    0.9391     |
| Consolidation        |   32      |    0.7793     |
| Edema                |   42      |    0.8957     |
| Effusion             |   32      |    0.9024     |
| Emphysema            |   22      |    0.9668     |
| Fibrosis             |   12      |    0.8619     |
| Hernia               |   12      |    0.9979     |
| Infiltration         |   32      |    0.7070     |
| Mass                 |   42      |    0.9068     |
| Nodule               |   12      |    0.7814     |
| Pleural_Thickening   |   22      |    0.8012     |
| Pneumonia            |   12      |    0.7009     |
| Pneumothorax         |   22      |    0.8881     |

**Summary:**
Seed 12 achieves the highest AUROC for Fibrosis, Hernia, Nodule, and Pneumonia. Seed 22 performs best for Cardiomegaly, Emphysema, Pleural Thickening, and Pneumothorax. Seed 32 yields the top AUROC for Atelectasis, Consolidation, Effusion, and Infiltration. Seed 42 is best for Edema and Mass. This highlights that while all seeds perform robustly, certain seeds may offer advantages for specific disease classes.
