
# DenseNet-121 (CLAHE) with Focal Loss — Random Seed Impact

This experiment evaluates the effect of different random seeds on the performance of the CLAHE-enhanced DenseNet-121 model trained with Focal Loss. All runs use the same architecture, preprocessing, and training protocol, varying only the random seed.

## Overall Metrics Comparison (per seed)

| Seed | Loss   | Avg AUROC | Avg F1  |
|------|--------|-----------|---------|
| 12   | 0.0420 | 0.8443    | 0.3693  |
| 22   | 0.0421 | 0.8475    | 0.3852  |
| 32   | 0.0422 | 0.8458    | 0.3679  |
| 42   | 0.0415 | 0.8514    | 0.3803  |

**Note:** The model achieves consistently high AUROC across all seeds, with seed 42 yielding the highest Avg AUROC and lowest loss.

## Per-Class AUROC Comparison (per seed)

| Class                | Seed 12 | Seed 22 | Seed 32 | Seed 42 |
|----------------------|:-------:|:-------:|:-------:|:-------:|
| Atelectasis          | 0.8047  | 0.8129  | 0.8187  | 0.8146  |
| Cardiomegaly         | 0.9320  | 0.9388  | 0.9406  | 0.9325  |
| Consolidation        | 0.7928  | 0.7772  | 0.7766  | 0.7871  |
| Edema                | 0.8841  | 0.8992  | 0.8806  | 0.8841  |
| Effusion             | 0.9010  | 0.8993  | 0.9014  | 0.9015  |
| Emphysema            | 0.9640  | 0.9681  | 0.9669  | 0.9656  |
| Fibrosis             | 0.8263  | 0.8441  | 0.8185  | 0.8207  |
| Hernia               | 0.9902  | 0.9801  | 0.9783  | 0.9936  |
| Infiltration         | 0.6975  | 0.6970  | 0.6983  | 0.7044  |
| Mass                 | 0.9027  | 0.8983  | 0.9029  | 0.9122  |
| Nodule               | 0.7666  | 0.7758  | 0.7653  | 0.7780  |
| Pleural_Thickening   | 0.8019  | 0.8002  | 0.7932  | 0.8124  |
| Pneumonia            | 0.6726  | 0.6974  | 0.7243  | 0.7229  |
| Pneumothorax         | 0.8831  | 0.8761  | 0.8757  | 0.8902  |

## Summary

- The CLAHE-enhanced DenseNet-121 with Focal Loss achieves high and stable AUROC across all random seeds, with only minor variation in per-class and overall metrics.
- Seed 42 achieves the highest overall AUROC and lowest loss, but all seeds perform strongly, demonstrating the robustness of this configuration.
- This model serves as a strong baseline for further experiments and comparisons with other loss functions or architectures.

## Model Performance by Disease (Summary)
## Best Seed per Class

| Class                | Best Seed | Highest AUROC |
|----------------------|:---------:|:-------------:|
| Atelectasis          |   32      |    0.8187     |
| Cardiomegaly         |   32      |    0.9406     |
| Consolidation        |   12      |    0.7928     |
| Edema                |   22      |    0.8992     |
| Effusion             |   42      |    0.9015     |
| Emphysema            |   22      |    0.9681     |
| Fibrosis             |   22      |    0.8441     |
| Hernia               |   42      |    0.9936     |
| Infiltration         |   42      |    0.7044     |
| Mass                 |   42      |    0.9122     |
| Nodule               |   42      |    0.7780     |
| Pleural_Thickening   |   42      |    0.8124     |
| Pneumonia            |   32      |    0.7243     |

**Summary:**
Seed 22 achieves the highest AUROC for Edema, Emphysema, and Fibrosis. Seed 32 performs best for Atelectasis, Cardiomegaly, and Pneumonia. Seed 42 yields the top AUROC for Effusion, Hernia, Infiltration, Mass, Nodule, and Pleural Thickening, showing strong performance across many classes. Seed 12 is best for Consolidation. This highlights that while all seeds perform robustly, certain seeds may offer advantages for specific disease classes.
