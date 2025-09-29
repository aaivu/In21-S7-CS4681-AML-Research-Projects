# DenseNet-121 (CLAHE, Attention) with Focal Loss — Random Seed Impact

This experiment evaluates the effect of random seed on the performance of the CLAHE-enhanced DenseNet-121 model with CBAM Attention and Focal Loss. Due to the high computational cost, only seed 42 was run for this configuration.

## Results (Seed 42)

| Metric   | Value   |
|----------|---------|
| Loss     | 0.0419  |
| Avg AUROC| 0.8480  |
| Avg F1   | 0.3787  |

### Per-Class AUROC (Seed 42)

| Class                | AUROC  |
|----------------------|:------:|
| Atelectasis          | 0.8138 |
| Cardiomegaly         | 0.9364 |
| Consolidation        | 0.7774 |
| Edema                | 0.8950 |
| Effusion             | 0.8991 |
| Emphysema            | 0.9606 |
| Fibrosis             | 0.8379 |
| Hernia               | 0.9973 |
| Infiltration         | 0.7058 |
| Mass                 | 0.9077 |
| Nodule               | 0.7648 |
| Pleural_Thickening   | 0.7932 |
| Pneumonia            | 0.7054 |
| Pneumothorax         | 0.8781 |

## Note

The attention-based method (CBAM) was only run for seed 42 due to the high time consumption required for training. Results for other seeds are not available for this configuration.
