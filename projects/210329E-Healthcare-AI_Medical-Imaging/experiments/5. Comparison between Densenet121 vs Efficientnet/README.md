# Experiment 5 - Comparison: DenseNet-121 (CLAHE) vs EfficientNet-B2

Summary
-------
This experiment compares two backbones for the reproduced DannyNet model using the same CLAHE-preprocessed images and a fixed random seed (`42`) for reproducibility:

- DenseNet-121 backbone (CLAHE-enhanced replication) — reference reproduction used in earlier experiments.
- EfficientNet-B2 backbone — new run using the same preprocessing and evaluation protocol.

Both runs used the same data splits, CLAHE preprocessing, and evaluation code (optimal thresholding per-class). Below I present overall metrics and per-class AUROC/F1 comparisons.

| Metric | DenseNet-121 (CLAHE replication) | EfficientNet-B2 |
|-------:|---------------------------------:|----------------:|
| Loss   | **0.0415**                       | 0.0423          |
| Avg AUROC | **0.8514**                    | 0.8482          |
| Avg F1 | **0.3803**                       | 0.3676          |

Overall metrics comparison
--------------------------
| Metric    | DenseNet-121 (CLAHE replication) | EfficientNet-B2 | EfficientNet-B3 |
|----------:|---------------------------------:|----------------:|----------------:|
| Loss      | **0.0415**                       | 0.0423          | 0.0431          |
| Avg AUROC | **0.8514**                       | 0.8482          | 0.8382          |
| Avg F1    | **0.3803**                       | 0.3676          | 0.3774          |


Per-class comparison (DenseNet-121 CLAHE replication vs EfficientNet-B2 vs EfficientNet-B3)
------------------------------------------------------------------------------------------
| Class                | DenseNet AUROC | EffNet-B2 AUROC | EffNet-B3 AUROC |
|----------------------|:--------------:|:---------------:|:---------------:|
| Atelectasis          | 0.8146         | 0.8168          | 0.8179          |
| Cardiomegaly         | 0.9325         | 0.9259          | 0.9358          |
| Consolidation        | 0.7871         | 0.7799          | 0.7693          |
| Edema                | 0.8841         | 0.9064          | 0.8931          |
| Effusion             | 0.9015         | 0.9032          | 0.9039          |
| Emphysema            | 0.9656         | 0.9612          | 0.9451          |
| Fibrosis             | 0.8207         | 0.8177          | 0.7671          |
| Hernia               | 0.9936         | 0.9733          | 0.9659          |
| Infiltration         | 0.7044         | 0.7093          | 0.6986          |
| Mass                 | 0.9122         | 0.8974          | 0.8975          |
| Nodule               | 0.7780         | 0.7848          | 0.7779          |
| Pleural_Thickening   | 0.8124         | 0.7845          | 0.7653          |
| Pneumonia            | 0.7229         | 0.7338          | 0.7234          |
| Pneumothorax         | 0.8902         | 0.8808          | 0.8736          |


Conclusions
-----------
- Among the three models, DenseNet-121 (CLAHE replication) achieves the best overall metrics (lowest loss, highest Avg AUROC and Avg F1), followed by EfficientNet-B2, with EfficientNet-B3 trailing slightly in Avg AUROC but showing competitive Avg F1.
- Per-class AUROC: DenseNet-121 leads on most classes, but EfficientNet-B2 and B3 each have individual class wins (e.g., Edema for B2, Cardiomegaly for B3). B3 does not consistently outperform B2 or DenseNet-121 on any group of classes, despite its larger size and higher input resolution.
- EfficientNet-B3 required substantially more computational resources (image size 300, longer training time) but did not yield a clear overall advantage in this setup.
- Decision: Because the CLAHE+DenseNet-121 replication achieves better average performance and lower loss in my runs, I will continue using the CLAHE-enhanced DenseNet-121 replication as the working baseline for subsequent experiments.


Notes
-----
- I kept the random seed fixed at `42` across all experiments to ensure deterministic splits and comparable results.
- Image-size / preprocessing note:
  - DenseNet-121 runs used an input image size of `224` (standard for DenseNet).
  - EfficientNet-B2 used `260` for better compatibility with EfficientNet scaling and preprocessing.
  - EfficientNet-B3 used `300` as its native image size, which increases computational and memory requirements. Training EfficientNet-B3 took noticeably more time and resources compared to B2 and DenseNet-121.
- See `Danneynet_with_efficientnetb2.ipynb` and `Danneynet_with_efficientnetb3.ipynb` in this folder for the exact code used to train and evaluate EfficientNet-B2 and EfficientNet-B3, respectively.
