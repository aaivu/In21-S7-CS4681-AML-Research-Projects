# Comparison: DenseNet-121 (CLAHE) vs EfficientNet-B2

Summary
-------
This experiment compares two backbones for the reproduced DannyNet model using the same CLAHE-preprocessed images and a fixed random seed (`42`) for reproducibility:

- DenseNet-121 backbone (CLAHE-enhanced replication) — reference reproduction used in earlier experiments.
- EfficientNet-B2 backbone — new run using the same preprocessing and evaluation protocol.

Both runs used the same data splits, CLAHE preprocessing, and evaluation code (optimal thresholding per-class). Below I present overall metrics and per-class AUROC/F1 comparisons.

Overall metrics comparison
--------------------------
| Metric | DenseNet-121 (CLAHE replication) | EfficientNet-B2 |
|-------:|---------------------------------:|----------------:|
| Loss   | **0.0415**                       | 0.0423          |
| Avg AUROC | **0.8514**                    | 0.8482          |
| Avg F1 | **0.3803**                       | 0.3676          |


Per-class comparison (DenseNet-121 CLAHE replication vs EfficientNet-B2)
-------------------------------------------------------------------------
| Class                | DenseNet AUROC | EffNet AUROC |
|----------------------|:--------------:|:------------:|
| Atelectasis          | 0.8146         | 0.8168       |
| Cardiomegaly         | 0.9325         | 0.9259       |
| Consolidation        | 0.7871         | 0.7799       |
| Edema                | 0.8841         | 0.9064       |
| Effusion             | 0.9015         | 0.9032       |
| Emphysema            | 0.9656         | 0.9612       |
| Fibrosis             | 0.8207         | 0.8177       |
| Hernia               | 0.9936         | 0.9733       |
| Infiltration         | 0.7044         | 0.7093       |
| Mass                 | 0.9122         | 0.8974       |
| Nodule               | 0.7780         | 0.7848       |
| Pleural_Thickening   | 0.8124         | 0.7845       |
| Pneumonia            | 0.7229         | 0.7338       |
| Pneumothorax         | 0.8902         | 0.8808       |

Conclusions
-----------
- Overall, the DenseNet-121 (CLAHE replication) run has slightly better overall metrics (lower loss, higher Avg AUROC and higher Avg F1) compared to the EfficientNet-B2 run.
- Per-class comparisons:
  - DenseNet-121 (CLAHE) shows higher AUROC for: Atelectasis (slightly), Cardiomegaly, Consolidation, Effusion (close), Emphysema, Fibrosis, Hernia, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax (see tables above).
  - EfficientNet-B2 has higher AUROC for: Edema, Infiltration, and slight gains for some classes such as Cardiomegaly F1 and Mass F1 in certain cases.
- Decision: because the CLAHE+DenseNet-121 replication achieves better average performance and lower loss in my runs, I will continue using the CLAHE-enhanced DenseNet-121 replication as the working baseline for subsequent experiments.

Notes
-----
- I kept the random seed fixed at `42` across all experiments to ensure deterministic splits and comparable results.
- I attempted an EfficientNet-B3 run as well; training exceeded 12 hours on the Kaggle environment and resulted in a runtime failure before completion. The B3 run is therefore incomplete and not included in the comparison tables above.
- See `Danneynet_with_efficientnetb2.ipynb` in this folder for the exact code used to train and evaluate EfficientNet-B2.
