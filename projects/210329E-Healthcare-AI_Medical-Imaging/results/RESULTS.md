# Main Results: Ensemble Model and Baseline Comparisons

## 1. Per-Class AUROC Comparison

| Disease             | CheXNet (Paper) | DannyNet (Paper) | DannyNet (Reproduced) | Deep Ensemble (Our Work) |
|---------------------|----------------:|-----------------:|----------------------:|----------------------:|
| Atelectasis         | 0.809           | 0.817            | 0.8181                | 0.8215                |
| Cardiomegaly        | 0.925           | 0.932            | 0.9280                | 0.9405                |
| Consolidation       | 0.790           | 0.783            | 0.7810                | 0.7834                |
| Edema               | 0.888           | 0.896            | 0.8782                | 0.8976                |
| Effusion            | 0.864           | 0.905            | 0.8975                | 0.9059                |
| Emphysema           | 0.937           | 0.963            | 0.9606                | 0.9705                |
| Fibrosis            | 0.805           | 0.814            | 0.8216                | 0.8448                |
| Hernia              | 0.916           | 0.997            | 0.9951                | 0.9937                |
| Infiltration        | 0.735           | 0.708            | 0.6986                | 0.7078                |
| Mass                | 0.868           | 0.919            | 0.9047                | 0.9126                |
| Nodule              | 0.780           | 0.789            | 0.7736                | 0.7862                |
| Pleural Thickening  | 0.806           | 0.801            | 0.7988                | 0.8073                |
| Pneumonia           | 0.768           | 0.740            | 0.7209                | 0.7193                |
| Pneumothorax        | 0.889           | 0.875            | 0.8831                | 0.8908                |

## 2. Overall Metrics Comparison

| Model                | Loss   | Avg AUROC | Avg F1  |
|----------------------|--------|-----------|---------|
| CheXNet (Paper)      |  —     | 0.8066    |    0.435     |
| DannyNet (Paper)     |   0.0416     | 0.8527    | 0.3861  |
| DannyNet (Reproduced)| 0.0419 | 0.8471    | 0.3705  |
| Deep Ensemble (Our Work)       | N/A    | 0.8559    | 0.3857  |

## 3.Deep Ensemble Model: Comprehensive Per-Class Metrics

| Disease             | AUROC  | F1_Score | Threshold | Brier_Score | ECE    | NLL    | TU_Mean | AU_Mean | EU_Mean |
|---------------------|--------|----------|-----------|-------------|--------|--------|---------|---------|---------|
| Atelectasis         | 0.8215 | 0.4084   | 0.3153    | 0.0841      | 0.0953 | 0.3037 | 0.4512  | 0.4267  | 0.0245  |
| Cardiomegaly        | 0.9405 | 0.5035   | 0.2786    | 0.0383      | 0.0508 | 0.1527 | 0.2716  | 0.2467  | 0.0249  |
| Consolidation       | 0.7834 | 0.2402   | 0.2115    | 0.0476      | 0.0868 | 0.2053 | 0.3664  | 0.3418  | 0.0246  |
| Edema               | 0.8976 | 0.2635   | 0.2556    | 0.0243      | 0.0570 | 0.1165 | 0.2448  | 0.2235  | 0.0213  |
| Effusion            | 0.9059 | 0.6245   | 0.3656    | 0.0885      | 0.0922 | 0.3062 | 0.4444  | 0.4209  | 0.0235  |
| Emphysema           | 0.9705 | 0.5584   | 0.2631    | 0.0208      | 0.0567 | 0.1057 | 0.2404  | 0.2165  | 0.0239  |
| Fibrosis            | 0.8448 | 0.1531   | 0.2368    | 0.0212      | 0.0679 | 0.1172 | 0.2687  | 0.2432  | 0.0255  |
| Hernia              | 0.9937 | 0.7500   | 0.2836    | 0.0021      | 0.0195 | 0.0241 | 0.0937  | 0.0805  | 0.0133  |
| Infiltration        | 0.7078 | 0.4145   | 0.3081    | 0.1402      | 0.1006 | 0.4524 | 0.5660  | 0.5467  | 0.0193  |
| Mass                | 0.9126 | 0.4917   | 0.3270    | 0.0430      | 0.0873 | 0.1877 | 0.3570  | 0.3287  | 0.0283  |
| Nodule              | 0.7862 | 0.3258   | 0.2939    | 0.0584      | 0.0865 | 0.2391 | 0.3974  | 0.3698  | 0.0276  |
| Pleural_Thickening  | 0.8073 | 0.2448   | 0.2296    | 0.0425      | 0.0736 | 0.1861 | 0.3357  | 0.3089  | 0.0268  |
| Pneumonia           | 0.7193 | 0.0683   | 0.1853    | 0.0183      | 0.0710 | 0.1150 | 0.2784  | 0.2524  | 0.0260  |
| Pneumothorax        | 0.8908 | 0.3537   | 0.3422    | 0.0395      | 0.0747 | 0.1704 | 0.3217  | 0.2957  | 0.0260  |


## 4. Deep Ensemble Model: Uncertainty Quantification Summary

| Uncertainty Type        | Mean Value |
|------------------------|------------|
| **Total Uncertainty**  | 0.3312     |
| **Aleatoric Uncertainty** | 0.3073  |
| **Epistemic Uncertainty** | 0.0240  |

| Metric                | Mean Value |
|-----------------------|------------|
| **Brier Score**       | 0.0478     |
| **ECE**               | 0.0728     |
| **NLL**               | 0.1916     |

## 5. Deep Ensemble Model: Mean Metrics Summary

| Metric                | Mean Value | Std Dev |
|-----------------------|------------|---------|
| AUROC                 | 0.8559     | 0.0855  |
| F1_Score              | 0.3857     | 0.1816  |
| Brier_Score           | 0.0478     | 0.0346  |
| ECE                   | 0.0728     | 0.0209  |
| NLL                   | 0.1916     | 0.1037  |
| Total_Uncertainty     | 0.3312     | 0.1106  |
| Aleatoric_Uncertainty | 0.3073     | 0.1094  |
| Epistemic_Uncertainty | 0.0240     | 0.0037  |

## 6. Plots

- **Ensemble Model Per-Class ROC Curves:**
  ![Ensemble Model Combined ROC Curve](images/Ensemble%20Model%20Combined%20ROC%20Curve.png)

- **Per-Class Confusion Matrices:**
  ![Per Class Confusion Matrices](images/Per%20Class%20Confusion%20Matrices.png)

- **Enhanced Uncertainty Analysis:**

  ![Enhanced Uncertainty Analysis](images/Enhanced%20Uncertainity%20Analysis.png)

## 7. Example GradCAM Visualizations

To provide interpretability and insight into the model's predictions, we include GradCAM visualizations for the Deep Ensemble. These highlight the regions of the chest X-ray that most influenced the model's decision for a given disease class.

- **Original Image:**
  ![Original Image](images/original_image.png)

- **Ensemble GradCAM for Mass:**
  ![Ensemble GradCAM for Mass](images/ensemble_gradcam_mass.png)

The GradCAM overlay shows the areas (in red/yellow) that contributed most to the ensemble's prediction for the "Mass" class, providing visual interpretability for clinical review.

---

## Conclusion: New State-of-the-Art (SOTA)

Our Deep Ensemble approach achieves new state-of-the-art (SOTA) results on the NIH Chest X-ray dataset. Since the CheXNet (Paper) results could not be reproduced, we focus our comparison and SOTA claim on DannyNet and our further improvements. The Deep Ensemble achieves the highest average AUROC and a competitive F1 score compared to DannyNet, demonstrating robust and consistent performance. The ensemble leverages model diversity and uncertainty quantification to deliver well-calibrated predictions across all disease classes. These results demonstrate the effectiveness of combining multiple architectures, loss functions, and attention mechanisms for improved chest X-ray diagnosis.
