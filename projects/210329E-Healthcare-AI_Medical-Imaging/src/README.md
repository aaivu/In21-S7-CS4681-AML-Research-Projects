# 1. Individual Model Testing and Comparison

This notebook evaluates all trained 14 models so far, including those with different architectures (DenseNet-121, EfficientNet-B2, EfficientNet-B3, DenseNet-121+CBAM), loss functions (Focal Loss, ZLPR Loss), and random seeds. Each model is tested on the same test set, and their test AUROC, F1, and per-class performances are compared.

Based on overall test accuracy, architectural diversity, and per-class performance variability, 9 models were selected for future ensembling:

| Model Name                                 | Test AUROC | Test F1 |
|---------------------------------------------|------------|---------|
| seed 22 - densnet 121 - focal loss.pth      | 0.8475     | 0.3852  |
| seed 22 - densnet 121 - ZLPR loss.pth       | 0.8468     | 0.3758  |
| seed 32 - densenet 121 - ZLPR loss.pth      | 0.8479     | 0.3762  |
| seed 32- densnet121 - focal loss.pth        | 0.8458     | 0.3679  |
| seed 42 - densnet 121 - focal loss.pth      | 0.8514     | 0.3803  |
| seed 42 - densnet 121 - ZLPR loss.pth       | 0.8462     | 0.3621  |
| seed 42 - densnet121- Attention - focal loss.pth | 0.8480 | 0.3787  |
| seed 42 - efficienet b3 - focal loss.pth    | 0.8117     | 0.3338  |
| seed 42 - efficinet b2 - focal loss.pth     | 0.8322     | 0.3528  |

These models represent a diverse set of architectures, loss functions, and random seeds, providing a strong foundation for subsequent ensemble experiments.

# 2. Ensemble: Simple Average

In the next stage, the selected models were combined using a simple average ensemble. Since a separate held-out validation dataset was not maintained (apart from the test set), it would not be fair to perform a weighted average ensemble (as weights would be tuned on the test set, leading to overfitting and optimistic results). Therefore, a simple (uniform) average of the model predictions was used for ensembling.

All the base models were trained using the same train, validation, and test splits. Since we do not have any additional dataset (beyond these) to tune ensemble weights fairly, a weighted ensembling approach is not appropriate here.

**What happens in the simple average ensemble:**

**Results:**

The table below compares the best individual model to the simple average ensemble:

| Metric    | Best Individual | Simple Average Ensemble | Improvement | Improvement (%) |
|-----------|-----------------|------------------------|-------------|-----------------|
| AUROC     | 0.8514          | 0.8559                 | 0.0045      | 0.53            |
| F1_Score  | 0.3852          | 0.3857                 | 0.0005      | 0.13            |


# 3. Uncertainty Quantification: Metrics and Results

The simple average ensemble provides a small but consistent improvement over the best individual model, especially in AUROC, demonstrating the benefit of combining diverse models.

**Uncertainty metrics used:**
- **Total Uncertainty (TU):** Measures the overall uncertainty in the prediction, combining both aleatoric and epistemic components.
- **Aleatoric Uncertainty (AU):** Captures the inherent data noise or ambiguity (uncertainty that cannot be reduced by collecting more data).
- **Epistemic Uncertainty (EU):** Captures the model's uncertainty due to limited knowledge or data (can be reduced with more data or better models).

**Equations:**

Let $p_k$ be the predicted probability for class $k$ from model $m$ in the ensemble, and $M$ be the number of models.

- **Total Uncertainty (TU):**
	$$ TU = -\sum_{k} \bar{p}_k \log \bar{p}_k $$
	where $\bar{p}_k = \frac{1}{M} \sum_{m=1}^M p_{k}^{(m)}$ is the mean predicted probability across models.

- **Aleatoric Uncertainty (AU):**
	$$ AU = \frac{1}{M} \sum_{m=1}^M \left( -\sum_{k} p_{k}^{(m)} \log p_{k}^{(m)} \right) $$

- **Epistemic Uncertainty (EU):**
	$$ EU = TU - AU $$

**Other metrics reported:**
- AUROC, F1 Score, Brier Score, Expected Calibration Error (ECE), Negative Log-Likelihood (NLL)

The table below summarizes the per-class results for the ensemble:

| Disease             | AUROC  | F1_Score | Brier_Score | ECE    | NLL    | TU_Mean | AU_Mean | EU_Mean |
|---------------------|--------|----------|-------------|--------|--------|---------|---------|---------|
| Atelectasis         | 0.8215 | 0.4084   | 0.0841      | 0.0953 | 0.3037 | 0.4512  | 0.4267  | 0.0245  |
| Cardiomegaly        | 0.9405 | 0.5035   | 0.0383      | 0.0508 | 0.1527 | 0.2716  | 0.2467  | 0.0249  |
| Consolidation       | 0.7834 | 0.2402   | 0.0476      | 0.0868 | 0.2053 | 0.3664  | 0.3418  | 0.0246  |
| Edema               | 0.8976 | 0.2635   | 0.0243      | 0.0570 | 0.1165 | 0.2448  | 0.2235  | 0.0213  |
| Effusion            | 0.9059 | 0.6245   | 0.0885      | 0.0922 | 0.3062 | 0.4444  | 0.4209  | 0.0235  |
| Emphysema           | 0.9705 | 0.5584   | 0.0208      | 0.0567 | 0.1057 | 0.2404  | 0.2165  | 0.0239  |
| Fibrosis            | 0.8448 | 0.1531   | 0.0212      | 0.0679 | 0.1172 | 0.2687  | 0.2432  | 0.0255  |
| Hernia              | 0.9937 | 0.7500   | 0.0021      | 0.0195 | 0.0241 | 0.0937  | 0.0805  | 0.0133  |
| Infiltration        | 0.7078 | 0.4145   | 0.1402      | 0.1006 | 0.4524 | 0.5660  | 0.5467  | 0.0193  |
| Mass                | 0.9126 | 0.4917   | 0.0430      | 0.0873 | 0.1877 | 0.3570  | 0.3287  | 0.0283  |
| Nodule              | 0.7862 | 0.3258   | 0.0584      | 0.0865 | 0.2391 | 0.3974  | 0.3698  | 0.0276  |
| Pleural_Thickening  | 0.8073 | 0.2448   | 0.0425      | 0.0736 | 0.1861 | 0.3357  | 0.3089  | 0.0268  |
| Pneumonia           | 0.7193 | 0.0683   | 0.0183      | 0.0710 | 0.1150 | 0.2784  | 0.2524  | 0.0260  |
| Pneumothorax        | 0.8908 | 0.3537   | 0.0395      | 0.0747 | 0.1704 | 0.3217  | 0.2957  | 0.0260  |


## Uncertainty Quantification Summary

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

# 4. Ensemble GradCAM Visualization

To interpret and explain the ensemble's predictions, GradCAM visualizations were generated for selected test images. This approach highlights the regions of chest X-ray images that contributed most to the ensemble's decision for each disease class.

**Key points:**
- GradCAM heatmaps are computed for each model in the ensemble and then averaged to produce an ensemble-level visualization for each disease class.
- For each test image, the visualization focuses on classes where the ensemble prediction probability exceeds 0.5.
- The notebook displays:
	- The original X-ray image
	- GradCAM overlays for each individual model and the ensemble
	- Bar charts of ensemble prediction scores for high-confidence classes
- This helps to:
	- Interpret which regions the models focus on for each disease
	- Compare attention patterns across models
	- Increase trust and transparency in the ensemble's predictions

Example visualizations are provided for several randomly selected test images, showing both the true labels and the predicted high-confidence classes.


========================================================================= 

**Note on Model Weights:**
Due to the large size of the model files, both the models used for comparison and the final models are provided via external download links. Please download the folders from the links below and place them in this directory:

- **All models** (used for comparison):
	- Download link: https://dms.uom.lk/s/cdxgzH6yARsCEjE
	- Place the folder as: `projects/210329E-Healthcare-AI_Medical-Imaging/src/All models/`

- **Final Models** (selected for ensemble):
	- Download link: https://dms.uom.lk/s/n8bGtHpjtMdRbAw
	- Place the folder as: `projects/210329E-Healthcare-AI_Medical-Imaging/src/Final Models/`

Ensure the folder names and locations match exactly for the code to run correctly.