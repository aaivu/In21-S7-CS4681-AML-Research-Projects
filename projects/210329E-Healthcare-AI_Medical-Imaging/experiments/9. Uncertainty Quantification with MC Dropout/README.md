# 9. Uncertainty Quantification with Monte Carlo Dropout (DenseNet-121)

## Implementation Summary

This experiment integrates uncertainty quantification into the DenseNet-121 model using Monte Carlo Dropout (MC Dropout). The same random seed (42), Focal Loss, and other training settings were used as in the baseline experiment to ensure a fair comparison:

- **Model:** DenseNet-121 (without CLAHE enhancement)(Model that was used under experiment 3).
- **MC Dropout Integration:**  
	- A Dropout layer is inserted before the final classifier.
	- At inference, dropout is kept active and the model performs T = 30 stochastic forward passes (`mc_forward_passes`).
	- For each input, the mean predicted probability across passes is used as the final prediction.
	- The variance (or standard deviation) across passes is used as an uncertainty score (epistemic uncertainty).
- **What this achieves:**  
	- MC Dropout approximates sampling from the posterior over model weights, providing a predictive distribution.
	- The mean prediction is the marginalised predictive probability.
	- The variance across runs reflects model (epistemic) uncertainty.
	- Note: This does not capture aleatoric (data) uncertainty.

## Results

### Overall Metrics

| Model                        | Loss   | Avg AUROC | Avg F1  |
|------------------------------|--------|-----------|---------|
| DenseNet-121 (baseline)      | 0.0419 | 0.8471    | 0.3705  |
| DenseNet-121 + MC Dropout    | 0.0426 | 0.8362    | 0.3713  |

### Per-Class AUROC Comparison

| Class                | Baseline | MC Dropout |
|----------------------|:--------:|:----------:|
| Atelectasis          | 0.8181   | 0.7929     |
| Cardiomegaly         | 0.9280   | 0.9115     |
| Consolidation        | 0.7810   | 0.7745     |
| Edema                | 0.8782   | 0.8841     |
| Effusion             | 0.8975   | 0.8938     |
| Emphysema            | 0.9606   | 0.9533     |
| Fibrosis             | 0.8216   | 0.7993     |
| Hernia               | 0.9951   | 0.9984     |
| Infiltration         | 0.6986   | 0.6996     |
| Mass                 | 0.9047   | 0.8859     |
| Nodule               | 0.7736   | 0.7430     |
| Pleural_Thickening   | 0.7988   | 0.7965     |
| Pneumonia            | 0.7209   | 0.6961     |
| Pneumothorax         | 0.8831   | 0.8779     |

> **Observation:**  
> After integrating MC Dropout, the model’s overall AUROC dropped slightly, and most per-class AUROC scores are marginally lower. This is expected, as MC Dropout introduces stochasticity and regularization, but provides valuable uncertainty estimates for each prediction.


## Next Steps

As a next step, I plan to explore ensemble methods for uncertainty quantification. Combining predictions and uncertainties from multiple diverse models is expected to further improve both predictive performance and the reliability of uncertainty estimates.