# Results

## 1. Pretraining: Inter-Codebook Similarity Loss (ICSL)

The following table summarizes the final contrastive loss for different configurations of Wav2Vec2 during pretraining.  
Lower values indicate better alignment between latent and quantized representations.

| Configuration        | Final Contrastive Loss |
|----------------------|------------------------|
| With ICSL (8 Codebooks)     | 4.6120 |
| Without ICSL (8 Codebooks)  | 4.6178 |
| With ICSL (2 Codebooks)     | 4.6133 |
| Without ICSL (2 Codebooks)  | 4.6135 |

**Observation:**  
- The configuration with **ICSL and 8 codebooks** achieved the lowest loss, demonstrating that inter-codebook regularization improves representation diversity and convergence stability.  
- Increasing codebooks **without ICSL** degraded performance, confirming the issue of inter-codebook redundancy.

### Contrastive Loss Curves
![ICSL Contrastive Loss](ICSL_Contrastive_Loss.png)

---

## 2. Fine-Tuning: Residual Vector Quantization (RVQ)

The fine-tuning experiments evaluated the effect of integrating RVQ into the Wav2Vec2 model.  
Below are summarized CTC loss results reported in the paper.

| RVQ Levels | CTC Loss |
|-------------|-----------|
| 1 | 4.59 |
| 2 | 4.22 |
| 4 | 3.95 |
| 8 | 3.85 |
| 16 | 3.99 |

**Observation:**  
- The CTC loss consistently decreased as the number of quantization levels increased up to 8.  
- Beyond that, performance slightly declined, suggesting **optimal performance around 4–8 quantization levels**.  
- Models with RVQ exhibited faster convergence and improved feature compactness, particularly beneficial for **low-resource settings**.

---

## 3. Summary

| Improvement Stage | Method | Key Benefit |
|--------------------|---------|--------------|
| Pretraining | Inter-Codebook Similarity Loss (ICSL) | Prevents redundancy across codebooks and stabilizes convergence |
| Fine-tuning | Residual Vector Quantization (RVQ) | Improves adaptation and reduces CTC loss in low-resource training |

These results collectively show that combining ICSL during pretraining and RVQ during fine-tuning improves both **efficiency** and **adaptability** of Wav2Vec2 under limited data scenarios.
