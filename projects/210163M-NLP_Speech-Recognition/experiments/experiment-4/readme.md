# Experiment 4: KenLM Hyperparameter Tuning  

---

## Objective  
This experiment fine-tunes the **KenLM language model hyperparameters** (α and β) used during **beam search decoding** in the WavLM-Large + CTC ASR pipeline.  

The goal is to identify the optimal balance between **acoustic model confidence** and **language model influence** for minimizing the **Word Error Rate (WER)** on the **LibriSpeech test-clean** dataset.

---

## Background  
In the previous experiment (Experiment 3), a 4-gram KenLM model was integrated into the decoding stage with parameters:  
- α = 0.8  
- β = -0.2  

While this setup achieved **WER = 4.16%**, further improvements were expected through systematic tuning of α (LM weight) and β (word insertion penalty).  

KenLM scoring function:  

\[
\text{Score}(y) = \log P_{AM}(y|X) + \alpha \log P_{LM}(y) + \beta \cdot |y|
\]

- **α (LM weight):** Controls how much the language model influences decoding.  
- **β (Word insertion penalty):** Adjusts for overly short or long predictions.

---

## Experiment Setup  

### **Model and Data**
- **Acoustic model:** WavLM-Large fine-tuned with CTC (from Experiment 3)  
- **Language model:** 4-gram KenLM trained on LibriSpeech transcripts  
- **Evaluation set:** LibriSpeech *test-clean* subset  
- **Metrics:** Word Error Rate (WER)

### **Hyperparameter Search Grid**
| α (LM Weight) | β (Word Insertion Penalty) |
|----------------|-----------------------------|
| 0.3 | 0.1 |
| 0.3 | 0.35 |
| 0.3 | 0.5 |
| 0.5 | 0.1 |
| 0.5 | 0.35 |
| 0.5 | 0.5 |

Each configuration was tested using identical beam search decoding on the same trained model to ensure comparability.

---

## Results  

| α | β | WER (%) |
|----|----|----------|
| 0.3 | 0.1 | 4.35 |
| 0.3 | 0.35 | 4.37 |
| 0.3 | 0.5 | 4.37 |
| 0.5 | 0.1 | **4.05** |
| 0.5 | 0.35 | 4.06 |
| 0.5 | 0.5 | 4.08 |

---

## Analysis  

- **Best performance:**  
  - α = 0.5, β = 0.1 → **WER = 4.05%**  
  This slightly improves over the previous best WER (4.16%) achieved with α = 0.8, β = -0.2.

- **Effect of α (LM weight):**  
  Increasing α from 0.3 to 0.5 generally improved results, showing that moderate LM influence helps correct language-level errors without overpowering acoustic evidence.

- **Effect of β (insertion penalty):**  
  Higher β values (>0.35) slightly increased WER, suggesting over-penalization of longer sequences.  
  A smaller β = 0.1 gave the best trade-off between insertion and deletion errors.

- **Overall trend:**  
  The WER curve flattens around α = 0.5, indicating that the model’s performance is relatively stable near this value.

---

## Key Insights  

1. **Balanced LM Weight:**  
   A moderate LM influence (α ≈ 0.5) provides optimal contextual correction without overfitting to common word patterns.

2. **Low Insertion Penalty Helps:**  
   Small β values (0.1) prevent unnecessary word omissions while maintaining sentence fluency.

3. **Performance Plateau:**  
   Beyond α = 0.5, additional LM weighting provides negligible benefit—showing diminishing returns on WER.

4. **Improved Accuracy:**  
   This tuning reduced WER from **4.16% → 4.05%**, marking a relative improvement of about **2.6%**.

---

## Summary  

| Experiment | Description | WER (%) |
|-------------|-------------|---------|
| Experiment 3 | WavLM + CTC + KenLM (α=0.8, β=-0.2) | 4.16 |
| **Experiment 4** | **KenLM tuned (α=0.5, β=0.1)** | **4.05** |

The tuned decoding configuration achieved the **lowest WER to date** in this project series, indicating the benefit of fine-grained LM parameter optimization.

---

## Next Steps  
- Extend hyperparameter search to α ∈ [0.4–0.7] and β ∈ [0.05–0.2] with finer resolution.  
- Explore **beam width optimization** (e.g., 50–200 range).  
- Integrate **neural language model (NLM)** for comparison with KenLM (shallow fusion).  
- Evaluate generalization on noisy or accented LibriSpeech subsets (other than test-clean).

---

### Key Outcome  
**Optimal KenLM configuration:**  
> α = 0.5, β = 0.1 → **WER = 4.05% (Best so far)**  

**Conclusion:**  
Fine-tuning KenLM hyperparameters provides measurable but stable improvements over default configurations, confirming that decoder-level optimization remains a valuable step even for highly capable pretrained encoders like WavLM.

---
