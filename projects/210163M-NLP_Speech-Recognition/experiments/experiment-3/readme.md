# Experiment 3 – Fine-Tuned WavLM-CTC with KenLM Decoder

This experiment focuses on enhancing the decoding stage of **WavLM-Large** fine-tuned with a CTC head by integrating a **KenLM language model**.  
The primary goal was to reduce **Word Error Rate (WER)** and **Character Error Rate (CER)** beyond what was achieved in previous experiments.

---

## Experiment Setup

- **Model Backbone:** WavLM-Large (fine-tuned with CTC head)  
- **Decoder:** Beam Search Decoder with **KenLM**  
- **Dataset:** LibriSpeech  
  - Training set: `train-clean-100` (100h)  
  - Validation set: `dev-clean`  
  - Test set: `test-clean`  
- **Metrics:**  
  - Word Error Rate (WER)  
  - Character Error Rate (CER)  

---

## Motivation

CTC-trained models can produce strong character-level predictions but often struggle with word boundaries, grammar, and homophones.  
By adding **KenLM**, we incorporate linguistic context into the decoding process, improving sequence-level accuracy and reducing errors.

---

## Results

### Validation Set
- **WER:** `0.0394` (≈ 3.94%)  
- **CER:** `0.0134` (≈ 1.34%)  

### Test Set
- **WER:** `0.0416` (≈ 4.16%)  
- **CER:** `0.0135` (≈ 1.35%)  

---

## Comparison with Previous Experiments

| Experiment | Decoder | Dataset | Best WER | CER |
|------------|---------|---------|----------|-----|
| **1** | Greedy CTC | 60% of LibriSpeech-100h | ~18.08%(Validation) | N/A |
| **2** | Greedy CTC | Full LibriSpeech-100h | **6.82%(Validation)** | N/A |
| **3** | **KenLM** | Full LibriSpeech-100h | **4.16% (Test)** <br> **3.94% (Validation)** | ~1.3% |

---

## Key Takeaways

1. **Language Model Decoding Improves Accuracy**  
   - KenLM reduced WER from ~6.8% (Experiment 2) to ~4.1%.  

2. **Low CER Across Sets**  
   - CER remains consistently low (~1.3%), showing strong character-level predictions.  

3. **Near WeNet Performance**  
   - WeNet baseline WER: ~2.7%  
   - WavLM+KenLM WER: ~4.1%  
   - Our approach is competitive, with potential for improvement using larger training data.  

---

## Next Steps

- Fine-tune with **LibriSpeech-460h** for further accuracy gains.  
- Tune **KenLM hyperparameters** (beam width, LM weight, word insertion penalty).  
- Explore **shallow fusion** with neural language models.  
- Benchmark against **full LibriSpeech-960h** training.  

