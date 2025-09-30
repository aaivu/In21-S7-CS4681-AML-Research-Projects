# Experiment-1: Fine-tuning WavLM-Large with CTC Head on LibriSpeech (Clean-100 Subset)

This repository contains my first experiment towards **outperforming WeNet** by leveraging **WavLM-large** for automatic speech recognition (ASR).  
The experiment focuses on **fine-tuning WavLM-large** with a custom **CTC head** using **60% of the LibriSpeech clean-100 dataset**.

---

## Objective
- Evaluate the performance of WavLM-large when fine-tuned on a smaller subset of LibriSpeech.
- Establish a strong baseline Word Error Rate (WER) before scaling to larger datasets (LibriSpeech 460h).
- Compare against **WeNet**, which reports a **2.7% WER** benchmark.

---

## Experiment Setup
- **Model**: WavLM-large + custom CTC head  
- **Dataset**: 60% of LibriSpeech *clean-100* subset (~60 hours)  
- **Training Epochs**: 3  
- **Training Time**: ~10 hours  
- **Loss Function**: CTC Loss  
- **Evaluation Metric**: WER (Word Error Rate)  

---

## Training Performance

| Step | Training Loss | Validation Loss | WER |
|------|---------------|-----------------|-----|
| 2500 | 0.6325 | 0.3654 | 0.3504 |
| 3000 | 0.5553 | 0.3028 | 0.2969 |
| 3500 | 0.4837 | 0.2681 | 0.2607 |
| 4000 | 0.4547 | 0.2426 | 0.2279 |
| 4500 | 0.4179 | 0.2239 | 0.2076 |
| 5000 | 0.4063 | 0.2103 | 0.1982 |
| 5400 | 0.3799 | 0.2047 | 0.1917 |
| 6000 | 0.3782 | 0.1972 | 0.1817 |
| 6400 | 0.3807 | 0.1960 | 0.1809 |

**Final WER**: **~18.1%** on validation after 3 epochs  

---

## Next Steps
- Scale training to **LibriSpeech-460h** dataset for better generalization.  
- Apply decoding improvements (e.g., **KenLM**, **beam search**) to reduce WER further.  
- Target performance close to or better than **WeNet’s 2.7% WER** benchmark.

---

## 🔗 References
- [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)  
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)  
- [WeNet](https://github.com/wenet-e2e/wenet)  

---
