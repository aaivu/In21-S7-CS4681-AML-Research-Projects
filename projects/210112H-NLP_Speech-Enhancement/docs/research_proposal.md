# Research Proposal: NLP:Speech Enhancement

**Student:** 210112H  
**Research Area:** NLP:Speech Enhancement  
**Date:** 2025-09-01

## Abstract

Single-channel speech enhancement (SE) aims to improve intelligibility and perceived quality of speech corrupted by real-world noise and reverberation. Classical approaches (spectral subtraction, Wiener filtering) struggle with non-stationary noise, motivating data-driven methods in the time–frequency (T-F) domain. This project proposes a lightweight **two-stage DPCRN**: Stage-1 predicts a magnitude mask to quickly raise SNR; Stage-2 refines the **complex** spectrum (real/imaginary) using dual-path recurrence. A **learnable spectral compression mapping (SCM)** preserves low/mid bands while compressing high frequencies for efficiency. Training will use MiniVCTK (clean) and DEMAND (noise) with broad SNR sampling; evaluation reports PESQ, STOI, and SNR improvement per DEMAND environment. We expect consistent SNRi gains, modest PESQ improvements, stable STOI, and favorable runtime suitable for deployment and future causalization.

## 1. Introduction

Robust SE is central to telephony, conferencing, hearing assistance, and on-device voice interfaces. Deep learning has shifted SE from handcrafted filtering to learned T-F mappings that generalize better to non-stationary conditions. In single-channel settings—where spatial cues are absent—models must leverage strong temporal and spectral priors under tight latency/compute budgets. This proposal targets a practical, research-ready system advancing **phase-aware** enhancement without sacrificing efficiency.

## 2. Problem Statement

Lightweight single-channel SE still faces three challenges: (i) **phase estimation** at low SNR is unstable; (ii) **uniform frequency allocation** wastes capacity on sparse high-frequency bins; (iii) **domain robustness** degrades under diverse real noises. We aim to design a compact model that stabilizes complex-spectrum learning, biases capacity toward perceptually salient bands, and yields reliable gains across DEMAND environments.

## 3. Literature Review Summary

Magnitude masking is simple but phase-blind. Complex ratio masking and **complex spectral mapping** improve perceptual quality, especially at low SNR. CRN/DPCRN backbones balance local spectral modeling and temporal context with strong latency–accuracy trade-offs; TF-attention can help but may increase compute. Composite losses (RI + multi-resolution STFT) stabilize training. Gaps remain in low-SNR phase recovery, frequency-aware allocation, and metric–perception alignment—motivating a **two-stage DPCRN with SCM**.

## 4. Research Objectives

### Primary Objective
Design and evaluate a **two-stage, frequency-aware DPCRN** that delivers robust phase-aware enhancement with competitive latency and parameter budgets.

### Secondary Objectives
- Quantify the benefit of **SCM** for efficiency and quality.  
- Compare **two-stage vs. single-stage** training on PESQ/STOI/SNRi across DEMAND categories.  
- Release an open, reproducible pipeline (data, training, evaluation) with ablations and analysis.

## 5. Methodology

We will construct mixtures from MiniVCTK (clean) and selected DEMAND categories (DKITCHEN, OOFFICE, STRAFFIC, TCAR) at 48 kHz. Front-end uses 25 ms Hann windows, 12.5 ms hop, FFT 1200. **SCM** maps 601→256 bins, passing the first 64 unchanged and learning a linear projection for high bins. **Stage-1** encoder–decoder CNN predicts a bounded magnitude mask in compressed space and lifts it back to full resolution. **Stage-2** dual-path RNN (intra BiLSTM + inter LSTM; hidden ≈127) refines RI components (or residuals). Loss: **RI MSE** plus optional **multi-resolution STFT**; curriculum emphasizes magnitude early, then RI. Baselines: no enhancement; single-stage DPCRN; single-stage + SCM; two-stage without SCM. Evaluation: PESQ, STOI, SNRi per environment, plus qualitative spectrograms/audio. Reproducibility: fixed seeds, version-pinned metrics, checkpoints, config-logged runs.

## 6. Expected Outcomes

- **Quantitative:** Consistent SNRi gains; modest PESQ improvements; STOI stable or slightly improved across DEMAND categories.  
- **Qualitative:** Cleaner harmonics, fewer musical-noise artifacts, better consonant recovery.  
- **Practical:** ≈5M-parameter model with clear path to causal/streaming deployment; open-source code, configs, scripts, and ablation results.

## 7. Timeline

| Week | Task                    |
|------|-------------------------|
| 1-2  | Literature Review       |
| 3-4  | Methodology Development |
| 5-8  | Implementation          |
| 9-12 | Experimentation         |
| 13-15| Analysis and Writing    |
| 16   | Final Submission        |

## 8. Resources Required

- **Datasets:** MiniVCTK (clean); DEMAND (noise); optional public RIRs.  
- **Compute:** 1× NVIDIA GPU (≥8–12 GB VRAM); ~15–20 GB storage.  
- **Software:** Python ≥3.10, PyTorch ≥2.1, Torchaudio, NumPy/SciPy, PESQ/STOI packages, tqdm, plotting libs; git.  
- **Artifacts:** Data download/extract scripts, training/eval pipelines, configs, documentation.

## References

- D. S. Williamson, Y. Wang, D. Wang, “Complex Ratio Masking for Monaural Speech Separation,” IEEE/ACM TASLP, 2016.  
- K. Tan, D. L. Wang, “A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement,” Interspeech, 2018.  
- K. Tan, D. L. Wang, “Complex Spectral Mapping with a CRN for Monaural Speech Enhancement,” ICASSP, 2019.  
- Y. Le, H. Chen, K.-J. Chen, J. Lu, “DPCRN: Dual-Path Convolution Recurrent Network for Single-Channel Speech Enhancement,” Interspeech, 2021.  
- Y. Wan et al., “Multi-Loss TF-Attention Model for Speech Enhancement,” ICASSP, 2022.  
- J. Thiemann, N. Ito, E. Vincent, “DEMAND: Diverse Environments Multichannel Acoustic Noise Database,” 2013.  
- C. Valentini-Botinhao et al., “Noisy Speech Corpora Using VCTK for SE,” 2016.

---

**Submission Instructions:**
1. Complete all sections above  
2. Commit your changes to the repository  
3. Create an issue with the label "milestone" and "research-proposal"  
4. Tag your supervisors in the issue for review
