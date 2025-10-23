# Methodology: NLP:Speech Enhancement

**Student:** 210112H  
**Research Area:** NLP:Speech Enhancement  
**Date:** 2025-09-01

## 1. Overview

This methodology details a single-channel speech enhancement (SE) pipeline centered on a lightweight Dual-Path Convolutional Recurrent Network (DPCRN) with two key ideas: (i) a **two-stage refinement** (Stage-1 magnitude masking to quickly raise SNR; Stage-2 complex RI refinement for fine detail), and (ii) a **learnable Spectral Compression Mapping (SCM)** that prioritizes low/mid frequencies while compressing high bands. Training uses mixtures of MiniVCTK speech and DEMAND noises across varied SNRs; evaluation emphasizes both objective metrics and qualitative analysis.

## 2. Research Design

We adopt a **quantitative, experimental** design with controlled data generation and ablations. Independent variables: architecture (with/without SCM; single- vs two-stage), loss composition (magnitude vs RI + MR-STFT), and data regimes (SNR ranges, optional reverberation). Dependent variables: PESQ, STOI, SNRi. Workflow: (1) dataset prep/verification, (2) model implementation, (3) training with fixed seeds and splits, (4) validation/tuning, (5) test evaluation by noise category, (6) ablations and error analysis.

## 3. Data Collection

### 3.1 Data Sources
- **Clean speech:** MiniVCTK (subset of VCTK; ~4 speakers × ~20 clips/speaker).
- **Noise:** DEMAND (environments: DKITCHEN, OOFFICE, STRAFFIC, TCAR; 48 kHz versions).
- *(Optional)* **Reverberation:** Public RIRs for realism if time allows.

### 3.2 Data Description
- **Sampling rate:** 48 kHz end-to-end (aligns with DEMAND 48k and selected windowing).
- **Clip lengths:** Clean ≈4–5 s; noise segments ≈5 min; for mixing, crop random 4 s windows from noise.
- **SNR schedule:** Uniform from {15, 10, 5, 0, −5} dB during training; held-out SNRs for validation.
- **Splits:** Train/Val/Test ≈ 80/10/10; keep speakers disjoint across splits when possible.

### 3.3 Data Preprocessing
- **Resampling/format:** Ensure WAV 48 kHz mono; downmix DEMAND (e.g., ch01.wav).
- **Normalization:** Per-utterance RMS normalization for stability.
- **Mixture generation:** On-the-fly pairing of clean/noise with random SNR; light gain and ±2% pitch perturbations.
- **Quality checks:** Scripts to count files, verify durations, visualize waveforms/spectrograms, and audition samples.

## 4. Model Architecture

- **Front-end STFT:** 25 ms Hann, 12.5 ms hop, FFT 1200 → 601 bins.
- **SCM:** Map 601→256 dims; pass first 64 bins unchanged; learn a linear mapping for high bins (frequency-aware capacity).
- **Stage-1 (Masking):** Encoder–decoder CNN (e.g., channels 16→32→48) predicts a bounded magnitude mask in the compressed space; up-projects to full resolution.
- **Stage-2 (Complex Refinement):** Dual-Path RNN (intra BiLSTM + inter LSTM; hidden ≈127) refines real/imag components or predicts RI residuals.
- **Losses:** RI MSE + optional multi-resolution STFT (MR-STFT). Early epochs may emphasize magnitude; later include full RI weighting (curriculum).
- **Complexity/latency:** Target ≈5M params for near real-time; causalization by replacing BiLSTM with LSTM if streaming is required.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **PESQ** (perceptual quality)
- **STOI** (intelligibility)
- **SNRi** (dB improvement)
- *(Optional)* **DNSMOS P.835** (non-intrusive quality proxy)

### 5.2 Baseline Models
- **Noisy passthrough** (no enhancement)
- **DPCRN baseline** (single-stage, no SCM, no complex refinement)
- **Ablations:**
  - Baseline + SCM only
  - Two-stage without SCM
  - Two-stage + SCM (**proposed**)
- *(Optional)* **RNNoise** (non-DPCRN, tiny RNN + DSP features) for real-time context

### 5.3 Hardware/Software Requirements
- **Hardware:** 1× NVIDIA GPU (≥8–12 GB VRAM) recommended; CPU feasible but slow
- **Software:** Python ≥3.10, PyTorch ≥2.1, Torchaudio, NumPy/SciPy, PESQ/STOI pkgs, tqdm, YAML/JSON config
- **Reproducibility:** Fixed seeds, deterministic flags where feasible; log hyperparams/metrics to CSV/JSON

## 6. Implementation Plan

| Phase   | Tasks                                                                                  | Duration | Deliverables                           |
|---------|-----------------------------------------------------------------------------------------|----------|----------------------------------------|
| Phase 1 | Data scripts, ingest & verification; waveform/spectrogram sanity checks                 | 1.5–2 w  | Cleaned datasets & verified splits     |
| Phase 2 | Implement Stage-1 masking, SCM, training loop, checkpointing                            | 2–3 w    | Stage-1 model & learning curves        |
| Phase 3 | Add Stage-2 dual-path refinement; composite losses; end-to-end training                 | 2–3 w    | Full two-stage model + ablation runs   |
| Phase 4 | Evaluation per noise category; stats, plots; qualitative audio/spectrogram analysis     | 1–1.5 w  | Results tables, plots, audio samples   |
| Phase 5 | Write-up (methods/experiments/discussion); code cleanup & tagged release                | 1 w      | Draft paper + reproducible repository  |

## 7. Risk Analysis

- **Data path/format errors**  
  *Mitigation:* Robust path resolution; automated checks; enforced 48 kHz resampling.

- **Metric/library instability**  
  *Mitigation:* Pin versions; small wrappers with unit tests; cache metric outputs.

- **Overfitting / weak generalization**  
  *Mitigation:* Diverse SNRs; light augmentation; early stopping; strict held-out test set.

- **Training instability (phase learning)**  
  *Mitigation:* Curriculum (mask-first); gradient clipping; lower LR; MR-STFT stabilization.

- **Compute/time limits**  
  *Mitigation:* Mini-datasets for rapid iteration; prioritized ablations; frequent checkpoints.

## 8. Expected Outcomes

- **Quantitative:** Consistent SNRi gains; modest but reliable PESQ improvements; stable or slightly higher STOI across DEMAND categories.
- **Qualitative:** Cleaner harmonics, fewer musical-noise artifacts, improved consonant clarity (spectrograms + audio).
- **Ablation evidence:** (i) SCM improves efficiency/accuracy by prioritizing low/mid bands; (ii) two-stage refinement stabilizes training and enhances high-frequency detail.
- **Artifacts:** Reproducible codebase (data, train, eval), configs, trained checkpoints, and clear experiment logs supporting paper claims.

---

**Note:** Update this document as the methodology evolves (e.g., add DNSMOS, causal/streaming tests, or additional datasets/reverberation).
