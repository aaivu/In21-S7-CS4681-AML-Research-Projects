# Research Proposal: NLP:ionSpeech Enhancement

**Student:** 210049U  
**Research Area:** NLP:ionSpeech Enhancement  
**Date:** 2025-09-01  

---

## Abstract

This research proposes a lightweight two-stage speech enhancement system that improves the perceptual quality of separated speech produced by transformer-based models such as SepFormer. Despite achieving high objective separation accuracy, models like SepFormer often leave residual artifacts and background noise that degrade speech quality and intelligibility. The proposed approach integrates a post-processing **lightweight CNN denoiser** to refine separated signals without retraining the base model. By emphasizing computational efficiency and modular design, the project aims to achieve measurable improvements in **SI-SDR**, **PESQ**, and **STOI** metrics while maintaining real-time feasibility. The outcome will contribute to more deployable speech enhancement systems for telecommunications, hearing aids, and ASR pipelines.

---

## 1. Introduction

Speech separation and enhancement are critical for many real-world applications including voice assistants, teleconferencing, and hearing devices. Recent transformer-based architectures such as **SepFormer** have achieved remarkable separation performance; however, residual artifacts remain a significant barrier to perceptual quality. Enhancing these separated signals through efficient post-processing offers a practical solution for real-time and resource-constrained systems.

---

## 2. Problem Statement

Although transformer-based models like SepFormer achieve high separation accuracy, **their outputs often contain residual noise, musical artifacts, and inter-speaker leakage**. These issues reduce intelligibility and perceptual quality. Retraining large-scale models is computationally expensive and impractical for many applications. Hence, there is a need for a **lightweight, model-agnostic post-processing framework** that enhances separated speech while maintaining real-time performance.

---

## 3. Literature Review Summary

Research in speech separation has evolved from traditional ICA and CASA methods to deep learning approaches such as **DPCL**, **Conv-TasNet**, and **DPRNN**. The **SepFormer** model introduced a dual-path transformer structure achieving state-of-the-art results on WSJ0-2mix.  
However, limited attention has been given to **post-separation enhancement**. Existing work primarily focuses on architecture optimization rather than refinement of separated outputs. This research addresses that gap by combining separation and denoising into a modular two-stage pipeline.

---

## 4. Research Objectives

### Primary Objective
To design and evaluate a lightweight convolutional post-processing network that improves the perceptual quality of transformer-based speech separation outputs.

### Secondary Objectives
- To analyze residual artifacts in SepFormer-separated speech.  
- To implement a CNN-based denoiser optimized for low computational cost.  
- To evaluate improvements using objective (SI-SDR, PESQ, STOI) and perceptual metrics.  
- To validate the approach for real-time and resource-limited deployment.  

---

## 5. Methodology

The proposed system consists of two main stages:  
1. **Speech Separation:** Using a pre-trained SepFormer model to extract individual sources from mixed speech.  
2. **Speech Enhancement:** Applying a lightweight CNN denoiser (two 1D convolutional layers, ~305 parameters) to suppress residual artifacts and enhance clarity.  

Experiments will be conducted using the **WSJ0-2mix dataset**, evaluating performance improvements over baseline SepFormer outputs. The implementation will utilize **PyTorch**, **SpeechBrain**, and **TorchAudio**, running on an NVIDIA GPU for efficiency testing.

---

## 6. Expected Outcomes

- Improved **speech quality (PESQ)** and **intelligibility (STOI)** in separated outputs.  
- Demonstration of a **computationally efficient, real-time denoising stage**.  
- Evidence supporting modular post-processing as a practical enhancement strategy.  
- Contribution to the design of deployable speech enhancement systems for real-world applications.  

---

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-8  | Implementation |
| 9-12 | Experimentation |
| 13-15| Analysis and Writing |
| 16   | Final Submission |

---

## 8. Resources Required

- **Dataset:** WSJ0-2mix speech corpus  
- **Tools & Frameworks:** Python, PyTorch, SpeechBrain, TorchAudio  
- **Hardware:** NVIDIA GPU (≥6 GB VRAM)  
- **Metrics:** SI-SDR, PESQ, STOI  
- **Version Control:** GitHub for code and documentation  

---

## References

1. M. Subakan et al., “Attention is all you need in speech separation,” *IEEE ICASSP*, 2021.  
2. Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal time–frequency masking,” *IEEE/ACM TASLP*, 2019.  
3. J. R. Hershey et al., “Deep clustering for speech separation,” *IEEE ICASSP*, 2016.  
4. S. Pascual et al., “SEGAN: Speech enhancement generative adversarial network,” *Interspeech*, 2017.  
5. Y. Zhang et al., “Multi-stage speech separation with iterative refinement,” *IEEE ICASSP*, 2020.  

---

**Submission Instructions:**
1. Complete all sections above  
2. Commit your changes to the repository  
3. Create an issue with the label **“milestone”** and **“research-proposal”**  
4. Tag your supervisors in the issue for review  
