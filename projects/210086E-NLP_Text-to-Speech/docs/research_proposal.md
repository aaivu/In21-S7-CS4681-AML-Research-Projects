# Research Proposal: NLP:Text-to-Speech

**Student:** 210086E
**Research Area:** NLP:Text-to-Speech
**Date:** 2025-09-01

## Abstract

This research project focuses on optimizing the VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture for improved real-time performance while maintaining speech quality. The primary objective is to address computational bottlenecks in neural TTS systems, particularly in the vocoder component, through frequency-domain synthesis techniques and model compression strategies. By implementing iSTFT-based vocoding and targeted architectural optimizations, the project aims to achieve 4-8× speedup over baseline VITS while preserving naturalness and intelligibility, validated on the multi-speaker VCTK dataset.

## 1. Introduction

Text-to-Speech synthesis has evolved dramatically with end-to-end neural architectures like VITS achieving impressive naturalness. However, these systems face computational challenges that hinder real-time deployment, particularly in resource-constrained environments. This research addresses these efficiency bottlenecks through systematic optimization of the most computationally expensive components.

## 2. Problem Statement

Modern neural TTS systems like VITS produce high-quality speech but suffer from computational inefficiency, particularly in the HiFi-GAN vocoder component which accounts for 40-60% of inference time. This limits their deployment in real-time applications and mobile/embedded environments where low latency and minimal resource consumption are critical requirements.

## 3. Literature Review Summary

Recent advances in TTS optimization demonstrate promising approaches through frequency-domain synthesis (FLY-TTS), model compression, and enhanced training strategies. Research gaps remain in validating these techniques on multi-speaker datasets and achieving the quality-efficiency trade-offs necessary for practical deployment. Detailed analysis is provided in `literature_review.md`.

## 4. Research Objectives

### Primary Objective
Develop an optimized VITS variant that achieves 4-8× speedup (RTF < 0.1 on CPU) while maintaining speech quality within 5% of baseline performance.

### Secondary Objectives
- Implement and validate iSTFT-based vocoder architecture
- Apply model compression through parameter sharing (25-35% reduction)
- Comprehensive benchmarking on VCTK multi-speaker dataset
- Document practical optimization strategies for TTS deployment

## 5. Methodology

The research follows a four-phase approach: (1) Baseline reproduction and bottleneck analysis, (2) Implementation of iSTFT vocoder and model compression, (3) Training and incremental validation, (4) Comprehensive evaluation and benchmarking. Detailed methodology is provided in `methodology.md`.

## 6. Expected Outcomes

- Optimized VITS implementation with 4-8× speedup
- 25-35% parameter reduction while maintaining quality
- Validation of frequency-domain techniques on VCTK
- Technical documentation and implementation guide

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Baseline Setup & Profiling |
| 3-4  | iSTFT Vocoder Implementation |
| 5-6  | Training & Validation |
| 7    | Evaluation & Documentation |

## 8. Resources Required

- **Dataset**: VCTK Corpus (44 hours, 109 speakers)
- **Hardware**: NVIDIA RTX GPU (12GB+ VRAM), multi-core CPU
- **Software**: PyTorch, Coqui TTS, standard audio libraries
- **Compute**: ~100 GPU hours for training

## References

See `literature_review.md` for comprehensive references.