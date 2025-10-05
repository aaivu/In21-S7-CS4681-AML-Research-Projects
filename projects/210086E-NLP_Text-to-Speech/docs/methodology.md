# Methodology: NLP:Text-to-Speech

**Student:** 210086E
**Research Area:** NLP:Text-to-Speech
**Date:** 2025-09-01

## 1. Overview

This methodology outlines a structured approach to enhancing the VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture for improved real-time performance while maintaining speech quality. The approach focuses on targeted architectural modifications and training optimizations based on identified computational bottlenecks in the baseline system. The project follows an incremental development strategy, implementing and validating each enhancement independently before integration, ensuring reliable attribution of performance improvements to specific modifications.

## 2. Research Design

The research adopts an empirical optimization approach with four distinct phases:

1. **Baseline Reproduction and Analysis**: Establish a reliable VITS baseline on VCTK dataset and profile computational bottlenecks
2. **Targeted Enhancement Implementation**: Apply architectural modifications and training optimizations based on literature findings
3. **Incremental Validation**: Evaluate each modification independently through controlled experiments
4. **Comprehensive Benchmarking**: Conduct final evaluation comparing optimized system against baseline

The methodology prioritizes practical improvements with measurable impact on inference speed while maintaining speech naturalness. Each enhancement is guided by existing literature findings, particularly focusing on vocoder optimization and model compression techniques that have demonstrated success in recent TTS research.

## 3. Data Collection

### 3.1 Data Sources
- **Primary Dataset**: VCTK Corpus (Voice Cloning Toolkit)
- **Baseline Reference**: Pre-trained VITS models from open-source implementations (Coqui TTS)

### 3.2 Data Description
The CSTR VCTK Corpus (Version 0.80) contains speech data from 109 English speakers with various accents. Each speaker reads approximately 400 sentences comprising:
- **Newspaper texts**: Selected from Herald Glasgow using greedy algorithm for contextual and phonetic coverage
- **Rainbow Passage**: Standard passage from International Dialects of English Archive  
- **Elicitation paragraph**: From speech accent archive for consistent phonetic coverage

**Recording specifications:**
- **Original format**: 96kHz, 24-bit using omni-directional microphone (DPA 4035)
- **Processed format**: 48kHz, 16-bit, manually end-pointed
- **Recording environment**: Hemi-anechoic chamber (University of Edinburgh)
- **Total duration**: Approximately 44 hours of high-quality speech data
- **License**: Open Data Commons Attribution License (ODC-By) v1.0

### 3.3 Data Preprocessing
- **Audio Processing**: Downsample from 48kHz to 22kHz, normalize amplitude, apply pre-emphasis
- **Text Processing**: Convert to phonemes using standard G2P (Grapheme-to-Phoneme) conversion
- **Feature Extraction**: Generate mel-spectrograms (80 dimensions, hop length 256, window length 1024)
- **Speaker Selection**: Focus on speaker subset for development iteration (due to large corpus size)
- **Data Splits**: 
  - Training: ~80% of speakers (~87 speakers)
  - Validation: ~10% of speakers (~11 speakers) 
  - Test: ~10% of speakers (~11 speakers)
- **Quality Control**: Utilize manually end-pointed recordings to ensure clean training data

## 4. Model Architecture

The enhanced VITS architecture incorporates targeted optimizations in two key areas:

### 4.1 Vocoder Optimization
- **iSTFT-based Decoder**: Replace HiFi-GAN's transposed convolutions with frequency-domain synthesis using inverse Short-Time Fourier Transform
- **Multi-band Processing**: Implement parallel sub-band generation for accelerated synthesis
- **Lightweight Convolutions**: Use ConvNeXt or grouped convolution blocks for feature prediction

### 4.2 Model Compression
- **Grouped Parameter Sharing**: Apply parameter tying across Transformer encoder layers to reduce model size by ~30%
- **Flow Module Optimization**: Reduce coupling layers in normalizing flows while maintaining alignment quality
- **Selective Fine-tuning**: Focus optimization on computationally intensive components (vocoder, text encoder)

### 4.3 Training Enhancements
- **Knowledge Distillation**: Use baseline VITS as teacher for maintaining quality in compressed models
- **Multi-objective Losses**: Incorporate perceptual losses using pre-trained speech models (WavLM)
- **Enhanced Discriminators**: Utilize multi-scale discriminators for improved adversarial training

The architecture maintains VITS's end-to-end design while addressing identified bottlenecks through systematic optimization of the most computationally expensive components.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
**Efficiency Metrics:**
- Real-Time Factor (RTF) on CPU and GPU (target: RTF < 0.1 on CPU)
- Model parameter count and memory usage
- Inference latency per utterance

**Quality Metrics:**
- Mean Opinion Score (MOS) through listening tests or proxy measures (PESQ, STOI)
- Mel Cepstral Distortion (MCD) for spectral similarity
- F0 Root Mean Square Error (RMSE) for pitch accuracy
- Word Error Rate (WER) using automatic speech recognition for intelligibility

### 5.2 Baseline Models
- **Primary Baseline**: Standard VITS trained on VCTK dataset
- **Reference Comparisons**: 
  - Original Tacotron 2 + HiFi-GAN pipeline
  - FastSpeech 2 (where applicable)
  - Published performance metrics from FLY-TTS and related optimized models

### 5.3 Hardware/Software Requirements
**Hardware:**
- GPU: NVIDIA RTX 3080/4080 (minimum 12GB VRAM)
- CPU: Multi-core processor for CPU benchmarking (Intel i7/i9 or equivalent)
- RAM: 32GB minimum for dataset handling

**Software:**
- PyTorch framework with CUDA support
- Coqui TTS or similar VITS implementation as starting point
- Standard audio processing libraries (librosa, scipy)
- Evaluation tools for MOS proxies and acoustic metrics

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | **Baseline Setup & Profiling**<br>- Reproduce VITS on VCTK<br>- Performance profiling<br>- Bottleneck identification | 1.5 weeks | Baseline model + performance analysis |
| Phase 2 | **Core Optimizations**<br>- Implement iSTFT vocoder<br>- Apply parameter sharing<br>- Basic training setup | 2 weeks | Optimized model architecture |
| Phase 3 | **Training & Validation**<br>- Train optimized models<br>- Incremental testing<br>- Quality assessment | 2 weeks | Trained models + initial results |
| Phase 4 | **Evaluation & Documentation**<br>- Comprehensive benchmarking<br>- Results analysis<br>- Final report preparation | 1 week | Complete evaluation + final report |

**Total Duration: 6.5 weeks** (Conservative timeline accounting for potential implementation challenges)

## 7. Risk Analysis

**Technical Risks:**
- **Quality Degradation**: Optimization may reduce speech naturalness
  - *Mitigation*: Implement incremental validation and use knowledge distillation
- **Implementation Complexity**: iSTFT vocoder integration may be challenging
  - *Mitigation*: Start with simpler optimizations (parameter sharing) first
- **Training Instability**: Modified architectures may be harder to train
  - *Mitigation*: Use pre-trained components and careful learning rate scheduling

**Resource Risks:**
- **Computational Limitations**: Limited GPU time for extensive experiments
  - *Mitigation*: Focus on most promising optimizations, use smaller model variants for testing
- **Time Constraints**: Complex modifications may exceed available time
  - *Mitigation*: Prioritize highest-impact changes, maintain fallback to simpler approaches

**Data/Evaluation Risks:**
- **Limited Subjective Evaluation**: MOS studies may not be feasible
  - *Mitigation*: Use established objective proxies (PESQ, STOI) validated in literature

## 8. Expected Outcomes

**Performance Improvements:**
- **Efficiency Gains**: Target 4-8× speedup over baseline VITS (RTF < 0.1 on CPU)
- **Model Compression**: 25-35% reduction in parameter count while maintaining quality
- **Quality Preservation**: MOS/proxy scores within 5% of baseline performance

**Research Contributions:**
- Validation of frequency-domain optimization techniques on multi-speaker dataset (VCTK)
- Empirical analysis of quality-efficiency trade-offs in real-time TTS systems
- Implementation guide for practical VITS optimization in resource-constrained environments

**Deliverables:**
- Optimized VITS implementation with documented performance improvements
- Comprehensive benchmark results comparing multiple optimization strategies
- Technical report detailing methodology, results, and recommendations for deployment

**Minimum Viable Outcome:**
Even if full optimization targets are not achieved, the project will provide valuable insights into VITS bottlenecks and practical optimization approaches, with documented results suitable for future research directions.

---

**Note:** Update this document as your methodology evolves during implementation.