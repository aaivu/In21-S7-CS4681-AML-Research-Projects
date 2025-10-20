# Literature Review: NLP:Text-to-Speech

**Student:** 210086E
**Research Area:** NLP:Text-to-Speech
**Date:** 2025-09-01

## Abstract

This literature review examines current advances in neural Text-to-Speech (TTS) synthesis, with particular focus on end-to-end models and real-time optimization techniques. The review covers key developments from 2018-2024, analyzing the evolution from two-stage TTS pipelines to unified architectures like VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech). Key findings reveal that while VITS established new benchmarks for naturalness and efficiency, significant bottlenecks remain in vocoder architectures, particularly in the HiFi-GAN decoder component. Recent research has demonstrated promising solutions through frequency-domain approaches, model compression techniques, and enhanced training strategies. The review identifies critical research gaps in real-time mobile deployment, streaming synthesis, and computational efficiency for low-resource environments. These insights inform the development of optimized TTS architectures that balance quality, speed, and resource constraints for practical applications.

## 1. Introduction

Text-to-Speech (TTS) synthesis has evolved dramatically over the past decade, transitioning from concatenative and parametric approaches to sophisticated neural architectures that produce human-like speech quality. Modern TTS systems face the dual challenge of achieving high naturalness while maintaining computational efficiency for real-world deployment scenarios, particularly in mobile and embedded environments.

This literature review focuses on recent advances in neural TTS, with emphasis on end-to-end architectures that integrate text processing and waveform generation into unified frameworks. The scope encompasses key developments from 2018-2024, examining the progression from two-stage pipelines (text-to-spectrogram followed by vocoding) to fully integrated models like VITS. Special attention is given to performance optimization techniques, including efficient vocoder architectures, model compression strategies, and enhanced training methodologies.

The review aims to identify current limitations in existing TTS systems, particularly computational bottlenecks that hinder real-time deployment, and examine recent solutions proposed in the literature. This analysis will inform the development of optimized TTS architectures that can deliver high-quality speech synthesis while meeting the stringent performance requirements of practical applications.

## 2. Search Methodology

### Search Terms Used
- "Text-to-Speech" OR "TTS" OR "Speech Synthesis"
- "VITS" OR "Variational Inference Text-to-Speech"
- "End-to-end TTS" OR "Neural TTS" OR "Neural Speech Synthesis"
- "HiFi-GAN" OR "Vocoder" OR "Neural Vocoder"
- "Real-time TTS" OR "Efficient TTS" OR "Fast TTS"
- "FLY-TTS" OR "FastSpeech" OR "Tacotron"
- "Transformer TTS" OR "Conformer TTS"
- "Multi-band synthesis" OR "Frequency-domain TTS"
- "Model compression" OR "Parameter sharing" OR "Lightweight TTS"
- "Adversarial training" OR "GAN-based TTS"

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] Other: Papers with Code, Semantic Scholar

### Time Period
2018-2024, with primary focus on developments from 2020-2024 to capture recent advances in end-to-end neural TTS architectures, particularly post-VITS innovations and optimization techniques.

## 3. Key Areas of Research

### 3.1 End-to-End Neural TTS Architectures
The shift from traditional two-stage pipelines to unified end-to-end architectures represents a fundamental advancement in TTS research. These models integrate text encoding, alignment, and waveform generation into single trainable frameworks, eliminating the need for intermediate representations and external vocoders.

**Key Papers:**
- Kim et al., 2021 - Introduced VITS, combining Transformer-based text encoding with HiFi-GAN generation through conditional variational autoencoders
- Ren et al., 2020 - FastSpeech 2 established parallel, non-autoregressive synthesis with explicit duration and pitch control
- Valle et al., 2020 - Tacotron 2 demonstrated the effectiveness of attention-based sequence-to-sequence architectures for TTS

### 3.2 Efficient Vocoder Architectures
Vocoding remains the primary computational bottleneck in modern TTS systems. Research has focused on replacing expensive transposed convolution operations with more efficient alternatives, particularly frequency-domain approaches.

**Key Papers:**
- FLY-TTS researchers, 2024 - Introduced ConvNeXt-based frequency-domain synthesis with multi-band processing, achieving 8.8× CPU speedup
- Kong et al., 2020 - HiFi-GAN established high-quality neural vocoding but with significant computational overhead
- Vocos team, 2023 - Demonstrated efficient Fourier-based vocoding with inverse STFT operations

### 3.3 Model Compression and Optimization
Reducing model size and computational complexity while maintaining quality is crucial for practical deployment, especially in resource-constrained environments.

**Key Papers:**
- Q-VITS researchers, 2024 - Introduced Conformer-based GAN decoder with 67% inference time reduction
- Parameter sharing studies, 2024 - Demonstrated 36% parameter reduction (28M to 17.9M) through grouped sharing in Transformer and flow modules
- Knowledge distillation approaches - Various works on transferring knowledge from large models to efficient student networks

### 3.4 Advanced Training Strategies
Enhanced training methodologies improve the efficiency-quality trade-off through auxiliary objectives, perceptual losses, and sophisticated discriminator architectures.

**Key Papers:**
- FNH-TTS team, 2024 - Mixture-of-Experts duration prediction with multi-band discriminators
- WavLM integration studies, 2024 - Pre-trained speech models as auxiliary discriminators for improved quality guidance
- Multi-discriminator training - Various works on adversarial training with multiple specialized discriminators

## 4. Research Gaps and Opportunities

### Gap 1: Real-time Mobile and Embedded Deployment
**Why it matters:** Current TTS systems, including optimized versions of VITS, still struggle to achieve true real-time performance on mobile and embedded devices. While CPU RTF of 0.12 (8× real-time) is achieved on high-end processors, mobile deployment requires much greater efficiency improvements.
**How your project addresses it:** Develop ultra-lightweight TTS architectures that combine aggressive model compression, novel vocoder designs, and hardware-aware optimizations to achieve sub-0.05 RTF on mobile CPUs while maintaining acceptable quality.

### Gap 2: Streaming and Low-Latency Synthesis
**Why it matters:** Most existing TTS systems operate on complete utterances, introducing latency that makes them unsuitable for real-time conversational applications. The need for streaming synthesis with minimal algorithmic delay remains largely unaddressed.
**How your project addresses it:** Design streaming-capable TTS architectures that can begin synthesis before complete text input is available, incorporating chunked processing and predictive alignment mechanisms to minimize end-to-end latency.

### Gap 3: Quality-Efficiency Trade-off Optimization
**Why it matters:** Current optimization approaches often sacrifice quality for speed or vice versa. There's a need for principled approaches that can navigate this trade-off space more effectively, particularly for specific deployment scenarios.
**How your project addresses it:** Develop adaptive TTS systems that can dynamically adjust their computational complexity based on available resources and quality requirements, using techniques like early exit mechanisms and progressive synthesis.

### Gap 4: Limited Evaluation on Diverse Datasets
**Why it matters:** Most optimization studies focus on single datasets (often LJSpeech), limiting the generalizability of findings. Multi-speaker, multi-accent datasets like VCTK remain underexplored for efficiency optimization.
**How your project addresses it:** Conduct comprehensive evaluation on VCTK dataset to validate optimization techniques across diverse speakers and accents, ensuring robustness of efficiency improvements.

## 5. Theoretical Framework

The theoretical foundation for this research rests on several key principles from machine learning and signal processing:

**Variational Inference and Generative Modeling:** VITS employs conditional variational autoencoders (CVAEs) to learn the mapping from text to speech latent representations. The variational framework allows for modeling the inherent one-to-many relationship between text and speech, capturing prosodic variations while maintaining computational tractability.

**Adversarial Training:** The integration of GAN-based discriminators ensures high-quality waveform generation by encouraging the generator to produce realistic audio that cannot be distinguished from natural speech. Multi-discriminator architectures provide specialized feedback for different aspects of audio quality.

**Flow-based Models:** Normalizing flows in VITS enable invertible transformations between text features and speech latent variables, providing both training stability and inference efficiency. The flow-based prior learns complex alignment patterns without explicit supervision.

**Frequency-Domain Processing:** Recent advances leverage the efficiency of Fast Fourier Transform (FFT) operations for audio synthesis, replacing expensive time-domain convolutions with frequency-domain manipulations that can be computed more efficiently on modern hardware.

**Information Theory and Compression:** Model compression techniques draw from information-theoretic principles to identify and eliminate redundant parameters while preserving essential information for high-quality synthesis.

## 6. Methodology Insights

**Evaluation Metrics:** The field employs both objective and subjective evaluation methods. Objective metrics include Real-Time Factor (RTF) for efficiency, Mean Opinion Score (MOS) for perceived quality, and various acoustic measures. The RTF metric has emerged as particularly critical for deployment considerations.

**Training Strategies:** Multi-stage training approaches are common, often involving pre-training of individual components followed by end-to-end fine-tuning. The use of auxiliary losses and perceptual objectives (such as pre-trained speech model features) has shown significant promise for maintaining quality in compressed models.

**Architecture Design Patterns:** Successful approaches combine multiple design principles: parallel processing for speed, hierarchical representations for quality, and modular architectures for flexibility. The trend toward frequency-domain processing represents a significant methodological shift from traditional time-domain approaches.

**Dataset Considerations:** Multi-speaker datasets like VCTK provide more robust evaluation compared to single-speaker datasets. The choice of dataset significantly impacts the generalizability of optimization results.

**Most Promising Approaches for This Work:**
1. Multi-band frequency-domain synthesis for vocoder optimization
2. Parameter sharing and structured pruning for model compression  
3. Auxiliary discriminators using pre-trained speech models
4. Progressive training with gradually increasing complexity
5. Hardware-aware optimization considering specific deployment constraints

## 7. Conclusion

This literature review reveals that while significant progress has been made in neural TTS, particularly through end-to-end architectures like VITS, substantial opportunities remain for optimization. The primary bottleneck consistently identified across studies is the vocoder component, specifically the computationally expensive upsampling operations in HiFi-GAN-based decoders.

Key findings indicate that frequency-domain approaches offer the most promising avenue for efficiency improvements, with demonstrated speedups of 4-9× while maintaining or improving quality. Model compression through parameter sharing and architectural optimization provides complementary benefits, achieving significant parameter reduction without quality degradation.

The research landscape suggests that future TTS systems should integrate multiple optimization strategies: efficient frequency-domain vocoders, compressed model architectures, and enhanced training methodologies. The identified research gaps in mobile deployment, streaming synthesis, and principled quality-efficiency trade-offs provide clear directions for impactful research contributions.

These insights directly inform the proposed research direction of developing an optimized VITS variant that combines the most promising techniques from recent literature: multi-band frequency-domain synthesis, strategic parameter compression, and auxiliary training objectives, validated on the challenging multi-speaker VCTK dataset to ensure robustness and practical applicability.

## References

[1] J. Kim, J. Kong, and J. Son, "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech," arXiv preprint arXiv:2106.06103, 2021. [Online]. Available: https://arxiv.org/abs/2106.06103.

[2] Coqui AI, Vits - conditional variational autoencoder with adversarial learning for end-to-end text-to-speech, https://docs.coqui.ai/en/latest/models/vits.html, Accessed: 2025.

[3] Y. Guo et al., "Fly-tts: Fast, lightweight and high-quality end-to-end text-to-speech synthesis," arXiv preprint arXiv:2407.00753, 2024. [Online]. Available: https://arxiv.org/abs/2407.00753.

[4] The Moonlight, Fly-tts: Fast, lightweight and high-quality end-to-end text-to-speech synthesis - literature review, https://www.themoonlight.io/en/review/fly-tts-fast-lightweight-and-high-quality-end-to-end-text-to-speech-synthesis, Accessed: 2025.

[5] Unknown, "Fnh-tts: A fast, natural, and human-like speech synthesis system with advanced prosodic modeling based on mixture of experts," arXiv preprint arXiv:2508.12001, 2025. [Online]. Available: https://arxiv.org/abs/2508.12001v2.

[6] H. Sun, J. Song, and Y. Jiang, "Fast inference end-to-end speech synthesis with style diffusion," Electronics, vol. 14, no. 14, p. 2829, 2025. doi: 10.3390/electronics14142829. [Online]. Available: https://www.mdpi.com/2079-9292/14/14/2829.

[7] D. Lim et al., "Jets: Jointly training fastspeech2 and hifi-gan for end to end text to speech," arXiv preprint arXiv:2203.16852, 2022. [Online]. Available: https://arxiv.org/abs/2203.16852.

---

**Notes:**
- Aim for 15-20 high-quality references minimum
- Focus on recent work (last 5 years) unless citing seminal papers
- Include a mix of conference papers, journal articles, and technical reports
- Keep updating this document as you discover new relevant work