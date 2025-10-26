# Literature Review: NLP:Speech Enhancement

**Student:** 210112H
**Research Area:** NLP:Speech Enhancement
**Date:** 2025-09-01

## Abstract
This review synthesizes recent advances in single-channel speech enhancement (SE) with emphasis on time-frequency (T-F) learning over STFT representations. It contrasts magnitude masking and complex spectral mapping, surveys efficient architectures (CRN/DPCRN, DPRNN, lightweight attention), and discusses loss design, datasets, metrics, and deployment constraints. Evidence points to phase-aware objectives and compact convolution–recurrent hybrids as the best accuracy–latency trade-off for real-time use. Persistent gaps include robust phase modeling at low SNR, domain generalization to real acoustics, and alignment between objective metrics and human perception—motivating staged enhancement and frequency-aware capacity allocation.

## 1. Introduction
Speech enhancement improves intelligibility and quality in noisy or reverberant conditions for telephony, conferencing, assistive hearing, and embedded voice interfaces. Traditional spectral subtraction and Wiener filtering struggle with non-stationary noise. Modern deep models learn to predict masks or complex spectra in the STFT domain, excelling in single-channel setups where spatial cues are unavailable. This review centers on single-channel, low-latency SE, highlighting design choices that balance performance and deployment feasibility.

## 2. Search Methodology

### Search Terms Used

“speech enhancement”, “single-channel”, “monaural”, “ratio mask”, “complex ratio mask”

“complex spectral mapping”, “CRN”, “DPCRN”, “DPRNN”, “attention”, “real-time”, “low-latency”

“PESQ”, “STOI”, “DNSMOS”, “DEMAND”, “VCTK”, “phase modeling”, “perceptual loss”

### Databases Searched
- [✓] IEEE Xplore
- [✓] ACM Digital Library
- [✓] Google Scholar
- [✓] ArXiv
- [ ] Other: ___________

## 3. Key Areas of Research
3.1 Masking vs. Complex Spectral Mapping

Magnitude-domain ratio masks estimate per-bin gains on |X| and are simple and robust, but they leave phase largely unchanged. Complex ratio masking adds phase correction, improving perceptual quality and reducing musical noise. Complex spectral mapping directly predicts real and imaginary (RI) components of the clean spectrum, aligning the objective with waveform reconstruction. Hybrid strategies—coarse magnitude masking followed by RI refinement—often stabilize training and recover high-frequency consonants better at low SNR.

Key Papers:

Williamson et al., 2016 — Introduced complex ratio masking, demonstrating phase-aware gains over magnitude-only masking.

Tan & Wang, 2019 — Complex CRN for monaural enhancement, showing benefits of complex mapping with recurrent context.

3.2 Architectures and Efficiency (CRN/DPCRN, DPRNN, Attention)

Convolutional recurrent networks (CRN) blend 2-D convs (local spectral patterns) with RNNs (temporal context) for real-time SE. DPCRN extends CRN with dual-path recurrence (intra-chunk + inter-chunk), scaling temporal context efficiently. DPRNN, popular in separation, inspired chunk-wise temporal modeling for SE. Attention mechanisms (TF/axial attention, conformers) focus computation on informative regions and can improve metrics, but naive self-attention increases latency/compute; lightweight or axial variants mitigate this.

Key Papers:

Tan & Wang, 2018 — CRN for real-time SE with strong accuracy–latency trade-off.

Le et al., 2021 — DPCRN baseline; efficient dual-path temporal modeling for SE.

Luo & Mesgarani, 2019 — DPRNN (separation); informs efficient long-context modeling in SE.

Wan et al., 2022 — Multi-loss TF-attention layered onto a DPCRN-style backbone.

3.3 Loss Design and Perceptual Objectives

Pure magnitude MSE is easy but correlates imperfectly with human quality. Adding RI losses promotes phase consistency. Multi-resolution STFT (MR-STFT) losses compare spectra at several window/hop sizes, improving stability and sharpness. Perceptual surrogates (e.g., MetricGAN, self-supervised speech embeddings) bias training toward human-relevant improvements, at the cost of added optimization complexity. Dynamic loss weighting or curriculum (mask-first, then complex mapping) mitigates artifact buildup in low SNR.

Key Papers:

Tan & Wang, 2019 — RI losses with complex mapping.

Fu et al., 2019 (MetricGAN) — Adversarial/perceptual optimization for SE.

3.4 Data, Augmentation, and Evaluation

SE often uses synthetic mixtures: clean speech (e.g., VCTK) mixed with real noise (e.g., DEMAND) across SNRs (−5 to +15 dB). Reverberation via RIR convolution improves realism; simple augmentations (gain/pitch perturbations) enhance robustness. Metrics typically include PESQ (quality), STOI (intelligibility), and SNR improvement (SNRi). The DNS Challenge popularized non-intrusive predictors (DNSMOS) that better reflect subjective ratings across diverse conditions.

Key Papers:

Thiemann et al., 2013 — DEMAND dataset: diverse real-world noises.

Valentini-Botinhao et al., 2016 — VCTK/Noisy speech corpora and protocols.

Reddy et al., 2020–2023 — DNS Challenge and DNSMOS.

3.5 Deployment: Real-Time and Edge Constraints

Causal processing, bounded latency, and modest memory/compute are essential on embedded devices and in live communications. CRN/DPCRN achieve favorable trade-offs via hierarchical downsampling and efficient recurrence. Attention is helpful when constrained (axial, low-rank) and integrated sparsely. Frequency-aware parameterization (e.g., sub-bands, spectral compression) concentrates capacity where speech energy and intelligibility cues reside (low/mid bands).

Key Papers:

Valin, 2018 — RNNoise: a tiny RNN + DSP features (non-DPCRN) illustrating strict real-time deployment.

Le et al., 2021; Tan & Wang, 2018 — Practical, low-latency backbones.

## 4. Research Gaps and Opportunities
Gap 1: Robust phase modeling at low SNR

Why it matters: Phase errors cause musical noise and smear transients, degrading perceived quality.
How your project addresses it: Two-stage pipeline—Stage-1 magnitude masking to lift SNR, followed by Stage-2 complex RI refinement with dual-path recurrence for fine temporal structure.

Gap 2: Frequency-aware capacity allocation

Why it matters: Uniform processing over-allocates parameters to sparse high-frequency bins.
How your project addresses it: Learnable spectral compression that preserves low bands and compresses highs, improving efficiency while retaining salient cues for intelligibility.

Gap 3: Domain generalization to real acoustics

Why it matters: Lab improvements may not transfer to diverse, non-stationary environments with reverberation.
How your project addresses it: Train on DEMAND with broad SNR sampling; add simple augmentations (gain, pitch, optional RIRs) to improve robustness without heavy compute.

Gap 4: Metric–perception mismatch

Why it matters: PESQ/STOI can diverge from human judgments, especially off-distribution.
How your project addresses it: Report PESQ/STOI/SNRi alongside qualitative spectrogram/audio checks; consider DNSMOS for non-intrusive perceptual correlation.

## 5. Theoretical Framework
Dual-path recurrence splits sequences into chunks; an intra-chunk RNN models short-range structure and an inter-chunk RNN models long-range context efficiently. Frequency-aware compression echoes psychoacoustics (critical bands, speech energy concentration) by prioritizing low/mid bands. Composite losses (RI + MR-STFT) encourage both envelope and phase fidelity.

## 6. Methodology Insights
Common practice: Hann windows (≈25 ms) with hops (≈10–12.5 ms), moderate FFT sizes (e.g., 1200 @ 48 kHz) for a balance of resolution and latency. Training uses on-the-fly mixing at varied SNRs, random noise pairing, and light augmentation. Effective models use 2–4 conv encoder stages and compact dual-path recurrence (hidden sizes ≈128–256). For losses, RI MSE plus MR-STFT typically outperforms magnitude-only MSE; perceptual surrogates help when stable. Evaluation should aggregate by noise type and report means/variances; include SNRi for interpretability and consider DNSMOS or listener checks when feasible.
## 7. Conclusion
Recent SE advances converge on phase-aware training, efficient convolution–recurrent backbones, and frequency-aware parameterization. Attention can improve results when carefully constrained for latency. Open challenges—phase at low SNR, domain robustness, and metric alignment—motivate staged enhancement strategies and better data design. A two-stage DPCRN-style model with learnable spectral compression addresses these needs by first improving SNR via masking, then refining RI components, while biasing capacity toward perceptually salient bands—well-suited to real-time and edge deployments.
## References

D. S. Williamson, Y. Wang, D. Wang, “Complex Ratio Masking for Monaural Speech Separation,” IEEE/ACM TASLP, 2016.

K. Tan, D. L. Wang, “A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement,” Interspeech, 2018.

K. Tan, D. L. Wang, “Complex Spectral Mapping with a CRN for Monaural Speech Enhancement,” ICASSP, 2019.

Y. Le, H. Chen, K.-J. Chen, J. Lu, “DPCRN: Dual-Path Convolution Recurrent Network for Single-Channel Speech Enhancement,” Interspeech, 2021.

Y. Luo, N. Mesgarani, “Dual-Path RNN: Efficient Long-Sequence Modeling,” IEEE/ACM TASLP, 2019.

Y. Wan et al., “Multi-Loss TF-Attention Model for Speech Enhancement,” ICASSP, 2022.

A. Valin, “RNNoise: Learning Noise Suppression,” 2018 (hybrid DSP/RNN).

C. K. Reddy et al., “DNS Challenge: Objective Speech Quality Assessment,” 2020–2023.

J. Thiemann, N. Ito, E. Vincent, “DEMAND: Diverse Environments Multichannel Acoustic Noise Database,” 2013.

C. Valentini-Botinhao et al., “Noisy Speech Corpora Using VCTK for SE,” 2016.

A. Défossez et al., “Demucs for Denoising/Separation,” 2021–2023.

A. Pandey, D. Wang, “TCN-Based Speech Enhancement,” IEEE/ACM TASLP, 2019.

P. C. Loizou, Speech Enhancement: Theory and Practice, 2013.

S. Pascual et al., “SEGAN: Adversarial Speech Enhancement,” Interspeech, 2017.

H. Chen et al., “FullSubNet: Full-Band and Sub-Band Fusion,” Interspeech, 2020.

K. Hu et al., “DCCRN: Deep Complex Conv Recurrent Network,” Interspeech, 2020.

Y. Fu et al., “MetricGAN: Generative Adversarial Metric Learning for SE,” Interspeech, 2019.

A. Martín-Doñas et al., “Perceptually Motivated Losses for SE,” 2020.

**Notes:**
- Aim for 15-20 high-quality references minimum
- Focus on recent work (last 5 years) unless citing seminal papers
- Include a mix of conference papers, journal articles, and technical reports
- Keep updating this document as you discover new relevant work
