# Literature Review: Multi-Modal Learning:Vision-Language

**Student:** 210299J
**Research Area:** Multi-Modal Learning:Vision-Language
**Date:** 2025-09-01

## Abstract

This literature review examines recent progress in multimodal vision–language learning, with a particular focus on image captioning and zero-shot generalization. Three major research directions are covered: (1) Contrastive Captioners (CoCa) that unify contrastive and generative objectives, (2) diffusion-based models that leverage synthetic image–text pairs and CLIP-based supervision for zero-shot captioning, and (3) the Nocaps benchmark, which drives evaluation for novel object captioning.
The review highlights the evolution from dual-encoder retrieval models to unified encoder–decoder architectures capable of fine-grained alignment and generative reasoning. It identifies a key limitation in current models—the lack of explicit alignment signals during inference—and motivates a lightweight inference-time reranking strategy combining CoCa’s generative likelihood with CLIPScore similarity. This hybrid approach aims to enhance caption faithfulness to unseen objects without retraining, addressing a core gap in multimodal understanding benchmarks.

## 1. Introduction

The intersection of computer vision and natural language processing has evolved rapidly through the development of multi-modal foundation models. These models aim to jointly understand and generate both visual and textual information, enabling applications such as image captioning, visual question answering (VQA), and cross-modal retrieval.
While early research treated vision and language as separate modalities combined through task-specific fusion layers, recent foundation models like CLIP, BLIP, SimVLM, and CoCa have demonstrated that unified pre-training on large-scale image–text corpora can yield powerful general-purpose representations.
This review focuses on recent advances in vision–language pre-training (VLP) and zero-shot image captioning, particularly through contrastive-captioning frameworks, diffusion-based synthetic data generation, and benchmark analysis on Nocaps. Together, these lines of research outline the current frontiers and challenges that motivate the proposed CLIP-guided reranking strategy.

## 2. Search Methodology

### Search Terms Used

“vision–language pretraining”, “image captioning”, “zero-shot captioning”, “contrastive captioner”, “CoCa model”, “CLIP alignment”, “diffusion prior captioning”, “Nocaps benchmark”, “novel object captioning”

Synonyms: multimodal transformer, cross-modal learning, text-to-image diffusion, CLIPScore reranking, generative vision–language model

### Databases Searched

- [ ] IEEE Xplore
- [ ] ACM Digital Library
- [ ] Google Scholar
- [ ] ArXiv
- [ ] Other: \***\*\_\_\_\*\***

### Time Period

2019 – 2025, emphasizing recent multimodal foundation models and zero-shot captioning literature.

## 3. Key Areas of Research

### 3.1 Contrastive Captioners (CoCa): Image–Text Foundation Models

Image-text foundation models have evolved through three paradigms: single-encoder classifiers (e.g., CNN, ViT), dual-encoder contrastive models (e.g., CLIP, ALIGN), and encoder–decoder captioners (e.g., SimVLM, BLIP).
CoCa unified these paradigms into a single transformer architecture capable of both contrastive alignment and caption generation in a single forward pass. It employs a unimodal text decoder for contrastive pre-training and a multimodal decoder with cross-attention for generative tasks.
Pre-trained jointly on large-scale web text and labeled data, CoCa achieved state-of-the-art zero-shot performance on Nocaps (122.4 CIDEr, 15.5 SPICE).
However, during inference CoCa uses only the generative log-likelihood pathway, ignoring the learned contrastive alignment. This limits caption grounding on novel objects.
The proposed CLIP-guided reranking addresses this by combining CoCa’s log-probability with CLIPScore similarity, thereby aligning generative fluency with semantic faithfulness during decoding.

**Key Papers:**

- Yu et al., 2022 – CoCa: Contrastive Captioners are Image-Text Foundation Models [1].

### 3.2 Diffusion-Based Approaches for Zero-Shot Image Captioning

Diffusion-based and CLIP-aligned models have redefined zero-shot image captioning (ZIC).
Earlier text-only decoders (MAGIC, CapDec, DeCap) attempted to reuse CLIP priors but suffered from modality gaps when substituting text embeddings with visual features.
PCM-Net (Luo et al., 2024) introduced Patch-wise Cross-modal Mix-up (PCM) to blend visual and textual features and a CLIP-weighted Cross-Entropy (CXE) loss to prioritize high-quality synthetic samples.
These innovations improved both in-domain and cross-domain captioning without paired supervision.
The diffusion-prior approach reinforces the role of CLIP-based alignment not just during training but also as an inference-time calibration mechanism—precisely the principle extended by the proposed CLIP-guided reranking of CoCa captions.

**Key Papers:**

- Luo et al., 2024 – Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning [2].

## 4. Research Gaps and Opportunities

[Identify gaps in current research that your project could address]

### Gap 1: [Description]

**Why it matters:** [Explanation]
**How your project addresses it:** [Your approach]

### Gap 2: [Description]

**Why it matters:** [Explanation]
**How your project addresses it:** [Your approach]

## 5. Theoretical Framework

[Describe the theoretical foundation for your research]

## 6. Methodology Insights

[What methodologies are commonly used? Which seem most promising for your work?]

## 7. Conclusion

[Summarize key findings and how they inform your research direction]

## References

[Use academic citation format - APA, IEEE, etc.]

1. [Reference 1]
2. [Reference 2]
3. [Reference 3]
   ...

---

**Notes:**

- Aim for 15-20 high-quality references minimum
- Focus on recent work (last 5 years) unless citing seminal papers
- Include a mix of conference papers, journal articles, and technical reports
- Keep updating this document as you discover new relevant work
