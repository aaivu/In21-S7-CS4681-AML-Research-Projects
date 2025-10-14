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

### 3.3 The Nocaps Benchmark for Novel Object Captioning

Nocaps (Agrawal et al., 2019) was introduced to measure models’ ability to caption novel visual concepts absent from training data.
Built from the Open Images V4 dataset, it contains 15 k images and 166 k human captions with 600+ object categories, including 400 novel ones unseen during training.
The benchmark defines three test splits—in-domain, near-domain, and out-of-domain—to assess generalization to unseen categories.
Baseline methods such as Neural Baby Talk (NBT) and Constrained Beam Search (CBS) improved coverage of novel words but produced awkward, weakly grounded captions.
Nocaps thus exposed the limitation of purely generative captioners and inspired hybrid models integrating object detection and pretrained vision–language encoders.
For the present study, Nocaps serves as the evaluation benchmark to validate whether CLIP-guided reranking enhances CoCa’s grounding on unseen categories.

**Key Paper:**

- Agrawal et al., 2019 – nocaps: Novel Object Captioning at Scale [3].

## 4. Research Gaps and Opportunities

[Identify gaps in current research that your project could address]

### Gap 1: Lack of Alignment Signal during Inference

**Why it matters:** Although multimodal models such as CoCa are trained with both contrastive and generative losses, only the generative pathway contributes at decoding time. This causes fluent yet weakly grounded captions.

**How your project addresses it:** Introduce an inference-time hybrid scoring mechanism combining CoCa’s log-probability with CLIPScore (semantic similarity) to better align language output with visual content.

### Gap 2: Limited Zero-Shot Generalization to Novel Objects

**Why it matters:** Models often fail on the out-of-domain Nocaps split where unseen categories appear.

**How your project addresses it:** By reranking multiple CoCa caption candidates using CLIPScore, captions most semantically aligned with visual input are selected—enhancing zero-shot generalization without retraining.

### Gap 3 – High Training Cost of Diffusion-based Captioners

**Why it matters:**
Diffusion models such as PCM-Net demand large compute and synthetic data generation.

**How this project addresses it:**
The proposed reranking approach is training-free, providing a lightweight yet effective alternative leveraging pretrained CLIP embeddings.

## 5. Theoretical Framework

This research is grounded in multimodal representation learning and contrastive learning theory.

Contrastive Learning: Encourages semantic alignment between heterogeneous modalities by maximizing agreement between matched image–text pairs while minimizing non-matching pairs.

Generative Modeling: Uses likelihood-based objectives to generate text conditioned on visual embeddings.

Hybrid Inference Theory: Combines these principles at inference by scoring both probabilistic fluency (log P) and semantic alignment (CLIPScore), normalized via z-scoring to prevent one modality from dominating.
The combined score is formalized as:

HybridScore(𝑐) = ℓ(𝑐)- 𝛼\*𝑠(𝑐)
where ℓ and s are z-normalized log-likelihood and CLIPScore values, and α balances alignment and fluency.

## 6. Methodology Insights

Common methodologies in multimodal captioning research include:

Encoder–Decoder Transformers for text generation (SimVLM, BLIP, CoCa).

Contrastive Pre-training on large-scale image–text datasets (CLIP, ALIGN).

Diffusion Priors for synthetic image–text generation (PCM-Net).

Reranking Mechanisms based on CLIPScore or cross-modal similarity for post-generation calibration.

For this project, the methodology focuses on inference-time reranking: generating N candidate captions per image via CoCa, computing log P and CLIPScore for each, applying z-score normalization, and combining them via the hybrid formula above to select the top-ranked caption.

## 7. Conclusion

The reviewed literature shows a clear trajectory in vision–language modeling—from separate encoders toward unified multimodal transformers that jointly learn contrastive and generative tasks. Despite significant progress, inference alignment remains under-explored.
Diffusion-based methods and CLIP-weighted training highlight the value of external semantic supervision, but they increase complexity.
The proposed CLIP-guided reranking thus occupies a promising middle ground: lightweight, training-free, and theoretically consistent with CoCa’s dual objectives.
Evaluating this approach on Nocaps directly addresses the long-standing challenge of novel object captioning, potentially improving caption grounding while maintaining fluency.

## References

1. Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., & Wu, Y. (2022). CoCa: Contrastive Captioners are Image-Text Foundation Models. arXiv:2205.01917 [cs.CV]. https://arxiv.org/abs/2205.01917
2. Luo, J., Chen, J., Li, Y., Pan, Y., Feng, J., Chao, H., & Yao, T. (2024). Unleashing Text-to-Image Diffusion Prior for Zero-Shot Image Captioning. arXiv:2501.00437 [cs.CV]. https://arxiv.org/abs/2501.00437
3. Agrawal, H., Desai, K., Wang, Y., Chen, X., Jain, R., Johnson, M., Batra, D., Parikh, D., Lee, S., & Anderson, P. (2019). nocaps: Novel Object Captioning at Scale. arXiv:1812.08658 [cs.CV]. https://arxiv.org/abs/1812.08658
   ...

---
