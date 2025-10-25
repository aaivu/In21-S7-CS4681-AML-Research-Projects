# Methodology: Multi-Modal Learning — Vision-Language

**Student:** 210299J  
**Research Area:** Multi-Modal Learning: Vision-Language  
**Date:** 2025-09-01

## 1. Overview

This project studies **zero-shot image captioning** by augmenting Contrastive Captioners (**CoCa**) with an **inference-time reranking** step guided by **CLIPScore**. Instead of relying only on generative log-likelihoods at decode time, we generate a pool of candidate captions (via OpenCLIP CoCa checkpoints) and select the final caption using a **hybrid score** that balances fluency (CoCa log-probability) and semantic alignment (CLIP image–text cosine similarity). We evaluate on **Nocaps** across in-/near-/out-of-domain splits with **CIDEr** and **SPICE** as primary metrics.

## 2. Research Design

We adopt an **experimental, comparative** design:

1. **Candidate generation:** For each image, decode \(N\) captions using a fixed CoCa variant (and optionally multiple CoCa variants for diversity).
2. **Alignment scoring:** Compute **CLIPScore** between the image and each candidate.
3. **Hybrid reranking:** For each candidate \(c\):  
   \[
   \text{Score}(c)\;=\;\log P\_{\text{CoCa}}(c\mid I)\;+\;\alpha\cdot \text{CLIPScore}(I,c),
   \]
   with per-image z-normalization of both terms and \(\alpha\) tuned on validation.
4. **Comparison:** Evaluate **Baseline CoCa** (greedy) vs **MaxLogP** (best by log-prob over the same pool) vs **Hybrid** (best by the hybrid score).
5. **Ablations:** \(\alpha\) sweep; beam vs sampling; number of candidates; optional CLIP backbones.

## 3. Data Collection

### 3.1 Data Sources

- **Nocaps** validation/test splits (images from Open Images V4 with multiple human references per image).
- Public **OpenCLIP CoCa** checkpoints (e.g., ViT-B/32, ViT-L/14; with/without MS-COCO fine-tuning) for candidate generation.

### 3.2 Data Description

- **Validation (4.5k images, 10 refs/image)** with three domains: **in-domain**, **near-domain**, **out-of-domain**; the last focuses on categories absent from MS-COCO. This split structure measures generalization to **novel objects**.

### 3.3 Data Preprocessing

- Image I/O via OpenCLIP transforms (resize, center-crop, normalization).
- Caption text normalization (lowercasing, unicode cleanup).
- Map `file_name → image_id`; ensure one prediction per image; evaluate on the **intersection** of IDs for fair comparisons.
- Candidate **de-duplication** (e.g., normalized edit distance or ROUGE-L threshold) to remove near-identical strings.
- Optional filters to drop artifacts (e.g., “stock photo”, stray digits).

## 4. Model Architecture

- **CoCa**: vision encoder (ViT) + two-stage decoder (unimodal text → multimodal cross-attn). We use **frozen** public CoCa checkpoints for candidate generation; decoding via **greedy** or **beam search**.
- **CLIP**: ViT-B/32 encoders compute L2-normalized embeddings; **CLIPScore** is cosine similarity between image and text embeddings.
- **Reranking**: compute z-scores per image for \(\log P\) and CLIPScore; select the caption maximizing the hybrid score above; tune \(\alpha\).

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Primary:** **CIDEr**, **SPICE** (semantic faithfulness / novel-object grounding).
- **Secondary:** **BLEU-4**, **METEOR**, **ROUGE-L** (fluency/overlap).

### 5.2 Baseline Models

- **CoCa Greedy:** single caption per image from a **fixed checkpoint** (primary baseline).
- **MaxLogP:** best candidate by length-normalized log-prob (**logp_mean**) over the same pool (control for reranking).
- **Hybrid (ours):** best by hybrid score with \(\alpha\in\{0.1,0.2,0.5,1.0\}\); report best \(\alpha\) overall and per split.

### 5.3 Hardware/Software Requirements

- **Hardware:** NVIDIA GPU (≥16 GB VRAM recommended), ≥16 GB RAM, ≥50 GB disk (images + caches).
- **Software:** Python 3.10+, PyTorch, **OpenCLIP**, `pycocotools`, **pycocoevalcap** (BLEU/METEOR/ROUGE/CIDEr/SPICE), Java (CoreNLP for SPICE), NumPy/Pandas/Matplotlib.

## 6. Implementation Plan

| Phase   | Tasks                                                  | Duration | Deliverables                   |
| ------- | ------------------------------------------------------ | -------: | ------------------------------ |
| Phase 1 | Data download, preprocessing, ID mapping, eval harness |  2 weeks | Clean dataset + evaluator      |
| Phase 2 | CoCa decoding (greedy/beam), CLIPScore module          |  3 weeks | Working generator + scorer     |
| Phase 3 | Reranking + ablations (\(\alpha\), beams, N)           |  2 weeks | Results (overall + split-wise) |
| Phase 4 | Analysis, plots, error taxonomy, write-up              |   1 week | Final report & assets          |

## 7. Risk Analysis

- **Hybrid underperforms baseline:** CLIP may up-rank generic captions.  
  _Mitigation:_ use **beam candidates**, length penalty, artifact filters, tune \(\alpha\), compare to **MaxLogP**.
- **Metric instability / coverage mismatch:** different image sets across files.  
  _Mitigation:_ evaluate on **intersection of image IDs**; add **bootstrap CIs** for CIDEr/SPICE.
- **SPICE latency / CoreNLP issues:** long first-run download; Java memory.  
  _Mitigation:_ cache CoreNLP; allocate sufficient heap; run SPICE last.
- **Compute constraints:** multiple checkpoints and beams increase cost.  
  _Mitigation:_ start with one strong CoCa checkpoint; scale up only if needed.

## 8. Expected Outcomes

- **Primary:** A **training-free** enhancement that is **most beneficial on out-of-domain** images where novel objects appear; anticipate small but consistent gains in **CIDEr/SPICE** with well-tuned \(\alpha\) and beam-generated candidates.
- **Secondary:** A reproducible pipeline (code + scripts) for **candidate reranking**, **split-wise evaluation**, and **ablations**, plus qualitative analyses illustrating improved grounding (e.g., correctly mentioning novel objects).

---
