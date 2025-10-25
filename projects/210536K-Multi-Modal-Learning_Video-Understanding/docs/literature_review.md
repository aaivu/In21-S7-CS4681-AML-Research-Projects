# Literature Review: Multi-Modal Learning: Video Understanding

**Student:** 210536K  
**Research Area:** Multi-Modal Learning: Video Understanding  
**Date:** 2025-09-15  

---

## Abstract

This literature review explores recent advancements in *Temporal Action Localization (TAL)* — a fundamental subdomain of video understanding that aims to identify, classify, and temporally localize human actions in untrimmed videos. The review focuses on the evolution of Transformer-based architectures, particularly *ActionFormer* and its successors, which introduced efficient attention mechanisms, multi-scale context modeling, and anchor-free localization. Key contributions such as *TriDet*, *Self-Feedback DETR*, *Long-Term Pre-training (LTP)*, and *CLTDR* have addressed challenges like temporal collapse, boundary ambiguity, and global dependency modeling. The review categorizes advances across proposal generation, DETR-inspired methods, and feature pyramid modeling, and identifies research gaps — especially in integrating long-term pre-training with attention feedback and task decoupling. The analysis concludes that combining pre-training, feedback mechanisms, and boundary-aware design offers a promising direction for robust, scalable TAL on long-form videos.

---

## 1. Introduction

Video understanding within the field of multi-modal learning has become vital for applications such as video surveillance, content summarization, sports analytics, and human-computer interaction. Unlike trimmed video recognition tasks, *Temporal Action Localization (TAL)* must process long, untrimmed sequences containing multiple actions and background noise.  

Recent Transformer-based architectures like *ActionFormer* (Zhang et al., 2022) demonstrated that a minimalist, local self-attention framework can outperform complex anchor-based and proposal-based methods. However, TAL still faces core issues such as over-smoothing (temporal collapse), limited global dependency capture, and inaccurate boundary predictions. This literature review synthesizes existing works addressing these challenges, emphasizing the transition toward unified and end-to-end models that balance efficiency, context modeling, and precision.

---

## 2. Search Methodology

### Search Terms Used
- "Temporal Action Localization"
- "Video Understanding"
- "Transformers for TAL"
- "Anchor-free Action Detection"
- "Boundary Modeling"
- "Temporal Attention Feedback"
- "Long-term Pre-training"
- Synonyms: "Temporal Action Detection", "Temporal Action Recognition", "Video Transformers", "End-to-End TAL"

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [ ] Other: SpringerLink  

### Time Period
**2018–2025**, focusing on recent Transformer-based and deep learning approaches for fully supervised TAL.

---

## 3. Key Areas of Research

### 3.1 Proposal Generation and Boundary Modeling

Early TAL methods relied on a two-stage structure involving temporal proposal generation and classification. *Boundary-Sensitive Network (BSN)* (Lin et al., 2018) introduced a boundary probability estimation mechanism to produce flexible-duration proposals. However, such pipelines were computationally heavy.

Recent single-stage approaches integrated boundary modeling directly into the localization framework. *TriDet* (Shi et al., 2023) introduced a Trident-head that models relative boundary distributions, offering uncertainty-aware boundary regression and outperforming *ActionFormer* in accuracy and efficiency. *Temporal Deformable Transformers* (2023) further improved flexibility by dynamically aggregating temporal context through deformable attention, addressing rigid contextual assumptions.  

**Key Papers:**
- Lin et al., 2018 — Introduced BSN for precise boundary-based proposal generation.  
- Shi et al., 2023 — Proposed TriDet with relative boundary modeling and scalable granularity perception.  
- Temporal Deformable Transformer, 2023 — Adapted deformable attention for dynamic temporal aggregation and improved efficiency.

---

### 3.2 DETR-Inspired End-to-End Methods

The DETR framework for object detection (Carion et al., 2020) inspired several TAL models that removed anchor dependency. *Self-Feedback DETR* (Kim et al., 2023) employed decoder-to-encoder attention feedback loops to maintain attention diversity and prevent temporal collapse. *Long-Term Pre-training (LTP)* (Kim et al., 2024) addressed data scarcity by designing pretext tasks such as temporal order prediction and masked action reconstruction.  

Recent work by *Ouyang et al.* (2022) introduced *Adaptive Perception Transformer (AdaPerFormer)*, using dual-branch attention to balance global and local perception. *Zhang et al.* (2025) extended this line of inquiry by comparing Transformer and Mamba (state-space model) blocks for TAL, demonstrating that efficient SSMs can match Transformer performance with lower computational cost.

**Key Papers:**
- Carion et al., 2020 — Introduced DETR, inspiring end-to-end temporal localization models.  
- Kim et al., 2023 — Self-Feedback DETR mitigates over-smoothing through cross-attention feedback.  
- Kim et al., 2024 — Long-Term Pre-training (LTP) enhances feature representations for long-form videos.  
- Ouyang et al., 2022 — AdaPerFormer fuses global and local cues through adaptive dual-branch attention.  
- Zhang et al., 2025 — Compared Transformers and Mamba for temporal modeling efficiency.

---

### 3.3 Feature Pyramid and Context Modeling

Handling varying action durations demands multi-scale context aggregation. *ActionFormer* (Zhang et al., 2022) proposed a multiscale Transformer encoder with local self-attention (window size 19), outperforming all prior TAL methods on THUMOS14 and ActivityNet. *HTNet* (Kang et al., 2022) extended this with hierarchical transformers and background feature sampling to enhance contextual sensitivity.

Simpler yet competitive designs like *TemporalMaxer* (Tran et al., 2023) achieved comparable accuracy using max pooling instead of attention. *Cross-Layer Task Decoupling and Refinement (CLTDR)* (Li et al., 2025) addressed task conflict by separating classification and regression heads across layers. *STMixer* (Wu et al., 2024) explored sparse spatio-temporal feature sampling, extending the one-stage paradigm to sparse detection.

**Key Papers:**
- Zhang et al., 2022 — ActionFormer established the baseline for single-stage Transformer TAL.  
- Kang et al., 2022 — HTNet used hierarchical transformers and BFS for improved feature hierarchies.  
- Tran et al., 2023 — TemporalMaxer simplified context modeling via temporal max pooling.  
- Li et al., 2025 — CLTDR introduced cross-layer task decoupling and gated multi-granularity refinement.  
- Wu et al., 2024 — STMixer combined sparse feature sampling with one-stage detection for efficiency.

---

## 4. Research Gaps and Opportunities

### Gap 1: Limited Global Temporal Dependency Modeling  
**Why it matters:** Local self-attention in ActionFormer restricts long-range context and leads to temporal collapse in long videos.  
**How your project addresses it:** Incorporates *self-attention feedback loops* and *long-term pre-training* to expand context and maintain feature diversity.

### Gap 2: Imprecise Boundary Predictions  
**Why it matters:** Current regression heads lack robustness in ambiguous transitions, affecting precision at high tIoU thresholds.  
**How your project addresses it:** Integrates *relative boundary modeling* and *task decoupling* (TriDet, CLTDR) for probabilistic and cross-layer refinement.

### Gap 3: Underexplored Integration of Pre-training and Attention Feedback  
**Why it matters:** Previous works treat pre-training and feedback independently.  
**How your project addresses it:** Unifies these mechanisms within an enhanced ActionFormer architecture for end-to-end optimization and improved generalization.

---

## 5. Theoretical Framework

The study builds on the theoretical foundations of the *Transformer architecture* (Vaswani et al., 2017), integrating:
- **Local and Hierarchical Attention:** (ActionFormer, HTNet) for multi-scale efficiency.  
- **Feedback Mechanisms:** (Self-DETR) to counter over-smoothing and enhance global dependencies.  
- **Probabilistic Boundary Regression:** (TriDet) for uncertainty modeling.  
- **Task Decoupling Principle:** (CLTDR) for reducing inter-task interference.

Together, these establish a unified theoretical base for modeling temporal dependencies and uncertainty-aware action boundaries.

---

## 6. Methodology Insights

Common methodologies in TAL include:
- **Backbones:** Pre-trained I3D, SlowFast, or ViViT encoders.  
- **Feature Pyramids:** Multi-scale temporal hierarchies for variable durations.  
- **Losses:** Focal loss (classification), DIoU (regression), and auxiliary distribution losses for boundary precision.  
- **Training:** Sliding window training with fixed sequence lengths (512–2304), center sampling, and Adam optimizer.  
- **Evaluation:** Mean Average Precision (mAP) across tIoU thresholds (e.g., [0.5:0.05:0.95]).

Emerging directions emphasize:
- Long-term pre-training for global representation learning.  
- Feedback-enhanced Transformers for collapse mitigation.  
- Sparse and deformable attention mechanisms for scalability.  
- Unified multi-task optimization for boundary consistency and classification accuracy.

---

## 7. Conclusion

The evolution of TAL research demonstrates a clear movement toward *end-to-end, anchor-free, and Transformer-driven* models. While *ActionFormer* established a powerful baseline, subsequent innovations in feedback attention, hierarchical transformers, deformable modeling, and decoupled optimization reveal substantial potential for improving long-form video understanding. Integrating *long-term pre-training* with *attention feedback* and *boundary-aware regression* represents a promising research frontier — one that this project directly addresses to enhance both precision and generalization in Transformer-based TAL systems.

---

## References

1. Zhang, C.-L., Wu, J., & Li, Y. (2022). *ActionFormer: Localizing moments of actions with transformers.* ECCV.  
2. Lin, T., Zhao, X., Su, H., Wang, C., & Yang, M. (2018). *BSN: Boundary sensitive network for temporal action proposal generation.* ECCV.  
3. Shi, D., Zhong, Y., Cao, Q., Ma, L., Li, J., & Tao, D. (2023). *TriDet: Temporal action detection with relative boundary modeling.* CVPR.  
4. Kim, J., Lee, M., & Heo, J.-P. (2023). *Self-feedback DETR for temporal action detection.* ICCV.  
5. Kim, J., Lee, M., & Heo, J.-P. (2024). *Long-term pre-training for temporal action detection with transformers.* arXiv:2408.13152.  
6. Li, Q., Liu, D., Kong, J., Xu, H., & Wang, J. (2025). *Temporal action localization with cross-layer task decoupling and refinement.* AAAI.  
7. Tran, T., Truong, T.-D., & Bui, Q.-H. (2023). *TemporalMaxer: Maximize temporal context with only max pooling for TAL.* arXiv:2303.09055.  
8. Kang, T.-K., Lee, G.-H., & Lee, S.-W. (2022). *HTNet: Anchor-free temporal action localization with hierarchical transformers.* arXiv:2207.09662.  
9. Ouyang, Y., Zhang, T., Gu, W., & Wang, H. (2022). *Adaptive Perception Transformer for Temporal Action Localization (AdaPerFormer).* arXiv:2208.11908.  
10. Zhang, Y., Palmero, C., & Escalera, S. (2025). *Transformer or Mamba for Temporal Action Localization?* SCITEPRESS.  
11. Temporal Deformable Transformer. (2023). *Dynamic temporal aggregation for TAL.* Springer LNCS.  
12. Wu, T., Cao, M., Gao, Z., Wu, G., & Wang, L. (2024). *STMixer: A one-stage sparse action detector.* arXiv:2404.09842.  
13. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention is all you need.* NeurIPS.  
14. Carion, N., Massa, F., Synnaeve, G., Usunier, A., Kirillov, A., & Zagoruyko, S. (2020). *End-to-end object detection with transformers (DETR).* ECCV.  
15. Wang, B., Zhao, Y., Yang, L., Long, T., & Li, X. (2023). *Temporal action localization in the deep learning era: A survey.* IEEE TPAMI.  
16. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). *SlowFast networks for video recognition.* ICCV.  
17. Carreira, J., & Zisserman, A. (2017). *Quo Vadis, Action Recognition?* CVPR.  
