# Research Proposal: Multi-Modal Learning: Video Understanding

**Student:** 210536K  
**Research Area:** Multi-Modal Learning: Video Understanding  
**Date:** 2025-09-01  

---

## Abstract

Temporal Action Localization (TAL) in untrimmed videos remains a critical challenge in video understanding due to the limitations of Transformer-based models. Current models, such as ActionFormer, achieve strong performance but suffer from temporal collapse, restricted global dependency capture, and inaccurate boundary predictions. This research proposes an enhanced ActionFormer framework that integrates three underexplored directions — long-term pre-training, self-attention feedback mechanisms, and boundary-aware task decoupling. The proposed model aims to enrich feature representations for long-form videos, reduce over-smoothing through dynamic feedback in attention layers, and improve temporal precision via relative boundary modeling. The study will evaluate the approach on benchmark datasets including ActivityNet 1.3 and EPIC-Kitchens 100, targeting a 1–3% improvement in mean Average Precision (mAP) over the baseline. The project contributes to advancing scalable and accurate TAL models with better generalization to complex, real-world video analysis applications.

---

## 1. Introduction

Video understanding is a central problem in multi-modal learning, enabling intelligent systems to recognize and localize human actions in complex, untrimmed video sequences. Temporal Action Localization (TAL) focuses on identifying the start and end times of action instances along with their class labels. This task supports numerous real-world applications, including video summarization, sports analysis, human–robot interaction, and surveillance.  

Recent advancements in Transformer architectures have transformed TAL research, with *ActionFormer* emerging as a key benchmark model. It employs a hierarchical Transformer with local self-attention to efficiently process temporal sequences. However, despite its success, it faces notable limitations when applied to long-form videos: temporal collapse in deep self-attention layers, limited capture of global dependencies, and imprecise boundary predictions. This research aims to overcome these challenges by enhancing the ActionFormer framework to achieve more accurate and scalable TAL for long-form, real-world videos.

---

## 2. Problem Statement

Existing Transformer-based TAL models such as ActionFormer and TriDet effectively model local temporal dependencies but fall short in capturing long-range temporal relationships and accurate action boundaries.  
Key issues include:
- **Over-smoothing (Temporal Collapse):** Deep self-attention layers produce homogenized feature representations, reducing the model’s ability to distinguish subtle temporal variations in long videos.  
- **Limited Global Dependency Modeling:** Local attention restricts the model’s receptive field, hindering contextual understanding of distant frames.  
- **Boundary Ambiguity:** Action boundaries are often subtle and imprecise, causing mislocalizations.  

These limitations reduce performance on long-form datasets like EPIC-Kitchens 100 and ActivityNet 1.3. Therefore, this research seeks to develop an enhanced TAL framework that improves global temporal modeling, refines boundary precision, and mitigates representation collapse.

---

## 3. Literature Review Summary

Early TAL methods relied on two-stage pipelines such as the Boundary-Sensitive Network (BSN), which separately generated proposals and classified them. Recent models like TriDet, Self-Feedback DETR, Long-Term Pre-training (LTP), and CLTDR represent a shift toward end-to-end Transformer frameworks.  

- **TriDet** introduced relative boundary modeling using probability distributions for improved temporal precision.  
- **Self-Feedback DETR** used cross-attention feedback loops to mitigate temporal collapse in deep Transformers.  
- **LTP** applied pretext-based pre-training to enrich feature representations for long-range temporal contexts.  
- **CLTDR** improved classification–localization consistency via cross-layer task decoupling.  

While these approaches improved aspects of TAL independently, none integrated pre-training, feedback mechanisms, and boundary-aware modeling into a unified Transformer framework. This research fills that gap by combining these innovations to improve accuracy, stability, and generalization for long-form videos.

---

## 4. Research Objectives

### Primary Objective
To develop an enhanced Transformer-based Temporal Action Localization model that integrates long-term pre-training, attention feedback mechanisms, and boundary-aware task decoupling to improve accuracy and scalability for long-form videos.

### Secondary Objectives
- **Enhance Global Temporal Dependency Modeling:** Incorporate self-attention feedback to capture long-range dependencies and reduce temporal collapse.  
- **Improve Boundary Precision:** Apply relative boundary modeling and decoupled task heads to handle ambiguous transitions.  
- **Leverage Long-Term Pre-training:** Use pretext tasks such as temporal order prediction to enrich feature representations and generalization.  
- **Benchmark Evaluation:** Achieve at least a 1–3% improvement in mAP on ActivityNet 1.3 and EPIC-Kitchens 100 compared to baseline ActionFormer.  
- **Analyze Synergistic Effects:** Study the combined benefits of pre-training and attention feedback within a single framework.

---

## 5. Methodology

The proposed methodology builds upon the ActionFormer architecture and introduces several extensions:

1. **Long-Term Pre-training (LTP) Adaptation:**  
   - Pre-train the Transformer encoder using unlabeled long-form videos with pretext tasks such as temporal order prediction and masked action reconstruction.  
   - Initialize the ActionFormer encoder with these pre-trained weights to improve feature richness and reduce over-smoothing.  

2. **Encoder Modifications with Self-Attention Feedback:**  
   - Integrate feedback loops from decoder cross-attention maps into the encoder layers.  
   - Dynamically adjust attention weights to maintain attention diversity and capture global dependencies efficiently.  

3. **Decoder Enhancements for Boundary Precision:**  
   - Replace standard regression heads with relative probability-based boundary modeling.  
   - Implement cross-layer task decoupling and refinement to align classification and localization tasks, improving consistency and precision.  

4. **Training and Inference Strategy:**  
   - Use combined loss functions (focal loss, DIoU, and boundary distribution loss).  
   - Employ pre-training followed by end-to-end fine-tuning on ActivityNet 1.3 and EPIC-Kitchens 100 datasets.  
   - Evaluate using average mAP across tIoU thresholds and boundary error analysis.  

5. **Tools and Frameworks:**  
   - Implemented in PyTorch 2.0+ using Kaggle/Google Colab GPUs.  
   - Datasets: THUMOS14, ActivityNet 1.3, and EPIC-Kitchens 100.  

This unified approach aims to enhance both feature quality and boundary accuracy while maintaining computational efficiency.

---

## 6. Expected Outcomes

- A Transformer-based TAL model capable of effectively capturing long-term dependencies without temporal collapse.  
- Improved action boundary prediction through uncertainty-aware regression and task decoupling.  
- 1–3% improvement in average mAP on ActivityNet 1.3 and EPIC-Kitchens 100 datasets compared to ActionFormer.  
- Comprehensive ablation results demonstrating the effectiveness of pre-training and feedback integration.  
- Scalable and resource-efficient framework deployable on commodity GPUs for real-world applications.

---

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review and Dataset Familiarization |
| 3-4  | Develop Long-Term Pre-training and Feedback Modules |
| 5-8  | Integrate and Implement Enhanced ActionFormer Model |
| 9-12 | Conduct Experiments and Ablation Studies |
| 13-15| Analyze Results and Prepare Research Report |
| 16   | Final Evaluation and Submission |

---

## 8. Resources Required

- **Datasets:** THUMOS14, ActivityNet 1.3, EPIC-Kitchens 100.  
- **Compute Resources:** Google Colab Pro / Kaggle GPUs (≥16 GB VRAM).  
- **Frameworks & Libraries:** PyTorch 2.0+, TorchVision, Hugging Face Transformers, MMAction2, NumPy, Pandas, Matplotlib.  
- **Storage:** Google Drive (~300 GB for datasets and checkpoints).  
- **Version Control:** GitHub repository for code and experiment tracking.  

---

## References

1. Zhang, C.-L., Wu, J., & Li, Y. (2022). *ActionFormer: Localizing moments of actions with transformers.* ECCV.  
2. Shi, D., Zhong, Y., Cao, Q., Ma, L., Li, J., & Tao, D. (2023). *TriDet: Temporal action detection with relative boundary modeling.* CVPR.  
3. Kim, J., Lee, M., & Heo, J.-P. (2023). *Self-feedback DETR for temporal action detection.* ICCV.  
4. Kim, J., Lee, M., & Heo, J.-P. (2024). *Long-term pre-training for temporal action detection with transformers.* arXiv:2408.13152.  
5. Li, Q., Liu, D., Kong, J., Xu, H., & Wang, J. (2025). *Temporal action localization with cross-layer task decoupling and refinement.* AAAI.  
6. Wang, B., Zhao, Y., Yang, L., Long, T., & Li, X. (2023). *Temporal action localization in the deep learning era: A survey.* IEEE TPAMI.  
7. Tran, T., Truong, T.-D., & Bui, Q.-H. (2023). *TemporalMaxer: Maximize temporal context with only max pooling for TAL.* arXiv:2303.09055.  

---

**Submission Instructions:**
1. Complete all sections above.  
2. Commit this file to your project repository as `research_proposal.md`.  
3. Create an issue with labels **milestone** and **research-proposal**.  
4. Tag your supervisors for review and feedback.  
