# Literature Review: Multi-Modal Learning\- Vision-Language

**Student:** 210417X
**Research Area:** Multi-Modal Learning\- Vision-Language
**Date:** 2025-09-01

## Abstract
This literature review explores recent advancements in multi-modal learning with a focus on vision-language integration, covering key areas such as Visual Question Answering (VQA), large-scale pre-training, and transformer-based architectures. It examines the evolution from early CNN-RNN models to state-of-the-art frameworks like Flamingo, highlighting challenges such as overfitting to small datasets, noisy feature integration, and generalization limitations. Key findings include the efficacy of frozen model approaches in few-shot settings and the potential of semantic alignment and noise reduction techniques to enhance performance, setting the stage for innovative enhancements in resource-constrained environments.

## 1. Introduction
The field of multi-modal learning, particularly vision-language integration, has emerged as a critical area in artificial intelligence, enabling machines to reason across visual and textual data. This review scopes recent developments in Vision-Language Models (VLMs) from 2018 to 2025, focusing on VQA as a benchmark task. It addresses the transition from traditional architectures to advanced pre-trained models, identifying persistent challenges and opportunities for improvement, especially in few-shot learning scenarios.

## 2. Search Methodology

### Search Terms Used
- Multi-modal learning
- Vision-language models
- Visual Question Answering (VQA)
- Few-shot learning
- Transformer architectures
- Semantic alignment
- Noise reduction
- Frozen models
- Gated cross-attention

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv

### Time Period
2018-2025, prioritizing recent developments with seminal works from earlier years included.

## 3. Key Areas of Research

### 3.1 Early VQA Models and Benchmarks
This area explores initial approaches to VQA, relying on CNN-RNN architectures for feature extraction and reasoning. Early models struggled with generalization and bias.
**Key Papers:**
- Agrawal et al., 2017 \cite{b1} - Introduced VQA as a benchmark, highlighting the need for balanced datasets like VQA v2.0.
- Goyal et al., 2017 \cite{b4} - Enhanced VQA v2.0 with balanced question types, addressing language priors.

### 3.2 Transformer-Based Vision-Language Models
Recent advancements leverage transformers for joint vision-language representation, improving performance through pre-training.
**Key Papers:**
- Lu et al., 2019 \cite{b9} - Proposed ViLBERT, a dual-stream transformer with 5-7% accuracy gains on VQA v2.0.
- Tan and Bansal, 2019 \cite{b10} - Introduced LXMERT, advancing cross-modality encoding.
- Alayrac et al., 2022 \cite{b2} - Developed Flamingo, integrating frozen LLMs with gated cross-attention for few-shot VQA.

### 3.3 Few-Shot Learning and Efficiency
Focuses on adapting VLMs to limited data, emphasizing efficiency and noise resilience.
**Key Papers:**
- Tsimpoukelli et al., 2021 \cite{b6} - Explored multimodal few-shot learning with frozen models.
- Awadalla et al., 2023 \cite{b13} - Presented OpenFlamingo, an open-source framework for VLM training.

## 4. Research Gaps and Opportunities

### Gap 1: Noise in Visual Feature Integration
**Why it matters:** Unfiltered visual features degrade VQA performance, especially in few-shot settings.
**How your project addresses it:** FLAMINGO-VQA’s QGFP module filters irrelevant patches using CLIP, improving input quality.

### Gap 2: Limited Generalization in Few-Shot VQA
**Why it matters:** Models overfit to small datasets, reducing real-world applicability.
**How your project addresses it:** SFS and SCV enhance context relevance and prediction stability, validated by an 8.0 pp accuracy gain.

## 5. Theoretical Framework
The research is grounded in multi-modal fusion theory, leveraging pre-trained representations (e.g., CLIP \cite{b3}) and attention mechanisms (e.g., gated cross-attention \cite{b2}). It builds on the hypothesis that optimizing input quality and output consistency can mitigate limitations of frozen VLMs in few-shot scenarios.

## 6. Methodology Insights
Common methodologies include supervised pre-training (ViLBERT, LXMERT) and zero-shot/few-shot adaptation (Flamingo). Promising approaches for this work include non-trainable modular enhancements (e.g., QGFP, SFS, SCV), validated through ablation studies on VQA v2.0, offering a scalable, efficient alternative.

## 7. Conclusion
This review highlights the evolution of vision-language models, identifying noise and generalization as key challenges. The findings inform FLAMINGO-VQA’s development, leveraging semantic alignment and stability techniques to achieve a 41.0% accuracy, guiding future research toward resource-efficient, real-world VQA solutions.

## References

## References

1. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D., “Making the V in VQA matter: Elevating the role of image understanding in visual question answering,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 6904–6913.
2. Tan, H. and Bansal, M., “LXMERT: Learning cross-modality encoder representations from transformers,” in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2019, pp. 5099–5110.
3. Lu, J., Batra, D., Parikh, D., and Lee, S., “ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 13–23.
4. Radford, A., et al., “Learning transferable visual models from natural language supervision,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.
5. Tsimpoukelli, M., et al., “Multimodal few-shot learning with frozen language models,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021, pp. 200–212.
6. Alayrac, J.-B., et al., “Flamingo: A visual language model for few-shot learning,” *arXiv preprint arXiv:2204.14198*, 2022.
7. Hu, E. J., et al., “LoRA: Low-rank adaptation of large language models,” *arXiv preprint arXiv:2106.09685*, 2021.
8. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D., “Making the V in VQA matter: Elevating the role of image understanding in visual question answering,” in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 6904–6913.
9. Lu, J., Batra, D., Parikh, D., and Lee, S., “ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 13–23.
10. Tan, H. and Bansal, M., “LXMERT: Learning cross-modality encoder representations from transformers,” in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2019, pp. 5099–5110.
11. Radford, A., et al., “Learning transferable visual models from natural language supervision,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2021, pp. 8748–8763.
12. Tsimpoukelli, M., et al., “Multimodal few-shot learning with frozen language models,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021, pp. 200–212.
13. Alayrac, J.-B., et al., “Flamingo: A visual language model for few-shot learning,” *arXiv preprint arXiv:2204.14198*, 2022.
14. Li, J., Li, D., Xiong, C., and Hoi, S., “BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” *arXiv preprint arXiv:2301.12597*, 2023.
15. Chen, X., et al., “PaLI-X: On scaling up multilingual multimodal models,” *arXiv preprint arXiv:2305.01278*, 2023.
16. Hu, E. J., et al., “LoRA: Low-rank adaptation of large language models,” *arXiv preprint arXiv:2106.09685*, 2021.
17. Guo, Z., Zhu, X., Li, H., et al., “From images to textual prompts: Zero-shot VQA with frozen large language models,” in *Proc. Annu. Meeting Assoc. Comput. Linguist. (ACL)*, 2022.
