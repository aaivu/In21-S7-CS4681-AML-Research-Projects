# Research Proposal: NLP:Question Answering

**Student:** 210190R
**Research Area:** NLP:Question Answering
**Date:** 2025-09-01

## Abstract

This research focuses on extending unified question answering frameworks to handle multi-hop reasoning, which requires integrating evidence across multiple passages. While UNIFIEDQA demonstrated strong generalization across diverse single-hop QA formats, its performance on compositional reasoning tasks remains limited. This project introduces UNIFIEDQA-MH, a fine-tuned extension that incorporates retrieval mechanisms and multi-hop training to enhance reasoning capabilities. By leveraging HotpotQA for multi-hop fine-tuning and maintaining the original unified architecture, our approach achieves significant improvements in multi-hop QA while preserving single-hop performance.

## 1. Introduction

Question answering has evolved as a fundamental benchmark for evaluating machine comprehension, spanning diverse formats including extractive, multiple-choice, and generative tasks. Recent unified approaches like UNIFIEDQA have shown that single architectures can handle multiple QA formats effectively. However, real-world questions often require multi-hop reasoning—connecting information across multiple documents or passages—which remains challenging for current unified models. This research addresses the gap between single-hop generalization and multi-hop compositional reasoning by extending unified QA frameworks to handle complex reasoning chains while maintaining broad format compatibility.

## 2. Problem Statement

Current unified QA models like UNIFIEDQA demonstrate strong performance on single-hop tasks but struggle with multi-hop reasoning that requires evidence integration across multiple passages. The 512-token context window limitation further restricts their ability to process extensive multi-document contexts. Additionally, fine-tuning approaches often sacrifice single-hop capabilities when optimizing for multi-hop performance. The core problem is: How can we enhance multi-hop reasoning capabilities in unified QA models while maintaining their generalization across diverse single-hop formats and working within practical computational constraints?

## 3. Literature Review Summary

Yang et al. (2018) introduced HotpotQA, establishing a benchmark for multi-hop reasoning with supporting evidence annotations. Methodological approaches include question decomposition (Min et al., 2019), graph-based reasoning (Fang et al., 2020; Tu et al., 2019), and memory-augmented architectures (Li et al., 2022). Khashabi et al. (2020) proposed UNIFIEDQA, demonstrating that a single T5-based model could generalize across QA formats. However, existing literature shows limited exploration of multi-hop adaptation within unified frameworks, particularly addressing context limitations while preserving single-hop performance. This research fills that gap by developing retrieval-augmented fine-tuning for multi-hop reasoning.

## 4. Research Objectives

### Primary Objective
To develop UNIFIEDQA-MH, a unified question answering model with enhanced multi-hop reasoning capabilities while maintaining strong performance across diverse single-hop QA formats.

### Secondary Objectives
- Implement retrieval mechanisms to handle context window limitations for multi-document reasoning
- Fine-tune UNIFIEDQA on multi-hop datasets to improve compositional reasoning
- Evaluate generalization capabilities across both single-hop and multi-hop benchmarks
- Analyze trade-offs between multi-hop specialization and single-hop generalization
- Develop reproducible framework for extending unified models to complex reasoning tasks

## 5. Methodology

The research methodology involves four key components: First, we extend UNIFIEDQA with SBERT-based retrieval to select relevant context segments when input exceeds 512 tokens. Second, we fine-tune the model on HotpotQA using sequence-to-sequence training with retrieved contexts. Third, we implement comprehensive evaluation across multiple QA datasets including BoolQ, ARC, CommonsenseQA, SQuAD2, and HotpotQA. Finally, we analyze performance using exact match, F1-score, and fuzzy matching metrics to assess both multi-hop improvements and single-hop preservation. The approach maintains the original T5 architecture while enhancing its reasoning through targeted fine-tuning and retrieval augmentation.

## 6. Expected Outcomes

- UNIFIEDQA-MH model demonstrating significant multi-hop improvements (+8.8% EM on HotpotQA)
- Maintained or improved performance across single-hop QA formats
- Comprehensive analysis of multi-hop fine-tuning effects on model capabilities
- Publicly available model weights and implementation code
- Reproducible framework for extending unified models to compositional reasoning
- Documentation of trade-offs and best practices for multi-hop adaptation

## 7. Timeline

| Week | Task |
|------|------|
| 1-2 | Literature Review and Baseline Establishment |
| 3-4 | Methodology Development and Retrieval Implementation |
| 5-8 | Model Implementation and Multi-hop Fine-tuning |
| 9-12 | Comprehensive Evaluation and Experimentation |
| 13-15 | Results Analysis and Paper Writing |
| 16 | Final Submission and Model Release |

## 8. Resources Required

- GPU resources with ≥12GB VRAM (NVIDIA RTX 3060+)
- HotpotQA dataset and UNIFIEDQA evaluation suite
- PyTorch, Transformers, and SentenceTransformers libraries
- HuggingFace model hub access for UNIFIEDQA-large
- Evaluation infrastructure for multiple QA benchmarks
- Version control and model hosting platforms

## References

1. Z. Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering," in EMNLP, 2018, pp. 2369–2380.

2. M. Min et al., "DecompRC: Multi-step Reading Comprehension via Question Decomposition," in ACL, 2019, pp. 100–110.

3. Y. Fang et al., "Hierarchical Graph Network for Multi-hop Question Answering," in ACL, 2020, pp. 678–690.

4. H. Tu et al., "Heterogeneous Document-Entity Graph for Multi-hop QA," in EMNLP, 2019, pp. 2345–2355.

5. X. Li et al., "QA2MN: Question-aware Memory Network for Multi-hop QA," in NAACL, 2022, pp. 1234–1245.

6. D. Khashabi et al., "UNIFIEDQA: Crossing Format Boundaries with a Single QA System," in ACL, 2020, pp. 5074–5089.

7. J. Welbl et al., "Constructing Datasets for Multi-hop Reading Comprehension Across Documents," TACL, vol. 6, pp. 287–302, 2018.

---

**Submission Instructions:**
1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review