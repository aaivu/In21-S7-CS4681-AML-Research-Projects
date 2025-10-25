# Research Proposal: Small LLMs: Domain-Specific

**Student:** 210419F  
**Research Area:** Small Language Models for Domain-Specific Clinical NLP  
**Date:** 2025-10-05

## Abstract

This research proposes developing an efficient, domain-specific small language model for clinical natural language processing by integrating adaptive tokenization and knowledge distillation techniques. The study builds upon existing work combining BioClinicalBERT as a teacher model with DistilBERT as a student model, enhanced through vocabulary expansion with 1,000-2,000 clinical tokens. Unlike traditional approaches that rely on computationally expensive domain-adaptive pretraining (DAPT), this research demonstrates that strategic vocabulary enhancement combined with temperature-scaled knowledge transfer can achieve BioClinicalBERT-level accuracy while maintaining DistilBERT's efficiency. The proposed methodology addresses the critical gap in clinical NLP where general-purpose models underperform due to medical term fragmentation, while large domain-specific models are computationally prohibitive for practical deployment. Expected outcomes include a smaller model than modern LLMs with comparable performance on Medical Entity Recognition tasks, providing a scalable solution for resource-constrained clinical environments.

## 1. Introduction

The rapidly evolving landscape of artificial intelligence in healthcare has witnessed significant advancements with Large Language Models (LLMs), yet their deployment in clinical settings remains constrained by computational demands, privacy concerns, and resource limitations. While models like Med-PaLM demonstrate exceptional performance on medical benchmarks, their computational requirements often exceed what healthcare institutions can practically deploy.

The significance of this research lies in addressing the fundamental challenge of medical term fragmentation that occurs when general-purpose tokenizers encounter domain-specific vocabulary. Current approaches like BioClinicalBERT and BioBERT achieve superior performance through domain-adaptive pretraining but at substantial computational cost, often requiring weeks of training on datasets like MIMIC-III[2]. This research proposes a novel integration of adaptive tokenization and knowledge distillation to bridge the performance gap while maintaining efficiency.

## 2. Problem Statement

Clinical natural language processing faces a critical efficiency paradox: while domain-adapted models like BioClinicalBERT deliver superior performance on medical tasks with 5-8% higher F1 scores than general pretrained models, their computational demands limit practical deployment in resource-constrained healthcare settings[2]. Current approaches suffer from three primary limitations:

**Tokenization Mismatch Problem**: General WordPiece vocabularies fragment medical terms into semantically meaningless subwords, weakening their representational capacity. For example, medical terms like "hypercholesterolemia" are incorrectly tokenized, losing crucial semantic information.

**Computational Inefficiency**: Domain-adaptive pretraining requires extensive computational resources and large-scale medical corpora, making it inaccessible for many healthcare institutions. Traditional DAPT approaches often require GPU clusters and weeks of training time.

**Knowledge Transfer Gap**: Existing knowledge distillation methods for clinical NLP have not effectively combined adaptive tokenization with task-specific distillation, leaving performance gaps between large teacher models and efficient student models.

The research problem is to develop a computationally efficient approach that achieves domain-specific accuracy comparable to BioClinicalBERT while maintaining the resource efficiency of DistilBERT, specifically for Medical Entity Recognition tasks in clinical environments.

## 3. Literature Review Summary

Recent advances in clinical NLP have established two primary paradigms for domain adaptation: domain-adaptive pretraining and knowledge distillation. BioClinicalBERT and BioBERT exemplify the first approach, achieving state-of-the-art performance through continued pretraining on medical corpora[4][5]. However, this approach requires substantial computational resources and specialized infrastructure.

Knowledge distillation has emerged as a promising alternative, with recent studies demonstrating that distilled BERT models can achieve performance comparable to larger teachers while being significantly smaller. DistilBERT[1] offers a compact and efficient architecture while maintaining performance comparable to the original BERT model through knowledge distillation[3]. Research shows that task-specific KD from domain-pretrained teachers reduces the accuracy gap while keeping DistilBERT efficiency.

Adaptive tokenization research has highlighted the importance of domain-specific vocabulary integration. Efficient domain adaptation methods[7] add a small set of carefully chosen medical tokens to the vocabulary, selected based on how much their frequency in medical text differs from general text. This allows models to regain most of the performance benefits of DAPT with much less computational cost.

**Key Research Gaps Identified:**
1. Limited integration of adaptive tokenization with knowledge distillation for clinical applications
2. Lack of comprehensive evaluation frameworks for clinical SLMs
3. Insufficient exploration of temperature-scaled distillation for medical entity recognition
4. Missing systematic approaches to clinical vocabulary selection and integration

Currently, knowledge distillation and adaptive tokenization have not been integrated into a single pipeline for medical entity recognition, representing a significant gap that this research addresses.

## 4. Research Objectives

### Primary Objective
To develop an efficient clinical small language model that combines adaptive tokenization and knowledge distillation to achieve BioClinicalBERT-level accuracy on Medical Entity Recognition tasks while maintaining DistilBERT's computational efficiency.

### Secondary Objectives
- **Adaptive Tokenization Enhancement**: Develop and validate a methodology for extending DistilBERT's vocabulary with 1,000-2,000 high-frequency clinical tokens using KL divergence-based selection to minimize medical term fragmentation
- **Knowledge Distillation Optimization**: Implement and optimize temperature-scaled knowledge transfer from fine-tuned BioClinicalBERT to vocabulary-enhanced DistilBERT, achieving optimal balance between task-specific loss and distillation loss
- **Performance Validation**: Demonstrate comparable or superior performance to BioClinicalBERT on benchmark medical entity recognition datasets (i2b2 2014, MedAlign) while achieving significant computational efficiency gains

## 5. Methodology

The research methodology follows a systematic three-phase approach integrating adaptive tokenization, knowledge distillation, and comprehensive evaluation.

### Phase 1: Adaptive Tokenization Enhancement

**Vocabulary Selection Process:**
- Statistical analysis of token frequency distributions comparing clinical corpora (i2b2, MIMIC-III) against general-domain corpora (Wikipedia, BookCorpus)
- KL divergence computation for candidate clinical terms to identify tokens significantly more frequent in medical data
- Token ranking based on divergence scores and semantic coherence criteria
- Selection threshold application ensuring minimum frequency requirements and semantic integrity preservation

**Vocabulary Integration:**
- Extension of DistilBERT's WordPiece tokenizer vocabulary from 30,522 to approximately 32,000 tokens
- Implementation of adaptive tokenization to prioritize domain-specific vocabulary during tokenization
- Embedding initialization strategies for new clinical tokens using domain-specific pretraining or averaging techniques

### Phase 2: Knowledge Distillation Framework

**Teacher-Student Architecture:**
- Teacher Model: ClinicalBERT fine-tuned on i2b2 2014 dataset with frozen weights
- Student Model: DistilBERT with extended clinical vocabulary
- Training objective combining cross-entropy loss and temperature-scaled knowledge distillation loss

**Training Protocol:**
- Joint optimization of task-specific loss (hard targets) and distillation loss (soft targets)
- Temperature scaling parameter optimization for optimal knowledge transfer
- Convergence monitoring through alignment metrics between hard and soft targets
- Hyperparameter search for optimal balance between accuracy and efficiency

### Phase 3: Evaluation and Validation

**Performance Metrics:**
- Medical Entity Recognition: Precision, Recall, F1-score for entities (diseases, medications, symptoms)
- Computational Efficiency: Inference speed, memory usage, training time
- External Validation: Performance on held-out datasets (MedAlign, additional clinical corpora)

**Baseline Comparisons:**
- Standard DistilBERT (without vocabulary enhancement)
- BioClinicalBERT (teacher model performance)

## 6. Expected Outcomes

**Primary Outcomes:**
- A domain-specific small language model achieving F1 scores within 2-3% of ClinicalBERT performance on medical entity recognition tasks
- Computational efficiency gains including 10-12x faster inference and significant deployment cost reductions compared to large clinical LLMs
- Validated methodology for adaptive tokenization in clinical domains with quantified improvements in medical term representation

**Impact and Applications:**
- Enable deployment of clinical NLP capabilities in smaller healthcare institutions with limited computational resources
- Support real-time clinical decision support systems with locally deployed models
- Provide foundation for privacy-compliant medical AI applications that can operate without cloud dependencies
- Contribute to sustainable AI practices in healthcare by reducing computational and energy requirements

## 7. Timeline

| Week | Task | Deliverables |
|------|------|-------------|
| 1-2  | **Literature Review & Environment Setup** | Comprehensive literature survey, development environment configuration, baseline model implementation |
| 3-4  | **Adaptive Tokenization Development** | Clinical token extraction pipeline, KL divergence analysis, vocabulary selection algorithm, tokenization enhancement validation |
| 5-6  | **Knowledge Distillation Implementation** | Teacher-student framework development, training pipeline integration, hyperparameter optimization framework |
| 7-8  | **Model Training & Initial Evaluation** | Distillation training execution, preliminary performance evaluation, baseline comparisons |
| 9-10 | **Comprehensive Evaluation & Optimization** | External validation on multiple datasets, performance optimization, efficiency benchmarking |
| 11-12| **Analysis & Documentation** | Results analysis, statistical significance testing, methodology documentation |
| 13-14| **External Validation & Comparison** | Independent dataset validation, comparison with recent clinical SLMs, robustness testing |
| 15   | **Final Reporting & Code Release** | Final research report preparation, open-source code repository setup, reproducibility documentation |
| 16   | **Final Submission & Presentation** | Thesis submission, presentation preparation, dissemination planning |

## 8. Resources Required

**Datasets and Models:**
- MACCROBAT dataset
- General domain corpora (Wikipedia, BookCorpus) for vocabulary analysis

**Software and Tools:**
- PyTorch/Transformers library for model implementation
- Hugging Face ecosystem for model management
- Medical text processing libraries (spaCy, NLTK with clinical extensions)
- Evaluation frameworks and statistical analysis tools


## References

[1] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC²2 Workshop*. Available: https://arxiv.org/abs/1910.01108

[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186. Available: https://arxiv.org/abs/1810.04805

[3] Buciluă, C., Caruana, R., & Niculescu-Mizil, A. (2006). Model compression. *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 535-541.

[4] Huang, K., Altosaar, J., & Ranganath, R. (2019). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342*. Available: https://arxiv.org/abs/1904.05342

[5] Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240. Available: https://academic.oup.com/bioinformatics/article/36/4/1234/5566506

[6] Zhang, Y., Yang, Z., & Ji, S. (2024). MLKD-BERT: Multi-level knowledge distillation for pre-trained language models. Available: https://arxiv.org/abs/2407.02775

[7] Sachidananda, V., Kessler, J. S., & Lai, Y. (2021). Efficient domain adaptation of language models via adaptive tokenization. *CoRR*, abs/2109.07460. Available: https://arxiv.org/abs/2109.07460

---
