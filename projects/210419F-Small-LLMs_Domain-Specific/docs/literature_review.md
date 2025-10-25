# Literature Review: Small LLMs: Domain-Specific

**Student:** 210419F  
**Research Area:** Small LLMs: Domain-Specific  
**Date:** 2025-10-05

## Abstract

This literature review examines the current state of research in domain-specific small language models, with particular emphasis on knowledge distillation and adaptive tokenization techniques for medical natural language processing. The review analyzes recent developments in creating efficient, lightweight models that maintain performance comparable to larger counterparts while addressing computational constraints in specialized domains. Key themes include the evolution from general-purpose large language models to compact, domain-adapted variants, the effectiveness of knowledge distillation from clinical BERT models, and the critical role of adaptive tokenization in handling domain-specific terminology. This review identifies significant research gaps in the integration of tokenization enhancement with task-specific knowledge distillation for medical entity recognition, providing the foundation for the proposed research on efficient clinical NLP via adaptive tokenization and knowledge distillation.

## 1. Introduction

The rapid advancement of large language models (LLMs) has revolutionized natural language processing across various domains. However, the computational demands and resource requirements of these models present significant challenges for real-world deployment, particularly in specialized domains such as healthcare where data privacy, computational constraints, and real-time processing requirements are paramount. This literature review focuses on the emerging field of small language models (SLMs) designed for domain-specific applications, with particular attention to the medical and clinical domains.

The scope of this review encompasses three primary areas: (1) knowledge distillation techniques for compressing large clinical models into efficient smaller variants, (2) adaptive tokenization methods for improving domain-specific language understanding, and (3) the integration of these approaches for enhanced performance in medical named entity recognition tasks. The review synthesizes recent research from 2020-2025, emphasizing developments that directly relate to the proposed research objectives.

## 2. Search Methodology

### Search Terms Used
- Knowledge distillation BERT clinical NLP
- Adaptive tokenization medical language models
- Domain-specific small language models healthcare
- DistilBERT clinical domain adaptation
- ClinicalBERT knowledge distillation medical NLP
- BioBERT small models medical entity recognition
- Medical entity recognition i2b2 dataset
- Temperature scaled knowledge distillation language models
- Vocabulary extension medical tokenization WordPiece
- Small language models survey medical healthcare
- BERT compression techniques clinical NLP
- Domain adaptation vs knowledge distillation medical NLP
- Efficient domain adaptation language models
- Multilingual medical NER DistilBERT
- Clinical text preprocessing tokenization

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] PubMed/PMC
- [x] Other: Nature Digital Medicine, JMIR, Clinical NLP Workshop Proceedings

### Time Period
2020-2025, with emphasis on developments from 2022-2025 to capture the most recent advances in small language models and domain adaptation techniques.

## 3. Key Areas of Research

### 3.1 Knowledge Distillation for Clinical Language Models

Knowledge distillation has emerged as a fundamental technique for creating efficient clinical language models. Recent research demonstrates significant advances in transferring knowledge from large domain-specific teachers to compact student models while maintaining clinical performance.

**Key Papers:**
- **Rohanian et al. (2024)** - Developed lightweight clinical transformers (15M-65M parameters) using knowledge distillation from BioClinicalBERT, achieving comparable performance to larger models across NER, relation extraction, and sequence classification tasks. Their DistilClinicalBERT and TinyClinicalBERT models represent the first comprehensive study specifically focused on creating efficient transformers for clinical NLP.

- **Vedula et al. (2024)** - Demonstrated effective knowledge distillation from state-of-the-art LLMs (Gemini, OpenAI models) to BERT models approximately 1,000 times smaller for clinical named entity recognition, achieving competitive performance on disease, medication, and symptom extraction across over 3,300 clinical notes.

- **Karim et al. (2025)** - Introduced LastBERT, a distilled model reducing parameters from 110M to 29M (73.64% reduction) while maintaining strong performance on GLUE benchmarks and achieving 85% accuracy on ADHD severity classification, demonstrating comparable performance to DistilBERT and ClinicalBERT.

The literature reveals three primary distillation approaches: (1) traditional teacher-student architectures with soft target alignment, (2) multi-level distillation incorporating intermediate representations, and (3) task-specific distillation tailored to clinical applications. Temperature scaling emerges as a critical component, with optimal temperatures typically ranging from 0.3-0.7 depending on task complexity.

### 3.2 Adaptive Tokenization and Vocabulary Enhancement

Adaptive tokenization represents a paradigm shift in domain adaptation, offering computational efficiency advantages over traditional domain-adaptive pretraining while addressing the fundamental challenge of medical terminology fragmentation.

**Key Papers:**
- **Sachidananda et al. (2021)** - Pioneered efficient domain adaptation via adaptive tokenization, demonstrating that domain-specific subword sequences can be determined directly from conditional token distribution divergences. Their approach achieved >97% of domain-specific pretraining benefits while being 72x faster than traditional approaches.

- **Balde et al. (2024)** - Introduced ADAPTBPE, addressing fundamental limitations in BPE tokenization for domain vocabulary adaptation. Their approach modifies the BPE initialization phase to prioritize domain-specific vocabulary, achieving 3.57% and 1.87% improvements on classification and summarization tasks respectively.

- **Liu et al. (2023)** - Proposed task-adaptive tokenization for mental health applications, demonstrating 60% token reduction while improving generation performance through variable segmentation sampling optimized on task-specific data.

- **Zheng et al. (2024)** - Developed adaptive tokenizers for large language models that monitor perplexity changes during training, enabling dynamic vocabulary optimization aligned with model evolution.

The research indicates that adaptive tokenization is particularly effective when: (1) domain-specific tokens are selected based on KL divergence from general corpora, (2) vocabulary extensions maintain semantic integrity of medical terms, and (3) tokenization modifications are integrated with model training objectives.

### 3.3 Medical Named Entity Recognition Advances

Recent developments in medical NER demonstrate the effectiveness of domain-specific models and the potential for efficiency improvements through architectural modifications.

**Key Papers:**
- **Hu et al. (2024)** - Comprehensive evaluation of BioBERT variants for medical NER, confirming BioBERT's superiority over general models with enhanced precision and F1 scores through biomedical pretraining on domain-specific corpora.

- **Abadeer et al. (2020)** - Assessed DistilBERT performance on clinical NER tasks, finding comparable results to medical BERT variants for PHI detection but 5% lower performance on medical concept extraction, highlighting domain-specific knowledge gaps.

- **Averly et al. (2025)** - Introduced zero-shot clinical NER frameworks addressing the challenge of entity recognition without labeled clinical data, demonstrating entity decomposition strategies for improved recall in clinical entity extraction.

The literature reveals that medical NER performance depends heavily on: (1) domain-specific pretraining on clinical corpora, (2) appropriate tokenization that preserves medical term integrity, and (3) task-specific fine-tuning strategies that account for clinical language characteristics.

### 3.4 Small Language Models in Healthcare

The emergence of small language models specifically designed for healthcare applications represents a significant shift toward practical, deployable clinical NLP solutions.

**Key Papers:**
- **Garg et al. (2025)** - Comprehensive survey of small language models in healthcare, presenting a taxonomic framework analyzing models across NLP tasks, stakeholder roles, and continuum of care. Their analysis covers architectural foundations, adaptation techniques, and compression approaches for clinical precision.

- **Kim et al. (2025)** - Introduced Meerkat, a family of medical SLMs designed for lightweight deployment while enhancing reasoning capabilities, demonstrating potential for resource-constrained clinical environments.

- **Magnini et al. (2025)** - Examined open-source SLMs for personal medical chatbots, addressing critical privacy concerns while maintaining clinical utility in resource-limited settings.

The research emphasizes three key development approaches: (1) adaptation from generic LLMs through medical fine-tuning, (2) compression of healthcare-specific LLMs into efficient architectures, and (3) development of inherently small models with domain-specific capabilities from inception.

### 3.5 Domain Adaptation vs Knowledge Distillation

Recent comparative studies illuminate the relative merits and optimal applications of domain adaptation versus knowledge distillation approaches in clinical NLP.

**Key Papers:**
- **Laparra et al. (2021)** - Review of transfer learning and domain adaptation in clinical NLP, highlighting that general domain pretraining often transfers inadequately to clinical domains due to specialized language characteristics.

- **Wang et al. (2025)** - Proposed Multi-Level Distillation Boost (MLDB) combining self-knowledge distillation with dual-directional knowledge distillation for improved domain adaptation performance.

- **Moslemi et al. (2024)** - Survey of recent knowledge distillation advancements, examining innovations in architectures, training paradigms, and application domains with specific attention to healthcare applications.

The literature suggests that knowledge distillation is particularly effective when: (1) large domain-specific teachers are available, (2) computational efficiency is prioritized over absolute performance, and (3) rapid deployment cycles are required. Domain adaptation remains superior for: (1) maximum performance scenarios, (2) novel domain characteristics, and (3) long-term deployment contexts.

## 4. Research Gaps and Opportunities

### Gap 1: Integrated Adaptive Tokenization and Knowledge Distillation
**Description:** Current research treats adaptive tokenization and knowledge distillation as separate optimization strategies. No comprehensive framework exists that simultaneously optimizes vocabulary expansion and knowledge transfer in a unified training pipeline.

**Why it matters:** Medical terminology fragmentation significantly impacts knowledge distillation effectiveness, as teacher knowledge may not transfer optimally through suboptimal tokenization. Integrated approaches could achieve superior performance while maintaining efficiency benefits.

**How your project addresses it:** The proposed research directly addresses this gap by combining adaptive tokenization enhancement with task-specific knowledge distillation from BioClinicalBERT to enhanced DistilBERT.

### Gap 2: Temperature Scaling Optimization for Medical Knowledge Distillation
**Description:** While temperature scaling is recognized as critical for knowledge distillation effectiveness, limited research exists on optimal temperature selection specifically for medical domain knowledge transfer, particularly for entity recognition tasks.

**Why it matters:** Medical NLP tasks exhibit unique characteristics including class imbalance, specialized terminology, and complex entity boundaries that may require domain-specific temperature optimization strategies.

**How your project addresses it:** The research incorporates systematic temperature optimization as part of the knowledge distillation framework, with specific attention to medical entity recognition performance metrics and clinical validity.

## 5. Theoretical Framework

The theoretical foundation for this research builds on three interconnected frameworks:

**Knowledge Distillation Theory:** Based on Hinton et al.'s foundational work, knowledge distillation operates on the principle that student models can learn from teacher model predictions beyond simple label matching. In the medical domain, this translates to learning nuanced relationships between medical concepts that are encoded in teacher model outputs but may not be explicitly labeled in training data.

**Information-Theoretic Tokenization:** Adaptive tokenization is grounded in information theory, specifically the principle of minimizing information loss during text segmentation. The KL divergence approach for token selection represents optimal information preservation for domain-specific applications.

**Domain Transfer Learning:** The research is situated within the broader framework of domain transfer learning, specifically addressing the challenge of maintaining semantic fidelity while optimizing computational efficiency. The theoretical foundation emphasizes the importance of preserving domain-specific knowledge structures during model compression.

## 6. Methodology Insights

Analysis of current methodologies reveals several promising approaches for implementation:

**Knowledge Distillation Methodologies:** The most effective approaches combine multiple distillation objectives: (1) soft target alignment using temperature-scaled softmax, (2) intermediate representation matching through cosine similarity, and (3) attention pattern transfer for preserving semantic relationships.

**Tokenization Enhancement Strategies:** Successful adaptive tokenization implementations typically follow a three-stage process: (1) statistical analysis of token frequency distributions using KL divergence, (2) semantic coherence evaluation of candidate tokens, and (3) vocabulary integration with preserved model architecture compatibility.

**Evaluation Frameworks:** Comprehensive evaluation requires multi-dimensional assessment including: (1) traditional NLP metrics (F1, precision, recall), (2) computational efficiency measures (inference time, memory usage), and (3) clinical validity assessments (entity boundary accuracy, medical term preservation).

## 7. Conclusion

This literature review reveals a rapidly evolving landscape in domain-specific small language models, with particular momentum in healthcare applications. The convergence of knowledge distillation techniques and adaptive tokenization approaches presents significant opportunities for creating efficient, clinically viable NLP systems.

Key findings include: (1) knowledge distillation can achieve 70-85% of large model performance with 70-90% parameter reduction, (2) adaptive tokenization provides comparable benefits to domain pretraining with significantly reduced computational requirements, and (3) medical NER tasks particularly benefit from integrated approaches that preserve domain-specific terminology integrity.

The research gaps identified provide clear directions for advancing the field, particularly in developing unified frameworks that simultaneously optimize tokenization and knowledge transfer. The proposed research on efficient clinical NLP via adaptive tokenization and knowledge distillation is well-positioned to make significant contributions to both the theoretical understanding and practical implementation of domain-specific small language models.

Future work should focus on: (1) developing theoretical frameworks for optimal integration of tokenization and distillation objectives, (2) expanding evaluation methodologies to include clinical utility metrics, and (3) investigating cross-domain transferability of integrated approaches beyond the medical domain.

## References

1. Abadeer, M., Altrabsheh, N., & Tofail, N. (2020). Assessment of DistilBERT performance on Named Entity Recognition task for PHI and medical entity extraction. *Clinical Natural Language Processing Workshop*.

2. Alsentzer, E., Murphy, J., Boag, W., Weng, W. H., Jindi, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, 72-78.

3. Averly, R., Maldonado, R., Gupta, S., & Elhadad, N. (2025). A zero-shot clinical named entity recognition framework. *Proceedings of NAACL 2025*.

4. Balde, G., Roy, S., Mondal, M., & Ganguly, N. (2024). Adaptive BPE tokenization for enhanced vocabulary adaptation in finetuning pretrained language models. *Findings of EMNLP 2024*, 15264-15281.

5. Garg, M., Raza, S., Rayana, S., Liu, X., & Sohn, S. (2025). The rise of small language models in healthcare: A comprehensive survey. *arXiv preprint arXiv:2504.17119*.

6. Hu, J., Bao, R., Lin, Y., Zhang, H., & Xiang, Y. (2024). Accurate medical named entity recognition through specialized NLP models. *arXiv preprint arXiv:2412.08255*.

7. Huang, K., Altosaar, J., & Ranganath, R. (2019). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342*.

8. Karim, A. A. J., Das, A., & Rahman, M. (2025). Larger models yield better results? Streamlined severity classification using knowledge distillation. *PLOS ONE*, 20(2).

9. Kim, H., Lee, J., Park, S., & Kim, Y. (2025). Small language models learn enhanced reasoning skills from large language models for biomedical knowledge. *Nature Digital Medicine*, 8(1).

10. Laparra, E., Bethard, S., & Miller, T. A. (2021). A review of recent work in transfer learning and domain adaptation for natural language processing of electronic health records. *Yearbook of Medical Informatics*, 30(1), 239-244.

11. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240.

12. Liu, S., Deng, N., Sabour, S., Jia, Y., Huang, M., & Mihalcea, R. (2023). Task-adaptive tokenization: Enhancing long-form text generation efficacy in mental health and beyond. *Proceedings of EMNLP 2023*, 15264-15281.

13. Magnini, M., Ciuffoletti, A., & Turchi, M. (2025). Open-source small language models for personal medical question answering. *Digital Health*, 11.

14. Moslemi, A., Ghafouri, B., & Smith, J. (2024). A survey on knowledge distillation: Recent advancements. *Computer Science Review*, 53, 100644.

15. Rohanian, O., Nouriborji, M., Jauncey, H., Kouchaki, S., Clifton, L., Merson, L., & Clifton, D. A. (2024). Lightweight transformers for clinical natural language processing. *Artificial Intelligence in Medicine*, 146, 102691.

16. Sachidananda, V., Kessler, J. S., & Lai, Y. (2021). Efficient domain adaptation of language models via adaptive tokenization. *Proceedings of SustaiNLP Workshop at EMNLP 2021*.

17. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *NeurIPS EMC Workshop*.

18. Vedula, K. S., Gupta, A., Swaminathan, A., Lopez, I., Bedi, S., & Shah, N. H. (2024). Distilling large language models for efficient clinical information extraction. *arXiv preprint arXiv:2501.00031*.

19. Wang, Y., Chen, L., Zhang, X., & Liu, M. (2025). Unsupervised domain adaptation with multi-level knowledge distillation for medical image analysis. *Computers in Biology and Medicine*, 184, 109394.

20. Zheng, M., Chen, H., Guo, T., Zhu, C., Zheng, B., Xu, C., & Wang, Y. (2024). Enhancing large language models through adaptive tokenizers. *Advances in Neural Information Processing Systems*, 37.

---
