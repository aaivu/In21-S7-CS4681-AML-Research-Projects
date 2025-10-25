# Methodology: NLP:Question Answering

**Student:** 210190R
**Research Area:** NLP:Question Answering
**Date:** 2025-09-01

## 1. Overview

This project investigates multi-hop question answering capabilities within unified frameworks, focusing on extending UNIFIEDQA for compositional reasoning across multiple passages. The objective is to develop UNIFIEDQA-MH, a fine-tuned variant that maintains strong single-hop performance while significantly improving multi-hop reasoning without architectural modifications. The study evaluates the effect of retrieval-augmented fine-tuning, multi-hop training data, and sequence-to-sequence adaptation on model performance across diverse QA formats.

## 2. Research Design

The research follows an experimental design with the following phases:

1. **Baseline establishment**: Evaluate original UNIFIEDQA on both single-hop and multi-hop datasets to establish performance baselines.

2. **Multi-hop adaptation**: Fine-tune UNIFIEDQA on HotpotQA with integrated retrieval mechanism to handle context window limitations.

3. **Comparative evaluation**: Analyze performance differences between baseline and improved setups using Exact Match (EM) and F1 metrics across diverse QA datasets.

4. **Generalization analysis**: Assess whether multi-hop fine-tuning compromises single-hop capabilities through comprehensive cross-dataset evaluation.

This design ensures controlled comparison under identical conditions, isolating the effects of multi-hop fine-tuning strategies.

## 3. Data Collection

### 3.1 Data Sources

- HotpotQA — primary multi-hop benchmark with Wikipedia-based questions requiring reasoning across multiple passages
- UNIFIEDQA evaluation suite — including BoolQ, ARC-Easy, ARC-Hard, CommonsenseQA, SQuAD2 for single-hop evaluation

### 3.2 Data Description

- **HotpotQA**: 113k question-answer pairs with supporting facts and reasoning chains
- **BoolQ**: 9.4k yes/no questions based on Wikipedia passages
- **ARC-Easy/ARC-Hard**: 7.8k science exam questions with difficulty stratification  
- **CommonsenseQA**: 12.1k questions requiring commonsense reasoning
- **SQuAD2**: 150k extractive QA pairs with unanswerable questions

### 3.3 Data Preprocessing

- Questions and contexts concatenated into single input sequences
- Text normalization: lowercase conversion, punctuation standardization, whitespace cleaning
- Context retrieval using SBERT embeddings when input exceeds 512 tokens
- Sequence length management through token-level truncation/padding
- Answer formatting normalization for consistent evaluation

## 4. Model Architecture

The proposed model extends UNIFIEDQA with multi-hop capabilities:

- **Base Architecture**: T5-based sequence-to-sequence transformer
- **Input Processing**: Concatenated question + retrieved context (up to 512 tokens)
- **Retrieval Component**: SentenceTransformer (all-MiniLM-L6-v2) for relevance scoring
- **Training Objective**: Standard sequence-to-sequence cross-entropy loss
- **Output Generation**: Direct answer generation in natural language

Multi-hop enhancements:
- **Retrieval Integration**: Dynamic context selection based on question similarity
- **Multi-hop Fine-tuning**: Exposure to compositional reasoning during training
- **Unified Formulation**: Single model handling both single-hop and multi-hop tasks
- **Efficiency Optimization**: Focused context attention within token limits

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Exact Match (EM)**: Primary metric for answer accuracy
- **F1-score**: Token-overlap based scoring for partial matches 

### 5.2 Baseline Models

| Model | Description | Training Data | Key Features |
|-------|-------------|---------------|--------------|
| UNIFIEDQA | Original T5-based unified QA | Multiple single-hop datasets | Format-agnostic, no multi-hop focus |
| UNIFIEDQA-MH | Our multi-hop extension | HotpotQA + original data | Retrieval augmentation, multi-hop fine-tuning |

### 5.3 Hardware/Software Requirements

| Category | Specification |
|----------|---------------|
| Frameworks | PyTorch, Transformers, SentenceTransformers |
| Libraries | HuggingFace, NumPy, Pandas, Scikit-learn |
| Pretrained Models | UNIFIEDQA-large, all-MiniLM-L6-v2 |
| Evaluation | Official dataset evaluation scripts |

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Dataset preparation & baseline evaluation | 2 weeks | Preprocessed datasets, baseline results |
| Phase 2 | Retrieval integration & multi-hop fine-tuning | 3 weeks | UNIFIEDQA-MH implementation |
| Phase 3 | Comprehensive evaluation across datasets | 2 weeks | Performance comparison results |
| Phase 4 | Analysis, visualization & paper preparation | 1 week | Final report, plots, and publication |

## 7. Risk Analysis

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| Context window limitations | High | SBERT retrieval for relevant segment selection |
| Multi-hop reasoning complexity | Medium | Gradual fine-tuning with retrieval support |
| Single-hop performance degradation | Medium | Careful multi-hop training with single-hop preservation |
| Computational resource constraints | Medium | Efficient batching, gradient accumulation |

## 8. Expected Outcomes

- A robust UNIFIEDQA-MH model demonstrating significant multi-hop QA improvements (+8.8% EM on HotpotQA)
- Maintained or improved performance across diverse single-hop QA formats
- A reproducible framework for extending unified models to compositional reasoning tasks
- Comprehensive analysis of multi-hop fine-tuning effects on model capabilities

---

**Note:** Update this document as your methodology evolves during implementation.