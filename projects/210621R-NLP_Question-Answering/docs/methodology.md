# Methodology: Multi-hop Question Answering with Hybrid Retrieval

**Student:** 210621R  
**Research Area:** NLP:Question Answering  
**Date:** 2025-09-01

## 1. Overview

This research proposes a hybrid retrieval framework for multi-hop question answering that combines dense neural representations with sparse lexical matching. The methodology focuses on developing and evaluating a normalized weighted scoring mechanism that integrates dense embeddings from pre-trained MDR models with BM25 lexical scores through weighted linear interpolation.

## 2. Research Design

Our approach follows an empirical methodology with the following key components:

1. Development of a hybrid scoring mechanism
2. Implementation of complementary retrieval systems
3. Comprehensive evaluation on benchmark dataset
4. Detailed error analysis and performance characterization

## 3. Data Collection

### 3.1 Data Sources

- HotpotQA dataset (Yang et al., 2018)
- Full Wikipedia corpus (5.23M paragraphs)
- Training set: 112,779 question-answer pairs
- Validation set: 7,405 questions with gold supporting document annotations

### 3.2 Data Description

- Multi-hop questions requiring reasoning over multiple documents
- Question types: Bridge and Comparison
- Supporting documents: 2+ Wikipedia paragraphs per question
- Full corpus: 5.23 million Wikipedia paragraphs

### 3.3 Data Preprocessing

1. Document indexing for BM25:
   - Construction of inverted index
   - Term frequency computation
   - Document length normalization

2. Dense embedding preparation:
   - RoBERTa tokenization
   - Document pre-encoding
   - Memory-mapped storage

## 4. Model Architecture

### 4.1 Dense Retrieval Component

- Model: RoBERTa-base encoders
- Query encoder: 768-dimensional embeddings
- Document encoder: 768-dimensional embeddings
- Training: Momentum-based contrastive learning
- Batch size: 128
- Learning rate: 2e-5

### 4.2 Sparse Retrieval Component

- Algorithm: BM25
- Parameters: k1=1.5, b=0.75
- Implementation: rank_bm25 library
- Index structure: Inverted index

### 4.3 Hybrid Scoring Function

- Min-max score normalization
- Weighted linear interpolation
- Adaptive parameter tuning
- Optimal α: 0.7 (dense-sparse balance)

## 5. Experimental Setup

### 5.1 Evaluation Metrics

1. Top-k Accuracy (k ∈ {1, 5, 10, 50})
2. Mean Reciprocal Rank (MRR)
3. Recall@k
4. Processing time per query

### 5.2 Baseline Models

1. Multi-hop Dense Retrieval (MDR)
2. BM25 baseline
3. Pure dense retrieval (α=1.0)
4. Pure sparse retrieval (α=0.0)

### 5.3 Hardware/Software Requirements

- CPU: Intel Xeon E5-2698v4
- RAM: 256GB
- GPU: NVIDIA V100
- Storage: 5GB for indices
- Frameworks: PyTorch, rank_bm25
- Dependencies: transformers, FAISS

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing, index building | 2 weeks | Processed corpus, indices |
| Phase 2 | Model implementation, score combination | 3 weeks | Working hybrid system |
| Phase 3 | Experiments, parameter tuning | 2 weeks | Results, optimal config |
| Phase 4 | Analysis, ablation studies | 1 week | Final report |

## 7. Risk Analysis

### Technical Risks

1. Computational scalability
   - Mitigation: Efficient indexing, batched processing
2. Memory constraints
   - Mitigation: Memory-mapped files, streaming
3. Performance degradation
   - Mitigation: Extensive parameter tuning

### Research Risks

1. Suboptimal integration
   - Mitigation: Comprehensive ablation studies
2. Domain generalization
   - Mitigation: Cross-dataset validation
3. Reproducibility challenges
   - Mitigation: Detailed documentation, public code

## 8. Expected Outcomes

1. Performance Improvements:
   - 3-4% gain in top-1 accuracy
   - 2-3% gain in top-10 accuracy
   - Improved MRR scores

2. Technical Contributions:
   - Novel hybrid scoring mechanism
   - Efficient implementation strategy
   - Comprehensive error analysis

3. Research Impact:
   - Guidelines for hybrid retrieval
   - Insights into retrieval paradigms
   - Open-source implementation

---

**Note:** This methodology will be refined based on experimental results and findings during implementation.