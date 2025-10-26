# Research Proposal: Multi-hop Question Answering with Hybrid Retrieval

**Student:** 210621R  
**Research Area:** NLP:Question Answering  
**Date:** 2025-09-01

## 1. Introduction

Multi-hop question answering represents a challenging frontier in natural language understanding, requiring systems to retrieve and reason over multiple interconnected documents. Current approaches primarily rely on dense neural retrievers, which face limitations in matching exact lexical patterns and rare entities. This research explores the potential of combining dense and sparse retrieval paradigms to improve document retrieval accuracy in multi-hop reasoning scenarios.

## 2. Problem Statement

While dense neural retrievers have shown promising results in multi-hop QA, they struggle with:
1. Exact matching of rare entities and specific constraints
2. Precise temporal and numerical comparisons
3. Computational efficiency at scale

This research addresses these limitations by developing an efficient hybrid retrieval framework that combines the semantic understanding of dense retrievers with the precision of sparse lexical matching.

## 3. Literature Review Summary

Recent advances in multi-hop QA have focused on dense retrieval methods, with the Multi-hop Dense Retrieval (MDR) framework achieving state-of-the-art results. However, analysis reveals persistent challenges in exact matching and computational efficiency. Traditional sparse methods like BM25 demonstrate complementary strengths in lexical precision but lack semantic understanding. The potential of combining these approaches for multi-hop QA remains unexplored.

## 4. Research Objectives

### Primary Objective

Develop and evaluate a hybrid retrieval framework that improves multi-hop QA accuracy by combining dense neural representations with sparse lexical matching, while maintaining computational efficiency.

### Secondary Objectives

1. Design a normalized weighted scoring mechanism for combining dense and sparse retrieval signals
2. Implement an efficient hybrid retrieval pipeline with minimal latency overhead
3. Analyze the effectiveness of hybrid retrieval across different question types and document characteristics
4. Identify optimal parameter settings for balancing semantic and lexical matching

## 5. Methodology

1. System Development
   - Implement dense retrieval using RoBERTa-based encoders
   - Set up BM25 sparse retrieval with efficient indexing
   - Develop normalized score combination mechanism

2. Experimental Setup
   - Dataset: HotpotQA (112,779 training, 7,405 validation samples)
   - Hardware: NVIDIA V100 GPU, 256GB RAM
   - Metrics: Top-k accuracy, MRR, Recall@k

3. Evaluation Strategy
   - Compare against MDR baseline
   - Conduct ablation studies
   - Perform detailed error analysis

## 6. Expected Outcomes

1. Technical Contributions
   - Novel hybrid scoring mechanism
   - Efficient implementation strategy
   - Comprehensive error analysis framework

2. Performance Improvements
   - 3-4% gain in top-1 retrieval accuracy
   - 2-3% gain in top-10 accuracy
   - Improved ranking quality (MRR)

3. Research Insights
   - Understanding of retrieval paradigm complementarity
   - Characterization of question types benefiting from hybrid approach
   - Guidelines for balancing semantic and lexical matching

## 7. Timeline

| Week | Task | Deliverables |
|------|------|-------------|
| 1-2 | Literature review and system design | Design document |
| 3-4 | Dense retriever implementation | Working dense retrieval system |
| 5-6 | Sparse retriever integration | BM25 implementation |
| 7-8 | Hybrid scoring mechanism | Score combination module |
| 9-10 | Parameter tuning and optimization | Optimized configuration |
| 11-12 | Comprehensive evaluation | Results and analysis |
| 13-14 | Error analysis and refinement | Detailed analysis report |
| 15-16 | Paper writing and submission | Research paper draft |

## 8. Resources Required

1. Computational Resources
   - NVIDIA V100 GPU
   - 256GB RAM
   - 5GB storage for indices

2. Software and Libraries
   - PyTorch
   - Transformers library
   - rank_bm25
   - FAISS

3. Datasets
   - HotpotQA
   - Wikipedia corpus (5.23M paragraphs)

## References

1. Yang, Z., et al. (2018). "HotpotQA: A dataset for diverse, explainable multi-hop question answering." EMNLP 2018.
2. Xiong, X., et al. (2021). "Answering complex open-domain questions with multi-hop dense retrieval." ICLR 2021.
3. Robertson, S., & Zaragoza, H. (2009). "The probabilistic relevance framework: BM25 and beyond." Found. Trends Inf. Retr.
4. Karpukhin, V., et al. (2020). "Dense passage retrieval for open-domain question answering." EMNLP 2020.
5. Luan, S., et al. (2021). "Sparse dense hybrid retrieval." SIGIR 2021.

