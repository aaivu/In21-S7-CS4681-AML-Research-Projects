# Literature Review: Multi-hop Question Answering with Hybrid Retrieval

**Student:** 210621R  
**Research Area:** NLP:Question Answering  
**Date:** 2025-09-01

## Abstract

This literature review explores the landscape of multi-hop question answering systems, focusing on document retrieval mechanisms that combine dense neural representations with sparse lexical matching. Key areas covered include multi-hop QA architectures, dense retrieval methods using pre-trained language models, traditional sparse retrieval approaches, and hybrid retrieval systems. The review reveals that while dense retrievers excel at semantic understanding, they struggle with exact matching of rare entities and specific constraints - critical elements in multi-hop reasoning. Sparse retrievers like BM25 provide complementary strengths through precise lexical matching but lack semantic generalization. Recent work suggests that hybrid approaches combining both paradigms can achieve superior performance, though their application to multi-hop QA remains underexplored.

## 1. Introduction

Multi-hop question answering represents a challenging frontier in natural language understanding, requiring systems to retrieve and reason over multiple interconnected documents to synthesize coherent answers. Unlike traditional single-hop QA, these systems must identify bridging entities that connect disparate information sources, making the retrieval task particularly complex. Recent advances in dense neural retrieval have shown promise, but face inherent limitations in matching exact lexical patterns and rare entities - characteristics frequently essential for establishing connections in multi-hop reasoning scenarios.

## 2. Search Methodology

### Search Terms Used

- "multi-hop question answering"
- "dense passage retrieval"
- "hybrid information retrieval"
- "sparse text retrieval"
- "neural information retrieval"
- "BM25 retrieval"
- "document ranking"
- "semantic search"
- "cross-document reasoning"
- Variations with: "multi-document", "cross-passage", "multi-step"

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] Other: OpenReview, Papers With Code

### Time Period

2018-2025, with focus on developments since the introduction of HotpotQA dataset (2018)

## 3. Key Areas of Research

### 3.1 Multi-hop Question Answering Systems

The introduction of HotpotQA by Yang et al. (2018) established a comprehensive benchmark for multi-hop QA, featuring questions requiring reasoning over multiple Wikipedia articles. This dataset sparked significant research into architectures specifically designed for multi-document reasoning.

**Key Papers:**

- Yang et al. (2018) - Introduced HotpotQA dataset with explicit supervision for supporting facts
- Xiong et al. (2021) - Developed Multi-hop Dense Retrieval (MDR) framework using iterative retrieval
- Min et al. (2019) - Proposed question decomposition approach for complex reasoning
- Asai et al. (2020) - Introduced graph-based reasoning over Wikipedia structure

### 3.2 Dense Retrieval Methods

Dense retrieval approaches leverage pre-trained language models to encode queries and documents into fixed-dimensional vector spaces. These methods excel at capturing semantic relationships but may struggle with precise matching.

**Key Papers:**

- Karpukhin et al. (2020) - Dense Passage Retrieval (DPR) with dual encoders
- Liu et al. (2019) - RoBERTa model serving as backbone for many retrievers
- Xiong et al. (2020) - ANCE with approximate nearest neighbor negative sampling
- Khattab & Zaharia (2020) - ColBERT's late interaction approach

### 3.3 Sparse and Traditional Retrieval

Traditional sparse retrieval methods, particularly BM25, remain competitive baselines due to their precise lexical matching capabilities and computational efficiency.

**Key Papers:**

- Robertson & Zaragoza (2009) - Comprehensive overview of BM25 framework
- Thakur et al. (2021) - BEIR benchmark showing BM25's robustness
- Formal et al. (2021) - SPLADE combining sparsity with neural learning
- Dai & Callan (2020) - Context-aware term importance estimation

### 3.4 Hybrid Retrieval Approaches

Recent work explores combining dense and sparse signals to leverage their complementary strengths, though primarily in single-hop scenarios.

**Key Papers:**

- Luan et al. (2021) - Score interpolation for first-stage ranking
- Wang et al. (2022) - E5 embeddings with hybrid training objectives
- Lin et al. (2021) - Dynamic weighting in conversational search

## 4. Research Gaps and Opportunities

Through this literature review, several key research gaps have been identified in the current state of multi-hop QA systems:

### Gap 1: Limited Integration of Retrieval Paradigms

**Why it matters:** While hybrid retrieval has shown promise in single-hop scenarios, its application to multi-hop QA remains underexplored. The unique challenges of multi-hop retrieval, including entity-centric reasoning and bridging information, suggest potential benefits from combining complementary retrieval approaches.

**How this research addresses it:** Developing a hybrid retrieval framework specifically optimized for multi-hop QA, with weighted score combination and adaptive parameter tuning.

### Gap 2: Efficiency-Effectiveness Trade-off

**Why it matters:** Current dense retrieval methods require significant computational resources, while sparse methods may miss semantic connections. A balanced approach is needed for practical applications.

**How this research addresses it:** Proposing a lightweight hybrid architecture that leverages existing indices and pre-computed embeddings to minimize computational overhead.

## 5. Theoretical Framework

The theoretical foundation for this research combines:

1. Probabilistic relevance framework underlying BM25
2. Neural information retrieval principles from dense encoders
3. Multi-hop reasoning chains for complex question answering
4. Score normalization and fusion techniques from information retrieval

## 6. Methodology Insights

Common methodologies in the field include:

1. Dual encoder architectures with contrastive learning
2. Iterative retrieval with query reformulation
3. Graph-based reasoning over document connections
4. Hybrid score combination through weighted interpolation

Most promising approaches leverage:

- Pre-trained language models for semantic understanding
- Efficient exact matching through inverted indices
- Explicit modeling of multi-hop reasoning paths
- Balanced integration of multiple retrieval signals

## 7. Literature Review Scope and Updates

This review has the following scope:

- Literature review focused on papers from 2018-2025
- Emphasis on top-tier conferences and journals
- Regular updates planned as new relevant work emerges

## References

1. Yang, Z., et al. (2018). "HotpotQA: A dataset for diverse, explainable multi-hop question answering." EMNLP 2018.
2. Xiong, X., et al. (2021). "Answering complex open-domain questions with multi-hop dense retrieval." ICLR 2021.
3. Karpukhin, V., et al. (2020). "Dense passage retrieval for open-domain question answering." EMNLP 2020.
4. Robertson, S., & Zaragoza, H. (2009). "The probabilistic relevance framework: BM25 and beyond." Found. Trends Inf. Retr.
5. Liu, Y., et al. (2019). "RoBERTa: A robustly optimized BERT pretraining approach." arXiv:1907.11692.
6. Luan, S., et al. (2021). "Sparse dense hybrid dense retrieval." SIGIR 2021.
7. Wang, L., et al. (2022). "Text embeddings by weakly-supervised contrastive pre-training." arXiv:2212.03533.
8. Min, S., et al. (2019). "Multi-hop reading comprehension through question decomposition and rescoring." ACL 2019.
9. Asai, A., et al. (2020). "Learning to retrieve reasoning paths over wikipedia graph for question answering." ICLR 2020.
10. Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT." SIGIR 2020.
11. Thakur, N., et al. (2021). "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models." NeurIPS 2021.
12. Formal, T., et al. (2021). "SPLADE: Sparse lexical and expansion model for first stage ranking." SIGIR 2021.
13. Lewis, P., et al. (2020). "Retrieval-augmented generation for knowledge-intensive NLP tasks." NeurIPS 2020.
14. Xiong, L., et al. (2020). "Approximate nearest neighbor negative contrastive learning for dense text retrieval." ICLR 2021.
15. Lin, S., et al. (2021). "In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval." SCAI 2021.

---

**Notes:**

- Literature review focused on papers from 2018-2025
- Emphasis on conference publications (EMNLP, ICLR, SIGIR, NeurIPS)
- Core papers in multi-hop QA, dense retrieval, and hybrid approaches
- Regular updates planned as new relevant work emerges