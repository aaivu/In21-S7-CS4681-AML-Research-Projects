# Research Proposal: NLP:Retrieval-Augmented Generation (RAG)

**Student:** 210063G  
**Research Area:** NLP: Retrieval-Augmented Generation (RAG)  
**Date:** 2025-10-20


## Abstract

This research proposes the development and evaluation of an Agentic LightRAG framework—a multi-agent orchestration system designed to enhance Retrieval-Augmented Generation capabilities for complex, multi-hop question answering. Current RAG systems employ static, single-pass retrieval pipelines that fail when confronted with intricate queries requiring reasoning across multiple information sources. By integrating cognitive architecture principles and multi-agent coordination, the proposed framework introduces query decomposition, specialized retrieval agents operating at multiple abstraction levels, and intelligent answer synthesis. The system will be evaluated against established baselines using the HotpotQA benchmark and standard evaluation metrics. Expected outcomes include quantified performance improvements (targeting 15-25% enhancement in answer correctness), interpretable reasoning traces that enhance system transparency, and a reusable architectural blueprint for agentic RAG systems. This work addresses critical limitations in current knowledge-intensive AI systems and contributes to building more robust, reliable, and human-aligned information retrieval systems.


## 1. Introduction

The emergence of Large Language Models (LLMs) has fundamentally transformed natural language processing, demonstrating unprecedented capabilities in understanding and generating human language. However, LLMs face two critical constraints that limit their real-world applicability. First, their knowledge is inherently static, bounded by their training data cutoff date, making them unsuitable for applications requiring current information. Second, LLMs are notoriously susceptible to "hallucination"—generating plausible but factually incorrect information with high confidence, undermining trust in mission-critical applications such as medical, legal, and financial domains.

Retrieval-Augmented Generation (RAG) emerged as an elegant solution to these constraints. RAG systems dynamically ground LLMs by retrieving relevant, up-to-date information from external knowledge sources before generation, thereby providing factual context that mitigates hallucination and ensures information timeliness. RAG has become the de facto standard for building reliable, knowledge-intensive applications and represents a critical research frontier.

However, the dominant "naive RAG" paradigm—characterized by a linear, single-shot retrieve-then-generate workflow—reveals significant brittleness when facing real-world query complexity. This simple pipeline operates under the assumption that a single, initial retrieval suffices to answer any query, an assumption that breaks down for complex, decomposable queries requiring synthesis from multiple disparate sources; multi-hop (sequential) queries where prerequisite sub-questions must be answered first; and ambiguous queries requiring clarification before effective retrieval.

Recent advances in agentic AI suggest that by treating information retrieval as a "tool" that agents can dynamically control, and by structuring the RAG process around human-like cognitive principles—planning, execution, and synthesis—systems can achieve substantially improved performance on these challenging query types. This proposal introduces Agentic LightRAG, a framework that operationalizes these principles through coordinated multi-agent architectures designed specifically for complex question answering.



## 2. Problem Statement

The research addresses the following core problem: **How can Retrieval-Augmented Generation systems be designed to intelligently handle complex, multi-hop, and ambiguous queries that require reasoning across multiple information sources and adaptive retrieval strategies?**

More specifically, the research tackles several interconnected sub-problems:

**Limitation 1: Rigid Information Flow**: Conventional RAG systems operate unidirectionally without feedback mechanisms. When initial retrieval fails to identify relevant documents, the system lacks recourse beyond poor context generation. This rigid, non-adaptive pipeline is fundamentally incapable of recognizing and recovering from retrieval failures.

**Limitation 2: Inability to Decompose Complex Queries**: Non-agentic RAG systems treat every query as a monolithic request for information. When confronted with a complex question like "Compare the economic policies of Country A with the environmental policies of Country B," the system cannot recognize that this represents at least four distinct information needs (policies of A, policies of B, economic aspects, environmental aspects) that should be resolved independently and then synthesized. This monolithic approach results in noisy, irrelevant context that confuses LLM generation.

**Limitation 3: Sequential Dependency Failure**: Multi-hop questions represent a particularly challenging query class where information required for subsequent reasoning steps depends on answers to prior steps. For example, "Who is the current head of government in the country where the company that developed GPT-3 is headquartered?" requires a three-step logical sequence: (1) identify the company, (2) identify its headquarters country, (3) identify the country's head of government. Non-agentic systems cannot model these dependencies and typically attempt to retrieve all required information simultaneously, leading to semantic confusion and systematic retrieval failure.

**Limitation 4: Lack of Interpretable Reasoning**: Current RAG systems provide limited transparency into their reasoning processes. Users receive answers without understanding how information was gathered, evaluated, or synthesized. This opacity undermines trust, particularly in high-stakes domains, and complicates error diagnosis and system improvement.

**Limitation 5: Suboptimal Performance Trade-offs**: When simple RAG systems fail, practitioners face a stark choice: accept poor performance or resort to expensive model fine-tuning. No principled, modular approach exists for improving performance through better reasoning and retrieval coordination without extensive retraining.

This research proposes that by structuring RAG systems around agentic principles inspired by human cognition, these limitations can be substantially mitigated, leading to more robust, interpretable, and effective information systems.


## 3. Literature Review Summary

Recent literature reveals a field in rapid evolution from foundational single-pass RAG to sophisticated agentic systems. The progression demonstrates clear performance benefits but also identifies critical gaps.

**Evolution of RAG Approaches**:
The RAG landscape progresses through three stages: Naive RAG (Lewis et al., 2020) provides the foundational retrieve-then-generate pipeline; Advanced RAG incorporates optimizations at pre-retrieval (query expansion), retrieval (hybrid search), and post-retrieval (re-ranking) phases; and Agentic RAG treats retrieval as a dynamic tool within LLM-based reasoning agents. Current state-of-the-art includes Self-RAG (Asai et al., 2023), which fine-tunes LLMs to generate reflection tokens assessing retrieval necessity and passage relevance, and FLARE (Jiang et al., 2023), which triggers active retrieval when generation confidence drops.

**Agentic AI Patterns**:
Foundational agentic patterns include ReAct (Yao et al., 2022), demonstrating that LLMs can solve complex tasks by interleaving reasoning traces with actions; Plan-and-Solve (Wang et al., 2023), showing benefits of explicit multi-step planning; and Self-Correction (Madaan et al., 2023), demonstrating value of agents critiquing their own outputs. These patterns provide proven building blocks for complex reasoning.

**Multi-Hop Question Answering**:
HotpotQA (Yang et al., 2018) established multi-hop QA as a critical evaluation domain. Research shows that questions requiring synthesis across multiple documents remain challenging for current systems and represent a key frontier for improvement.

**Key Research Gaps**:
1. Limited framework explicitly combining planning, execution, and synthesis components for RAG
2. Insufficient exploration of multi-agent coordination mechanisms for information retrieval
3. Lack of approaches balancing improved reasoning with practical latency constraints
4. Underexplored integration of graph-based retrieval with agentic coordination
5. Limited analysis of interpretability improvements and their user impact

The literature demonstrates clear momentum toward agentic approaches but reveals gaps in modular architectures that balance interpretability, performance, and practical deployment. This research fills these gaps through a comprehensive framework combining cognitive architecture principles with multi-agent coordination.


## 4. Research Objectives

### Primary Objective

**Design, implement, and evaluate an Agentic LightRAG framework that achieves substantial performance improvements on multi-hop question answering tasks through principled multi-agent coordination inspired by human cognitive processes, while maintaining interpretability and practical feasibility for deployment.**

### Secondary Objectives

- **Develop a modular multi-agent architecture** with specialized agents for query decomposition, dual-level retrieval, and answer synthesis that can be independently optimized and evaluated, providing reusable components for future RAG systems.

- **Achieve quantified performance improvements** targeting 15-25% enhancement in Answer Correctness and 5-15% improvement in Response Relevancy compared to strong non-agentic baselines, demonstrating the practical value of agentic decomposition for complex queries.

- **Create interpretable reasoning traces** that provide transparent insight into system decision-making, enabling users to understand information retrieval strategies, understand agent reasoning, and identify failure modes.

- **Establish practical performance-latency trade-offs** by measuring end-to-end latency, identifying bottlenecks, and proposing optimization strategies that enable deployment of agentic systems within reasonable response time constraints.

- **Generate comprehensive architectural documentation** detailing design decisions, implementation patterns, and learned lessons that provide a blueprint for future agentic RAG research and facilitate reproducibility and extension by other researchers.

- **Conduct comparative analysis** of when agentic decomposition provides advantages and when simpler approaches suffice, providing practitioners with decision criteria for choosing appropriate approaches for different query types.


## 5. Methodology

The research employs a systematic, iterative design science methodology combining architecture-driven development with empirical evaluation:

**Phase 1: Architecture Design (Weeks 5-6)**: Complete formalization of the agentic LightRAG architecture specifying four agent types, coordination mechanisms, state management, and inter-agent communication protocols. Review against cognitive architecture principles and design patterns from literature.

**Phase 2: Component Implementation (Weeks 7-10)**: Iteratively implement each agent component—Query Decomposition Agent for strategic planning, Low-Level and High-Level Retrieval Agents for tactical operations, Answer Synthesis Agent for result generation—with unit testing and integration testing at each stage.

**Phase 3: System Integration (Weeks 9-10)**: Integrate agents with graph-based knowledge index, implement communication protocols, and establish state management. Validate end-to-end system functionality on sample queries.

**Phase 4: Empirical Evaluation (Weeks 11-14)**: Conduct comprehensive evaluation on HotpotQA benchmark (200 test queries) using RAGAs framework. Measure Answer Correctness, Response Relevancy, Context Precision/Recall, and latency. Compare against Naive RAG and LightRAG baselines.

**Phase 5: Analysis & Reporting (Weeks 13-14)**: Conduct statistical analysis of results, perform case studies examining system behavior on representative queries, analyze failure modes by query type, and document insights about agentic decomposition effectiveness.

**Evaluation Metrics**: Answer Correctness (semantic similarity with ground truth), Response Relevancy (relevance of answer to query), Context Precision/Recall (retrieval efficiency), Query Success Rate (proportion of answerable queries), Latency (end-to-end response time), Reasoning Quality (interpretability of agent traces).

**Baselines**:LightRAG (non-agentic graph-based retrieval)



## 6. Expected Outcomes

**Technical Outcomes**:
- A fully functional multi-agent agentic RAG system demonstrating coordinated agent operations
- Quantified performance improvements on HotpotQA benchmark (target: 15-25% Answer Correctness improvement)
- Comprehensive reasoning traces for all system operations enabling full interpretability
- Detailed performance analysis documenting latency-accuracy trade-offs

**Research Contributions**:
- Novel modular multi-agent architecture for agentic RAG with reusable components
- Empirical validation on established benchmarks demonstrating framework effectiveness
- Clear documentation of when agentic decomposition provides advantages
- Foundation for future research on hybrid agentic approaches and cross-domain generalization

**Knowledge Contributions**:
- Practical understanding of cognitive architecture principles applied to AI systems
- Insights into multi-agent coordination mechanisms for information retrieval
- Analysis of interpretability improvements from explicit reasoning traces
- Guidelines for practitioners choosing between agentic and non-agentic approaches

**Documentation Deliverables**:
- Comprehensive final project report (6-8 pages) including all research components
- Complete system architecture documentation with implementation details
- Case studies demonstrating system capabilities on complex queries
- Analysis of limitations and recommendations for future research



## 7. Timeline

| Week | Task |
|------|------|
| 5-6  | Finalize agentic LightRAG architecture design; Set up development environment; Configure data preprocessing and indexing infrastructure |
| 7-8  | Implement Query Decomposition Agent and Low-Level/High-Level Retrieval Agents; Unit testing and validation |
| 9-10 | Integrate agents with graph-based knowledge index; Develop inter-agent communication protocols; Implement state management layer |
| 11-12| Implement Answer Synthesis Agent; Conduct initial experiments on 100-150 test questions; Debugging and performance tuning |
| 13-14| Complete final evaluation on full test set (200 queries); Comprehensive result analysis; Case studies and failure analysis |



## 8. Resources Required

**Software and Libraries**:
- Python 3.10+ 
- LangChain for LLM orchestration and agent management
- OpenAI API access for GPT-4o-mini and text-embedding-3-large
- NetworkX for graph operations
- RAGAs evaluation framework for standardized metrics
- Git, Jupyter, and standard development tools

**Datasets**:
- HotpotQA dataset (open-source, publicly available)

**External Services**:
- OpenAI API subscription for LLM and embedding services (cost approximately $10 for experimental evaluation)
- GitHub for version control and project hosting

**Knowledge Prerequisites**:
- Strong understanding of NLP, embeddings, and language models
- Familiarity with RAG systems and retrieval techniques
- Programming proficiency in Python
- Understanding of agent-based systems and multi-agent architectures


## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Nogueira, R., He, H., Chen, D., Yih, W., & Komeili, M. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

2. Thayasivam, U., & Bandara, K. (2025). CogniRAG: A human-cognition-inspired agentic framework for enhancing retrieval-augmented generation. *Proceedings of the 2025 International Conference on Computational Linguistics*, Department of Computer Science and Engineering, University of Moratuwa.

3. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *arXiv preprint arXiv:2310.11511*.

4. Jiang, Z., Liska, M., Gu, L., Gu, R., Yavuz, S., Luan, D., Zhao, J., Yao, S., & Du, N. (2023). Active retrieval augmented generation. *arXiv preprint arXiv:2305.06983*.

5. Yao, S., Zhao, J., Yu, D., Du, N., Durmus, I., Liska, M., Gu, L., Luan, D., & Gu, R. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*.

6. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2369-2380.

7. Es, S., James, J., Laskar, S., Gerit, L., Vaidya, V., & Jacob, A. (2023). RAGAs: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.

8. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., & others. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

9. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1-38.

10. Luan, Y., Piktus, A., Cvicek, F., He, H., Gunaratna, K., Lee, K., Yih, W., & Chang, M. (2021). Sparse, dense, and attentive representations for text retrieval. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 5756-5771.

11. Nogueira, R., Yang, W., Cho, K., & Lin, J. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

12. Ma, X., Gong, Y., Jiang, A., Liu, P., Wang, H., Yan, B., & Chen, N. (2023). Query rewriting for retrieval-augmented large language models. *arXiv preprint arXiv:2305.14283*.

13. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

14. Kwiatkowski, T., Palomaki, J., Redfield, D., Collins, A., Parikh, A., Alberti, C., Epstein, D., Polevychenko, I., Kelcey, M., & Grave, E. (2019). Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7, 452-466.

15. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Guo, Q. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

16. Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., & others. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

17. Madaan, A., Tandon, N., Gupta, A., Halliday, P., Converse, A. K., Huang, Y., Sap, M., Rashkin, H., & Schlichtkrull, M. (2023). Self-refine: Iterative refinement with self-feedback. *arXiv preprint arXiv:2303.17651*.

18. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*.

19. Welbl, J., Levy, O., Dagan, I., & Schwartz, R. (2018). Constructing datasets for multi-hop reading comprehension across documents. *Transactions of the Association for Computational Linguistics*, 6, 285-297.

20. Min, S., Zhong, V., Soares, L., & Zettlemoyer, L. (2019). Compositional questions do not necessitate compositional structure of their answers. *arXiv preprint arXiv:1906.07381*.

