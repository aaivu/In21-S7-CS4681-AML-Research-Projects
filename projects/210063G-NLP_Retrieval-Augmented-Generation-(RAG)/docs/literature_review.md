# Literature Review: NLP - Retrieval-Augmented Generation (RAG)

**Student:** 210063G  
**Research Area:** NLP: Retrieval-Augmented Generation (RAG)  
**Date:** 2025-10-20


## Abstract

This literature review examines the landscape of Retrieval-Augmented Generation (RAG) systems, a critical technique for grounding Large Language Models in factual, external knowledge. The review traces the evolution from naive RAG through advanced optimization techniques to the current frontier of agentic RAG systems. Key findings indicate that while conventional RAG architectures achieve foundational improvements in factual accuracy, they struggle with complex, multi-hop reasoning tasks. Contemporary agentic frameworks that incorporate planning, decomposition, and self-critique mechanisms demonstrate substantial performance improvements. The review identifies critical research gaps in latency optimization, expanded tool integration, and hybrid agentic models, positioning these as key opportunities for future development.


## 1. Introduction

Retrieval-Augmented Generation (RAG) has emerged as a transformative approach to addressing fundamental limitations of Large Language Models (LLMs), particularly their propensity for hallucination and static knowledge constraints. RAG systems dynamically ground LLMs by providing relevant, up-to-date information retrieved from external knowledge sources before generation. This literature review examines the current state of RAG research, exploring the progression from foundational naive RAG approaches to sophisticated agentic frameworks that emulate human cognitive processes.

The scope of this review encompasses the theoretical foundations of RAG, existing system architectures, evaluation methodologies, current limitations, and emerging research directions. Special emphasis is placed on understanding how human-cognition-inspired agentic approaches can enhance RAG performance on complex, multi-hop reasoning tasks that challenge conventional pipeline-based systems.


## 2. Search Methodology

### Search Terms Used

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs) and Knowledge Grounding
- Agentic AI and Reasoning
- Multi-Hop Question Answering
- Query Decomposition and Planning
- Self-Correction and Reflection in LLMs
- Vector Retrieval and Semantic Search
- Graph-Based Retrieval Systems
- Information Extraction and Knowledge Bases
- Chain-of-Thought Reasoning
- Tool Use and LLM Agents
- Advanced Retrieval Optimization

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] NeurIPS, EMNLP, and ACL Conference Proceedings
- [x] GitHub and Open-Source Repositories

### Time Period

2018-2025, with emphasis on developments from 2020 onwards, particularly focusing on the emergence of agentic RAG systems in 2023-2025.


## 3. Key Areas of Research

### 3.1 Evolution and Stages of RAG

RAG research can be categorized into three distinct evolutionary stages:

**Naive RAG**: The foundational paradigm introduced by Lewis et al. (2020) consists of a simple, sequential pipeline. A user query is encoded, used to retrieve a fixed number of text chunks from a vector database via semantic search, and these chunks are concatenated with the original query as context for LLM generation. While transformative in addressing hallucination, its performance is highly sensitive to the quality of the initial retrieval.

**Advanced RAG**: This stage involves optimizations at pre-retrieval, retrieval, and post-retrieval phases. Pre-retrieval techniques include query expansion and rewriting. During retrieval, hybrid search methods combining keyword-based BM25 with semantic search are employed. Post-retrieval techniques include re-ranking with cross-encoders and context compression. These methods improve context relevance without altering the fundamental single-pass, non-reasoning nature.

**Agentic RAG**: The current frontier treats RAG as a dynamic, reasoning-driven task executed by an LLM-based agent. The agent uses retrieval as a tool, iteratively deciding what information is needed, retrieving it, and assessing sufficiency. This stage represents the focus of contemporary research.

**Key Papers:**
- Lewis et al., 2020 - Introduced foundational RAG framework for knowledge-intensive NLP tasks
- Zhang et al., 2023 - Query Rewriting for Retrieval-Augmented LLMs
- Gao et al., 2023 - Advancements in hybrid retrieval approaches
- Asai et al., 2023 - Self-RAG framework incorporating self-reflection mechanisms

### 3.2 Agentic AI Patterns and Frameworks

Several foundational agentic patterns have emerged that are directly applicable to RAG:

**ReAct (Reason and Act)**: Proposed by Yao et al. (2022), this pattern demonstrates that LLMs can solve complex tasks by interleaving reasoning traces (Thought) with actions (Act). The observation from actions feeds back into subsequent thought processes, creating a Thought → Action → Observation loop that forms the foundation for many agentic systems.

**Plan-and-Solve Pattern**: Involves agents creating explicit, multi-step plans before execution, decoupling strategic thinking from tactical execution and leading to more robust performance on complex tasks.

**Self-Correction and Reflection**: Frameworks like Self-RAG incorporate LLM-generated "reflection tokens" to assess retrieved passage relevance and generation quality, allowing models to decide whether to retrieve more information or refine answers.

**Tool Use**: The principle of augmenting LLMs with external tools, where retrieval systems themselves become primary tools and agent intelligence lies in learning when and how to use them effectively.

**Key Papers:**
- Yao et al., 2022 - ReAct: Synergizing Reasoning and Acting in Language Models
- Asai et al., 2023 - Self-RAG: Learning to Retrieve, Generate, and Critique
- Jiang et al., 2023 - FLARE: Active Retrieval Augmented Generation
- Thayasivam & Bandara, 2025 - CogniRAG: Human-Cognition-Inspired Agentic Framework

### 3.3 Multi-Hop and Complex Reasoning

Multi-hop question answering represents a critical challenge for RAG systems, requiring sequential reasoning across multiple pieces of information.

**Characteristics of Complex Queries:**
- Complex, decomposable queries requiring synthesis from multiple disparate sources
- Multi-hop (sequential) queries where answering prerequisite sub-questions is necessary
- Ambiguous or vague queries requiring clarification before effective retrieval

**Key Papers:**
- Yang et al., 2018 - HotpotQA: Dataset for Diverse, Explainable Multi-Hop Question Answering
- Welbl et al., 2018 - Constructing Datasets for Multi-Hop Reading Comprehension
- Min et al., 2019 - Compositional Questions Do Not Necessitate Compositional Structure

### 3.4 Knowledge Representation and Retrieval Infrastructure

Modern RAG systems employ sophisticated knowledge representations combining vector and graph-based approaches:

**Vector-Based Retrieval**: Semantic embeddings enable similarity-based retrieval. Models like text-embedding-3-large provide high-performance semantic representation.

**Graph-Based Retrieval**: Knowledge graphs capture relational structure, enabling structured querying and improved precision for entity-relationship questions.

**Hybrid Retrieval**: Combining dense vector retrieval with sparse lexical methods (BM25) improves both recall and precision.

**Key Papers:**
- Luan et al., 2021 - Sparse, Dense, and Attentive Representations for Text Retrieval
- Nogueira et al., 2019 - Passage Re-Ranking with BERT
- Devlin et al., 2018 - BERT: Pre-training of Deep Bidirectional Transformers

### 3.5 Evaluation Methodologies

Standardized evaluation frameworks are essential for comparing RAG systems:

**RAGAs Framework**: The open-source RAGAs framework provides automated, objective evaluation using metrics including:
- Answer Correctness: Measures factual accuracy by comparing against ground-truth answers using semantic similarity
- Response Relevancy: Evaluates conciseness and on-topic focus by generating potential questions from answers
- Context Precision and Recall: Assess retrieval quality

**Dataset Benchmarks:**
- HotpotQA: Multi-hop question answering with diverse reasoning requirements
- Natural Questions: Real user queries with Wikipedia-based answers
- MSMARCO: Large-scale Information Retrieval dataset

**Key Papers:**
- Es et al., 2023 - RAGAs: Automated Evaluation of Retrieval Augmented Generation
- Kwiatkowski et al., 2019 - Natural Questions: A Benchmark for Question Answering
- Nguyen et al., 2016 - MS MARCO: A Human Generated MAchine Reading COmprehension Dataset

### 3.6 Limitations and Challenges of Current RAG Systems

**Static Information Flow**: Non-agentic systems operate unidirectionally without feedback loops. When retrieval fails to find relevant documents, the system lacks recourse beyond poor context generation.

**Inability to Decompose**: Traditional RAG treats every query as monolithic, unable to recognize that complex questions represent multiple independent information needs.

**Failure on Sequential Dependencies**: Multi-hop questions depend on prior answers, but non-agentic systems attempt retrieval simultaneously, causing semantic confusion.

**Latency Trade-offs**: Agentic approaches involve multiple LLM calls, inherently slower than single-pass systems, posing challenges for real-time applications.

**LLM Dependency**: System performance heavily depends on the reasoning capabilities of underlying LLMs. Less capable models may struggle with effective planning and critique.

## 4. Research Gaps and Opportunities

### Gap 1: Latency Optimization in Agentic RAG

**Description:** While agentic RAG systems demonstrate superior performance on complex reasoning tasks, their iterative nature involving multiple LLM calls introduces significant latency compared to single-pass systems. This creates a performance-latency trade-off that limits real-world applicability for time-sensitive applications.

**Why it matters:** Real-time applications and user-facing systems require responses within strict latency constraints. Current agentic frameworks may be impractical for production deployments requiring sub-second response times. Reducing latency while maintaining reasoning quality is critical for adoption.

**How your project addresses it:** Investigate strategies to reduce LLM calls through more efficient planning mechanisms, such as enabling the Planner to create complex, multi-action plans simultaneously rather than sequentially, or employing smaller, distilled models for critique loops that maintain reasoning quality at lower computational cost.

### Gap 2: Limited Tool Ecosystem and Integration

**Description:** Most current agentic RAG systems are limited to a single tool (retrieval). While modular architectures theoretically support expansion, practical integration of diverse tools remains underdeveloped. Real-world complex queries often require computation, real-time web access, database queries, and code execution.

**Why it matters:** Many complex questions cannot be answered through retrieval alone. For example, comparative analysis across multiple sources may require computation, real-time data requires web search, and some queries benefit from structured database queries rather than semantic search. Limited tool availability severely constrains system capabilities.

**How your project addresses it:** Extend the Executor to use a wider array of tools beyond retrieval, including code interpreters for calculations, web search APIs for real-time information, and database query engines. Develop a meta-layer for tool selection and composition based on query analysis.

### Gap 3: Hybrid Planning and Fine-Grained Critique Mechanisms

**Description:** Current agentic approaches typically employ either explicit planning (like CogniRAG) or token-level self-correction (like Self-RAG), but rarely combine both. This creates a gap between coarse-grained strategic planning and fine-grained token-level quality control.

**Why it matters:** Combining explicit planning with fine-grained critique could enable systems to verify factual accuracy at the token level while maintaining overall strategic coherence. This hybrid approach could significantly improve both accuracy and efficiency by catching errors at multiple abstraction levels.

**How your project addresses it:** Explore architectures combining CogniRAG's explicit planning with Self-RAG's reflection tokens. Develop mechanisms to integrate token-level critique into the Planner's decision loop, allowing for both strategic and tactical error correction.

### Gap 4: Generalization Across Query Types and Domains

**Description:** Most current RAG systems are evaluated on specific benchmarks (primarily HotpotQA) or within particular domains. Generalization capabilities across diverse query types, domains, and knowledge sources remain poorly understood.

**Why it matters:** Real-world deployments must handle heterogeneous queries across domains without task-specific fine-tuning. Understanding how RAG systems generalize, and developing techniques to improve cross-domain performance, is critical for building robust, versatile systems.

**How your project addresses it:** Conduct comprehensive evaluation across multiple benchmarks and diverse domains. Analyze failure modes across different query types and develop adaptive mechanisms that adjust retrieval and reasoning strategies based on query characteristics.

### Gap 5: Interpretability and Explainability

**Description:** While agentic RAG provides greater interpretability through explicit reasoning traces, the quality of explanations to end-users remains underexplored. Many systems generate internal reasoning logs but struggle to translate these into clear, human-understandable explanations.

**Why it matters:** For high-stakes applications (legal, medical, financial), users need to understand not just answers but the reasoning behind them. Poor explanation quality undermines trust and adoption. Developing better explanation mechanisms is crucial for transparency.

**How your project addresses it:** Develop specialized explanation modules that transform technical reasoning traces into clear, natural language explanations. Implement interactive explanation systems allowing users to drill down into specific reasoning steps and question the system's decisions.


## 5. Theoretical Framework

RAG systems operate at the intersection of several theoretical domains:

**Information Retrieval Theory**: Builds on classical IR principles of relevance ranking, precision-recall trade-offs, and query reformulation. Modern RAG extends these with semantic similarity metrics and neural retrieval approaches.

**Cognitive Science**: The most recent advancements, particularly CogniRAG, draw inspiration from human problem-solving processes. The framework models human cognition through three stages: planning (understanding and strategy formation), execution (tactical operations), and synthesis (narrative formation). This cognitive inspiration suggests that better alignment with human reasoning processes may lead to more robust AI systems.

**Agent Theory and Control Flow**: Agentic RAG builds on established agent architectures from AI research, where agents maintain state, reason about goals, select actions, and observe outcomes. The ReAct framework formalizes this through the Thought-Action-Observation loop.

**LLM Prompting and In-Context Learning**: Current implementations leverage instruction-tuning and prompt engineering to guide LLM behavior. The effectiveness of agentic systems depends critically on the quality of prompts that structure the Planner, Executor, and Generator components.



## 6. Methodology Insights

### 6.1 Common Methodologies in RAG Research

**Empirical Evaluation**: Most RAG research employs empirical evaluation on benchmark datasets like HotpotQA, using automated metrics from frameworks like RAGAs. This approach provides quantitative comparison but may not capture all dimensions of system quality.

**Qualitative Analysis**: Researchers increasingly supplement quantitative metrics with qualitative analysis, examining specific failure cases and reasoning traces to understand system behavior and limitations.

**Ablation Studies**: Systematic removal of components helps isolate the contribution of specific mechanisms (e.g., knowledge refinement, self-correction) to overall performance.

**User Studies**: While less common in academic papers, some recent work includes user evaluations assessing whether systems' explanations are helpful and whether users trust the recommendations.

### 6.2 Most Promising Approaches for Your Work

**Agentic Architectures with Modular Design**: CogniRAG's three-component architecture (Planner, Executor, Generator) shows promise for handling complex reasoning while maintaining interpretability. This separation of concerns enables independent optimization of each component.

**Iterative Critique Mechanisms**: The critique loop where agents evaluate their own outputs before proceeding offers a principled approach to error correction. Integrating this into strategic planning decisions appears more effective than single-pass generation.

**Hybrid Retrieval Methods**: Combining vector-based semantic search with graph-based structured retrieval and lexical methods provides complementary strengths. Systems employing multiple retrieval modes outperform single-method approaches.

**Multi-Level Evaluation**: Using multiple evaluation metrics (answer correctness, relevancy, precision, recall) provides more complete assessment than any single metric. The RAGAs framework offers a promising standardized approach.

**Query Type Classification**: Explicitly identifying query archetypes (direct, decomposable, sequential) and applying tailored strategies for each shows substantial performance improvements compared to one-size-fits-all approaches.


## 7. Conclusion

The RAG literature reveals a field in rapid evolution from foundational single-pass pipelines to sophisticated agentic systems that model human cognitive processes. The progression through naive RAG, advanced RAG with optimized retrieval techniques, to agentic RAG demonstrates how incorporating reasoning, planning, and self-critique mechanisms substantially improves performance on complex tasks.

CogniRAG and similar recent frameworks show that explicit modeling of cognitive processes—planning, execution, and synthesis—yields significant improvements. Empirical results demonstrate 24.1% improvements in answer correctness and 8.5% improvements in response relevancy compared to strong non-agentic baselines, particularly on multi-hop reasoning tasks that challenge conventional approaches.

However, substantial research opportunities remain. Latency optimization, expanded tool integration, and hybrid approaches combining explicit planning with token-level critique represent critical frontiers. Additionally, gaps in cross-domain generalization, improved interpretability, and comprehensive evaluation methodologies remain to be addressed.

The trajectory suggests that RAG systems will increasingly adopt agentic approaches inspired by human cognition. Future work should focus on making these systems practical through latency reduction, more versatile through tool expansion, and more trustworthy through enhanced explainability and cross-domain evaluation.


## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Nogueira, R., He, H., Chen, D., Yih, W., & Komeili, M. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

2. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., & others. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

3. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1-38.

4. Yao, S., Zhao, J., Yu, D., Du, N., Durmus, I., Liska, M., Gu, L., Luan, D., & Gu, R. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*.

5. Es, S., James, J., Laskar, S., Gerit, L., Vaidya, V., & Jacob, A. (2023). RAGAs: Automated evaluation of retrieval augmented generation. *arXiv preprint arXiv:2309.15217*.

6. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. *arXiv preprint arXiv:2310.11511*.

7. Jiang, Z., Liska, M., Gu, L., Gu, R., Yavuz, S., Luan, D., Zhao, J., Yao, S., & Du, N. (2023). Active retrieval augmented generation. *arXiv preprint arXiv:2305.06983*.

8. Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., & others. (2023). The rise and potential of large language model based agents: A survey. *arXiv preprint arXiv:2309.07864*.

9. Luan, Y., Piktus, A., Cvicek, F., He, H., Gunaratna, K., Lee, K., Yih, W., & Chang, M. (2021). Sparse, dense, and attentive representations for text retrieval. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 5756-5771.

10. Nogueira, R., Yang, W., Cho, K., & Lin, J. (2019). Passage re-ranking with BERT. *arXiv preprint arXiv:1901.04085*.

11. Ma, X., Gong, Y., Jiang, A., Liu, P., Wang, H., Yan, B., & Chen, N. (2023). Query rewriting for retrieval-augmented large language models. *arXiv preprint arXiv:2305.14283*.

12. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2369-2380.

13. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

14. Kwiatkowski, T., Palomaki, J., Redfield, D., Collins, A., Parikh, A., Alberti, C., Epstein, D., Polevychenko, I., Kelcey, M., & Grave, E. (2019). Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7, 452-466.

15. Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016). MS MARCO: A human generated MAchine Reading COmprehension dataset. *arXiv preprint arXiv:1611.09268*.

16. Thayasivam, U., & Bandara, K. (2025). CogniRAG: A human-cognition-inspired agentic framework for enhancing retrieval-augmented generation. *Proceedings of the 2025 International Conference on Computational Linguistics*, Department of Computer Science and Engineering, University of Moratuwa.

17. Welbl, J., Levy, O., Dagan, I., & Schwartz, R. (2018). Constructing datasets for multi-hop reading comprehension across documents. *Transactions of the Association for Computational Linguistics*, 6, 285-297.

18. Min, S., Zhong, V., Soares, L., & Zettlemoyer, L. (2019). Compositional questions do not necessitate compositional structure of their answers. *arXiv preprint arXiv:1906.07381*.

19. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Guo, Q. (2023). Retrieval-augmented generation for large language models: A survey. *arXiv preprint arXiv:2312.10997*.

20. Wolfram, S., & Heaton, M. (2023). What we've learned from a year of making neural nets smarter. *Wolfram Blog*, 2023.

