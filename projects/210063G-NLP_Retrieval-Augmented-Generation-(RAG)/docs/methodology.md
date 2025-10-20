# Methodology: NLP:Retrieval-Augmented Generation (RAG)

**Student:** 210063G  
**Research Area:** NLP: Retrieval-Augmented Generation (RAG)  
**Date:** 2025-10-20

## 1. Overview

This research proposes an Agentic LightRAG framework that enhances traditional Retrieval-Augmented Generation systems through multi-agent orchestration inspired by human cognitive processes. The methodology combines query decomposition, specialized retrieval agents operating at multiple abstraction levels, graph-based knowledge indexing, and intelligent answer synthesis. The framework is designed to address limitations of naive RAG systems when handling complex, multi-hop, and ambiguous queries by introducing planning, decomposition, and iterative refinement mechanisms. The project follows a structured 10-week implementation timeline from weeks 5-14, progressing from architectural design through component integration to comprehensive evaluation and analysis.

## 2. Research Design

The research employs an empirical, comparative methodology grounded in the design science research paradigm. The approach consists of the following key elements:

**Architecture-Driven Development**: The project constructs a novel multi-agent agentic RAG system based on cognitive architecture principles. The design separates concerns across four specialized agent types, each handling distinct responsibilities in the question-answering pipeline.

**Iterative Implementation**: The system is developed incrementally, with components built sequentially and integrated progressively. This allows for early validation of design decisions and incorporation of learnings from earlier phases.

**Comparative Evaluation**: The implemented system is evaluated against established baselines using standardized metrics and benchmark datasets. Performance improvements are quantified through multiple evaluation dimensions including answer correctness, response relevancy, and retrieval quality metrics.

**Benchmark-Driven Testing**: Evaluation uses established multi-hop question-answering benchmarks (HotpotQA) to ensure comparability with prior work and to assess performance on the specific challenge domain for which the framework is designed.

**Mixed-Method Analysis**: Both quantitative metrics and qualitative analysis are employed. Quantitative results provide performance comparisons, while qualitative analysis through case studies examines system behavior, reasoning quality, and failure modes.


## 3. Data Collection

### 3.1 Data Sources

- **HotpotQA Dataset**: Primary benchmark for multi-hop question answering. Provides diverse, explainable questions requiring reasoning across multiple Wikipedia articles. Retrieved from the official HotpotQA repository.
- **Wikipedia Knowledge Base**: The underlying document collection for HotpotQA, providing factual information and structured content for retrieval operations.
- **External Knowledge Graphs**: Optional supplementary knowledge sources for enriching the graph-based index with semantic relationships and entity connections.
- **Open-Source Document Collections**: Additional datasets for robustness testing and cross-domain evaluation if time permits.

### 3.2 Data Description

**HotpotQA Dataset Characteristics**:
- Total samples: 113,000+ question-answer pairs (using subset of 200-500 for primary evaluation based on computational constraints)
- Question types: Multi-hop questions requiring synthesis of information from 2-4 Wikipedia documents
- Annotation style: Questions include supporting facts indicating which documents and sentences contribute to the correct answer
- Answer format: Short factoid answers (typically 1-3 words) with binary classification for yes/no questions

**Wikipedia Knowledge Base**:
- Approximately 5.9 million documents covering diverse topics
- Structured with clear document boundaries and section hierarchies
- Format: Plain text articles with title, content, and hyperlink information

**Graph Index Format**:
- Entity-relationship representation capturing connections between topics
- Node types: entities, concepts, documents
- Edge types: contains, references, related_to, has_property
- Enables both semantic and structured retrieval approaches

### 3.3 Data Preprocessing

**Document Chunking**: Wikipedia articles are segmented into semantically meaningful chunks of approximately 512 tokens with 128-token overlap. Chunk boundaries respect section boundaries when possible to maintain coherence.

**Vector Embedding Generation**: All document chunks are embedded using OpenAI's text-embedding-3-large model, generating 3,072-dimensional dense vectors capturing semantic information. Embeddings are stored in a vector index for similarity-based retrieval.

**Graph Index Construction**: Document chunks, entities, and their relationships are extracted using named entity recognition and relation extraction techniques. A graph structure is built with entities as nodes and semantic relationships as edges, enabling structured graph-based retrieval.

**Query Normalization**: User queries are cleaned through lowercasing, removal of extra whitespace, and basic punctuation normalization. Named entities in queries are identified and preserved for use in structured retrieval.

**Train-Test Split**: The HotpotQA dataset is divided into 80% training/development subset (for system parameter tuning) and 20% held-out test set (for final evaluation) to ensure unbiased performance assessment.

**Metadata Enrichment**: Questions are preprocessed to identify query type (direct, decomposable, sequential) based on linguistic patterns and constituent analysis. Ground-truth supporting document information is extracted for evaluation purposes.


## 4. Model Architecture

The Agentic LightRAG framework comprises four specialized agent components operating within a coordinated multi-agent system:

**Query Decomposition Agent**: Acts as the strategic meta-agent responsible for analyzing incoming user queries and formulating decomposition strategies. Performs initial query classification into archetypes (direct, decomposable, sequential), generates sub-queries for complex questions, and maintains a dynamic retrieval plan. This agent is implemented as a task-specific LLM prompt-based agent operating with chain-of-thought reasoning.

**Low-Level Retrieval Agent**: Executes granular, fact-seeking retrieval operations in response to specific atomic queries from the decomposition agent. Searches vector and graph indices to locate precise factual information. Implements query refinement strategies and synonym expansion for improved recall. Returns ranked results with relevance scores and retrieval confidence metrics.

**High-Level Retrieval Agent**: Performs broader, context-seeking retrieval operations to gather comprehensive background information and multi-document context. Executes at the document or section level rather than fact level. Implements document clustering and multi-document relevance ranking. Provides broader narrative context to support answer synthesis.

**Answer Synthesis Agent**: Responsible for transforming retrieved information fragments into coherent, user-facing answers. Performs information fusion across multiple retrieved passages, handles conflicting information through confidence-based ranking, and generates natural language answers with explicit reasoning justification. Implements answer verification against the original query to ensure relevance.

**Coordination Layer**: A state management and scheduling component managing inter-agent communication, maintaining the reasoning trace, implementing iteration termination protocols, and coordinating resource allocation across agents. Implemented using a queue-based message passing architecture with temporal state snapshots.


## 5. Experimental Setup

### 5.1 Evaluation Metrics

**Answer Correctness**: Measures factual accuracy of generated answers against ground-truth responses using both exact match scoring and semantic similarity via embedding-based comparison. Calculated as the cosine similarity between answer embeddings and ground truth with threshold of 0.7 for correctness determination. Provides primary metric of functional system success.

**Response Relevancy**: Evaluates whether generated answers directly address the original query without extraneous information. Implemented by generating potential questions from the answer using a question-generation model and measuring cosine similarity with the original query. Ranges from 0 to 1, with 1 indicating perfect relevancy.

**Context Precision**: Measures the proportion of retrieved documents that contribute to the correct answer. Calculated as: (number of retrieved supporting documents) / (total number of retrieved documents). Indicates efficiency of retrieval process.

**Context Recall**: Measures proportion of relevant documents successfully retrieved. Calculated as: (number of retrieved supporting documents) / (total available supporting documents). Indicates completeness of information gathering.

**Reasoning Quality**: Qualitative assessment of whether agent reasoning traces are logical, interpretable, and demonstrate appropriate problem decomposition. Evaluated through expert review of selected examples.

**Latency**: Total end-to-end response time from query input to answer generation. Measured in seconds. Tracks computational overhead introduced by multi-agent coordination.

**Query Success Rate**: Proportion of questions where the system produces a valid answer versus failure or no-answer responses. Reflects robustness of the decomposition strategy.

### 5.2 Baseline Models

**Naive RAG Baseline**: Standard retrieve-then-generate pipeline without agent orchestration. Single retrieval pass using vector similarity, concatenates top-k documents with query, passes directly to LLM for generation. Uses identical embedding model and LLM as agentic framework for fair comparison.

**LightRAG (Non-Agentic)**: Graph-based retrieval system without agent coordination. Implements dual-level retrieval using graph indices but operates in single-pass mode without iterative refinement. Represents current state-of-art non-agentic graph-based retrieval.

**Self-RAG (Reference Comparison)**: LLM fine-tuned for generating reflection tokens that assess retrieval necessity and passage relevance. Represents alternative approach to incorporating reasoning. Used for qualitative comparison of reasoning strategies rather than direct quantitative comparison due to training requirements.

### 5.3 Hardware/Software Requirements

**Computational Infrastructure**:
- GPU: NVIDIA A100 or equivalent (40GB VRAM minimum) for model inference and embedding generation
- CPU: 16+ cores for parallel retrieval operations
- RAM: 64GB minimum system memory
- Storage: 2TB for document collection, embeddings, and graph indices

**Software Stack**:
- Python 3.10+
- Light Rag Framework
- LangChain for LLM abstraction and agent orchestration
- OpenAI API for GPT-4o-mini and embedding models
- NetworkX for graph operations and traversal
- RAGAs framework for automated evaluation

**Development Tools**:
- Git for version control
- Jupyter notebooks for experimentation and analysis
- OpenAI for LLM and emdebbing models
- pytest for unit testing



## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1: Design & Setup (Weeks 5-6) | Finalize literature review; Complete agentic LightRAG architecture design; Set up development environment; Establish data preprocessing pipelines; Configure vector and graph indices | 2 weeks | Final architecture document; Preprocessed HotpotQA dataset; Configured indexing infrastructure; Development environment documentation |
| Phase 2: Core Agent Implementation (Weeks 7-8) | Implement Query Decomposition Agent with query classification and sub-query generation; Implement Low-Level Retrieval Agent with fact-seeking strategies; Implement High-Level Retrieval Agent with document-level retrieval | 2 weeks | Query decomposition module with test cases; Low-level retrieval agent passing unit tests; High-level retrieval agent passing unit tests; Inter-agent communication protocol specification |
| Phase 3: Integration & Protocols (Weeks 9-10) | Integrate agents with graph-based knowledge index; Develop and refine communication protocols between agents; Implement state management and coordination layer; Build reasoning trace logging system | 2 weeks | Integrated multi-agent system; Communication protocol documentation; State management module; Reasoning trace logs for sample queries |
| Phase 4: Answer Synthesis & Testing (Weeks 11-12) | Implement Answer Synthesis Agent with information fusion and answer generation; Conduct initial experiments on 50-100 HotpotQA test questions; Perform iterative debugging and performance tuning; Generate preliminary results | 2 weeks | Answer synthesis module with verification; Initial experimental results on test subset; Debugging report with identified issues; Performance tuning recommendations |
| Phase 5: Evaluation & Reporting (Weeks 13-14) | Complete full evaluation on complete test set (200 queries); Comprehensive performance analysis against baselines; Qualitative analysis of system behavior and failure modes; Write final project report with findings and insights | 2 weeks | Complete evaluation results with statistical analysis; Comparative performance analysis; Case studies of complex queries; Final project report with conclusions and future work |


## 7. Risk Analysis

**Risk 1: High Latency of Multi-Agent System**
- **Description**: The iterative nature of multi-agent coordination may introduce latency that makes the system impractical compared to single-pass baselines.
- **Probability**: Medium
- **Impact**: High - could limit practical applicability
- **Mitigation Strategies**: Implement parallel agent execution where feasible; use smaller, faster models for intermediate reasoning steps; implement caching mechanisms for repeated queries; establish latency budgets per agent

**Risk 2: LLM Reasoning Quality Limitations**
- **Description**: The Decomposition and Synthesis agents rely on LLM reasoning quality. Weaker LLMs may fail to decompose complex queries appropriately or synthesize answers coherently.
- **Probability**: Medium
- **Impact**: High - directly affects system correctness
- **Mitigation Strategies**: Extensive prompt engineering and few-shot examples; use GPT-4o-mini (strong reasoning model); implement fallback strategies for reasoning failures; include explicit constraints and output formatting requirements

**Risk 3: Graph Index Construction Errors**
- **Description**: Entity extraction and relationship identification may contain errors, leading to a corrupted knowledge graph that provides poor retrieval.
- **Probability**: Medium
- **Impact**: Medium - reduces retrieval quality but doesn't prevent system operation
- **Mitigation Strategies**: Implement validation checks on extracted entities and relationships; use multiple NER models and reconcile results; validate graph structure against known relationships; implement fallback to vector-only retrieval

**Risk 4: Limited Computational Resources**
- **Description**: GPU memory constraints may limit batch sizes, embedding dimensions, or ability to process large queries.
- **Probability**: Low-Medium
- **Impact**: Medium - affects scalability but manageable through optimization
- **Mitigation Strategies**: Implement model quantization if needed; use parameter-efficient methods; optimize batch processing; consider cloud GPU resources if local resources insufficient

**Risk 5: Dataset Bias and Limited Generalization**
- **Description**: Evaluation on HotpotQA alone may not demonstrate generalization to other domains or query types.
- **Probability**: Low-Medium
- **Impact**: Low-Medium - affects claimed generalizability
- **Mitigation Strategies**: If time permits, evaluate on additional datasets; analyze failure modes by query type; document limitations explicitly; suggest cross-domain evaluation for future work

**Risk 6: Communication Protocol Complexity**
- **Description**: Inter-agent communication protocols may become overly complex, introducing bugs and making the system difficult to debug.
- **Probability**: Low
- **Impact**: Medium - affects development velocity
- **Mitigation Strategies**: Use established message formats (JSON); implement comprehensive logging; create clear protocol documentation; implement unit tests for protocol handling

## 8. Expected Outcomes

**Functional System Deliverable**: A working multi-agent agentic RAG system capable of decomposing complex queries, coordinating specialized retrieval operations, and synthesizing coherent answers. The system should demonstrate principled agent coordination with interpretable reasoning traces.

**Performance Improvements**: Quantitative improvements over non-agentic baselines on the HotpotQA benchmark, with targets of 15-25% improvement in Answer Correctness (in line with CogniRAG results showing 24.1% improvement) and 5-15% improvement in Response Relevancy. Improved Context Recall and Precision indicating more effective multi-hop information gathering.

**Architectural Insights**: Clear documentation of how cognitive architecture principles (planning, execution, synthesis) translate into practical agent designs. Analysis of what types of queries benefit most from agentic decomposition and which baseline approaches remain competitive.

**Interpretability Enhancement**: Comprehensive reasoning traces and agent decision logs that provide transparency into system operations. Evidence that explicit agent reasoning improves user trust and enables error diagnosis compared to black-box approaches.

**Scalability and Practicality Assessment**: Clear analysis of latency-accuracy trade-offs. Documentation of computational requirements and practical deployment considerations. Identification of optimization opportunities and directions for reducing latency overhead.

**Research Contributions**: A novel multi-agent framework for agentic RAG with documented design decisions and architectural patterns. Empirical validation on established benchmarks. Clear articulation of framework advantages and limitations compared to existing approaches. Foundation for future work on hybrid agentic approaches and cross-domain generalization.

**Comprehensive Documentation**: Final project report including literature synthesis, detailed methodology, experimental results with statistical analysis, case studies demonstrating system capabilities, limitations discussion, and recommendations for future research directions.

