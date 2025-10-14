# Literature Review: AI Evaluation: Agentic Evaluation

**Student:** 210363C  
**Research Area:** AI Evaluation: Agentic Evaluation  
**Date:** 2025-09-01

## Abstract

This review explores agentic evaluation methods for LLM agents, focusing on: (i) benchmark-oriented, outcome-only evaluation (e.g., AgentBench), (ii) process-aware frameworks (e.g., DeepEval, IBM’s Agentic Evaluation Toolkit), (iii) limitations of current approaches (traceability, overheads, and robustness gaps), and (iv) emerging directions combining unified trace logging with robustness-oriented metrics. Key finding: accuracy alone obscures critical dimensions such as efficiency, stability, control-flow health, and sensitivity to stochastic variations. Lightweight trace logging and structured metrics offer a practical path to more transparent and reproducible agent evaluation.

## 1. Introduction

Agentic evaluation has shifted from solely assessing task success to examining how agents solve tasks. While outcome-based benchmarks enable standardized comparisons, they overlook reasoning efficiency, error recovery, and robustness. Recent frameworks assess process quality but often impose integration overheads and lack unified, lightweight traceability and robustness analysis.

## 2. Search Methodology

### Search Terms Used

- AI Agent Evaluation
- LLM Reasoning Evaluation

### Databases Searched

- [✔] IEEE Xplore
- [ ] ACM Digital Library
- [ ] Google Scholar
- [✔] ArXiv
- [ ] Other: -

### Time Period

Focused on recent developments.

## 3. Key Areas of Research

### 3.1 Benchmark-Oriented (Outcome-Only) Evaluation

**Focus:** Standardized environments measuring task success/accuracy; enables fair cross-system comparison but ignores inefficiencies and adaptability.

**Key Papers:**

- Liu et al., 2023 (AgentBench): Introduces standardized agent tasks to measure goal completion; strong comparability; limited visibility into reasoning processes.
- Huang et al., 2023: Benchmarks LLMs as agents; highlights the need to go beyond accuracy when assessing agent behaviors.

### 3.2 Process-Aware Frameworks

**Focus:** Reasoning steps, decision-making quality, error handling, and consistency.

**Key Papers:**

- Kıcıman et al., 2024 (DeepEval): Continuous testing for correctness/reliability/consistency of LLM applications; improved process visibility; integration overheads remain.
- IBM Research, 2024 (Agentic Evaluation Toolkit): Enterprise-oriented evaluation of reasoning and error handling; emphasizes stability and trustworthiness; lacks unified lightweight trace logging.

## 4. Research Gaps and Opportunities

### Gap 1: Lack of lightweight, unified traceability in agent evaluations

**Why it matters:** Without uniform, low-overhead traces, cross-run and cross-tool comparisons are hard; insights into reasoning dynamics remain opaque.  
**How your project addresses it:** Develop lightweight, unified trace logging to enable transparent, reproducible evaluations.

### Gap 2: Weak robustness analysis under stochastic variations

**Why it matters:** Agents can be sensitive to seeds, temperatures, and turn limits; accuracy parity may mask instability or cost inefficiency.  
**How your project addresses it:** Systematically vary seeds/temperatures/limits and report robustness-oriented metrics (e.g., token efficiency vs. accuracy trade-offs, retry patterns, abnormal terminations).

## 5. Theoretical Framework

- **Process-aware evaluation:** Evaluate how outcomes are achieved via traces (efficiency, stability, control-flow health).
- **Robustness analysis:** Treat evaluation as sensitivity testing over controlled stochastic parameters to reveal hidden trade-offs.
- **Reproducibility:** Fix seeds/configuration and centralize parameters to enable consistent, comparable runs.

## 6. Conclusion

Current agentic evaluation benefits from benchmarks but requires transparent, reproducible visibility into process dynamics and robustness. Lightweight unified trace logging with structured metrics addresses traceability and sensitivity gaps, enabling more trustworthy comparisons and deployment-relevant insights.

## References

- Yehudai, A., et al. (2025). _Survey on Evaluation of LLM‐based Agents._ A comprehensive survey mapping evaluation benchmarks, frameworks, and gaps in agent evaluation. arXiv.
- Mohammadi, M., et al. (2025). _Evaluation and Benchmarking of LLM Agents: A Survey._ Introduces a two-dimensional taxonomy for agent evaluation (objectives vs. process) and systematic overview of existing evaluation methods. arXiv.
- _Evaluating LLM‐based Agents for Multi-Turn Conversations: A Survey_ (2025). Focused on conversational agents; discusses what to evaluate (memory, planning, tool use) and how (metrics, datasets). arXiv.
- _AGENTBENCH: Evaluating LLMs as Agents_ (ICLR 2024). A benchmark framework for evaluating LLMs in agentic settings, with a toolkit for unified evaluation across tasks. Proceedings ICLR.
- _AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents_ (2024). Provides a benchmark + framework emphasizing progress metrics (incremental gains) and multi-round interaction analysis. NeurIPS.
- _MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research_ (2025). A benchmark for evaluating agents that undertake scientific/research tasks; includes stepwise and end-to-end evaluation. arXiv.
- _LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners_ (2025). Proposes evaluation of long-term learning/knowledge accumulation in agent settings over multiple tasks. arXiv.
- _Agent-SafetyBench: Evaluating the Safety of LLM Agents_ (2024). Focuses on safety evaluation across failure modes in agent interactions and tool use. arXiv.
- Lunardi et al., 2025. _On Robustness and Reliability of Benchmark-Based Evaluations._ Tests how paraphrasing benchmarks affects evaluation stability — useful reference for discussing robustness. arXiv.
- _Magenta: Metrics and Evaluation Framework for Generative Agents._ Proposes a holistic benchmark and evaluation framework combining agent-level and model-level metrics. AHFE Open Access.
- _FLEX: A Benchmark for Evaluating Robustness of Fairness in LLMs_ (2025). While focused on fairness, demonstrates how robustness evaluation across contexts can be operationalized. aclanthology.org.
- _Review: Evaluating large language models and agents in healthcare._ Domain-specific survey of LLM and agent evaluation practices in the medical domain. ScienceDirect.
- _Evaluate Amazon Bedrock Agents with RAGas and LLM-as-a-Judge._ A practical blog + applied case showing how agent evaluation uses trace instrumentation, LLM-as-judge techniques, and dashboards. Amazon Web Services, Inc.
- _The future of AI agent evaluation — IBM Research blog._ A forward-looking view of agent benchmarks, competencies, and evaluation trends (e.g., “Agent SATs”).
