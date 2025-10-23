# Research Proposal: AI Evaluation:Agentic Evaluation

**Student:** 210363C
**Research Area:** AI Evaluation:Agentic Evaluation
**Date:** 2025-09-01

## Abstract

The proposed research introduces **AutoGen-TraceKit**, a lightweight and reproducible framework designed to evaluate large language model (LLM) agents using process-aware and robustness-oriented metrics. Unlike traditional methods that focus solely on accuracy or task completion, this study emphasizes understanding _how_ agents achieve outcomes by capturing fine-grained traces of reasoning, efficiency, stability, and control-flow behavior. The framework integrates unified trace logging with derived efficiency indicators such as **Token Efficiency (TE)**, **Call Efficiency (CE)**, and **Latency Efficiency (LE)**. It also incorporates sensitivity analysis under varying conditions (e.g., temperature, random seed, and turn limits) to uncover hidden trade-offs between accuracy and stability. Using the **MATH benchmark** and the **DeepSeek-R1-Distill-Llama-70B model** via the Groq API, this research aims to advance AI agent evaluation by ensuring transparency, reproducibility, and deeper behavioral insights into LLM reasoning processes.

## 1. Introduction

Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of complex reasoning and decision-making. Traditional evaluation methods primarily focus on task success rates, often overlooking critical behavioral aspects such as reasoning efficiency, stability under stochastic conditions, and error recovery. This research proposes a process-aware evaluation framework that measures _how_ agents achieve their results, not just whether they succeed. By addressing these gaps, the study aims to enhance the reliability, reproducibility, and trustworthiness of AI agent behavior, contributing to safer real-world deployments.

## 2. Problem Statement

Current agent evaluation methods are predominantly outcome-based, offering limited insights into internal reasoning dynamics, efficiency, and robustness. There is a lack of unified, lightweight frameworks capable of:

- Capturing detailed agent interactions without altering model behavior.
- Quantifying process-level metrics such as efficiency and control-flow health.
- Testing robustness under stochastic variations.

This research addresses the absence of such a reproducible, process-aware, and low-overhead evaluation framework for LLM agents.

## 3. Literature Review Summary

Benchmark-based frameworks like **AgentBench** (Liu et al., 2023) standardize performance evaluation but fail to capture reasoning pathways. Process-aware tools such as **DeepEval** (Kıcıman et al., 2024) and IBM’s **Agentic Evaluation Toolkit** (IBM Research, 2024) provide reasoning insights but impose integration overheads and lack robustness testing. Recent surveys (Yehudai et al., 2025; Srivastava et al., 2024) emphasize the need for lightweight, reproducible, and robustness-sensitive evaluation methods. This project fills that gap by proposing **AutoGen-TraceKit**, which combines unified trace logging with robustness-oriented metrics for deeper insights into LLM agent behavior.

## 4. Research Objectives

### Primary Objective

To design and implement a process-aware, robustness-oriented evaluation framework that captures and analyzes fine-grained reasoning traces of LLM agents.

### Secondary Objectives

- Develop a unified trace-logging mechanism that records all agent interactions without affecting performance.
- Derive efficiency and stability metrics (e.g., TE, CE, LE, retry frequency, control-flow health).
- Conduct sensitivity analyses under variable seeds, temperatures, and turn limits to quantify robustness.
- Compare the proposed framework against outcome-only baselines for interpretability and transparency.

## 5. Methodology

The methodology consists of two main stages:

### Stage I – Baseline Evaluation

Agents are executed under fixed conditions (seed=42, temperature=0.7, turn limit=5, max tokens=1024). Unified trace logs capture all inputs, outputs, token usage, retries, and elapsed times. Baseline metrics (accuracy, efficiency, stability) are established.

### Stage II – Sensitivity Analysis

Controlled variations in seeds and temperatures ({42,123,999} × {0.5,0.7,0.9}) test robustness. The resulting traces are compared to the baseline to measure variability, resilience, and trade-offs between accuracy and efficiency.

**Setup:**

- **Dataset:** MATH benchmark (120 problems, JSONL format)
- **Model:** DeepSeek-R1-Distill-Llama-70B via Groq API
- **Metrics:** Token Efficiency (TE), Call Efficiency (CE), Latency Efficiency (LE), Control-Flow Health

## 6. Expected Outcomes

- A fully operational, lightweight evaluation framework (**AutoGen-TraceKit**) for LLM agents.
- Quantitative insights into hidden trade-offs (e.g., efficiency vs. accuracy; stability vs. adaptability).
- Demonstrated reproducibility through fixed seeds and centralized configurations.
- Enhanced transparency for researchers and practitioners evaluating LLM agents.
- Contribution toward establishing next-generation evaluation standards for agentic systems.

## 7. Timeline

| Week | Task                                             |
| ---- | ------------------------------------------------ |
| 1-2  | Literature Review                                |
| 3-4  | Methodology Development                          |
| 3-6  | Implementation of TraceKit Framework             |
| 5-7  | Experiments on Baseline and Sensitivity Analysis |
| 7-8  | Data Analysis and Report Writing                 |
| 9    | Final Submission                                 |

## 8. Resources Required

- **Datasets:** MATH benchmark subset (120 problems, JSONL format)
- **Model/API:** DeepSeek-R1-Distill-Llama-70B (Groq API)
- **Software:** Python 3.x, AutoGen framework, NumPy, Pandas, Matplotlib

## References

1. Liu, Z., et al. (2023). AgentBench: Evaluating LLMs as Agents. _arXiv:2308.03688_.
2. Kıcıman, J., et al. (2024). DeepEval: Benchmarking LLMs for Reasoning and Evaluation. GitHub: _confident-ai/deepeval_.
3. IBM Research. (2024). Agentic Evaluation Toolkit. IBM Documentation (_watsonx_).
4. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. _arXiv:2308.08155_.
5. Srivastava, A., et al. (2024). Beyond Accuracy: Evaluating Efficiency and Robustness in LLM Agents. _arXiv:2506.11111v2_.
6. Yehudai, A., et al. (2025). Survey on Evaluation of LLM-based Agents. _arXiv:2503.16416_.
7. Pan, X., et al. (2023). Robustness Testing of LLMs under Prompt and Parameter Perturbations. _arXiv:2507.16407_.
8. Huang, A., et al. (2023). Benchmarking Large Language Models as AI Agents. _arXiv:2310.03302_.
9. Chen, Y., et al. (2023). Evaluating Reasoning in Large Language Models. _arXiv:2404.01869_.

---

**Submission Instructions:**

1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review
