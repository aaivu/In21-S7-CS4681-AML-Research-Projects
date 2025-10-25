# Methodology: AI Evaluation:Agentic Evaluation

**Student:** 210363C  
**Research Area:** AI Evaluation:Agentic Evaluation  
**Date:** 2025-09-01

## 1. Overview

This study proposes AutoGen-TraceKit, a lightweight and reproducible framework for evaluating large language model (LLM) agents beyond accuracy. The methodology emphasizes process-aware and robustness-oriented evaluation by integrating unified trace logging with structured metrics. Rather than focusing solely on whether an agent completes a task, it analyzes how the result is achieved—capturing efficiency, stability, control-flow health, and resilience to stochastic variations such as random seeds and sampling temperatures.

## 2. Research Design

The research follows a two-stage experimental design combining baseline evaluation and sensitivity analysis.

1. Stage I – Baseline Evaluation: Agents are executed under fixed, controlled settings (seed = 42, temperature = 0.7, max tokens = 1024) to establish a reproducible performance reference.
2. Stage II – Sensitivity Analysis: Parameters such as random seed and temperature are systematically varied to assess robustness and identify trade-offs between efficiency, accuracy, and stability.

This process-aware design enables reproducible comparisons while minimizing integration overhead.

## 3. Data Collection

### 3.1 Data Sources

MATH benchmark dataset  
[Hendrycks MATH Benchmark Dataset](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark)

### 3.2 Data Description

The MATH dataset subset consists of 120 problems stored in JSONL format.  
Each entry includes:

- Problem ID
- Problem text
- Ground-truth solution
- Final answer

This subset ensures consistent and interpretable comparisons across runs.

### 3.3 Data Preprocessing

To ensure reproducibility and consistency, the following preprocessing steps were applied:

1. **Standardization**: Problems were standardized and stored in JSONL format.
2. **Configuration Management**: Random seeds, temperature, and token limits were predefined in a central configuration file (`config.py`).
3. **Logging**: Logs were generated for each run, capturing:
   - Token usage
   - Retry attempts
   - Elapsed time
   - Termination details

These logs were subsequently aggregated to facilitate metric computation and analysis.

## 4. Model Architecture

The experiments use DeepSeek-R1-Distill-Llama-70B accessed via the Groq API. No modifications were made to the model; AutoGen-TraceKit passively recorded agent–environment interactions. The architecture of TraceKit includes:

1. Agent Interaction Layer – handles task execution via AutoGen pipelines.
2. Trace Logger – captures inputs, outputs, retries, tokens, timestamps.
3. Metrics Engine – computes quantitative indicators such as token efficiency and control-flow health.
4. Reporting Module – generates summaries and dashboards for analysis.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

Metrics captured by TraceKit include:

- Tokens per Solve (TPS)
- Executions per Solve (EPS)
- Token Efficiency (TE) – solved tasks per 1k tokens
- Call Efficiency (CE) – solved tasks per LLM call
- Latency Efficiency (LE) – solved tasks per minute
- Control-flow indicators – retry frequency, abnormal terminations

### 5.2 Baseline Models

The outcome-only evaluation (accuracy) serves as the baseline for comparison against the process-aware metrics from AutoGen-TraceKit.

### 5.3 Hardware/Software Requirements

- Environment: Python-based standalone scripts
- API: Groq API
- Model: DeepSeek-R1-Distill-Llama-70B
- System: Controlled configuration via `config.py`
- Output Storage: `experiments/logs/` with consistent run naming (`runID_seed_temp.csv`)

## 6. Implementation Plan

Phase Tasks Duration Deliverables  
Phase 1 Data preprocessing 1 week Clean JSONL dataset  
Phase 2 Model & TraceKit implementation 3 weeks Working trace-logging framework  
Phase 3 Experiments (baseline + sensitivity) 2 weeks Logged results & metrics  
Phase 4 Analysis & reporting 1 week Final report & visual dashboards

## 7. Risk Analysis

Potential risks and mitigation strategies:

- API or model instability: Cache results and maintain fixed seeds for reproducibility.
- Computation overheads: TraceKit is lightweight and non-intrusive, ensuring minimal performance degradation.
- Metric bias due to noise: Average across multiple seeds to stabilize observations.

## 8. Expected Outcomes

The research expects to produce:

- A lightweight, reproducible framework for evaluating LLM agents beyond accuracy.
- Quantitative insights into efficiency, stability, and robustness trade-offs.
- A foundation for next-generation agentic evaluation methods integrating trace logging and sensitivity analysis.

---

**Note:** Update this document as your methodology evolves during implementation.
