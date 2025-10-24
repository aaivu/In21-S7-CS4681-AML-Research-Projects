# Methodology: Human-AI Collab:Interactive Problem Solving

**Student:** 210735U  
**Research Area:** Human-AI Collab:Interactive Problem Solving  
**Date:** 2025-09-01  

---

## 1. Overview

This research focuses on improving **Python code generation** in open-source Large Language Models (LLMs) using **lightweight inference-time techniques**. The methodology involves designing, implementing, and evaluating inference-time strategies that enhance accuracy and efficiency without retraining models. The goal is to support more effective human-AI collaboration during interactive programming and problem-solving.

---

## 2. Research Design

A **quantitative experimental research design** is adopted. The study compares baseline model outputs with results obtained after applying inference-time optimization techniques.  
The design involves four main stages:

1. **Baseline Evaluation:** Assess the default performance of selected open-source LLMs on Python code tasks.  
2. **Technique Implementation:** Integrate lightweight inference-time optimization methods into model pipelines.  
3. **Performance Comparison:** Evaluate the optimized outputs using predefined metrics (accuracy, correctness, and execution success).  
4. **Statistical Validation:** Apply statistical methods to determine the significance of observed performance differences.

---

## 3. Data Collection

### 3.1 Data Sources
- **HumanEval** – A benchmark dataset for evaluating functional correctness of generated code.  
- **MBPP (Mostly Basic Python Problems)** – Tests reasoning and problem-solving ability.  
- **Custom Evaluation Set** – Contains additional real-world coding challenges to validate generalization.

### 3.2 Data Description
Each dataset includes natural-language problem descriptions paired with ground-truth Python code solutions. Problems vary in complexity, covering algorithmic and functional reasoning tasks.

### 3.3 Data Preprocessing
- Removed duplicates and incomplete samples.  
- Standardized formatting for inputs and outputs.  
- Tokenized data using the Hugging Face tokenizer for each model.  
- Filtered problems based on runtime feasibility and testability.

---

## 4. Model Architecture

The experiments use **open-source transformer-based LLMs** fine-tuned for code:
- **Code Llama**  
- **StarCoder**  
- **Phi-3-mini**

These models are integrated within the Hugging Face Transformers framework.  
Each model is evaluated under identical conditions to ensure fair comparison, focusing on inference-level optimizations rather than architectural changes.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **Pass@k:** Measures likelihood of producing at least one correct output in *k* attempts.  
- **Syntax Validity (%):** Percentage of syntactically valid code outputs.  
- **Execution Accuracy:** Ratio of code that executes correctly and passes test cases.  
- **Inference Time (s):** Measures efficiency and latency impact.

### 5.2 Baseline Models
Baseline models are unmodified versions of Code Llama, StarCoder, and Phi-3-mini, evaluated under default inference settings for direct comparison.

### 5.3 Hardware/Software Requirements
- **Hardware:** NVIDIA GPU (≥12GB VRAM), 16GB RAM  
- **Software:** Python 3.10, PyTorch, CUDA, Hugging Face Transformers, OpenAI Eval toolkit

---

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean, validated dataset |
| Phase 2 | Model implementation | 3 weeks | Configured baseline models |
| Phase 3 | Experiments | 2 weeks | Performance results and logs |
| Phase 4 | Analysis | 1 week | Comparative report and visualization |

---

## 7. Risk Analysis

| Risk | Description | Mitigation |
|------|--------------|-------------|
| Computational limits | GPU constraints may slow experiments | Use batch inference and smaller subsets |
| Dataset imbalance | Uneven task difficulty | Stratified sampling and normalization |
| Metric bias | Overreliance on functional correctness | Include multiple evaluation criteria |
| Prompt variance | Unclear prompts may reduce reliability | Use standardized and optimized prompt templates |

---

## 8. Expected Outcomes

- Demonstrated improvement in **code accuracy and reasoning** through inference-time optimization.  
- Identification of **optimal decoding parameters** (temperature, top-k, top-p) for code generation.  
- Establishment of **a reproducible, lightweight methodology** for performance improvement without retraining.  
- Contribution to the broader goal of **enhancing human-AI collaborative problem-solving** through adaptive, efficient code generation systems.
 

 
