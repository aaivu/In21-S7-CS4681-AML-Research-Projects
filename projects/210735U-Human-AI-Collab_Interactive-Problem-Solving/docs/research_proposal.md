# Research Proposal: Human-AI Collab: Interactive Problem Solving

**Student:** 210735U  
**Research Area:** Human-AI Collab: Interactive Problem Solving  
**Date:** 2025-09-01  

---

## Abstract

This research explores improving Python code generation in open-source Large Language Models (LLMs) through **lightweight inference-time optimization techniques**. Rather than retraining models, which is resource-intensive, this work investigates how real-time inference strategies—such as prompt engineering, adaptive sampling, and temperature tuning—can enhance code accuracy, syntax validity, and execution success. The study evaluates models like CodeGen2, StarCoderBase, and GPT-Neo on benchmarks including HumanEval and MBPP. By analyzing the impact of different inference configurations, this research aims to identify efficient, reproducible methods that bridge the performance gap between open-source and proprietary models. The outcomes are expected to contribute to the development of **cost-effective optimization frameworks** that empower researchers and developers to deploy high-quality code-generation systems without requiring large-scale computational resources.

---

## 1. Introduction

Large Language Models (LLMs) such as GPT-4, Codex, and StarCoder have significantly advanced the field of **automated code generation**, allowing developers to transform natural language descriptions into functional source code. However, open-source models often lag behind their proprietary counterparts in terms of correctness, efficiency, and robustness. This limitation stems from smaller training datasets, reduced parameter sizes, and lack of continuous fine-tuning.  

This research investigates how **inference-time optimization**—techniques applied during model output generation—can improve the quality of code produced by open-source LLMs. Unlike fine-tuning, these methods do not require retraining, making them ideal for academic and low-resource environments.

---

## 2. Problem Statement

Open-source code generation models such as CodeGen, StarCoderBase, and GPT-Neo frequently generate **syntactically correct but functionally invalid** Python code. Retraining or fine-tuning these models to correct such errors is computationally expensive and inaccessible to many researchers.  

Hence, this research aims to address the following question:  

> **Can inference-time optimization techniques improve the functional correctness and execution success rate of open-source Python code generation models without retraining?**

---

## 3. Literature Review Summary

Existing studies highlight that **LLM inference-time adjustments** (e.g., prompt engineering, temperature control, and sampling strategies) can meaningfully influence model performance.  
- Chen et al. (2021) introduced *Codex*, showing the potential of LLMs in code synthesis.  
- Li et al. (2023) demonstrated that *self-consistency decoding* and *majority voting* improve reasoning-based tasks.  
- Open-source research, such as Hugging Face’s *StarCoder* (2023), shows promising results but lacks systematic evaluation of inference-time optimizations.  

The literature gap lies in **quantitatively comparing** and **benchmarking lightweight inference-time techniques** specifically for open-source code models on standard datasets.

---

## 4. Research Objectives

### Primary Objective
To enhance the functional correctness and execution reliability of open-source Python code generation models using **lightweight inference-time optimization techniques**.

### Secondary Objectives
- Evaluate baseline performance of open-source LLMs on HumanEval and MBPP benchmarks.  
- Implement and assess prompt optimization using manual and DSPy-based structured prompts.  
- Apply adaptive multi-sampling, temperature, and top-p/top-k tuning methods.  
- Compare performance using Pass@k, syntax validity, and execution accuracy metrics.  
- Develop reproducible guidelines and scripts for future open-source research.

---

## 5. Methodology

This research adopts a **quantitative experimental design**. Open-source models (CodeGen2-1B, StarCoderBase-1B, PolyCoder-0.4B, GPT-Neo-1.3B) will be evaluated using Python benchmarks (HumanEval, MBPP).  

The steps include:
1. **Baseline Evaluation:** Test models using default inference parameters.  
2. **Technique Integration:** Apply prompt optimization, adaptive sampling, and temperature tuning.  
3. **Metric Evaluation:** Measure performance using Pass@k, syntax correctness, and execution success.  
4. **Comparative Analysis:** Use statistical tests (t-test, ANOVA) to validate significance of improvements.  
5. **Result Documentation:** Summarize findings and propose best-practice inference configurations.

---

## 6. Expected Outcomes

- Improved **Pass@1** and **Pass@10** metrics compared to baseline models.  
- Demonstrated effectiveness of **structured prompting** and **sampling optimizations**.  
- A reproducible, open-source **evaluation and optimization framework**.  
- Insights into model behavior that inform future work in **human-AI collaborative code generation**.

---

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development and Dataset Preparation |
| 5-8  | Implementation and Baseline Evaluation |
| 9-12 | Experimentation and Optimization |
| 13-15| Data Analysis and Report Writing |
| 16   | Final Submission and Review |

---

## 8. Resources Required

- **Hardware:** GPU-enabled system (NVIDIA RTX A5000 or higher)  
- **Software:** Python 3.10, PyTorch, Hugging Face Transformers, DSPy, OpenAI Eval Framework  
- **Datasets:** HumanEval, MBPP, Custom Python Tasks  
- **Libraries:** Pandas, Matplotlib, NumPy for analysis  
- **Documentation:** GitHub for version control, Markdown for reporting  

---

## References

1. Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.  
2. Li, X., et al. (2023). *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models.* arXiv:2203.11171.  
3. Allal, L., et al. (2023). *StarCoder: May the Source Be with You!*. Hugging Face Research.  
4. Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.  
5. OpenAI. (2023). *GPT-4 Technical Report.* arXiv:2303.08774.  
6. Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling.* arXiv:2101.00027.  

