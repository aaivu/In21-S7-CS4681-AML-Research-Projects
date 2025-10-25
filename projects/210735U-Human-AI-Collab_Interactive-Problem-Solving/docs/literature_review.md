# Literature Review: Evolution of Large Language Models for Code Generation

**Student:** 210735U  
**Research Area:** Human-AI Collaboration: Interactive Problem Solving  
**Date:** 2025-09-01  

---

## Abstract

This literature review explores the evolution of large language models (LLMs) for code generation, tracing their development from early neural sequence models to advanced transformer-based systems. It highlights key model architectures (e.g., Codex, CodeGen2, StarCoder), common challenges in reasoning and correctness, and emerging inference-time optimization strategies that enhance performance without retraining. Findings suggest growing potential for lightweight techniques in improving code-generation quality and interactivity—critical to advancing human-AI collaborative problem solving.

---

## 1. Introduction

Large language models have transformed the way humans interact with AI systems in programming and problem-solving contexts. Early models such as RNNs and LSTMs laid the foundation for sequential learning, but the introduction of the Transformer architecture enabled more effective long-range dependency handling.  
This review focuses on how LLMs evolved for code generation, key developments in open and proprietary models, and inference-time optimization techniques relevant to enhancing human-AI collaboration in interactive coding tasks.

---

## 2. Search Methodology

### Search Terms Used
- “large language models for code generation”
- “transformer architecture programming”
- “Codex”, “CodeGen2”, “StarCoder”
- “inference-time optimization”
- “prompt tuning”, “LLM decoding strategies”

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [ ] Other: Hugging Face Papers

### Time Period
2018–2024, focusing on the rapid evolution of transformer-based models and inference optimization methods.

---

## 3. Key Areas of Research

### 3.1 Evolution of Large Language Models for Code Generation
Early sequence models such as **RNNs** and **LSTMs** (Graves, 2014) showed that neural networks could process sequential data but struggled with long-term dependencies.  
The **Transformer architecture** (Vaswani et al., 2017) marked a paradigm shift, enabling global attention across input sequences and forming the foundation for modern LLMs.

**Key Papers:**
- Vaswani et al. (2017) – Introduced the Transformer model, replacing recurrence with self-attention mechanisms.  
- Graves (2014) – Demonstrated RNN-based sequence generation but highlighted limitations in scalability and memory.  

---

### 3.2 Code-Specific LLM Architectures

**OpenAI Codex (Chen et al., 2021)**  
- Fine-tuned from GPT-3 on 159 GB of GitHub code.  
- Achieved **28.8% pass@1** and **72.3% pass@100** on HumanEval.  
- Introduced the **pass@k** metric for functional accuracy.

**CodeGen2 (Nijkamp et al., 2023)**  
- Salesforce’s multilingual open-source code model.  
- Python-specialized “CodeGen2-1B P” variant optimized for efficiency.

**StarCoder (Li et al., 2023)**  
- Built by the BigCode project using a **3TB Stack dataset**.  
- Supports 80+ programming languages with open-access licensing.

**PolyCoder (Xu et al., 2022)**  
- Focused on C-family languages; achieved high efficiency with smaller architectures.

**GPT-Neo (EleutherAI, 2021)**  
- Trained on “The Pile” dataset; performs well in text but weaker in reasoning for code.

---

## 4. Research Gaps and Opportunities

### Gap 1: Lack of Lightweight Optimization Methods  
**Why it matters:** Retraining LLMs for improved code reasoning is computationally expensive.  
**How your project addresses it:** Focuses on inference-time optimization techniques (e.g., adaptive sampling, prompt tuning) that require no model retraining.

### Gap 2: Limited Human-AI Interaction Studies in Code Generation  
**Why it matters:** Current models often lack responsiveness to iterative human input during debugging or code refinement.  
**How your project addresses it:** Investigates interactive inference loops that allow human feedback to refine LLM outputs dynamically.

---

## 5. Theoretical Framework

This research is grounded in **transformer-based attention theory** (Vaswani et al., 2017) and **human-AI collaborative frameworks** emphasizing adaptive feedback.  
It draws upon **interactive machine learning (IML)** principles, where human inputs guide iterative AI adjustments during problem solving.

---

## 6. Methodology Insights

Common methodologies in existing studies include:  
- **Benchmark-driven evaluation** (e.g., HumanEval, MBPP).  
- **Prompt engineering and decoding analysis** for improving output quality.  
- **Comparative performance studies** across open-source and proprietary models.  

For this project, inference-time experimentation—varying temperature, top-p, and multi-sampling—is identified as a practical and scalable approach for optimizing model behavior.

---

## 7. Conclusion

Existing research shows that LLMs for code generation have evolved rapidly but still face challenges in compositional reasoning, correctness, and interpretability.  
Inference-time optimization provides a resource-efficient pathway for enhancing performance, aligning with the broader goal of enabling **human-AI collaborative problem solving** in programming environments.

---

## References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.  
2. Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code (Codex).* OpenAI.  
3. Nijkamp, E. et al. (2023). *CodeGen2: Lessons for Training LLMs on Programming and Natural Languages.* Salesforce AI.  
4. Li, R. et al. (2023). *StarCoder: May the Source Be with You!* BigCode Project.  
5. Xu, F. F. et al. (2022). *A Systematic Evaluation of Large Language Models of Code.* CMU.  
6. Laban, S. et al. (2021). *Mostly Basic Python Problems (MBPP).* arXiv:2108.07732.  
7. EleutherAI. (2021). *The Pile Datasets and GPT-Neo Models.*  
