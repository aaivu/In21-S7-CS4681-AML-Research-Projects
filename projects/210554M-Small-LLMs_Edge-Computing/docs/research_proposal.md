# Research Proposal: Small LLMs: Edge Computing

**Student:** 210554M  
**Research Area:** Small LLMs: Edge Computing  
**Date:** 2025-10-16

## Abstract

Small Large Language Models (LLMs) offer promising avenues for on-device natural language reasoning, yet their deployment on mobile and edge devices is constrained by computational, memory, and power limitations. Quantization, which reduces numerical precision, has emerged as an effective approach to compress models but often degrades reasoning accuracy. This research investigates quantization strategies for small LLMs, using Qwen 0.5B as a representative case study. The study benchmarks float32, post-training quantization (PTQ), and quantization-aware training (QAT) to analyze trade-offs between accuracy, memory, and inference efficiency on CPU-based devices. An enhancement is proposed by combining QAT with Knowledge Distillation (KD) from a larger teacher model (Qwen 1.5B), aiming to align the outputs of the quantized student model with the softened outputs of the teacher to improve reasoning performance while retaining efficiency. The project ultimately aims to inform practical deployment of small LLMs on resource-constrained edge devices.

## 1. Introduction

Large Language Models (LLMs) demonstrate remarkable capabilities in natural language understanding and reasoning but require immense computational resources. This limits deployment on mobile or embedded systems. Cloud-based inference introduces latency and privacy concerns, motivating on-device optimization. Quantization reduces numerical precision to achieve compression, though often at the cost of accuracy. Quantization-Aware Training (QAT) and Knowledge Distillation (KD) together can help bridge this performance gap, making small LLMs viable for edge reasoning applications.

## 2. Problem Statement

How can quantization techniques be adapted to small LLMs to achieve practical, accurate, and efficient on-device inference without sacrificing reasoning quality?

## 3. Literature Review Summary

Recent studies highlight that 4-bit quantization (Wallace et al., 2025; Skial et al., 2025) significantly reduces energy use while retaining acceptable accuracy. Quantization-Aware Training (Bondarenko et al., 2024; Dettmers et al., 2023) allows models to adapt to quantized operations, achieving near-float precision results. Knowledge Distillation (Bhardwaj et al., 2024) further mitigates performance degradation by aligning student and teacher distributions. Frameworks like Hugging Face Transformers and BitsAndBytes enable accessible 4-bit QAT and inference pipelines. However, existing literature lacks systematic evaluation of combined QAT + KD for small LLMs targeting CPU/edge deployment, which this research addresses.

## 4. Research Objectives

### Primary Objective
To evaluate and enhance quantization strategies for small LLMs to enable efficient and accurate on-device inference.

### Secondary Objectives
- Benchmark Qwen 0.5B under float32, PTQ, and QAT configurations.  
- Implement QAT combined with Knowledge Distillation (Qwen 1.5B → 0.5B).  
- Evaluate across reasoning and language modeling benchmarks (WikiText-2, BoolQ, PIQA).  
- Analyze trade-offs in latency, accuracy, and memory for edge deployment feasibility.

## 5. Methodology

1. **Baseline Setup:** Evaluate Qwen 0.5B in full precision (FP32) as a reference.  
2. **Post-Training Quantization (PTQ):** Quantize trained weights to int8/int4 and assess performance drop.  
3. **Quantization-Aware Training (QAT):** Fine-tune with simulated quantization using fake quantization modules and Straight-Through Estimation.  
4. **Knowledge Distillation Enhancement:** Integrate KD loss using Qwen 1.5B as a teacher to improve quantized model accuracy.  
5. **Evaluation:** Compare models using metrics such as accuracy (BoolQ, PIQA), perplexity (WikiText-2), latency (ms/token), and memory (MB).

## 6. Expected Outcomes

- Quantized (int4/int8) LLMs achieving near-FP32 accuracy.  
- Demonstration of QAT + KD as an effective enhancement for quantized reasoning models.  
- Benchmarks validating practical deployment of Qwen 0.5B on CPU/edge hardware.  
- Reproducible results with documented training scripts and evaluation metrics.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Baseline Setup |
| 5-6  | Quantization Experiments (PTQ, QAT) |
| 7-9  | QAT + KD Enhancement |
| 10-12| Evaluation & Benchmarking |
| 13-15| Analysis and Paper Writing |
| 16   | Final Submission |

## 8. Resources Required

- GPU for QAT fine-tuning and KD training  
- CPU/microcontroller for deployment benchmarks  
- Hugging Face Transformers, BitsAndBytes, PEFT libraries  
- Datasets: WikiText-2, BoolQ, PIQA, GSM8K  
- Python environment with PyTorch and Accelerate

## References

1. Jacob, B. et al. (2017). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *arXiv:1712.05877.*  
2. Wallace, T. et al. (2025). Optimization strategies for enhancing resource efficiency in transformers large language models. *arXiv:2502.00046.*  
3. Skial, S. et al. (2025). LLM compression: How far can we go in balancing size and performance? *arXiv:2508.11318.*  
4. Bondarenko, Y. et al. (2024). Low-rank quantization-aware training for LLMs. *arXiv:2406.06385.*  
5. Dettmers, T. et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023.*  
6. Bhardwaj, K. et al. (2024). Improving quantized knowledge distillation via signal propagation analysis. *arXiv:2403.18159.*  
7. Liu, Y. et al. (2024). Evaluating the generalization ability of quantized LLMs. *arXiv:2406.12928.*  
8. Husom, E. et al. (2025). Sustainable LLM inference for edge AI. *arXiv:2504.03360.*  
9. Lu, Z. et al. (2025). Small Language Models: Survey, measurements, and insights. *arXiv:2409.15790.*

---