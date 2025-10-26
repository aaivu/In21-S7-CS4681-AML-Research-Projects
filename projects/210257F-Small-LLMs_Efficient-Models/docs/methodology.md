# Methodology: EdgeMIN — A Systematic Compression Pipeline for Efficient Transformer Deployment on Edge Devices

**Student:** 210257F <br>
**Research Area:** NLP: Model Compression and Edge Optimization <br>
**Date:** 2025-09-26 <br>

---

## 1. Overview

This methodology outlines the systematic approach used in **EdgeMIN**, a three-stage transformer compression pipeline designed to enable efficient deployment of large language models on **resource-constrained edge devices** such as smartphones, IoT sensors, and embedded systems.

Large transformer models (e.g., BERT, RoBERTa, DistilBERT) achieve outstanding natural language understanding performance but are computationally expensive and memory-intensive, making them impractical for edge applications. EdgeMIN addresses this challenge by **combining relational knowledge distillation, structured attention head pruning, and aggressive post-training quantization** to drastically reduce model size and inference latency while maintaining competitive accuracy.

The pipeline follows a reproducible design focusing on measurable CPU metrics — file size, parameter count, FLOPs, and latency — ensuring that improvements are verifiable without specialized hardware.

---

## 2. Research Design

The research follows an **experimental and stage-wise compression design**, progressively transforming a pretrained transformer model into a lightweight, edge-optimized version.

The main objectives are to:
- Compress large transformer models without major accuracy loss.
- Integrate complementary compression methods (distillation, pruning, and quantization) into one systematic pipeline.
- Evaluate model efficiency on **CPU-only environments** to simulate edge deployment.
- Quantify tradeoffs between compression ratio, latency, and task accuracy.

The design includes three sequential stages:

1. **Stage 1 – Relational Knowledge Distillation:** Transfers semantic and attention-based knowledge from a larger teacher model (DistilBERT) to a smaller MiniLMv2-based student.
2. **Stage 2 – Structured Attention Head Pruning:** Removes computationally redundant attention heads using gradient-based importance scoring.
3. **Stage 3 – Post-Training Dynamic Quantization:** Applies aggressive INT8 quantization with implicit feed-forward layer pruning for maximum size and latency reduction.

---

## 3. Data Collection

### 3.1 Data Sources
- **Hugging Face Datasets:** GLUE benchmark datasets (SST-2, MNLI, QQP).
- **Hugging Face Models:** Pretrained models from the `transformers` library (`distilbert-base-uncased`, `MiniLM-L12-H384-uncased`).

### 3.2 Data Description
Experiments were primarily conducted using the **SST-2 dataset** (Stanford Sentiment Treebank), a binary sentiment classification benchmark from GLUE.  
Zero-shot evaluations were also performed on **MNLI** (natural language inference) and **QQP** (paraphrase detection) to validate cross-task generalizability.

| **Dataset** | **Task Type** | **Train Samples** | **Validation Samples** | **Input Length** |
|--------------|----------------|-------------------|-------------------------|------------------|
| SST-2 | Sentiment Analysis | 67,349 | 872 | 128 tokens |
| MNLI | Natural Language Inference | 392,702 | 9,815 | 128 tokens |
| QQP | Paraphrase Detection | 363,870 | 40,430 | 128 tokens |

### 3.3 Data Preprocessing
All datasets were processed using a standardized tokenization and preprocessing pipeline:
- Text cleaning to remove non-printable characters and excessive whitespace.
- Tokenization using the **Hugging Face DistilBERT tokenizer**.
- Dynamic padding and truncation to maintain a fixed sequence length of 128 tokens.
- Lowercasing and subword tokenization consistent with pretrained vocabulary.

This preprocessing ensures consistent model input representation and compatibility with both teacher and student architectures.

---

## 4. Model Architecture

The **EdgeMIN pipeline** is based on a **teacher–student framework** where a large transformer (DistilBERT) transfers knowledge to a compact MiniLM-based student model.

### 4.1 Teacher and Student Models
- **Teacher:** DistilBERT-base-uncased (6 layers, 768 hidden size, 12 attention heads, ~67M parameters)
- **Student:** MiniLM-L12-H384-uncased (12 layers, 384 hidden size, 12 attention heads, ~33M parameters)

### 4.2 Compression Pipeline Design

#### **Stage 1: MiniLMv2 Relational Knowledge Distillation**
- Transfers **self-attention relation matrices** (Q-Q, K-K, V-V) from teacher to student using KL divergence loss.
- Enables flexible head and dimension alignment between models.
- Trained for 500 steps using AdamW (learning rate 5×10⁻⁵, batch size 8).
- Produces a smaller model retaining most of the teacher’s reasoning capability.

#### **Stage 2: Structured Attention Head Pruning**
- Calculates per-head importance based on gradient magnitude:
  \[
  I_h = ||\nabla_{a_h} L_{task}||^2
  \]
- Removes the bottom 20% least important attention heads.
- Followed by 2 epochs of fine-tuning (learning rate 2×10⁻⁵).
- Reduces parameters, FLOPs, and latency while maintaining accuracy.

#### **Stage 3: Post-Training Dynamic Quantization**
- Converts FP32 linear weights to INT8 using `torch.quantization.quantize_dynamic`.
- Achieves ~4× memory compression and ~37% CPU latency improvement.
- Implicitly prunes redundant feed-forward (FFN) components, further reducing parameters to 11.9M.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **Accuracy (%)** – Measured on GLUE validation sets.
- **Model Size (MB)** – File size of saved model weights.
- **Parameter Count (M)** – Total trainable parameters.
- **FLOPs (Billions)** – Estimated using `thop.profile` for sequence length 128.
- **CPU Latency (ms/sample)** – Measured on Colab CPU over 100 runs.
- **Throughput (samples/sec)** – Derived from latency measurements.

### 5.2 Hardware/Software Configuration
- **Platform:** Google Colab (CPU-only)
- **Frameworks:** PyTorch 2.0.1, Transformers 4.36.0, Python 3.10
- **Libraries:** `torch`, `thop`, `time`, `os`
- **Seed:** 42 for reproducibility

---

## 6. Implementation Plan

| **Phase** | **Tasks** | **Duration** | **Deliverables** |
|------------|-----------|---------------|------------------|
| Preparation | Literature review on model compression and setup of training environment | Week 1–2 | Initial design and baseline model setup |
| Stage 1 Implementation | Apply MiniLMv2 relational distillation from DistilBERT to MiniLM | Week 3 | Distilled student model |
| Stage 2 Implementation | Perform structured attention head pruning and fine-tune | Week 4–5 | Pruned model with reduced parameters |
| Stage 3 Implementation | Apply post-training quantization and benchmark CPU latency | Week 6 | Fully compressed and quantized model |
| Evaluation & Documentation | Compare performance, generate ablation results, finalize papers | Week 7–8 | Results tables, plots, final report and presentation |

---

## 7. Risk Analysis

| **Risk** | **Description** | **Mitigation Strategy** |
|-----------|----------------|--------------------------|
| Accuracy Degradation | Cumulative effects from pruning and quantization may lower task accuracy | Apply minimal pruning (20%), use fine-tuning to recover lost accuracy |
| Limited Hardware | Evaluation limited to CPU-only setup | Use smaller datasets and batch sizes, simulate edge-like environments |
| Implementation Complexity | Combining three compression stages requires compatibility checks | Modular pipeline design and staged evaluation |
| Quantization Overhead | Dynamic quantization can increase CPU latency due to dequantization | Future optimization with static quantization or INT8-accelerated hardware |

---

## 8. Expected Outcomes

- **Model Efficiency:** 1.96× reduction in model size, 2.8× reduction in parameters, and 2.6× reduction in FLOPs.
- **Performance Retention:** Maintains ~90% accuracy on SST-2 with less than 1.5% drop from the teacher.
- **Latency Improvement:** ~37% faster inference on standard CPU.
- **Deployment Readiness:** Final model (~65MB, 11.9M parameters) suitable for real-time edge inference on devices with <2GB RAM.
- **Reproducibility:** Fully reproducible pipeline implemented using PyTorch and Hugging Face libraries.


