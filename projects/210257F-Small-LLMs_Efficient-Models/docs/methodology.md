# Methodology: EdgeMIN - Systematic Pipeline for Compressing Transformers Towards Edge Optimization

**Student:** 210257F - Nipuni Jayathilake  
**Research Area:** Small LLMs: Efficient Models (Model Compression)  
**Supervisor:** Dr. Uthayasanker Thayasivam  
**Date:** 2025-01-24  

---

## I. Overview and Research Objective

This methodology outlines **EdgeMIN**, a systematic three-stage pipeline designed to compress transformer models for deployment on **resource-constrained edge devices** (e.g., IoT sensors, smartphones). The goal is to achieve maximal size and computational reduction (8× theoretical compression) with minimal accuracy degradation ($<2\%$ loss).

### A. Model Baselines

| Model | Role | Architecture | Key Dimensions | Parameters (M) |
| :--- | :--- | :--- | :--- | :--- |
| **DistilBERT-base-uncased** | Teacher ($E_T$) | 6 Layers, 12 Heads | 768 Hidden Dim | 66.96M |
| **MiniLM-like** | Student ($E_S$) | 12 Layers, 12 Heads | 384 Hidden Dim | 33.36M |

### B. Validation Strategy

The pipeline is validated on the **SST-2** sentiment classification task from the GLUE benchmark. Efficiency is measured using practical metrics on a **Google Colab standard CPU** to simulate edge constraints: actual file size (MB), parameter count (M), theoretical FLOPs (Billion), and CPU inference latency (ms).

---

## II. The EdgeMIN Pipeline (Sequential Stages)

EdgeMIN applies compression in a strict order: **Distill $\rightarrow$ Prune Heads $\rightarrow$ Quantize/Prune FFN**.

### A. Stage 1: MiniLMv2 Relational Distillation

**Goal:** Create a strong student model by transferring core semantic understanding.

**Method:** Relation-based Knowledge Distillation (KD) is used. It minimizes the KL divergence between the teacher and student's **self-attention relation matrices** ($R^{(i)}$) for Queries (Q), Keys (K), and Values (V):

$$
L_{\text{distill}} = \sum_{l=1}^{L_S} \sum_{i \in \{Q,K,V\}} \text{KL}(R^{(i)}_{T,l} \parallel R^{(i)}_{S,l}) \tag{1}
$$

**Rationale:** This focuses on token interactions, providing flexibility to handle the student's different hidden dimension (384 vs. 768).

### B. Stage 2: Structured Attention Head Pruning

**Goal:** Remove computationally redundant attention heads, reducing parameters and FLOPs.

**Method:** **Structured pruning** based on **gradient magnitude** is employed. The importance score ($I_h$) for each head $h$ is computed from the task loss ($L_{\text{task}}$):

$$
I_h = \parallel \nabla_{a_h} L_{\text{task}} \parallel_2 \tag{2}
$$

The lowest-scoring **20%** of heads are permanently removed. This is followed by a short recovery fine-tuning phase (2 epochs).

**Rationale:** Structured removal creates a smaller, dense model, yielding direct **FLOPs reduction** and minimal accuracy impact.

### C. Stage 3: Aggressive Post-Training Dynamic Quantization (PTQ)

**Goal:** Drastically reduce memory footprint and achieve maximal parameter reduction.

**Method:** **Dynamic PTQ** converts FP32 linear layer weights to **INT8** format. This theoretically provides $4\times$ memory compression.

$$
w_{\text{INT8}} = \text{clamp}(\text{round}(w/\text{scale} + \text{zero\_point}), q_{\min}, q_{\max}) \tag{3}
$$

**Aggressiveness & Key Result:** The quantization process leads to an observed substantial parameter drop (to **11.94M**), indicating **implicit pruning** of near-zero neurons within the Feed-Forward Network (FFN) layers, beyond simple INT8 conversion.

---

## III. Experimental Protocol and Evaluation

### A. Pipeline Order Justification

The sequence is chosen for progressive refinement:
1.  **Distill First:** Ensures optimal knowledge transfer to the full student capacity.
2.  **Prune Heads Second:** Reduces complexity before the final, precision-altering step.
3.  **Quantize Last:** Applies maximum memory reduction and aggressive FFN pruning to the already compacted model.

### B. Efficiency Metrics Measurement

All models are evaluated under identical conditions on the Colab CPU:

| Metric | Measurement Details |
| :--- | :--- |
| **File Size (MB)** | Measured via $\text{os.path.getsize}$ on the saved model file. |
| **Parameter Count (M)** | Sum of $\text{p.numel()}$ for all $\text{model.parameters()}$. |
| **CPU Latency (ms)** | Average time over **100 runs** on a single sample (Batch Size = 1), following **10 warm-up inferences**. |
| **FLOPs (Billion)** | Measured using $\text{thop.profile}$ on a single input sequence (length 128). |
