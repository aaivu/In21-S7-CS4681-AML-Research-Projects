# Methodology: AI Efficiency:Embedding Normalization
**Student:** 210207E **Research Area:** AI Efficiency:Dataset Efficiency **Date:** [Date corresponding to Week 8]

## 1. Overview

The methodology centers on a rigorous, controlled ablation study to evaluate the impact of Embedding Normalization (LayerNorm or RMSNorm) when applied directly after the token and positional embeddings in a Pre-LayerNorm (Pre-LN) GPT-style Transformer. The goal is to conclusively characterize the stability-accuracy trade-off at a resource-constrained scale.

## 2. Research Design

The research employs a **comparative experimental design** with three strictly controlled architectural variants (Baseline, LayerNorm, RMSNorm). All training runs are performed under an **equal computational budget** and are validated using **five distinct random seeds** to ensure statistical robustness and isolate the performance effect solely to the normalization layer.

## 3. Data Collection

### 3.1 Data Sources
*   The primary data source is the **character-level Shakespeare dataset**, derived from the complete works of Shakespeare (as commonly used in character-level language modeling benchmarks).

### 3.2 Data Description
*   **Total Size:** Approximately 1.1 million characters.
*   **Vocabulary Size:** 65 unique characters.
*   **Split:** 90% for training (1,003,854 characters) and 10% for validation (111,540 characters).
*   **Task:** Next-token prediction (Character-level Language Modeling).

### 3.3 Data Preprocessing
*   Characters are mapped to integers (token IDs) using a custom character-to-index (`stoi`) mapping.
*   The entire text is tokenized into a sequence of unsigned 16-bit integers (`np.uint16`).
*   The data is loaded in mini-batches, where each batch contains a stack of sequences of length $L=128$, sampled randomly from the training set.

## 4. Model Architecture

The model is a custom-implemented, **decoder-only Pre-LayerNorm (Pre-LN) Transformer**, designed to replicate the architectural principles of GPT-3 at a small scale.

| Component | Specification | Total Parameters |
| :--- | :--- | :--- |
| **Model Type** | GPT-style Pre-LN Transformer | $\approx$ 1.6 Million |
| **Depth ($N$)** | 8 Transformer layers | |
| **Dimensionality ($D$)** | 128 embedding/hidden size | |
| **Attention Heads ($H$)** | 8 attention heads | |
| **Context Length ($L$)** | 128 tokens | |
| **Core Principle** | Pre-LN blocks, tied input/output embeddings, standard initialization. | |

### Experimental Variants:
1.  **Baseline (A):** $X_{input} = E_{tok}(t) + E_{pos}(p)$ (No change).
2.  **LayerNorm (LN):** $X_{input} = \text{LayerNorm}(E_{tok}(t) + E_{pos}(p))$ (Centering and scaling).
3.  **RMSNorm (RMS):** $X_{input} = \text{RMSNorm}(E_{tok}(t) + E_{pos}(p))$ (Magnitude-only scaling).

## 5. Experimental Setup

### 5.1 Evaluation Metrics
| Metric | Category | Goal |
| :--- | :--- | :--- |
| **Validation Perplexity (PPL)** | Primary Performance | $e^{\text{Validation Loss}}$. Lower is better. |
| **Final Total Gradient Norm** | Stability Metric | L2 norm of the total model gradient after final step. Lower is better. |
| **Mock Macro Accuracy** | Secondary Performance | Next-token prediction accuracy (proxy for zero-shot capability). |
| **Mean Embedding Norm ($||X_{input}||$)** | Diagnostic Metric | Average L2 norm of the input to the first block. Confirms layer function. |

### 5.2 Baseline Models
The primary baseline is the **un-normalized Baseline (Variant A)**, which is the SOTA standard for stable GPT-style training. The normalized variants (LN and RMS) are compared against this Baseline to measure performance deltas and stability changes.

### 5.3 Hardware/Software Requirements
*   **Hardware:** Single GPU (e.g., NVIDIA T4 or equivalent).
*   **Software:** Python 3.x, PyTorch (GPU enabled), NumPy, Pandas, Matplotlib.
*   **Training Parameters:** Batch Size=16, Learning Rate=3e-4, AdamW optimizer.

## 6. Implementation Plan

| Phase | Tasks | Duration (Weeks) | Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Baseline model architecture (1.6M) and data pipeline setup. | 2 (Completed Week 6) | Working Baseline model, `train.bin`, `val.bin`. |
| **Phase 2** | Implement LN and RMS variants. Run 3-seed, 150-iter prelim experiments. | 2 (Completed Week 8) | Mid-Evaluation Short Paper, Preliminary Results (20.64 PPL for LN). |
| **Phase 3** | Execute **Final Rigor runs**: 5-seed, 300-iteration training for all 3 variants. | 3 (Completed Week 11) | `results_final_project_summary.csv`, all model checkpoints. |
| **Phase 4** | Statistical analysis (paired bootstrap), plotting, and paper finalization. | 1 | Final Research Paper, Code Repository, Conference Submission Proof. |

## 7. Risk Analysis

| Risk | Mitigation Strategy |
| :--- | :--- |
| **Computational Resource Limits** | Used a highly efficient 'nanoGPT' (1.6M parameters) on a character-level task; strictly matching the compute budget across all 15 runs. |
| **Non-Transferability of Findings** | Using multiple variants (LN vs RMS) and multiple scales (implicit comparison to large-scale literature) to ensure the finding's context is well-defined. |
| **Training Instability** | The Pre-LN architecture is inherently stable; the gradient norm metric is specifically monitored to detect and characterize any stability issues introduced by the new layers. |

## 8. Expected Outcomes

The primary expected outcome is the **conclusive characterization of the stability-accuracy trade-off** for embedding normalization in a stable, small-scale GPT-style Transformer.

*   **Contribution:** Provide new empirical evidence that either strongly validates the recommendation to **avoid embedding normalization** (due to accuracy regression) or, less likely but possible, demonstrates a conditional performance benefit at this specific scale.
*   **Deliverables:** A conference-ready research paper with statistically robust results, diagnostic plots, and clear best-practice guidance.

---
