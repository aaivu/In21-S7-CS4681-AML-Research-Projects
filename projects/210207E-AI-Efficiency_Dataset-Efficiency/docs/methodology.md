# Methodology: AI Efficiency:Dataset Efficiency

**Student:** 210207E
**Research Area:** AI Efficiency:Dataset Efficiency (Embedding Normalization)
**Date:** 2025-10-20 (Refined after Preliminary Experiments)

## 1. Overview

The methodology details a rigorous, controlled ablation study designed to evaluate the impact of **Embedding Normalization** (LayerNorm or RMSNorm) when applied immediately post-embedding lookup in a GPT-3–style, **Pre-LayerNorm Transformer**. The study is conducted on a small, custom-implemented **'nanoGPT'** model (0.8 million parameters) to ensure high statistical rigor and resource feasibility through multi-seed validation. The core objective is to characterize the stability-accuracy trade-off at this scale and validate or contradict findings from billion-parameter studies.

## 2. Research Design

The design is a **Controlled Ablation Study** comparing three model variants under an identical computational budget:

*   **Design Type:** A/B/C comparison (Baseline vs. LayerNorm vs. RMSNorm).
*   **Control Mechanism:** Strictly fixed token budget (initial: 150 iterations, final: 300 iterations) and identical hyperparameters across all variants.
*   **Validation:** Multi-seed validation (preliminary: 3 seeds; final: 5 seeds) to distinguish true architectural effects from random noise.
*   **Objective:** Next-token prediction on a character-level language modeling task.

## 3. Data Collection

### 3.1 Data Sources
*   **Training and Evaluation Data:** Character-level Shakespeare dataset (approximately 1.1 million characters).

### 3.2 Data Description
*   A small, consistent, and complex dataset that serves as a high-fidelity benchmark for analyzing model convergence and learning dynamics. The use of a character-level task and small scale is intentional to make multi-seed validation feasible.

### 3.3 Data Preprocessing
*   **Tokenization:** Character-level tokenization.
*   **Context Length:** Fixed maximum context length of $L = 128$ tokens.
*   **Batching:** Fixed batch size of $B = 16$.

## 4. Model Architecture

The experiments use a custom-implemented, **0.8 million parameter** Pre-LN Transformer model (referred to as 'Max Mini-GPT' or 'nanoGPT' architecture).

*   **Configuration:**
    *   Depth ($N$): 8 Transformer layers.
    *   Embedding/Hidden State Dimensionality ($D$): 128.
    *   Attention Heads ($H$): 8.
    *   Core Components: Maintains Pre-LN blocks, tied input/output embeddings, and standard weight initialization schemes.

### Experimental Variants (Corrected Input Flow Formatting)

| Variant | Name | Input Flow | Description |
| :---: | :---: | :--- | :--- |
| **A** | **Baseline** | $\mathbf{X}_{\text{input}} = \mathbf{E}_{\text{tok}}(t) + \mathbf{E}_{\text{pos}}(p) \to \text{Transformer Block}_1$ | Control group. Embeddings summed and fed directly. |
| **B** | **LayerNorm (LN)** | $\mathbf{X}_{\text{input}} = \text{LN}(\mathbf{E}_{\text{tok}}(t) + \mathbf{E}_{\text{pos}}(p)) \to \text{Transformer Block}_1$ | Adds standard LayerNorm (zero mean, unit variance) post-embedding. |
| **C** | **RMSNorm (RMS)** | $\mathbf{X}_{\text{input}} = \text{RMSNorm}(\mathbf{E}_{\text{tok}}(t) + \mathbf{E}_{\text{pos}}(p)) \to \text{Transformer Block}_1$ | Adds simpler RMSNorm (magnitude-only scaling) post-embedding. |

## 5. Experimental Setup

### 5.1 Evaluation Metrics

The evaluation uses a suite of four distinct metrics to measure both performance and training stability:

1.  **Primary Metric (Performance):** **Validation Perplexity (PPL)** ($\downarrow$). The gold standard for language model quality; lower is better.
2.  **Stability Metric:** **Final Total Gradient Norm** ($\downarrow$). Measures the magnitude of weight updates at the end of training; lower suggests a smoother, more stable minimum.
3.  **Secondary Metric (Performance Proxy):** **Mock Macro Accuracy** ($\uparrow$). Simple next-token prediction accuracy.
4.  **Diagnostic Metric:** **Mean Embedding Norm** ($\Vert \mathbf{X}_{\text{input}} \Vert$). Confirms the normalization layers are actively modulating the feature vector magnitude.

### 5.2 Baseline Models

The **Baseline (Variant A)** serves as the sole baseline model for all comparisons. Performance and stability metrics for Variants B (LN) and C (RMS) are measured relative to the Baseline's un-normalized performance.

### 5.3 Hardware/Software Requirements

*   **Optimizer:** AdamW with a learning rate of $3 \times 10^{-4}$.
*   **Budget Control:** Fixed training iterations (300 in the final phase) to ensure an exact match on computational budget.
*   **Software:** Custom implementation (likely PyTorch/TensorFlow based) focused on architectural fidelity to the GPT-3 design.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1: Preliminary Experiment** (Completed) | Train all three variants for 150 iterations on 3 random seeds. Perform initial analysis of PPL, Gradient Norm, and Mock Accuracy. | 3 Weeks | Preliminary Results (Table 1 & Figures 1-3). Validation of Methodology. |
| **Phase 2: Extended Rigor** | Extend training for all variants to **300 iterations** (fixed final budget). Expand validation to **5 distinct random seeds**. | 6 Weeks | Complete Training Logs for 5 Seeds/300 Iterations. Saved Checkpoints. |
| **Phase 3: Trade-off Characterization** | Conduct full statistical analysis: paired bootstrap testing on PPL and accuracy deltas; Holm-Bonferroni correction; plot loss/norm trends over time. | 4 Weeks | Statistical Analysis Report. Diagnostic Visualizations (Embedding/Gradient Norms). |
| **Phase 4: Finalization** | Synthesize all results and diagnostics into a coherent conclusion. Complete the conference-ready research paper and all project documentation. | 3 Weeks | Final Research Paper. Complete `src/` and `results/` folders. |

## 7. Risk Analysis

| Potential Risk | Mitigation Strategy |
| :--- | :--- |
| **Non-Transferability of Results** | Acknowledge in the Discussion that findings are specific to the 'nanoGPT' scale; recommend validation at a larger scale as Future Work. |
| **Catastrophic Instability** | The consistency of gradient norms across the preliminary runs suggests low risk; continuous monitoring of loss and gradient norms for early warning. |
| **Misinterpretation of Performance** | Use the comprehensive **PPL** as the primary metric (not Mock Accuracy) to measure model quality, mitigating risk of relying on a simplistic proxy. |
| **Insufficient Statistical Power** | Commit to **5 seeds** in the final phase and use paired bootstrap testing to maximize confidence in performance deltas. |

## 8. Expected Outcomes

The final outcome will be a **Conference-Ready Research Paper** providing a conclusive, data-driven answer on whether embedding normalization is beneficial or detrimental at this specific scale and architectural context. The key contribution is the characterization of the **stability-accuracy trade-off** and a supported recommendation to practitioners for either adopting or avoiding this technique.

---

**Note:** This document reflects the refined methodology following the preliminary experiments where the Baseline was outperformed by the LayerNorm variant, necessitating a full 5-seed, 300-iteration study to confirm the stability-accuracy trade-off.
