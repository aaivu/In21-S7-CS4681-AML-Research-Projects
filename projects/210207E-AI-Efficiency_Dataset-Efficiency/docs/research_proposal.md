# Research Proposal: Testing Embedding Normalization for Stability and Performance in GPT-Style Transformers

**Student Index Number:** 210207E
**Student Name:** J. Harismenan
**Course:** CS4681 - Advanced Machine Learning Project Assignment
**Date:** [Assuming Week 4 of the timeline, e.g., Sept 2025]

## 1. Project Objective and Scope

The primary objective is to conduct a rigorous, controlled evaluation of **Embedding Normalization**—the introduction of a single normalization layer (LayerNorm or RMSNorm) immediately following the combined token and positional embedding—in a GPT-3-style, decoder-only Pre-LayerNorm (Pre-LN) Transformer architecture.

The core question is: Does this architectural enhancement improve training stability and/or final model performance in a resource-constrained, but highly controlled, setting?

### Objectives:
1.  **Quantitative Assessment:** Measure the impact of embedding normalization on the primary metric (Perplexity) and a proxy for zero-shot accuracy (Mock Macro Accuracy) under strictly controlled, equal-compute training conditions.
2.  **Stability Analysis:** Investigate whether LayerNorm (LN) and RMSNorm (RMS) variants improve training stability compared to the un-normalized Baseline, using Total Gradient Norm as the stability metric.
3.  **Trade-off Characterization:** Characterize the stability-accuracy trade-off to provide guidance on when embedding normalization is justified.
4.  **Best Practice Guidance:** Formulate actionable recommendations for its use in Pre-LN Transformer training pipelines.

## 2. Baseline Model and Variants

The experiment will use a custom-implemented, small-scale GPT-style architecture, often referenced as a 'nanoGPT', to enable multi-seed ablation studies which are computationally infeasible on the original GPT-3 scale.

| Component | Setting (Initial/Mid-Evaluation) | Setting (Final/Target) |
| :--- | :--- | :--- |
| **Architecture** | Pre-LN Transformer (GPT-3 style) | Pre-LN Transformer (GPT-3 style) |
| **Model Size** | 0.8 million parameters | **1.6 million parameters** |
| **Configuration** | 8 layers, 128 dim, 8 heads, 128 context length | 8 layers, 128 dim, 8 heads, 128 context length |
| **Dataset** | Character-level Shakespeare (1.1 million characters) | Character-level Shakespeare (1.1 million characters) |
| **Training Budget** | 150 iterations, 3 random seeds | **300 iterations**, **5 distinct random seeds** |

### Experimental Variants:
1.  **Variant A (Baseline):** No normalization after embedding. $X_{input} = E_{tok}(t) + E_{pos}(p) \rightarrow \text{Transformer Block}_1$
2.  **Variant B (LayerNorm):** Standard LayerNorm (LN) immediately post-embedding. $X_{input} = \text{LayerNorm}(E_{tok}(t) + E_{pos}(p)) \rightarrow \text{Transformer Block}_1$
3.  **Variant C (RMSNorm):** RMSNorm (RMS) immediately post-embedding. $X_{input} = \text{RMSNorm}(E_{tok}(t) + E_{pos}(p)) \rightarrow \text{Transformer Block}_1$

## 3. Key Metrics

| Metric Category | Metric Name | Purpose |
| :--- | :--- | :--- |
| **Primary Performance** | Validation Perplexity (PPL) | Core measure of language model quality (Lower is better). |
| **Stability** | Final Total Gradient Norm | Measures how settled the model is at the end of training (Lower is better). |
| **Secondary Performance** | Mock Macro Accuracy | Proxy for zero-shot generalization. |
| **Diagnostic** | Mean Embedding Norm | Confirms the normalization layers are functioning as intended. |

## 4. Initial Hypothesis (Based on Literature Review)

**Risk Hypothesis ($H_{1a}$):** Following large-scale studies (e.g., Le Scao et al.), it is expected that embedding normalization will be **detrimental to zero-/few-shot performance** (higher PPL) even if it provides stability benefits (lower Grad Norm) for a stable Baseline. The focus will be on quantifying this potential **stability-accuracy trade-off**.

## 5. Deliverables

*   Project timeline and methodology outline (Progress Evaluation).
*   Short paper submission with preliminary results (Mid-Evaluation).
*   Complete research paper (6-8 pages, conference format).
*   Code implementation, including training logs and configurations.
*   Statistical analysis reports (means, standard deviations, confidence intervals).

---
