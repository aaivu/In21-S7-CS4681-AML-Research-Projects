# Research Proposal: AI Efficiency - Dataset Efficiency

**Student:** J. HARISMENAN (210207E)
**Research Area:** AI Efficiency: Dataset Efficiency (Specifically, Embedding Normalization in Large Transformers)
**GitHub:** @Janarthanan-Harismenan
**Email:** janarthanan.21@cse.mrt.ac.lk

## 1. Project Objectives

This project aims to conduct a rigorous empirical evaluation of embedding normalization techniques in GPT-3–style decoder-only Pre-LayerNorm Transformers. The study focuses on adding a single normalization layer (LayerNorm or RMSNorm) immediately following the token embedding and positional embedding lookup.

The overarching objectives include:

*   **Quantitative Assessment:** Measure the impact of embedding normalization on macro zero- and few-shot accuracy across a comprehensive suite of language tasks under strictly controlled, equal-compute training conditions.
*   **Stability Analysis:** Investigate whether embedding normalization improves training stability, particularly in settings showing early divergence or instability.
*   **Trade-off Evaluation:** Characterize the trade-offs between potential stability improvements and possible regressions in accuracy or calibration, guiding decisions on whether embedding normalization is justified.
*   **Best Practice Guidance:** Produce actionable recommendations regarding embedding normalization use in large Transformer training pipelines supported by reproducible, statistically robust empirical evidence.

## 2. Scope and Deliverables

The project is bounded by the following key architectural and training choices:

### Baseline Model and Setup
*   **Architecture:** GPT-3–style, decoder-only Transformer using Pre-LayerNorm architecture.
*   **Objective:** Trained on next-token prediction.
*   **Embeddings:** Learned token and absolute positional embeddings; tied input/output embeddings.
*   **Optimization:** AdamW optimizer with warmup and cosine decay schedule.
*   **Controls:** Fixed tokenizer, data pipeline, model configurations, hyperparameters, and training token budgets (e.g., 50 billion tokens).

### Experimental Variants
*   **Embedding Normalization Variants:** Introduce a single normalization layer (LayerNorm or RMSNorm) applied immediately after the embedding lookup and positional addition, before the first Transformer block. All other architectural and training settings remain identical to the baseline.

### Evaluation Protocol
*   **Frameworks:** Use LM-Eval or T0-Eval frameworks with fixed prompts.
*   **Metrics:** Zero- and few-shot evaluation metrics, including macro accuracy, perplexity, and calibration scores.

### Deliverables
1.  Well-documented training logs and configurations for reproducibility.
2.  Statistical analysis reports including effect sizes, confidence intervals, and multiple hypothesis corrections.
3.  Diagnostic visualizations detailing embedding norm behavior, gradient statistics, and inference calibration.
4.  A conference-ready research paper summarizing methodology, results, and practical recommendations.

## 3. Hypotheses

*   **H0 (Null):** Embedding normalization does not affect zero-/few-shot macro accuracy when training is stable without it.
*   **H1a (Risk Hypothesis):** Embedding normalization decreases zero-/few-shot accuracy despite smoothing or stabilizing training dynamics [referencing Le Scao et al., 2022].
*   **H1b (Conditional Benefit):** If baseline training shows instability or early divergence, embedding normalization may increase stability but at the potential expense of some downstream performance, necessitating a stability-accuracy tradeoff analysis.
