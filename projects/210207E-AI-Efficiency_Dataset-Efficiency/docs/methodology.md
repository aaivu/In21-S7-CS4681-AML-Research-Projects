# Detailed Methodology: Embedding Normalization in GPT-Style Transformers

**Student Index Number:** 210207E
**Project Phase:** Final Submission (Reflects the full 5-seed, 300-iteration plan)

## 1. Model Architecture (1.6 Million Parameter nanoGPT)

The model is a custom-implemented, decoder-only Pre-LayerNorm (Pre-LN) Transformer, designed as a faithful, small-scale replica of the GPT-3 design principles.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **n_layer** | 8 | Depth of the Transformer. |
| **n_embd** | 128 | Embedding and hidden state dimensionality. |
| **n_head** | 8 | Number of attention heads. |
| **block_size** | 128 | Maximum context length (L). |
| **vocab_size** | 65 | Character vocabulary size (Shakespeare). |
| **Key Architectural Choices** | Pre-LN blocks, tied input/output embeddings, standard initialization. | GPT-3 convention maintained throughout the internal blocks. |

## 2. Experimental Variants

The research tests three variants, all strictly controlled for architecture and training budget:

| Variant Name | Normalization Location | Normalization Type | Input Flow |
| :--- | :--- | :--- | :--- |
| **Baseline (A)** | None | None | $E_{tok}(t) + E_{pos}(p) \rightarrow \text{Transformer Block}_1$ |
| **LayerNorm (LN)** | Post-Embedding | LayerNorm ($\mu=0, \sigma=1$) | $\text{LN}(E_{tok} + E_{pos}) \rightarrow \text{Transformer Block}_1$ |
| **RMSNorm (RMS)** | Post-Embedding | RMSNorm (Magnitude-only) | $\text{RMS}(E_{tok} + E_{pos}) \rightarrow \text{Transformer Block}_1$ |

## 3. Experimental Setup and Controls (Rigor)

To ensure the highest level of experimental rigor for comparison, the following controls are strictly enforced:

| Control Area | Specification | Rationale |
| :--- | :--- | :--- |
| **Dataset** | Character-level Shakespeare (1.1M characters). | Consistent, high-fidelity benchmark for convergence studies. |
| **Training Budget** | **300 iterations** (Final Phase). | Matched compute budget (Chinchilla best practice) to isolate architectural effects. |
| **Statistical Rigor** | **Five distinct random seeds** per variant. | Robust validation against random noise; final analysis via paired bootstrap testing (planned). |
| **Hyperparameters** | Identical (Batch Size=16, Context Length=128, AdamW, LR=3e-4). | Guarantees differences are solely due to the normalization layer. |

## 4. Evaluation Protocol and Metrics

Evaluation is performed after the final optimization step (iteration 300) on the validation set.

| Metric | Type | Calculation / Goal |
| :--- | :--- | :--- |
| **Validation Perplexity (PPL)** | Primary Performance | $e^{\text{Validation Loss}}$. Lower PPL is better next-token prediction performance. |
| **Final Total Gradient Norm** | Stability Metric | L2 norm of the total model gradient after the final optimization step. Lower norm indicates smoother convergence. |
| **Mock Macro Accuracy** | Secondary Performance | Percentage of correctly predicted next tokens (proxy for zero-shot accuracy). |
| **Mean Embedding Norm ** | Diagnostic Metric | Average L2 norm of the tensor entering the first Transformer block. Confirms the layers' successful operation. |

## 5. Planned Analysis (Final Step)

1.  **Performance Delta:** Quantify the PPL reduction/increase over the Baseline with confidence intervals.
2.  **Trade-off Visualization:** Plot loss/norm trends over time to visually characterize the stability-accuracy trade-off.
3.  **Statistical Validation:** Conduct full statistical analysis (paired bootstrap testing) on accuracy and PPL deltas across the five seeds to confirm the robustness of the findings.
4.  **Mechanistic Analysis:** Analyze the Mean Embedding Norm results to explain why the LN variant's centering effect (or lack thereof) leads to the observed performance difference compared to RMSNorm.

---
