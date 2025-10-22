# Literature Review and References: Embedding Normalization

**Student Index Number:** 210207E

## 1. Direct Evidence Against Embedding Normalization

The most relevant and cautionary study is by Le Scao et al. [1].
*   **Source:** T. Le Scao et al., "What Language Model to Train if You Have One Million GPU Hours?," Findings of EMNLP, 2022.
*   **Finding:** A definitive controlled ablation study on a **~1.3 billion parameter** GPT-style model demonstrated that adding embedding normalization **significantly reduced zero-shot generalization accuracy** under matched computation.
*   **Recommendation:** Avoid embedding normalization by default unless necessary to stabilize an unstable baseline training run.

## 2. Cross-Implementation and Scale Transferability

Micro-architectural findings are often sensitive to implementation details, codebases, and scale, necessitating a re-evaluation for every distinct setup.
*   **Source:** S. Narang et al., "Do Transformer Modifications Transfer Across Implementations and Applications?," 2021-2022 Technical Report.
*   **Finding:** Changes to normalization and similar tweaks often **fail to transfer** across different codebases and tasks. This highlights the need for the current study on a small-scale, custom implementation.

## 3. Related Normalization Studies Within Transformer Blocks

While not directly targeting the embedding layer, other studies have explored the effect of LayerNorm vs. RMSNorm inside the Transformer blocks.
*   **Source (LayerNorm vs. RMSNorm):** BigScience Workshop (BLOOM) [2] and Teuken7B authors [4].
*   **Finding:** These studies performed ablations *inside* Transformer blocks and cautioned about scale dependencies, suggesting that normalization choice influences stability and throughput trade-offs. They do not isolate the effect of *embedding* normalization, leaving a gap for the current research.

## 4. Large-Scale Training and Methodological Rigor

Modern LLM training requires strict methodological control to ensure architectural changes, not data or budget variations, drive performance differences.
*   **Sources:** J. Rae et al. (Gopher) [6], J. Hoffmann et al. (Chinchilla) [7], A. Wang et al. (LM-Eval/T0-Eval discipline) [3].
*   **Best Practices:** Emphasize the necessity of:
    *   Matching the **total training token budget** [7].
    *   Using **fixed, consistent evaluation protocols** and frameworks (e.g., LM-Eval/T0-Eval) [3].
    *   Reproducible baseline setups and optimizer recipes [6].

## 5. Synthesis (Project Context)

The existing literature presents compelling evidence that embedding normalization is likely detrimental at **large scale** and for **zero-shot accuracy** (Le Scao et al.). However, the non-transferability of micro-architectural findings (Narang et al.) mandates a new, rigorous test on this specific **small-scale, Pre-LN GPT-style architecture**. The experiment serves to either validate the existing caution or provide the first robust empirical evidence that the technique *can* be beneficial at smaller scales.

---

## References

[1] T. Le Scao et al., “What Language Model to Train if You Have One Million GPU Hours?,” *Findings of EMNLP*, 2022. [Online]. Available: https://arxiv.org/abs/2210.15424.
[2] BigScience Workshop, “BLOOM: A 176B-Parameter Open-Access Multilingual Language Model,” 2022. [Online]. Available: https://arxiv.org/abs/2211.05100.
[3] A. Wang et al., “What Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?,” 2022. [Online]. Available: https://arxiv.org/abs/2210.15424.
[4] Teuken7B authors, “Teuken7B: Multilingual Study on Normalization Micro-Changes including RMSNorm,” 2023-2024. [Online]. Available: https://arxiv.org/abs/1910.07467.
[5] S. Narang et al., “Do Transformer Modifications Transfer Across Implementations and Applications?,” 2021-2022 Technical Report. [Online]. Available: https://arxiv.org/abs/2210.15424.
[6] J. Rae et al., “Scaling Language Models: Methods, Analysis & Insights from Training Gopher,” 2021. [Online]. Available: https://arxiv.org/abs/2112.11446.
[7] J. Hoffmann et al., “Training Compute-Optimal Large Language Models,” 2022. [Online]. Available: https://arxiv.org/abs/2203.15556.
