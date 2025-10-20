# Literature Review: AI Evaluation:Task Completion

**Student:** 210706H
**Research Area:** AI Evaluation:Task Completion
**Date:** 2025-09-01

## Abstract

This literature review covers the critical task of evaluating toxic degeneration in Large Language Models (LLMs). It begins by examining the foundational benchmark, RealToxicityPrompts (RTP), and its reliance on the commercial Perspective API. The review then synthesizes a growing body of research that identifies critical flaws in this black-box approach, including systemic bias, context insensitivity, and non-stationarity, which undermine scientific reproducibility . Key findings show that while alternative datasets like Jigsaw and fairness metrics like BPSN/BNSP AUC have been developed to address these issues, a significant research gap remains in creating an integrated, transparent, and reproducible evaluation framework. This review concludes that the most promising path forward involves fine-tuning state-of-the-art transformers on human-annotated, bias-aware data, a direction directly pursued by the CAFE project.

## 1. Introduction

The task of evaluating Large Language Models (LLMs) is multifaceted, but one of the most critical challenges is assessing their propensity for toxic degeneration—the generation of harmful content from benign prompts. The safe and ethical deployment of these models hinges on our ability to complete this evaluation task accurately and fairly. This literature review focuses on the evolution of methodologies for this specific task. It scopes the review from the establishment of the RTP benchmark  to the subsequent identification of its flaws and the development of more nuanced, fairness-aware techniques for toxicity classification. The central goal is to map the existing landscape and identify the gaps that necessitate the creation of a new, transparent evaluation framework.

## 2. Search Methodology

### Search Terms Used
- Toxicity evaluation in language models
- RealToxicityPrompts benchmark
- Perspective API bias
- Jigsaw unintended bias
- Fairness metrics in NLP" (BPSN, BNSP, Subgroup AUC)
- Debiasing text classifiers
- Context-aware toxicity detection
- Reproducibility in AI evaluation

### Databases Searched
- IEEE Xplore
- ACM Digital Library
- Google Scholar
- ArXiv

### Time Period
2018-2025, focusing on the period following the introduction of the RTP benchmark and the Jigsaw Unintended Bias competition.

## 3. Key Areas of Research

### 3.1 Benchmarking Toxic Degeneration
The primary task in this area has been to quantify the extent to which LLMs produce toxic content.

**Key Papers:**
- Gehman et al., 2020 - Introduced RealToxicityPrompts (RTP), the de facto benchmark for evaluating toxic degeneration. It pairs 100K web prompts with model continuations scored by Google's Perspective API.
- Perspective API (Google Jigsaw) - The commercial, black-box tool that underpins the RTP benchmark and much of the early work in toxicity scoring.


### 3.2 Auditing and Identifying Flaws in Black-Box Evaluators
A significant research thrust has been dedicated to uncovering the limitations of relying on single, opaque APIs for evaluation.

**Key Papers:**
- Sap et al., 2019 - Demonstrated the risk of racial bias in hate speech detectors, finding that models often associate identity terms like "black" with toxicity.
- Pozzobon et al., 2023 - Highlighted the challenges of non-stationarity in commercial APIs, where unannounced model updates make longitudinal comparisons and reproducible research impossible.
- Hosseini et al., 2017 - Showed the adversarial fragility of the Perspective API, where minimal perturbations can drastically change toxicity scores.

### 3.3 Development of Fairness-Aware Datasets and Metrics
To counter the flaws of early models, researchers focused on creating better data and more nuanced evaluation metrics.

**Key Papers:**

- Borkan et al., 2019 - Introduced the Jigsaw Unintended Bias in Toxicity Classification dataset, a large-scale corpus designed specifically to help models distinguish between genuine toxicity and the mere mention of identity terms. This work also formalized the key fairness metrics: Subgroup AUC, BPSN AUC, and BNSP AUC.
- Hartvigsen et al., 2022 - Developed ToxiGen, a dataset focused on implicit and adversarial hate speech, moving beyond the explicit toxicity common in other datasets.


## 4. Research Gaps and Opportunities


### Gap 1: Reliance on Opaque, Non-Reproducible Oracles
**Why it matters:** The dominant evaluation paradigm (RTP) relies on a commercial, proprietary API whose model can change at any time. Why it matters: This makes scientific findings impossible to reproduce reliably over time and hides the specific biases of the evaluator from scrutiny.
**How your project addresses it:** CAFE is constructed as an open, transparent "glass-box" artifact. By using a public dataset (Jigsaw) and an open-source model (DeBERTa), it creates a static, reproducible, and auditable evaluation tool.


### Gap 2: Lack of Integrated Context and Fairness in Evaluation
**Why it matters:** Most existing evaluators treat text literally, failing on sarcasm, irony, and reclaimed slurs. Furthermore, fairness is often an afterthought rather than a core part of the model's training objective. Why it matters: This leads to models that are brittle and biased, incorrectly penalizing non-toxic speech from minority groups and failing to understand real-world context.
**How your project addresses it:** CAFE addresses this directly through its core design. The multi-task learning objective forces the model to learn richer, context-dependent representations , while the fairness-weighted loss function explicitly optimizes the model to mitigate known biases during training.

## 5. Theoretical Framework

The theoretical foundation of this research lies in the field of Algorithmic Fairness. The project's methodology is directly informed by principles aiming to achieve group fairness. While not strictly enforcing Demographic Parity (which demands equal average scores across groups), the fairness-weighting scheme is a practical implementation designed to reduce the disparities measured by the BPSN and BNSP AUC metrics. These metrics are themselves grounded in the idea of measuring and mitigating specific types of error rate disparities between demographic subgroups.

## 6. Methodology Insights

The most successful and promising methodologies for completing the task of fair toxicity evaluation, as seen in the winning solutions for the Jigsaw competition and adopted by this project, converge on several key techniques:


- Fine-tuning Large Transformer Models: State-of-the-art architectures like DeBERTa have proven most effective at capturing the necessary nuance for this task .


- Multi-Task Learning: Using auxiliary targets (e.g., insult, threat) acts as a powerful regularizer, forcing the model to learn more generalizable and context-rich representations.


- Sample Weighting: Explicitly increasing the loss contribution of challenging fairness cases (BPSN/BNSP samples) is a direct and effective way to guide the model toward less biased outcomes.


- Ensembling: Combining predictions from multiple models trained via cross-validation is a proven method for improving robustness and overall performance.

## 7. Conclusion

The literature reveals a clear trajectory: from an initial reliance on a convenient but flawed black-box API to a more mature understanding of the need for transparent, reproducible, and fairness-aware evaluation. The key research gap is the lack of a single, integrated framework that is both methodologically sound and publicly accessible. The insights from prior work strongly suggest that a solution built on a state-of-the-art transformer, trained on a bias-aware dataset like Jigsaw with explicit fairness objectives, is the most promising direction. This directly informs and validates the research direction of the CAFE project, which aims to produce such a framework and use it to audit the very systems it seeks to replace.

## References

1. Gehman, S., Gururangan, S., Sap, M., Choi, Y., \& Smith, N. A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In arXiv [cs.CL]. http://arxiv.org/abs/2009.11462.
2. Perspective API. (n.d.). Perspectiveapi.com. Retrieved August 23, 2025, from https://perspectiveapi.com/.
3. TensorFlow datasets. (n.d.). TensorFlow. Retrieved August 23, 2025, from https://www.tensorflow.org/datasets/catalog/civil\_comments.
4. Dixon, L., Li, J., Sorensen, J., Thain, N., \& Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 67–73.
5. Borkan, D., Dixon, L., Sorensen, J., Thain, N., \& Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. In arXiv [cs.LG]. http://arxiv.org/abs/1903.04561.
6. Sap, M., Card, D., Gabriel, S., Choi, Y., \& Smith, N. A. (2019). The risk of racial bias in hate speech detection. In A. Korhonen, D. Traum, \& L. Màrquez (Eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 1668–1678). Association for Computational Linguistics.
7. Nogara, G., Pierri, F., Cresci, S., Luceri, L., Törnberg, P., \& Giordano, S. (2023). Toxic bias: Perspective API misreads German as more toxic. In arXiv [cs.SI]. http://arxiv.org/abs/2312.12651.
8. Pozzobon, L., Ermis, B., Lewis, P., \& Hooker, S. (2023). On the challenges of using black-box APIs for toxicity evaluation in research. In arXiv [cs.CL]. http://arxiv.org/abs/2304.12397.
9. Hosseini, H., Kannan, S., Zhang, B., \& Poovendran, R. (2017). Deceiving Google’s Perspective API built for detecting toxic comments. In arXiv [cs.LG]. https://labs.ece.uw.edu/nsl/papers/view.pdf.
10. Van Hee, C., Lefever, E., \& Hoste, V. (2018). SemEval-2018 task 3: Irony detection in English tweets. In M. Apidianaki, S. M. Mohammad, J. May, E. Shutova, S. Bethard, \& M. Carpuat (Eds.), Proceedings of The 12th International Workshop on Semantic Evaluation (pp. 39–50). Association for Computational Linguistics.
11. Association for Computational Linguistics. (2020). Proceedings of the Second Workshop on Figurative Language Processing (FigLang 2020). Association for Computational Linguistics. https://aclanthology.org/volumes/2020.figlang-1/.
12. Hartvigsen, T., Gabriel, S., Palangi, H., Sap,M., Ray, D., \& Kamar, E. (2022). ToxiGen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. In arXiv [cs.CL]. http://arxiv.org/abs/2203.09509.
13. Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., \& Mukherjee, A. (2020). HateXplain: A benchmark dataset for explainable hate speech detection. In arXiv [cs.CL]. http://arxiv.org/abs/2012.10289.
14. Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., \& Rajani, N. F. (2021). GeDi: Generative discriminator guided sequence generation. Findings of the Association for Computational Linguistics: EMNLP 2021.
15. Inan, H., Upasani, K., Chi, J., Rungta, R., Iyer, K., Mao, Y., Tontchev, M., Hu, Q., Fuller, B., Testuggine, D., \& Khabsa, M. (2023). Llama Guard: LLM-based input-output safeguard for Human-AI conversations. In arXiv [cs.CL]. http://arxiv.org/abs/2312.06674.
16. Barocas, S., Hardt, M., \& Narayanan, A. (2023). Fairness and machine learning: Limitations and opportunities. MIT Press. https://mitpress.mit.edu/9780262048613/fairness-and-machine-learning/.
17. Mondal, P., Ansari, F., \& Das, S. (2025). APFEx: Adaptive Pareto Front Explorer for Intersectional Fairness (arXiv:2509.13908v2). arXiv. https://arxiv.org/abs/2509.13908v2.