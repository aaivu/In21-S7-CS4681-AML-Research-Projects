# Experiment 7 — Loss comparison: FocalLoss (baseline) vs ZLPR / FZLPR

Overview & Primary references
-----------------------------
- ZLPR paper (arXiv): https://arxiv.org/pdf/2208.02955
- Supporting result cited (PubMed): https://pubmed.ncbi.nlm.nih.gov/40564410/

ZLPR Loss (Zero-Loss Plateau Regularization) is a recent loss function designed for multi-label classification. Unlike standard losses that keep reducing easy sample losses toward zero, ZLPR introduces a “zero-loss plateau,” preventing overfitting on already confident predictions while keeping gradients for hard cases. It combines a zero-bounded log-sum-exp with pairwise rank-based formulation, making it effective when the number of target labels is uncertain. ZLPR also accounts for label correlations, giving it an advantage over simple binary relevance approaches, while remaining computationally efficient compared to label powerset methods. Experiments show it improves calibration, generalization, and stability, with a soft version supporting tricks like label smoothing.

Why try ZLPR here
------------------
- The PubMed work linked above reports that FZLPR (a practical variant of ZLPR) outperforms other common losses on the NIH Chest X-ray14 dataset, including BCE and focal loss. Motivated by those findings we run a controlled comparison on our CLAHE-enhanced DenseNet-121 replication to verify gains in AUROC/F1 and per-class behaviour on this dataset.

Experiment setup
----------------
- Backbone / preprocessing: same CLAHE-enhanced DenseNet-121 replication used across prior experiments (see experiments/6). 
- Random seed: `42` (kept constant across both runs for reproducibility).
- Training budget: `25` epochs, batch size `8`, optimizer `AdamW` (wd=1e-5), image size `224`.
- Note: the focal-loss integrated baseline referenced here corresponds to the CLAHE DenseNet-121 replication used earlier (labeled "baseline" in experiment-6 README) which used `FocalLoss(alpha=1,gamma=2)`.


=== Per-Class Metrics ===

| Class                | AUROC  | F1     | Threshold |
|----------------------|:------:|:------:|:---------:|
| Atelectasis          | 0.8030 | 0.3868 | 0.2088    |
| Cardiomegaly         | 0.9249 | 0.4803 | 0.2823    |
| Consolidation        | 0.7764 | 0.2227 | 0.1428    |
| Edema                | 0.8957 | 0.2734 | 0.2450    |
| Effusion             | 0.8990 | 0.6060 | 0.3433    |
| Emphysema            | 0.9655 | 0.5301 | 0.1033    |
| Fibrosis             | 0.8564 | 0.1455 | 0.1017    |
| Hernia               | 0.9830 | 0.6087 | 0.2337    |
| Infiltration         | 0.6983 | 0.4155 | 0.2878    |
| Mass                 | 0.9068 | 0.4631 | 0.4138    |
| Nodule               | 0.7671 | 0.3122 | 0.2540    |
| Pleural_Thickening   | 0.8015 | 0.2400 | 0.0897    |
| Pneumonia            | 0.6817 | 0.0385 | 0.1669    |
| Pneumothorax         | 0.8873 | 0.3463 | 0.2852    |

Comparison tables (FocalLoss-integrated baseline vs ZLPR)
------------------------------------------------------
Overall metrics

| Metric    | CLAHE DenseNet-121 (FocalLoss baseline) | ZLPR run |
|----------:|-----------------------------------------:|--------:|
| Loss      | 0.0415                                   | 1.6268  |
| Avg AUROC | 0.8514                                   | 0.8462  |
| Avg F1    | 0.3803                                   | 0.3621  |

Per-class AUROC comparison (Baseline = CLAHE DenseNet-121 with FocalLoss) — Delta = (ZLPR - Baseline)

| Class                | Baseline | ZLPR   | Delta    | Winner  |
|----------------------|:--------:|:------:|:--------:|:-------:|
| Atelectasis          | 0.8146   | 0.8030 | -0.0116  | Baseline |
| Cardiomegaly         | 0.9325   | 0.9249 | -0.0076  | Baseline |
| Consolidation        | 0.7871   | 0.7764 | -0.0107  | Baseline |
| Edema                | 0.8841   | 0.8957 |  0.0116  | ZLPR     |
| Effusion             | 0.9015   | 0.8990 | -0.0025  | Baseline |
| Emphysema            | 0.9656   | 0.9655 | -0.0001  | Baseline |
| Fibrosis             | 0.8207   | 0.8564 |  0.0357  | ZLPR     |
| Hernia               | 0.9936   | 0.9830 | -0.0106  | Baseline |
| Infiltration         | 0.7044   | 0.6983 | -0.0061  | Baseline |
| Mass                 | 0.9122   | 0.9068 | -0.0054  | Baseline |
| Nodule               | 0.7780   | 0.7671 | -0.0109  | Baseline |
| Pleural_Thickening   | 0.8124   | 0.8015 | -0.0109  | Baseline |
| Pneumonia            | 0.7229   | 0.6817 | -0.0412  | Baseline |
| Pneumothorax         | 0.8902   | 0.8873 | -0.0029  | Baseline |

Conclusions & observations
---------------------------
- Summary: on overall metrics the CLAHE DenseNet-121 baseline (trained with FocalLoss) remains slightly stronger in Avg AUROC (0.8514 vs 0.8462) and Avg F1 (0.3803 vs 0.3621) compared to this ZLPR run. 
- Per-class: ZLPR shows clear gains for `Fibrosis` (+0.0357) and `Edema` (+0.0116), indicating ZLPR may help rarer or harder classes where label correlations / ranking matter. Most other classes remain better with the baseline FocalLoss in this pair of runs.
- Loss values differ in scale; don't over-interpret the larger ZLPR loss magnitude — prefer AUROC/F1 comparisons.

Practical notes
---------------
- Compute & runtime: this ZLPR run required ~10–11 hours on Kaggle with `T4 * 2` GPUs (reported by the run). Expect similar training times for the focal baseline under the same config.

References
----------
- ZLPR (arXiv): https://arxiv.org/pdf/2208.02955
- FZLPR / PubMed notice: https://pubmed.ncbi.nlm.nih.gov/40564410/