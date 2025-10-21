# TBT-Former: Learning Temporal Boundary Distributions for Action Localization

**Authors:** Thisara Rathnayaka, Dr. Uthayasanker Thayasivam  
**Affiliation:** Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka  
**Code:** [GitHub Repository](https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects/tree/main/projects/210536K-Multi-Modal-Learning_Video-Understanding)

---

## Abstract

Temporal Action Localization (TAL) is a crucial task in video understanding that involves identifying the start time, end time, and class of actions in untrimmed videos. While recent Transformer-based models like ActionFormer have set a high standard, they often struggle with localizing actions that have ambiguous ("fuzzy") boundaries and with effectively fusing information from multiple temporal scales. This paper introduces the **Temporal Boundary Transformer (TBT-Former)**, a new architecture that directly addresses these issues. TBT-Former improves upon the strong ActionFormer baseline with three key contributions: **(1)** a higher-capacity **Scaled Transformer Backbone** for more powerful feature extraction; **(2)** a **Cross-Scale Feature Pyramid Network (CS-FPN)** that better integrates high-level context with fine-grained details; and **(3)** a novel **Boundary Distribution Regression (BDR) Head**. Inspired by Generalized Focal Loss (GFL), the BDR head reframes boundary prediction as a probability distribution learning problem, allowing the model to explicitly handle boundary uncertainty. TBT-Former establishes a new state-of-the-art on the challenging **THUMOS14** and **EPIC-Kitchens 100** datasets and achieves competitive performance on **ActivityNet-1.3**.

**Keywords:** Temporal Action Localization, Transformers, Video Understanding, Boundary Regression, Feature Pyramid Network

---

## 1. Introduction

Temporal Action Localization (TAL) is a cornerstone of video understanding. Modern approaches have shifted from complex two-stage methods to more efficient single-stage, anchor-free models. A leading example, **ActionFormer**, uses a minimalist Transformer architecture to achieve state-of-the-art results. However, it faces two primary challenges:

1.  **Imprecise Boundary Regression:** Standard regression methods predict a single, deterministic start and end time. This approach struggles when action boundaries are ambiguous or gradual, leading to noisy training and inaccurate localization.
2.  **Limited Multi-Scale Fusion:** ActionFormer's feature pyramid is built in a simple feed-forward manner, which prevents rich, high-level semantic information from refining more precise, low-level temporal features.

To overcome these limitations, we propose **TBT-Former**, which enhances the ActionFormer baseline with three targeted innovations:
* A **Scaled Transformer Backbone** with more attention heads and a larger MLP dimension to capture complex temporal dependencies.
* A **Cross-Scale Feature Pyramid (CS-FPN)** with a top-down pathway to enrich fine-grained features with global context.
* A **Boundary Distribution Regression (BDR) Head** that learns a probability distribution over possible boundary locations, allowing the model to represent and reason about boundary uncertainty.



Our experiments show that TBT-Former significantly advances the state-of-the-art, demonstrating that explicitly modeling boundary uncertainty is a powerful technique for improving temporal action localization.

---

## 2. Method

TBT-Former retains the efficient single-stage, anchor-free structure of ActionFormer but integrates three significant architectural enhancements.

### Scaled Transformer Backbone

To boost the model's representational power, we scale up the Transformer encoder. Specifically, we increase the number of attention heads in each Multi-Head Self-Attention (MSA) block from 8 to **16** and expand the hidden dimension of the feed-forward network (FFN) to **6x** the input dimension (up from 4x). This provides the model with greater capacity to learn intricate temporal patterns.



### Cross-Scale Feature Pyramid (CS-FPN)

The original ActionFormer uses a simple feed-forward feature pyramid. We introduce a **CS-FPN** that adds a top-down pathway with lateral connections. This allows information to flow from coarse, semantically rich feature maps to fine-grained, temporally precise ones. For each pyramid level $P_i$, the feature map is computed by merging the upsampled map from the coarser level $P_{i+1}$ with the backbone map $C_i$ via element-wise addition:

$$P_i = \text{Conv}_{1 \times 1}(C_i) + \text{Upsample}(P_{i+1})$$

This ensures that every level of the pyramid contains a rich blend of both high-level semantics and precise temporal details, improving localization of actions with varying durations.



### Boundary Distribution Regression (BDR) Head

This is our main contribution. Instead of regressing a single start/end offset, the BDR head predicts a **probability distribution** over a range of possible offsets. This approach, inspired by Generalized Focal Loss (GFL), allows the model to represent its uncertainty about a boundary's precise location.

The final continuous offset, $\hat{d^s}$, is calculated as the expectation of the learned start-time distribution $P_s = \{p_s(0), p_s(1),..., p_s(W-1)\}$:

$$\hat{d^s} = \sum_{i=0}^{W-1} i \cdot p_s(i)$$

This probabilistic formulation provides a more stable and accurate learning signal, especially for actions with fuzzy boundaries. We train the BDR head using **Distribution Focal Loss (DFL)**, which encourages the model to concentrate probability mass around the true boundary location. The total model loss combines the standard Focal Loss for classification ($\mathcal{L}_{cls}$) with DFL for the start and end boundaries:

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda (\mathcal{L}_{DFL}(P_s, d^s_{gt}) + \mathcal{L}_{DFL}(P_e, d^e_{gt}))$$

---

## 3. Experiments and Results

We evaluated TBT-Former on three standard TAL benchmarks: **THUMOS14**, **ActivityNet-1.3**, and **EPIC-Kitchens 100**.

### Main Results

* **THUMOS14:** TBT-Former sets a new state-of-the-art, achieving an average mAP of **68.0%**, outperforming the ActionFormer baseline by 1.2 points and all other prior methods across all tIoU thresholds.

| Model | Type | Avg. mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ActionFormer | Single-Stage | 66.8 | 82.1 | 77.8 | 71.0 | 59.4 | 43.9 |
| **TBT-Former (Ours)** | Single-Stage | **68.0** | **82.5** | **79.0** | **72.4** | **60.6** | **45.3** |

* **ActivityNet-1.3:** On this large-scale dataset, TBT-Former achieves a highly competitive average mAP of **36.8%**, performing on par with ActionFormer while showing improved localization precision at stricter tIoU thresholds (0.75 and 0.95).

| Model | mAP@0.5 | mAP@0.75 | mAP@0.95 | Avg. mAP |
| :--- | :--- | :--- | :--- | :--- |
| ActionFormer | 54.7 | 37.8 | 8.4 | 36.6 |
| **TBT-Former (Ours)** | 53.9 | **38.2** | **8.5** | **36.8** |

* **EPIC-Kitchens 100:** TBT-Former consistently outperforms the baseline, with average mAP improvements of **1.0% for Verbs** and **1.2% for Nouns**. This highlights its effectiveness on short, object-centric actions common in egocentric video.

| Task | Model | Avg. mAP |
| :--- | :--- | :--- |
| **Verb** | ActionFormer | 23.5 |
| | **TBT-Former (Ours)** | **24.5** |
| **Noun** | ActionFormer | 21.9 |
| | **TBT-Former (Ours)** | **23.1** |

### Ablation Studies

We conducted extensive ablation studies on THUMOS14 to validate our design choices.

* **Component Contributions:** Each proposed component provides a clear benefit. The Scaled Backbone, CS-FPN, and BDR Head contribute +0.4, +0.3, and +0.8 mAP, respectively. Together, they achieve a synergistic improvement of **+1.2 mAP** over the baseline.

| \# | Model Configuration | Avg. mAP | $\Delta$ |
| :- | :--- | :--- | :--- |
| 1 | Baseline (ActionFormer) | 66.8 | - |
| 2 | + Scaled Backbone | 67.2 | +0.4 |
| 3 | + Cross-Scale FPN | 67.1 | +0.3 |
| 4 | + Boundary Distribution Head | 67.6 | +0.8 |
| 5 | **Full Model (TBT-Former)** | **68.0** | **+1.2** |

* **Local Attention Window:** We found that a local attention window size of **30** provides the optimal balance of contextual reach and model focus for our scaled architecture, improving performance over the baseline size of 19.
* **Alternative Backbones:** We explored replacing the Transformer backbone with a hybrid **Transformer-Mamba** architecture and a fully convolutional **SGP-based** backbone. While both showed competitive performance, they did not surpass our optimized Scaled Transformer Backbone, confirming its effectiveness for TAL.

---

## 4. Conclusion

This paper introduced **TBT-Former**, a novel anchor-free model that advances the state-of-the-art in temporal action localization. By integrating a scaled backbone, a cross-scale feature pyramid, and a novel boundary distribution regression head, TBT-Former successfully improves upon the formidable ActionFormer benchmark.

Our key finding is that **explicitly modeling temporal boundaries as probability distributions is a powerful and promising paradigm** for handling ambiguity and improving localization precision. The superior performance on THUMOS14 and EPIC-Kitchens 100 validates our approach. Furthermore, our exploration of alternative backbones like Mamba and SGP, while not outperforming our final model, highlights them as valuable directions for future research in building more efficient and powerful TAL systems.
