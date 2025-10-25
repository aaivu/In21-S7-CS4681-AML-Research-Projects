# Literature Review: Time Series Forecasting

**Student:** 210434V  
**Research Area:** Time Series Forecasting  
**Date:** 2025-10-13

## Abstract

This literature review provides a comprehensive analysis of the evolution of time series forecasting methodologies, charting a course from traditional statistical models to the current deep learning frontier. We begin by examining the transition from linear models like ARIMA to sequence-aware architectures such as RNNs and LSTMs. The review then delves into the paradigm shift initiated by Transformer-based models, exemplified by Informer, and the subsequent critical re-evaluation spurred by the surprising efficacy of simpler linear models like DLinear. A significant focus is placed on the rise of self-supervised learning (SSL) for creating universal time series representations, with a detailed exposition of the state-of-the-art TS2Vec framework. Our analysis reveals a critical disconnect: the contrastive objectives that make SSL models powerful for general representation learning are often misaligned with the specific requirements of forecasting, which heavily relies on deterministic patterns like seasonality and trend. This review identifies this "objective mismatch" as a primary research gap. We conclude by surveying emerging hybrid approaches and architectural innovations, such as PatchTST and TimesNet, positioning the proposed TS2Vec-Ensemble as a timely and logical solution that bridges this gap by strategically fusing powerful learned dynamics with essential, explicit temporal features.

## 1. Introduction

The task of time series forecasting, which involves predicting future values based on historical observations, remains a cornerstone of decision-making processes across a vast spectrum of scientific and industrial domains. Its applications are both ubiquitous and critical, ranging from managing energy consumption and predicting financial market movements to modeling climate change and optimizing industrial manufacturing processes. Within this field, the challenge of Long-Term Time Series Forecasting (LTSF) is particularly acute. LTSF demands that models possess a high predictive capacity, enabling them to capture complex, multi-scale temporal dynamics that unfold over extended horizons. These dynamics often include a mixture of long-range dependencies, underlying trends, and intricate, multi-periodic seasonalities, making accurate prediction a formidable task.

The methodological landscape for tackling this challenge has undergone a profound evolution. For decades, the field was dominated by classical statistical models, such as the Autoregressive Integrated Moving Average (ARIMA) family. While foundational, these methods are predicated on often-violated assumptions of linearity and stationarity, limiting their applicability to the increasingly complex and high-dimensional datasets of the modern era. This has catalyzed a decisive shift towards deep learning paradigms. Architectures capable of autonomously learning hierarchical patterns from raw data, beginning with Recurrent Neural Networks (RNNs) and later evolving to more sophisticated models, have demonstrated a superior ability to model the nonlinear dependencies inherent in real-world time series.

The primary objective of this literature review is to critically survey the state-of-the-art in LTSF, with a particular focus on the dynamic interplay and inherent tension between model complexity and predictive performance. This review will trace the trajectory of recent advancements, beginning with the adaptation of the Transformer architecture, which promised to revolutionize the field, and the subsequent, sobering counter-argument presented by surprisingly effective simple linear models. It will then explore the rise of self-supervised learning (SSL) as a powerful paradigm for learning universal data representations, culminating in a detailed analysis of the TS2Vec framework. The ultimate purpose of this comprehensive survey is to expose the strengths and, more critically, the limitations of these leading paradigms. Through this analysis, a precise and well-motivated research gap is identified, establishing the intellectual context and necessity for the TS2Vec-Ensemble framework, which is designed to address these specific shortcomings.

## 2. Search Methodology

This review was conducted through a systematic search of prominent academic databases and preprint archives, focusing on foundational and recent high-impact research in time series forecasting.

### Search Terms Used

**Primary:** "time series forecasting", "long-term time series forecasting", "multivariate time series"

**Architectures:** "Transformer time series", "Informer", "Autoformer", "FEDformer", "DLinear", "PatchTST", "TimesNet", "Recurrent Neural Network forecasting", "LSTM forecasting"

**Paradigms:** "self-supervised learning time series", "contrastive learning time series", "representation learning time series", "TS2Vec"

**Concepts:** "time series decomposition", "seasonality", "trend", "ensemble forecasting"

### Databases Searched
- [X] IEEE Xplore
- [X] ACM Digital Library
- [X] Google Scholar
- [X] ArXiv
- [X] Other: Papers from top-tier ML/AI conferences (e.g., NeurIPS, ICLR, AAAI, ICML, KDD)

### Time Period
2018-2024, with the inclusion of seminal papers (e.g., Box et al., 2015) for foundational context. This period was chosen to capture the rapid evolution from RNNs to the dominance of Transformer-based models and the emergence of SSL.

## 3. Key Areas of Research

### 3.1 From Statistical Baselines to Sequential Deep Learning

The history of time series forecasting is characterized by a progressive increase in model capacity to handle increasingly complex temporal patterns. The journey began with statistical methods that provided a rigorous mathematical foundation but were limited by strong assumptions, leading to the adoption of deep learning models designed for sequential data.

#### Foundational Statistical Models (ARIMA)

The cornerstone of classical time series analysis is the Autoregressive Integrated Moving Average (ARIMA) model, a class of models that captures temporal structures in data using linear relationships. As formalized in the seminal work by Box et al. (2015), ARIMA models operate by combining three components: an autoregressive (AR) part that models the dependency between an observation and a number of lagged observations; an integrated (I) part that uses differencing of raw observations to make the time series stationary; and a moving average (MA) part that models the dependency between an observation and a residual error from a moving average model applied to lagged observations. 

ARIMA excels at modeling linear dependencies in stationary or stationarized time series and has long served as a crucial benchmark. However, its fundamental limitation lies in its inherent assumption of linearity. Real-world datasets, such as the Electricity Transformer Temperature (ETT) benchmark, are replete with intricate nonlinear dependencies, non-stationarity, and complex seasonality that cannot be adequately captured by a linear framework, thereby restricting ARIMA's efficacy in modern LTSF tasks.

#### The Advent of Recurrent Architectures (RNNs/LSTMs)

The inherent limitations of linear statistical models directly motivated the field's pivot towards deep learning. Recurrent Neural Networks (RNNs) emerged as a natural architectural choice for modeling sequential data like time series. Unlike feedforward networks, RNNs possess cycles that allow them to maintain an internal state, or "memory," of past information, enabling them to learn temporal dependencies. However, standard RNNs suffer from the well-known vanishing and exploding gradient problems, which make it difficult for them to learn long-range dependencies.

This deficiency was largely overcome by the introduction of more sophisticated recurrent units, most notably the Long Short-Term Memory (LSTM) network. LSTMs incorporate a gating mechanism—comprising input, forget, and output gates—that meticulously controls the flow of information into and out of a cell state. This architecture allows the network to selectively remember information over long periods and forget irrelevant data, making it far more effective at capturing long-term dependencies than simple RNNs. The adoption of RNNs and LSTMs represented the first major paradigm shift in time series analysis, moving beyond linear assumptions to autonomously learn complex, dynamic patterns directly from the data.

### 3.2 The Transformer Era and Its Discontents

While RNNs provided a significant leap forward, their inherently sequential nature of processing—one time step at a time—creates a computational bottleneck and still poses challenges for capturing extremely long-range dependencies. The introduction of the Transformer architecture, with its parallelizable self-attention mechanism, marked the next major evolutionary step, initiating a new era in sequence modeling that was quickly adapted for time series forecasting.

#### The Transformer Revolution in LTSF (Informer, Autoformer, FEDformer)

Originally developed for natural language processing tasks, the Transformer architecture's core innovation is the self-attention mechanism, which allows the model to weigh the importance of all other time steps when processing a given time step. This enables the direct modeling of dependencies regardless of their distance in the sequence, effectively reducing the maximum signal path length to O(1) and overcoming the primary limitations of RNNs. However, the canonical self-attention mechanism has a time and memory complexity of O(n²), where n is the sequence length, making it computationally prohibitive for the very long sequences required in LTSF.

This challenge spurred a wave of research into more efficient Transformer variants. A pioneering and highly influential model in this space is **Informer**, proposed by Zhou et al. (2021). Informer introduced three key innovations to make Transformers viable for LTSF:

1. **ProbSparse Self-Attention:** This mechanism reduces the complexity from O(n²) to O(n log n) by allowing each key to only attend to the most significant queries, based on a probabilistic measure of sparsity. This avoids computing the full dot-product matrix for all query-key pairs.

2. **Self-Attention Distilling:** To handle extremely long input sequences, Informer employs a "distilling" operation in its encoder, which progressively shortens the sequence length in deeper layers, highlighting the most dominant attention patterns and reducing memory footprint.

3. **Generative Decoder:** Instead of the standard autoregressive, step-by-step decoding process, Informer uses a one-shot generative decoder that predicts all future time steps simultaneously. This drastically improves inference speed, a critical factor in real-world applications.

Informer and its successors, such as **Autoformer** and **FEDformer**, established a new state-of-the-art, demonstrating that complex, attention-based models could achieve superior performance on challenging LTSF benchmarks.

#### The Linear Counter-Revolution (DLinear)

Just as the field appeared to be converging on increasingly complex Transformer-based solutions, a pivotal 2023 paper by Zeng et al. titled "Are Transformers Effective for Time series Forecasting?" introduced a simple one-layer linear model, **DLinear**, that fundamentally challenged this trajectory. The paper presented a powerful critique, arguing that the core strength of the Transformer—the permutation-invariant self-attention mechanism—is theoretically ill-suited for time series, which are inherently ordered. While techniques like positional encodings attempt to inject ordering information, the fundamental nature of self-attention is to find semantic relationships between tokens, not to model temporal evolution.

The authors hypothesized that the success of complex Transformers on popular benchmarks was not due to their sophisticated attention mechanisms but rather their incidental ability to perform trend-seasonal decomposition, a task that a far simpler model could execute more efficiently. To test this, they proposed DLinear, which first decomposes the input time series into a trend component (using a moving average) and a seasonal (remainder) component. It then applies two separate single-layer linear networks to these components and sums their outputs to produce the final forecast.

The results were striking: this "embarrassingly simple" linear model consistently and significantly outperformed sophisticated Transformer-based models like Informer and FEDformer across all nine benchmark datasets, often by a large margin. This work acted as a moment of reckoning, forcing the research community to re-evaluate the assumption that greater complexity necessarily leads to better performance in forecasting. It suggested that for many real-world time series, the dominant predictable signal comes from trend and seasonality, and a model with a strong, explicit inductive bias for decomposition is more effective than a complex, general-purpose sequence model.

### 3.3 Self-Supervised Representation Learning for Time Series

Parallel to the developments in supervised forecasting models, a separate paradigm shift was occurring in the broader machine learning landscape: the rise of self-supervised learning (SSL). SSL provides a powerful framework for learning rich, general-purpose data representations from vast quantities of unlabeled data, which can then be adapted for various downstream tasks. This approach has proven highly effective for time series, where labeled data can be scarce but unlabeled data is often abundant.

#### The SSL Paradigm

The core idea of SSL is to create a "pretext task" for which labels can be generated automatically from the input data itself. The model is trained on this pretext task, forcing it to learn meaningful underlying patterns in the data. Two dominant SSL approaches are generative modeling and contrastive learning. Generative approaches, such as masked signal modeling, train a model to reconstruct or predict masked portions of the input. Contrastive learning, which has been particularly influential, trains a model to pull representations of "positive pairs" (augmented views of the same instance) closer together in an embedding space while pushing apart "negative pairs" (representations from different instances). By learning to discriminate between similar and dissimilar samples, the model learns a powerful and robust representation of the data.

#### TS2Vec: A Universal Framework

Among the various SSL methods adapted for time series, **TS2Vec**, proposed by Yue et al. (2022), has emerged as a state-of-the-art framework for learning universal time series representations. Its design addresses several key limitations of prior methods and is built on a few core innovations that make it particularly robust and flexible.

**Encoder Architecture:** The TS2Vec encoder is designed to generate fine-grained, timestamp-level representations. It consists of three main components:
1. **Input Projection Layer:** A fully connected layer that maps raw observations to a high-dimensional latent space, allowing for a universal mask token (zero vector) to be used in the latent space.
2. **Timestamp Masking:** Randomly masks latent vectors to generate augmented views of the series.
3. **Dilated CNN Module:** Ten residual blocks with exponentially increasing dilation rates create a large receptive field and efficiently extract contextual representations.

**Hierarchical Contrastive Learning:** TS2Vec's central innovation is its hierarchical learning objective. Instead of learning a single representation, it applies a contrastive loss at multiple scales, generated by applying max-pooling along the time axis. At each scale, it employs a dual contrastive loss function:

1. **Temporal Contrastive Loss:** Encourages discriminative representations over time. For a given time series, the representations of the same timestamp from two different augmented views are treated as a positive pair, while representations of different timestamps are treated as negative pairs.
2. **Instance-wise Contrastive Loss:** Distinguishes one time series from others in the same batch. For a given timestamp, representations from the same instance form a positive pair, while representations from all other instances in the batch are negative pairs.

**Contextual Consistency:** To generate positive pairs, TS2Vec introduces the concept of contextual consistency. Instead of relying on transformations or temporal proximity, it posits that the representation of the same timestamp should remain consistent across two different augmented context views of the series. These views are generated through a combination of random cropping and timestamp masking.

The power of TS2Vec lies in its demonstrated universality. The learned representations have been shown to achieve state-of-the-art results on a wide array of downstream tasks—including classification, anomaly detection, and forecasting—by simply training a lightweight linear model on top of the frozen encoder's outputs.

### 3.4 Architectures for Multi-Pattern Modeling

The critical insights from the DLinear paper—namely, that pure self-attention struggles with temporal ordering and that explicit structural priors are crucial—did not spell the end for complex deep learning models. Instead, they catalyzed a new wave of architectural innovation. Researchers began to design models that either fundamentally re-engineered the Transformer to be more suitable for time series or abandoned the 1D sequential paradigm altogether in favor of representations that could better capture temporal patterns.

#### PatchTST: Rethinking Tokenization for Transformers

**PatchTST**, proposed by Nie et al. (2023), is a direct and highly effective response to the critiques of applying Transformers to time series. The authors argued that the primary issue was not the Transformer itself, but how the time series was being fed into it. Instead of treating each time point as an individual token (as in NLP), PatchTST introduces two key architectural modifications:

1. **Patching:** The model first segments the input time series into subseries-level "patches" of a fixed length. These patches, which retain local sequential information, are then used as the input tokens for the Transformer encoder. This simple change reduces the input sequence length, making the attention mechanism more computationally efficient, and embeds local temporal context before the attention mechanism is even applied.

2. **Channel-Independence:** For multivariate time series, PatchTST treats each channel (i.e., each variable) as an independent univariate time series. A single, shared Transformer backbone is applied to each channel separately, and the learned representations are concatenated before the final forecasting head.

The empirical success of PatchTST was significant. It consistently outperformed not only previous Transformer variants but also the strong DLinear baseline, demonstrating that with the right input representation and structural priors, Transformers could indeed be highly effective for LTSF.

#### TimesNet: A 2D-Variation Perspective

While PatchTST sought to fix the Transformer for 1D sequences, **TimesNet**, proposed by Wu et al. (2023), took a more radical approach by reframing the problem itself. Inspired by the success of 2D Convolutional Neural Networks (CNNs) in computer vision, TimesNet transforms the 1D time series into a 2D representation to better model its complex periodic patterns. Its core ideas are:

1. **1D to 2D Transformation:** The model first uses Fast Fourier Transform (FFT) to identify the most significant periods in the time series. For each identified period, it reshapes the flat 1D sequence into a 2D tensor where each column represents the time steps within a single period, and each row represents the time steps at the same phase across consecutive periods.

2. **Modeling 2D Variations:** This 2D representation elegantly disentangles complex temporal variations. Patterns that occur within a period (e.g., daily fluctuations) manifest as intraperiod-variations along the columns, while patterns that evolve across periods (e.g., weekly trends) manifest as interperiod-variations along the rows.

3. **Vision Backbones:** By transforming the problem into the 2D domain, TimesNet can leverage powerful and highly optimized 2D CNN backbones, such as Inception blocks, to model these 2D variations simultaneously.

TimesNet's ability to achieve state-of-the-art performance across five distinct time series tasks (forecasting, imputation, classification, anomaly detection, and imputation) established it as a powerful and versatile foundation model.

### Comparative Analysis of State-of-the-Art Architectures

| Model | Core Architecture | Key Innovation | Modeling Paradigm | Strengths | Limitations |
|-------|------------------|----------------|------------------|-----------|-------------|
| **Informer** | Transformer with ProbSparse Attention | Efficiency for long sequences via sparse attention and generative decoder | Supervised End-to-End | Captures long-range dependencies; computationally efficient for long inputs | High model complexity; permutation-invariance can lose temporal order |
| **DLinear** | Linear Layer | Simplicity and explicit trend-seasonal decomposition | Supervised End-to-End | Highly efficient and interpretable; very strong baseline on decomposable series | May fail on series without clear trend/seasonality; limited capacity for complex patterns |
| **TS2Vec** | Dilated CNN | Hierarchical contrastive learning for universal representations | Self-Supervised Pre-training | Learns robust, general-purpose features from unlabeled data; versatile across tasks | Contrastive objective is not optimized for forecasting; can miss deterministic patterns |
| **PatchTST** | Patch-based Transformer | Patch-based tokenization and channel-independence | Supervised End-to-End | Retains local temporal context; makes Transformers effective for time series | Less interpretable than linear models; performance depends on patch size |
| **TimesNet** | 2D CNN (Inception) | 1D-to-2D transformation based on periodicity | Supervised End-to-End | Models intra- and inter-period variations; leverages powerful vision backbones | Assumes clear periodicity exists; performance may degrade on aperiodic series |
| **TS2Vec-Ensemble** | Ridge Regression Ensemble | Adaptive fusion of SSL representations and explicit time features | Hybrid SSL + Supervised | Fuses implicit dynamics and explicit seasonality; adapts to forecast horizon | Requires careful tuning of ensemble weights; adds a layer of complexity over baseline |

## 4. Research Gaps and Opportunities

A critical analysis of the current state-of-the-art reveals specific limitations and unresolved tensions within the field. While models like TS2Vec have revolutionized representation learning and architectures like PatchTST have revitalized Transformers, their application to forecasting is not without challenges. This section identifies two primary research gaps that directly motivate the development of the TS2Vec-Ensemble framework.

### Gap 1: The Objective Mismatch in Contrastive Learning for Forecasting

**Description:** A fundamental disconnect exists between the training objectives of leading self-supervised models and the specific requirements of the forecasting task. State-of-the-art frameworks like TS2Vec are built upon contrastive learning objectives. The goal of these objectives is primarily discriminative: the model is trained to distinguish between different time series instances (instance-wise contrast) and between different moments in time within the same series (temporal contrast). This process yields powerful, general-purpose representations that are robust to augmentations and capture the unique dynamic signature of a time series. However, this objective function does not explicitly encourage the model to learn or prioritize the deterministic, repeating patterns—such as seasonality and trend—that are often the most crucial drivers of predictability, especially in long-horizon forecasting scenarios.

**Why it matters:** This "objective mismatch" is a critical issue. A model optimized to learn instance-discriminative features may inadvertently learn to treat predictable, recurring seasonal cycles as uninformative, as they are common across many instances and do not help in distinguishing one from another. In essence, the very signals that a forecasting model must exploit to make accurate long-term predictions can be down-weighted or ignored by an encoder trained with a purely contrastive loss. This results in learned representations that, while rich in capturing complex, stochastic dynamics, are suboptimal for the specific downstream task of predicting future values based on stable, periodic patterns.

**How the TS2Vec-Ensemble addresses it:** The TS2Vec-Ensemble project directly confronts this gap by acknowledging the strengths and weaknesses of the pre-trained encoder. Instead of attempting to fundamentally alter the successful but mismatched contrastive objective, it proposes to complement the learned representations at the forecasting stage. The framework achieves this by creating a dual-headed architecture. One regression head leverages the pure TS2Vec embeddings, which are rich in learned dynamics. A second, parallel regression head is explicitly fed engineered seasonal features (e.g., sinusoidal encodings for time of day) in addition to the embeddings. This architecture directly injects the missing deterministic information that the contrastive objective overlooks, allowing the final ensembled model to draw upon both the powerful, implicitly learned dynamics and the reliable, explicit temporal priors.

### Gap 2: The Inefficacy of Naive Feature Fusion

**Description:** Given the objective mismatch, a seemingly straightforward solution would be to integrate the modeling of deterministic patterns directly into the representation learning phase. This could involve augmenting the contrastive loss with a generative, reconstruction-based objective (like Masked Signal Modeling, MSM) or simply concatenating engineered time features to the input of a single forecasting model. However, ablation studies demonstrate that these naive fusion strategies are not only ineffective but can be actively detrimental to performance. The TS2Vec+MSM model, which combined contrastive and reconstruction losses, yielded the worst results, with an error rate nearly double that of the baseline TS2Vec model.

**Why it matters:** This finding is of significant practical and theoretical importance because it invalidates the simplest and most intuitive solutions to the problem identified in Gap 1. The substantial underperformance of the TS2Vec+MSM model suggests a fundamental conflict between the pretext tasks. The contrastive loss encourages the encoder to learn global, scale-invariant, and discriminative representations. In contrast, a reconstruction loss forces the encoder to capture high-fidelity, localized information to accurately rebuild the masked inputs. These competing pressures appear to compromise the integrity of the learned representation, forcing a specialization that harms its ability to generalize to the multi-horizon regression task of forecasting.

**How the TS2Vec-Ensemble addresses it:** The TS2Vec-Ensemble framework is designed as a direct and sophisticated response to the failure of naive fusion. Instead of forcing a single encoder to learn from conflicting objectives, it employs a late fusion ensemble architecture. This approach maintains the integrity of the powerful TS2Vec representations by keeping the pre-trained encoder frozen. It then trains two separate and specialized Ridge regression heads in parallel: one becomes an expert on the implicit dynamics captured by the pure TS2Vec embeddings, while the other becomes an expert on the explicit patterns present in the embeddings concatenated with sinusoidal time features. The final prediction is a weighted average of these two specialists' outputs. Crucially, the ensemble weights are optimized independently for each forecast horizon on a validation set. This allows the model to adaptively prioritize which expert to trust more, dynamically balancing the influence of learned dynamics (which might dominate short-term forecasts) against deterministic seasonal patterns (which are critical for long-term stability).

## 5. Theoretical Framework

The proposed research is grounded in two well-established theoretical areas of machine learning, which together form the foundation for the TS2Vec-Ensemble methodology. The synergy between these frameworks allows the proposed model to leverage the strengths of modern representation learning while retaining the robustness of classical modeling principles.

### Contrastive Self-Supervised Learning

The foundational component of the architecture, the TS2Vec encoder, is built upon the principles of contrastive self-supervised learning. This paradigm operates by learning an embedding space where the model's objective is to maximize the agreement between different, augmented views of the same data point (a "positive pair") relative to its agreement with views of other data points ("negative pairs"). This is typically achieved by minimizing a contrastive loss function, such as InfoNCE. The TS2Vec framework implements this principle through a sophisticated dual objective that combines temporal and instance-wise contrastive losses. Its novel "contextual consistency" strategy, which defines positive pairs as the representation of the same timestamp under two different augmented contexts (generated via random cropping and masking), is a key theoretical contribution that adapts the contrastive paradigm specifically for the nuances of time series data. This framework enables the encoder to learn rich, discriminative, and highly generalizable features from unlabeled time series, forming a powerful base for downstream tasks.

### Ensemble Learning

The core enhancement proposed in the TS2Vec-Ensemble framework is a direct application of ensemble learning theory. This principle posits that the combination of predictions from multiple individual models, often referred to as "experts" or "base learners," can yield superior predictive performance, increased robustness, and reduced generalization error compared to any single constituent model. Ensembles are effective because different models may learn different aspects of the data and make different types of errors; by averaging their predictions, these errors can be mitigated. The TS2Vec-Ensemble framework embodies this theory by constructing an ensemble of two specialized Ridge regression models. One model is trained to be an expert in capturing the complex, data-driven dynamics present in the raw TS2Vec embeddings. The other is trained to be an expert in modeling the deterministic, periodic patterns anchored by explicit time features. The use of an adaptive, validation-optimized weighting scheme is a mechanism for intelligently and dynamically combining the outputs of these experts, tailoring the final prediction to the specific demands of each forecast horizon.

## 6. Methodology Insights

The surveyed literature reveals a significant methodological bifurcation in the application of deep learning to time series forecasting, primarily divided between end-to-end supervised approaches and two-stage self-supervised paradigms. The evolution of these methodologies provides crucial context for positioning the proposed work.

### End-to-End Supervised Models

The majority of prominent models in recent years, including the entire lineage of Transformer-based architectures (Informer, Autoformer, FEDformer) and their challengers (DLinear, PatchTST, TimesNet), are trained end-to-end in a fully supervised manner. In this paradigm, the model's parameters are optimized directly and exclusively to minimize a forecasting loss function, such as Mean Squared Error (MSE), on a labeled training dataset. This approach is conceptually straightforward and highly task-focused, as every component of the model is geared towards the singular goal of making accurate predictions. However, this method can be data-hungry, requiring large labeled datasets to avoid overfitting, and the learned features may not generalize well to datasets with different statistical properties or to different downstream tasks.

### Two-Stage SSL Paradigm (Pre-train, Fine-tune/Probe)

In contrast, TS2Vec champions a two-stage approach that decouples representation learning from the downstream task. In the first stage, a powerful encoder is pre-trained on a large (and potentially unlabeled) dataset using a self-supervised objective, such as the hierarchical contrastive loss of TS2Vec. This stage is computationally intensive but needs to be performed only once per dataset. In the second stage, the pre-trained encoder is frozen, and its weights are no longer updated. It is then used as a feature extractor to generate representations for the labeled data. A simple, lightweight model, typically a linear regressor or a shallow MLP, is trained on top of these fixed representations for the specific downstream task. This "linear probing" protocol is the standard method for evaluating the quality and universality of the learned SSL representations. The primary advantage of this paradigm is its data efficiency and generalization capability; the powerful representations can be adapted to various tasks with minimal labeled data.

### Promising Direction: Hybrid SSL-Supervised Methods

The TS2Vec-Ensemble framework represents a promising and logical evolution of the two-stage paradigm, creating a hybrid methodology. It begins by adopting the powerful pre-trained representations from the SSL stage, thereby inheriting the benefits of generalization and data efficiency. However, it explicitly acknowledges the limitations of these general-purpose representations for the specific task of forecasting, as identified in the research gaps. The proposed ensemble architecture can be viewed as a sophisticated, task-specific "head" that goes far beyond simple linear probing. It intelligently integrates domain knowledge (in the form of engineered time features) with the pre-trained features in a flexible, adaptive, and principled manner. This hybrid approach, which leverages the complementary strengths of both self-supervised representation learning and supervised, feature-aware modeling, appears to be the most promising direction for achieving robust, state-of-the-art performance in time series forecasting.

## 7. Conclusion

This comprehensive literature review has traced the dynamic and often-contentious trajectory of time series forecasting methodologies, from the foundational principles of classical statistical models to the complex deep learning architectures that define the current state-of-the-art. The journey through the "Transformer Era," marked by models like Informer, and the subsequent, disruptive challenge posed by the simplicity and efficacy of DLinear, has illuminated a central and recurring theme in the field: the critical need to strike a delicate balance between a model's capacity to learn complex patterns and its ability to explicitly model fundamental time series components like trend and seasonality. The advent of self-supervised learning, epitomized by the universal representation capabilities of TS2Vec, has introduced a powerful new paradigm for feature extraction from unlabeled data, further enriching the methodological toolkit.

However, our critical analysis has identified a significant research gap at the intersection of self-supervised learning and forecasting. The contrastive objective at the heart of TS2Vec, while exceptionally effective for learning general-purpose, discriminative representations, is fundamentally misaligned with the specific predictive needs of long-horizon forecasting, which relies heavily on stable, deterministic patterns. Furthermore, this review has highlighted that naive attempts to bridge this gap—for instance, by combining contrastive and reconstructive objectives within a single encoder—have been shown to be ineffective and can even degrade performance, indicating a fundamental tension between these learning paradigms.

The proposed TS2Vec-Ensemble framework is situated directly within this identified gap, offering a sophisticated and principled resolution to this conflict. It embraces the strengths of both paradigms without succumbing to their individual weaknesses. By employing an adaptive ensemble to intelligently fuse the rich, implicitly learned dynamics from a pre-trained TS2Vec encoder with the robust, explicit signal from engineered temporal features, the work provides a compelling and empirically validated path forward. This hybrid strategy, which respects both the immense pattern-recognition capabilities of deep representation learning and the enduring importance of statistical priors, represents a significant and logical next step in the ongoing pursuit of more accurate, robust, and reliable time series forecasting.

## References

1. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv preprint arXiv:1803.01271*.

2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

4. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *International Conference on Machine Learning*, 1597–1607.

5. Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting. In *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 2352–2359.

6. Franceschi, J.-Y., Dieuleveut, A., & Jaggi, M. (2019). Unsupervised Scalable Representation Learning for Multivariate Time Series. In *Advances in Neural Information Processing Systems*, 32.

7. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 16000–16009.

8. Lai, G., Chang, W., Yang, Y., & Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*, 95–104.

9. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In *International Conference on Learning Representations*.

10. Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In *International Conference on Learning Representations*.

11. Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181–1191.

12. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. In *International Conference on Learning Representations*.

13. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems*, 30.

14. Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In *International Conference on Learning Representations*.

15. Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. In *Advances in Neural Information Processing Systems*, 34, 22419-22430.

16. Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B. (2022). TS2Vec: Towards Universal Representation of Time Series. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8), 8980-8987.

17. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? In *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11131-11139.

18. Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C. (2021). A Transformer-Based Framework for Multivariate Time Series Representation Learning. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*, 2114–2124.

19. Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

20. Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. In *International Conference on Machine Learning*, 27268-27286.
