# Literature Review: Time Series Classification

**Student:** 210429K
**Research Area:** Time Series Classification
**Date:** 2025-10-19

## Abstract

This literature review examines unsupervised representation learning for time series classification, with a focus on the evolution from autoencoder-based methods to modern contrastive learning approaches. The review covers key developments in self-supervised learning techniques, architectural choices for temporal modeling, and few-shot learning methodologies. Major findings indicate that contrastive learning methods, particularly those leveraging temporal structure like TNC, TS2Vec, and BTSF, have surpassed traditional reconstruction-based approaches. However, a significant research gap exists in integrating powerful self-supervised encoders with few-shot classification frameworks, presenting an opportunity to develop sample-efficient solutions for time series classification in low-label regimes.

## 1. Introduction

Unsupervised representation learning for time series has recently gained attention as a critical research area in machine learning. Unlike supervised learning, which requires large amounts of labeled data, unsupervised learning focuses on extracting meaningful patterns without labels. This is especially valuable in real-world applications such as healthcare, where obtaining high-quality annotations is costly or impractical. 

Time series data present unique challenges such as high dimensionality, non-stationarity, and variable lengths, making representation learning both difficult and important. This literature review surveys major approaches to unsupervised time series representation learning, from early autoencoder-based methods to state-of-the-art contrastive learning frameworks. Additionally, it examines few-shot learning techniques and identifies opportunities for integrating self-supervised representations with sample-efficient classification methods.

## 2. Search Methodology

### Search Terms Used
- "unsupervised time series representation learning"
- "contrastive learning time series"
- "self-supervised time series classification"
- "few-shot learning time series"
- "temporal neighborhood coding"
- "time series autoencoders"
- "prototypical networks time series"
- "metric learning temporal data"

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [ ] Other: ___________

### Time Period
2016-2025, with emphasis on recent developments (2018-2024) in contrastive learning and self-supervised methods

## 3. Key Areas of Research

### 3.1 Autoencoder-Based Methods

Early research applied autoencoders and sequence-to-sequence models to time series representation learning. These models jointly train an encoder and decoder to reconstruct the input, forcing the encoder to capture compressed representations that preserve essential information about the original signal.

**Key Papers:**
- **Choi et al., 2016** [1] - Introduced autoencoder-based models for learning low-dimensional representations of medical time series data, demonstrating the feasibility of unsupervised learning in healthcare applications.
- **Amiriparian et al., 2017** [2] - Explored multi-channel autoencoders for audio-visual emotion recognition, showing how reconstruction objectives can capture temporal patterns.
- **Malhotra et al., 2017** [3] - Applied LSTM networks with autoencoder architecture for anomaly detection in time series, highlighting the importance of sequential modeling.

Variational autoencoders (VAEs) were later introduced to encourage disentangled and interpretable features. However, the fundamental limitation of autoencoder-based approaches lies in their pixel-level reconstruction objective, which may not align with downstream task requirements. While reconstruction forces the encoder to preserve information, it does not guarantee that the learned representations are discriminative or semantically meaningful for classification tasks. Additionally, autoencoders struggle with high-frequency noise and complex temporal dynamics, often learning to reproduce noise rather than extracting meaningful patterns. These limitations motivated the development of alternative self-supervised objectives that do not rely on explicit reconstruction.

### 3.2 Contrastive Learning Approaches

Contrastive learning has emerged as the dominant self-supervised strategy for time series representation learning. The core principle is to bring representations of similar samples (positives) closer in the embedding space while pushing apart dissimilar ones (negatives). Although initially popular in computer vision, adapting contrastive frameworks to time series required careful treatment of temporal dependencies and augmentations.

**Key Papers:**

- **Oord et al., 2018 - Contrastive Predictive Coding (CPC)** [4] - Introduced the idea of predicting future latent representations from past contexts, maximizing mutual information between context and future representations. CPC uses a unidirectional architecture that predicts multiple future steps. While CPC demonstrated strong performance on audio and vision tasks, its application to time series revealed limitations in handling abrupt state changes and long-range dependencies, particularly on highly non-stationary signals.

- **Franceschi et al., 2019 - Triplet Loss (T-Loss)** [5] - Proposed a triplet-based sampling strategy tailored for time series, using causal dilated convolutions for variable-length inputs. The method samples anchor, positive, and negative triplets based on temporal proximity. However, this simple heuristic can introduce false negatives when similar states occur at distant time points, and the approach often fails when states are generated from similar dynamics.

- **Tonekaboni et al., 2021 - Temporal Neighborhood Coding (TNC)** [6] - Addressed the limitations of previous contrastive methods by defining temporal neighborhoods as segments with locally stationary properties. TNC employs a debiased contrastive loss inspired by Positive-Unlabeled learning, treating non-neighboring samples as unlabeled rather than strictly negative to reduce false negatives. The key innovation is using statistical tests like the Augmented Dickey-Fuller test to dynamically determine temporal neighborhoods, allowing the model to adapt to local signal properties. TNC demonstrated superior performance over CPC and T-Loss, particularly in medical applications where non-stationarity is common.

- **Eldele et al., 2021 - TS-TCC** [7] - Extended contrastive methods with a dual-module framework combining temporal and contextual contrasting, often with a Transformer backbone. The temporal contrasting module learns representations by distinguishing between different temporal augmentations of the same instance, while the contextual contrasting module operates across different instances. This dual approach provides complementary supervision signals that improve representation quality, outperforming CPC and SimCLR, especially in few-label and transfer scenarios.

- **Yue et al., 2021 - TS2Vec** [8] - Introduced hierarchical contrastive learning at multiple semantic levels (timestamp, instance). TS2Vec applies contrastive learning across different temporal scales, from individual timestamps to entire sequences, enabling the model to capture both fine-grained and global temporal patterns. Demonstrated strong improvements across classification, forecasting, and anomaly detection, achieving state-of-the-art performance on both UEA and UCR benchmarks.

- **Yang & Hong, 2022 - BTSF** [9] - Incorporated both temporal and spectral features via bilinear fusion, arguing that spectral properties are often ignored in prior work. By combining time-domain and frequency-domain representations, BTSF captures complementary aspects of temporal dynamics that pure time-domain methods may miss, improving performance across multiple tasks.

SimCLR, originally developed for computer vision, has also been adapted for time series by applying various augmentation techniques such as jittering, scaling, rotation, and permutation. However, the effectiveness of these augmentations varies significantly across different types of time series data, and careful selection is required to avoid destroying semantic information.

### 3.3 Architectures for Time Series Representation Learning

Different frameworks adopt different encoder architectures, each with specific advantages and limitations for temporal modeling.

**Recurrent Neural Networks (RNNs):**
- Intuitive for sequential data but suffer from vanishing gradients and limited parallelization
- Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) address vanishing gradients, enabling capture of longer-range dependencies
- Remain fundamentally sequential, limiting computational efficiency on modern parallel hardware

**Convolutional Neural Networks (CNNs):**
- Dilated CNNs, used in T-Loss and TS2Vec, capture long-range dependencies more efficiently
- Dilated convolutions exponentially expand receptive field without increasing parameters
- Residual connections improve gradient flow and enable training of very deep architectures
- Temporal Convolutional Networks (TCNs) combine dilated convolutions with residual connections, achieving strong performance while maintaining computational efficiency

**Transformers:**
- Enable direct modeling of long-range dependencies through self-attention
- Explored in TS-TCC with success in capturing global patterns
- Quadratic complexity with respect to sequence length poses computational challenges
- For very long time series (thousands of time steps), CNN-based architectures often provide better performance-cost trade-offs

**Hybrid Architectures:**
- Recent work explores combinations leveraging strengths of different encoder types
- Examples: CNNs for local feature extraction followed by Transformers for global dependency modeling
- RNNs for temporal encoding with attention mechanisms for selective information aggregation

### 3.4 Few-Shot Learning in Time Series

Few-shot learning (FSL) addresses scenarios where only a handful of labeled samples are available per class, which is common in specialized domains and emerging applications.

**Key Approaches:**

**Metric-Based Learning:**
- **Prototypical Networks (Snell et al., 2017)** [10] - Compute class prototypes and classify queries based on distance in embedding space
- Matching Networks - Extend prototypical approach with attention mechanisms and episodic training
- Relation Networks - Learn to compare samples using neural networks rather than fixed distance metrics
- Particularly effective when embedding space naturally clusters similar examples

**Optimization-Based Learning:**
- Model-Agnostic Meta-Learning (MAML) - Learns initial parameters that can be quickly fine-tuned to new tasks with minimal data
- Extensions like Reptile and ANIL simplify meta-learning objective while maintaining competitive performance
- Require task distributions during meta-training and optimize for rapid adaptation

**Data Augmentation Approaches:**
- Expand limited labeled data through transformations
- Time series strategies: time warping, magnitude scaling, jittering, window slicing, mixup techniques
- Risk: aggressive augmentation can destroy temporal semantics

**Hallucination-Based Methods:**
- Generate synthetic samples from limited examples using generative models
- Aim to expand support set but risk introducing unrealistic or biased samples

In time series, metric-based approaches are particularly attractive due to their simplicity and robustness under small data regimes. They do not require multiple gradient steps per episode, making them computationally efficient. The episodic training paradigm naturally aligns with real-world deployment scenarios where models encounter new classes with minimal examples.

Recent time series-specific few-shot methods include ProtoTime (combining prototypical networks with learned time warping) and TapNet (introducing attentive prototypes that weight time steps by discriminative importance). However, most few-shot time series methods rely on supervised pre-training rather than self-supervised representations, limiting applicability when large labeled pre-training datasets are unavailable.

## 4. Research Gaps and Opportunities

Despite strong progress in both self-supervised time series representation learning and few-shot classification, a critical gap exists at their intersection.

### Gap 1: Integration of Self-Supervised Representations with Few-Shot Classification

**Description:** Most existing self-supervised time series models, including TNC, were not explicitly designed for few-shot classification. Their typical evaluation relies on training a linear classifier on top of frozen embeddings, which works well when ample labels are available but becomes suboptimal under extreme data scarcity.

**Why it matters:** 
- Real-world deployment scenarios often involve limited labeled data, especially in specialized domains like rare disease diagnosis or emerging industrial processes
- Self-supervised methods learn from abundant unlabeled data but fail to fully leverage these representations in low-label regimes
- Current evaluation protocols do not reflect realistic constraints where only 1-5 examples per class are available

**How your project addresses it:** 
This project integrates few-shot learning methods, specifically Prototypical Networks, directly into the TNC framework. By replacing the standard linear classifier with a prototypical head, the system can better exploit learned representations in low-label settings, providing a more sample-efficient solution to time series classification.

### Gap 2: Lack of Few-Shot Methods Leveraging Self-Supervised Pre-training

**Description:** Most few-shot time series classification methods assume supervised pre-training on related tasks, which may not be feasible in specialized domains. The combination of self-supervised representation learning with few-shot classification remains underexplored.

**Why it matters:**
- Supervised pre-training requires large labeled datasets from related domains, which may not exist for specialized applications
- Self-supervised learning can extract general temporal patterns from unlabeled data, then rapidly adapt to new classification tasks with minimal labels
- This combination offers a promising path to practical deployment in data-scarce scenarios

**How your project addresses it:**
The project explores enhanced prototypical methods that introduce learnable components to better leverage TNC representations, including learnable distance metrics, feature transformations, and adaptive architectures. This creates a unified framework that learns general representations unsupervised, then efficiently adapts to new tasks with few examples.

## 5. Theoretical Framework

The theoretical foundation for this research rests on three interconnected pillars:

### 5.1 Self-Supervised Representation Learning

Self-supervised learning creates supervision signals from the data itself, without requiring manual annotations. In the context of time series, this involves exploiting temporal structure to define pretext tasks that encourage the encoder to learn meaningful representations. The fundamental assumption is that good representations should preserve temporal continuity and discriminate between different dynamic regimes.

**Contrastive Learning Theory:**
Contrastive methods maximize agreement between differently augmented views of the same data while minimizing agreement between different data points. Formally, this maximizes a lower bound on mutual information between representations. The InfoNCE loss used in CPC and related methods provides a tractable objective for this optimization.

**Temporal Neighborhood Hypothesis:**
TNC builds on the assumption that nearby time segments from the same signal are more likely to share similar underlying dynamics than distant or unrelated segments. By defining neighborhoods based on local stationarity rather than fixed windows, TNC adapts to the non-stationary nature of real-world time series.

### 5.2 Metric Learning and Few-Shot Classification

Metric learning aims to learn an embedding space where distances correspond to semantic similarity. Prototypical Networks formalize this by representing each class with a prototype (typically the mean of support examples) and classifying queries based on distance to prototypes.

**Episodic Training:**
Few-shot methods employ episodic training where each episode simulates a few-shot task by sampling N classes with K examples each. This meta-learning approach teaches the model to generalize across tasks rather than memorizing specific classes.

**Embedding Space Properties:**
Effective few-shot learning requires embedding spaces with good inductive biases: similar examples should cluster, class boundaries should be well-separated, and the metric should generalize across different task distributions.

### 5.3 Integration Framework

The integration of self-supervised learning with few-shot classification creates a two-stage learning paradigm:

1. **Unsupervised Pre-training:** Learn general temporal representations from abundant unlabeled data using contrastive objectives
2. **Few-Shot Adaptation:** Fine-tune or apply metric-based classification on limited labeled examples

This framework assumes that temporal patterns learned during pre-training transfer to downstream classification tasks, and that metric-based methods can effectively leverage these pre-trained representations even with minimal labels.

## 6. Methodology Insights

### 6.1 Common Evaluation Protocols

Representation learning methods are typically evaluated on downstream tasks to assess the quality and generalizability of learned embeddings:

**Classification:**
- Primary metric: Accuracy and AUPRC (Area Under Precision-Recall Curve)
- Benchmarks: UCR Time Series Classification Archive (univariate datasets spanning medical diagnostics, motion tracking, sensor readings) and UEA archive (multivariate time series)
- Standard protocol: Train linear classifiers or nearest-neighbor methods on frozen representations to assess quality independently of classifier complexity

**Clustering:**
- Metrics: Silhouette score and Davies-Bouldin index
- Purpose: Measure how well learned representations naturally group similar time series without supervision
- Higher silhouette scores indicate better-separated and more compact clusters
- Lower Davies-Bouldin indices indicate better cluster separation

**Forecasting:**
- Metric: Mean Squared Error (MSE)
- Benchmarks: ETT (Electricity Transformer Temperature) dataset, Weather datasets
- Purpose: Evaluate whether learned representations capture predictive temporal patterns

**Anomaly Detection:**
- Metrics: Precision, Recall, and F1 score
- Benchmarks: Yahoo KPI, SWaT (Secure Water Treatment)
- Purpose: Test whether representations can distinguish normal patterns from abnormal events
- Critical for applications like industrial monitoring and cybersecurity

**Transfer Learning:**
- Measure performance when representations learned on one dataset are applied to different but related datasets
- Assesses generalization capability across domains

### 6.2 Promising Methodological Directions

Based on the literature review, several methodological approaches show particular promise for this project:

**1. Temporal Neighborhood Coding (TNC):**
- Adaptive neighborhood definition based on local stationarity
- Debiased contrastive loss reduces false negatives
- Strong performance on non-stationary signals
- Provides a robust foundation for self-supervised pre-training

**2. Prototypical Networks:**
- Simple and computationally efficient for few-shot scenarios
- Do not require multiple gradient steps per episode
- Episodic training aligns with real-world deployment
- Natural compatibility with pre-trained embeddings

**3. Hybrid Approaches:**
- Combining self-supervised pre-training with metric-based few-shot learning
- Enhanced prototypical methods with learnable distance metrics
- Feature transformations specific to temporal data
- Adaptive architectures that adjust to task characteristics

**4. Data Augmentation for Time Series:**
- Careful selection of augmentations that preserve temporal semantics
- Time warping, magnitude scaling, jittering, window slicing
- Validation that augmented samples remain realistic and preserve class labels

### 6.3 Architecture Selection

For time series representation learning, the choice of encoder architecture significantly impacts performance:

- **For long sequences (>1000 time steps):** Dilated CNNs or TCNs offer best efficiency-performance trade-off
- **For complex temporal dependencies:** Hybrid CNN-Transformer architectures
- **For real-time applications:** CNNs provide lower latency than RNNs or Transformers
- **For interpretability:** Attention-based mechanisms can highlight important time steps

The TNC framework's flexibility in encoder choice (RNN or CNN) allows experimentation to find the optimal architecture for specific datasets and tasks.

## 7. Conclusion

The literature reveals a clear evolution in unsupervised time series representation learning, from early reconstruction-based autoencoders to sophisticated contrastive learning frameworks. Modern approaches like TNC, TS2Vec, and BTSF represent the current state-of-the-art, successfully addressing challenges of non-stationarity, long-range dependencies, and multi-scale temporal patterns.

Key findings from this review include:

1. **Contrastive learning superiority:** Methods based on contrastive objectives consistently outperform reconstruction-based approaches, particularly when carefully designed for temporal data characteristics.

2. **Importance of temporal structure:** Successful methods like TNC that explicitly model temporal neighborhoods and local stationarity achieve better performance than those using generic contrastive frameworks.

3. **Architecture matters:** While Transformers show promise, dilated CNNs and TCNs often provide better efficiency-performance trade-offs for long time series.

4. **Evaluation gap:** Most self-supervised methods evaluate using abundant labeled data, not reflecting realistic low-label scenarios.

5. **Few-shot learning opportunity:** Metric-based few-shot methods, particularly Prototypical Networks, offer computational efficiency and natural compatibility with pre-trained embeddings.

**Critical Research Gap:**
Despite advances in both self-supervised representation learning and few-shot classification, their integration remains underexplored. Existing self-supervised time series models were not designed for extreme data scarcity, while most few-shot methods assume supervised pre-training. This gap presents a significant opportunity to develop unified frameworks that learn general temporal patterns from unlabeled data and efficiently adapt to new tasks with minimal labels.

**Project Direction:**
This research addresses the identified gap by enhancing TNC with Prototypical Network-based classification, creating a framework optimized for few-shot time series classification. By exploring learnable distance metrics, feature transformations, and adaptive architectures, the project aims to maximize the value of self-supervised representations in low-label regimes. This approach has practical implications for specialized domains like healthcare, industrial monitoring, and emerging applications where labeled data is scarce but unlabeled temporal data is abundant.

The integration of powerful self-supervised encoders with sample-efficient few-shot methods represents a promising direction for making deep learning more accessible and practical for real-world time series applications.

## References

1. Choi, J., Chiu, C., & Sontag, D. (2016). Learning low-dimensional representations of medical time series data using autoencoder-based models. *Proceedings of the Machine Learning for Healthcare Conference*.

2. Amiriparian, S., Schmitt, M., & Weninger, F. (2017). Audio-visual emotion recognition via multi-channel autoencoders. *Proceedings of the International Conference on Affective Computing and Intelligent Interaction*.

3. Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2017). Long Short Term Memory networks for anomaly detection in time series. *Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*.

4. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

5. Franceschi, J.-Y., et al. (2019). Unsupervised representation learning for time series with triplet loss. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

6. Tonekaboni, S., et al. (2021). Temporal Neighborhood Coding (TNC): Unsupervised representation learning for non-stationary time series. *arXiv preprint arXiv:2106.00750*.

7. Eldele, E., et al. (2021). TS-TCC: A framework for unsupervised representation learning of time series. *Proceedings of the AAAI Conference on Artificial Intelligence*.

8. Yue, C., Zhao, R., & Li, J. (2021). TS2Vec: Towards universal representation of time series. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD)*.

9. Yang, Z., & Hong, B. (2022). Unsupervised time-series representation learning via iterative bilinear temporal-spectral fusion (BTSF). *IEEE Transactions on Neural Networks and Learning Systems*.

10. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems (NeurIPS)*.

---