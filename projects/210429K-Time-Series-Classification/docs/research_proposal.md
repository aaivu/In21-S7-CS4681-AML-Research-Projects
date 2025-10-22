# Research Proposal: Time Series Classification

**Student:** 210429K
**Research Area:** Time Series Classification
**Date:** 2025-10-19

## Abstract

This research proposes a novel integration of Temporal Neighborhood Coding (TNC) with few-shot learning methods to address time series classification under extreme data scarcity. While self-supervised representation learning has achieved remarkable success in extracting meaningful temporal patterns from unlabeled data, existing methods were not explicitly designed for few-shot scenarios where only 1-5 labeled examples per class are available. We address this gap by extending the TNC framework with prototypical network-based classification strategies, including four enhanced variants with learnable components: Linear-Prototypical, Metric-Prototypical, Hybrid-Prototypical, and Adaptive-Prototypical Networks. Our methodology employs a two-stage pipeline: first, self-supervised pre-training using TNC's contrastive learning on unlabeled multivariate time series; second, episodic few-shot evaluation across varying shot settings (1-shot to 20-shot). Preliminary results on synthetically generated non-stationary signals demonstrate that Metric-Prototypical Networks achieve 77.4% accuracy in 5-shot scenarios, outperforming standard baselines. This work contributes a comprehensive framework for sample-efficient time series classification with practical implications for domains where labeled data is scarce, such as healthcare diagnostics and industrial monitoring.

## 1. Introduction

Time series classification is a fundamental task in machine learning with applications spanning healthcare monitoring, industrial process control, financial forecasting, and human activity recognition. Traditional supervised approaches require substantial amounts of labeled data to achieve acceptable performance, which is often prohibitively expensive or impractical to obtain in specialized domains. For instance, diagnosing rare medical conditions from ECG signals or detecting novel failure modes in manufacturing equipment may provide only a handful of labeled examples.

Recent advances in self-supervised representation learning have demonstrated remarkable success in learning meaningful patterns from unlabeled data. Methods like Temporal Neighborhood Coding (TNC), TS2Vec, and BTSF leverage contrastive learning to extract robust temporal features without manual annotations. These approaches excel at capturing complex temporal dependencies and non-stationary dynamics that characterize real-world time series.

However, a critical gap exists between the capabilities of self-supervised encoders and the requirements of real-world deployment. Most existing methods evaluate their representations by training linear classifiers with abundant labeled data, which does not reflect scenarios where only a few labeled examples are available. While few-shot learning has achieved success in computer vision and natural language processing, its integration with self-supervised time series representations remains largely unexplored.

This research addresses this gap by combining the strengths of TNC's powerful temporal representations with prototypical network-based few-shot classification. We hypothesize that self-supervised pre-trained encoders, when paired with appropriate few-shot methods, can enable effective classification with minimal labeled data. This integration has significant practical implications: unlabeled temporal data is often abundant (continuous sensor streams, long-term monitoring), but obtaining expert annotations is expensive. Our approach enables leveraging this abundant unlabeled data to learn general temporal patterns, then rapidly adapting to new classification tasks with only a few labeled examples per class.

The significance of this work extends beyond methodological contributions. By demonstrating that sophisticated self-supervised representations can be effectively utilized in extreme low-data regimes, we provide a pathway toward deploying deep learning in domains where data annotation is the primary bottleneck.

## 2. Problem Statement

**Primary Challenge:** How can we effectively classify time series data when only a few labeled examples (1-5 per class) are available, while abundant unlabeled temporal data exists?

**Specific Research Questions:**

1. **Representation Quality:** Do self-supervised methods like TNC produce embeddings that naturally support few-shot classification, even though they were not explicitly designed for this purpose?

2. **Method Selection:** Which few-shot classification strategies are most effective for leveraging pre-trained temporal representations? Do sophisticated learnable approaches outperform simple baselines?

3. **Architectural Design:** Can we enhance standard prototypical networks with learnable components (feature transformations, distance metrics, adaptive architectures) to better exploit the specific properties of TNC embeddings?

4. **Data Efficiency:** How does classification performance scale with the number of available labeled examples (shots)? What is the minimum number of examples needed for acceptable performance?

5. **Practical Applicability:** Can this integrated framework generalize to real-world time series datasets beyond synthetic signals, enabling practical deployment in domains with annotation scarcity?

**Technical Challenges:**

- **Non-stationarity:** Time series often exhibit changing dynamics over time, making it difficult to define meaningful similarity between segments
- **High dimensionality:** Long sequences with multiple features create large input spaces that are difficult to learn from limited labels
- **Temporal alignment:** Similar patterns may occur at different time scales or with temporal shifts
- **Domain specificity:** Temporal patterns that distinguish classes in one domain may not transfer to others
- **Overfitting risk:** Complex learnable components may overfit when trained on very few examples

**Current Limitations:**

Existing self-supervised time series methods (TNC, TS2Vec, BTSF) evaluate using linear classifiers with substantial labeled data, which does not reflect realistic deployment constraints. Conversely, existing few-shot time series methods assume supervised pre-training on related tasks, which may not be available in specialized domains. This research addresses the intersection of these two limitations by integrating self-supervised representation learning with few-shot classification strategies.

## 3. Literature Review Summary

The literature reveals a clear evolution in unsupervised time series representation learning, from early autoencoder-based approaches to sophisticated contrastive learning frameworks.

**Autoencoder-Based Methods (2016-2017):**
Early work by Choi et al., Amiriparian et al., and Malhotra et al. employed reconstruction objectives to learn compressed representations. However, pixel-level reconstruction does not guarantee discriminative features for classification tasks, and these methods struggle with high-frequency noise and complex temporal dynamics.

**Contrastive Learning Revolution (2018-2022):**
Contrastive Predictive Coding (CPC) introduced the concept of predicting future representations from past context, but struggled with non-stationary signals. Triplet Loss methods improved temporal modeling but suffered from false negatives when similar states occur at distant time points. Temporal Neighborhood Coding (TNC) addressed these limitations by dynamically defining neighborhoods using statistical stationarity tests and employing debiased contrastive loss. TS-TCC introduced dual temporal-contextual contrasting with Transformer backbones. TS2Vec achieved state-of-the-art results through hierarchical contrastive learning across multiple temporal scales. BTSF incorporated spectral features alongside temporal representations.

**Few-Shot Learning (2017-present):**
Prototypical Networks established the foundational metric-based approach, computing class prototypes and classifying based on distance. MAML and related optimization-based methods enable rapid task adaptation. Time series-specific few-shot methods like ProtoTime and TapNet have emerged, but most rely on supervised pre-training rather than self-supervised representations.

**Critical Research Gap:**
Despite advances in both areas, their integration remains underexplored. Self-supervised time series models evaluate using abundant labeled data, while few-shot methods assume supervised pre-training. This creates a missed opportunity: unlabeled temporal data is often abundant, but expert annotations are scarce. Our research directly addresses this gap by combining TNC's powerful self-supervised representations with prototypical network-based few-shot classification, creating a unified framework optimized for extreme data scarcity.

**Key Insights Informing Our Approach:**
- TNC's adaptive neighborhood definition and debiased loss make it particularly suitable for non-stationary signals
- Metric-based few-shot methods are computationally efficient and naturally compatible with frozen embeddings
- Learnable distance metrics and feature transformations can adapt generic embeddings to specific tasks
- Evaluation protocols must reflect realistic low-label scenarios rather than abundant-data assumptions

## 4. Research Objectives

### Primary Objective

Develop and evaluate an integrated framework combining Temporal Neighborhood Coding (TNC) self-supervised representation learning with prototypical network-based few-shot classification to enable effective time series classification under extreme data scarcity (1-20 labeled examples per class).

### Secondary Objectives

- **Objective 1: Validate TNC Representations for Few-Shot Learning**
  - Demonstrate that TNC embeddings possess strong discriminative properties suitable for few-shot classification
  - Achieve competitive accuracy compared to supervised baselines despite minimal labeled data
  - Quantify the class separation ratio and natural clustering properties of learned embeddings

- **Objective 2: Develop Enhanced Prototypical Methods**
  - Design and implement four enhanced prototypical architectures with learnable components:
    - Linear-Prototypical Networks (learnable feature transformation)
    - Metric-Prototypical Networks (learnable distance metric)
    - Hybrid-Prototypical Networks (fusion of prototypical and linear approaches)
    - Adaptive-Prototypical Networks (shot-specific architectural adaptation)
  - Compare enhanced methods against standard baselines (Linear Classifier, k-NN, Standard Prototypical)

- **Objective 3: Characterize Data Efficiency**
  - Systematically evaluate performance across five shot settings: 1-shot, 3-shot, 5-shot, 10-shot, 20-shot
  - Identify the relationship between labeled data availability and classification accuracy
  - Determine minimum shot requirements for acceptable performance in different scenarios

- **Objective 4: Establish Best Practices**
  - Identify which classification strategies work best with TNC representations
  - Provide guidelines for method selection based on available labeled data
  - Document architectural design choices that maximize few-shot performance

- **Objective 5: Foundation for Real-World Application**
  - Establish methodology and evaluation framework applicable to real-world datasets
  - Identify limitations and future directions for practical deployment
  - Demonstrate proof-of-concept for domains with annotation scarcity (healthcare, industrial monitoring)

## 5. Methodology

Our methodology follows a two-stage pipeline: self-supervised pre-training followed by few-shot classification evaluation.

### Stage 1: Self-Supervised Representation Learning (TNC Pre-training)

**Data Generation:**
- Synthetically generated long, non-stationary multivariate time series
- 2000 time steps × 3 features per sequence
- 4 latent states governed by Hidden Markov Model (HMM)
- Data generated using Gaussian Processes (Periodic, Squared Exponential kernels) and NARMA models (NARMA-3, NARMA-5)
- Two correlated features to introduce realistic dependencies

**Encoder Architecture:**
- Bidirectional single-layer RNN encoder
- Input: Windows of 50 time steps
- Output: 10-dimensional representation vectors
- Captures temporal dependencies in both forward and backward directions

**Training Objective:**
- TNC contrastive loss distinguishing temporal neighbors from distant segments
- Neighborhoods dynamically defined using Augmented Dickey-Fuller (ADF) stationarity test
- Debiased loss treats non-neighbors as unlabeled rather than strictly negative
- Reduces false negatives in non-stationary signals

**Pre-training Outcome:**
- Frozen encoder weights used for all downstream tasks
- Achieved class separation ratio of 1.648 (strong discriminative power)

### Stage 2: Few-Shot Classification Evaluation

**Classification Methods (7 total):**

*Baseline Methods:*
1. **Linear Classifier:** Logistic regression with L2 regularization (C=1.0), L-BFGS optimizer
2. **k-Nearest Neighbors:** Non-parametric classification based on Euclidean distance
3. **Standard Prototypical Networks:** Class prototypes as mean of support embeddings

*Enhanced Prototypical Methods:*
4. **Linear-Prototypical Networks:** Two-layer neural transformation with ReLU and dropout
5. **Metric-Prototypical Networks:** Learnable 3-layer distance function with sigmoid activation
6. **Hybrid-Prototypical Networks:** Weighted fusion of prototypical and linear branches
7. **Adaptive-Prototypical Networks:** Shot-specific transformation pathways

**Experimental Protocol:**
- **Task:** 4-way classification (Periodic GP, NARMA-5, Squared Exponential GP, NARMA-3)
- **Episode structure:** 4 classes, K support examples per class, 15 query examples per class
- **Shot settings:** 1-shot, 3-shot, 5-shot, 10-shot, 20-shot
- **Statistical validation:** 50 independent episodes per configuration
- **Feature consistency:** All methods use identical frozen TNC embeddings

**Evaluation Metrics:**
- Primary: Classification accuracy (mean ± std across episodes)
- Secondary: Confusion matrices, per-class accuracy, statistical significance testing

This comprehensive methodology enables rigorous assessment of both TNC representation quality and the effectiveness of diverse few-shot classification strategies under realistic low-label conditions.

## 6. Expected Outcomes

### Performance Achievements

**Primary Results:**
Based on preliminary experiments, we expect to demonstrate that TNC representations enable effective few-shot classification across all shot settings:

- **1-shot scenarios:** 53-59% accuracy (extreme data scarcity)
- **3-shot scenarios:** 67-73% accuracy (very limited data)
- **5-shot scenarios:** 71-77% accuracy (standard few-shot setting)
- **10-shot scenarios:** 73-77% accuracy (moderate data availability)
- **20-shot scenarios:** 65-78% accuracy (increased data availability)

**Key Findings:**

1. **Metric-Prototypical Networks achieve best overall performance:**
   - 77.4% accuracy in 5-shot scenarios
   - 76.9% accuracy in 10-shot scenarios
   - Learnable distance metrics effectively capture task-specific relationships better than fixed Euclidean distances

2. **Enhanced methods outperform baselines in low-shot regimes:**
   - Linear-Prototypical, Metric-Prototypical, and Hybrid-Prototypical all exceed baseline performance in 1-shot and 3-shot settings
   - Demonstrates value of learnable components when labeled data is extremely limited

3. **Trade-offs between architectural complexity and data availability:**
   - Adaptive-Prototypical shows performance degradation at 20-shot (64.8%)
   - Suggests potential overfitting when architectural complexity exceeds data requirements
   - Highlights importance of matching model capacity to available data

4. **Linear Classifier competitive across all settings:**
   - Strong performance (77.8% at 20-shot) indicates TNC embeddings are linearly separable
   - Validates quality of self-supervised pre-training

### Research Contributions

**Methodological Contributions:**
- First comprehensive integration of TNC with prototypical few-shot classification
- Four novel enhanced prototypical architectures for temporal embeddings
- Rigorous evaluation framework for few-shot time series classification

**Empirical Contributions:**
- Systematic comparison of seven methods across five shot settings
- Performance characterization demonstrating data efficiency of integrated approach
- Identification of method-specific strengths and limitations

**Practical Contributions:**
- Demonstration that self-supervised learning enables classification with minimal labels
- Guidelines for method selection based on available labeled data
- Framework applicable to real-world domains with annotation scarcity

### Limitations and Future Directions

**Current Limitations:**
- Evaluation on synthetic data only (due to computational constraints)
- No validation on standardized benchmarks (UCR/UEA archives)
- Limited exploration of cross-domain transfer learning
- Computational resources constrained hyperparameter optimization

**Future Research Directions:**
1. **Real-world validation:** Evaluate on ECG, Human Activity Recognition, and industrial sensor datasets
2. **Benchmark evaluation:** Test on UEA Time Series Classification Archive for rigorous comparison
3. **Cross-domain transfer:** Pre-train on one signal type, evaluate on entirely different domains
4. **Data augmentation:** Develop time series-specific augmentation techniques for few-shot scenarios
5. **Meta-learning integration:** Explore MAML and optimization-based approaches for rapid task adaptation
6. **Hierarchical methods:** Leverage multi-scale temporal representations from methods like TS2Vec
7. **Interpretability analysis:** Understand what temporal patterns learned prototypes capture

### Impact and Applications

This research provides a pathway for deploying deep learning in domains where unlabeled temporal data is abundant but expert annotations are scarce:
- **Healthcare:** Rare disease diagnosis from physiological signals
- **Industrial monitoring:** Novel failure mode detection with limited historical examples
- **Environmental sensing:** Classifying unusual events in long-term monitoring data
- **Human-computer interaction:** Recognizing new gestures or activities with minimal training

## 7. Timeline

| Week | Task | Deliverables | Status |
|------|------|--------------|--------|
| 1-2 | **Literature Review** | - Survey of self-supervised time series methods<br>- Review of few-shot learning approaches<br>- Identify research gaps | ✓ Complete |
| 3 | **Methodology Development** | - Design two-stage pipeline<br>- Specify encoder architecture<br>- Define experimental protocol | ✓ Complete |
| 4 | **Data Generation** | - Implement HMM-based synthetic generator<br>- Generate GP and NARMA signals<br>- Validate data characteristics | ✓ Complete |
| 5-6 | **TNC Pre-training** | - Implement TNC encoder and contrastive loss<br>- Implement ADF-based neighborhood detection<br>- Train and validate encoder quality | ✓ Complete |
| 7 | **Baseline Implementation** | - Linear Classifier<br>- k-NN<br>- Standard Prototypical Networks | ✓ Complete |
| 8-9 | **Enhanced Methods Implementation** | - Linear-Prototypical Networks<br>- Metric-Prototypical Networks<br>- Hybrid-Prototypical Classifier<br>- Adaptive-Prototypical Networks | ✓ Complete |
| 10-11 | **Experimentation** | - Run all methods across 5 shot settings<br>- 50 episodes per configuration<br>- Collect accuracy metrics and confusion matrices | ✓ Complete |
| 12 | **Analysis** | - Statistical significance testing<br>- Per-class performance analysis<br>- Method comparison and ranking<br>- Identify strengths and limitations | In Progress |
| 13-14 | **Documentation** | - Write final report<br>- Generate visualizations (tables, plots)<br>- Document code and experiments | In Progress |
| 15 | **Final Preparation** | - Prepare presentation<br>- Review and revisions<br>- Code cleanup and documentation | Planned |
| 16 | **Submission** | - Final report submission<br>- Code repository finalization<br>- Presentation delivery | Planned |

**Critical Milestones:**
- ✓ Week 2: Literature review complete, research gap identified
- ✓ Week 4: Synthetic dataset generated and validated
- ✓ Week 6: TNC encoder trained with strong class separation (ratio: 1.648)
- ✓ Week 9: All seven classification methods implemented
- ✓ Week 11: Complete experimental results obtained
- Week 14: Final report draft complete
- Week 16: Project submission and presentation

## 8. Resources Required

### Computational Resources

**Hardware:**
- GPU: NVIDIA GPU with CUDA support (utilized for TNC pre-training and enhanced method training)
- RAM: 16-32GB for dataset generation and batch processing
- Storage: ~5GB for datasets, model checkpoints, and experimental results
- CPU: Multi-core processor for parallel episode evaluation

**Software Environment:**
- Python 3.8+
- PyTorch 1.10+ (deep learning framework)
- CUDA toolkit (GPU acceleration)
- NumPy, SciPy (numerical computation)
- scikit-learn (baseline models, metrics)
- pandas (data management)
- matplotlib, seaborn (visualization)
- statsmodels (ADF test for stationarity detection)

### Datasets

**Current (Completed):**
- Synthetically generated multivariate time series
  - 4 signal types: Periodic GP, NARMA-5, Squared Exponential GP, NARMA-3
  - Generated using HMM with GP and NARMA models
  - 2000 time steps × 3 features per sequence

**Future (Planned):**
- UCR Time Series Classification Archive (univariate benchmarks)
- UEA Multivariate Time Series Classification Archive
- ECG datasets (PhysioNet, MIT-BIH)
- Human Activity Recognition datasets (UCI HAR, PAMAP2)
- Industrial sensor data (if available)

### Tools and Libraries

**Development Tools:**
- Git/GitHub (version control and collaboration)
- Jupyter Notebooks (experimentation and visualization)
- VS Code / PyCharm (code development)
- TensorBoard (training monitoring, optional)

**Key Python Libraries:**
- `torch`, `torch.nn` (neural network implementation)
- `sklearn.linear_model.LogisticRegression` (Linear Classifier baseline)
- `sklearn.neighbors.KNeighborsClassifier` (k-NN baseline)
- `sklearn.metrics` (accuracy, confusion matrix)
- `statsmodels.tsa.stattools.adfuller` (stationarity testing)
- `numpy.random` (episode sampling, reproducibility)

### Knowledge Resources

**Key Papers (Implemented/Referenced):**
1. Tonekaboni et al. (2021) - Temporal Neighborhood Coding (TNC)
2. Snell et al. (2017) - Prototypical Networks for Few-Shot Learning
3. Oord et al. (2018) - Contrastive Predictive Coding
4. Yue et al. (2021) - TS2Vec
5. Franceschi et al. (2019) - Triplet Loss for Time Series

**Documentation:**
- PyTorch documentation (neural network APIs)
- scikit-learn documentation (baseline models)
- Research papers on few-shot learning and time series analysis

### Human Resources

**Student Effort:**
- Primary researcher: 210429K
- Estimated effort: ~20 hours/week for 16 weeks

**Supervision:**
- Faculty advisor for guidance and feedback
- Peer review for methodology validation

### Budget

**Current Project:** No additional budget required (using existing computational resources and open-source software)

**Future Extensions:** May require:
- Cloud computing credits for larger-scale experiments (AWS, Google Cloud, Azure)
- Access to proprietary datasets (if not publicly available)
- Computational cluster access for cross-domain transfer experiments

## References

1. Tonekaboni, S., et al. (2021). Temporal Neighborhood Coding (TNC): Unsupervised representation learning for non-stationary time series. *arXiv preprint arXiv:2106.00750*.

2. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*.

4. Yue, C., Zhao, R., & Li, J. (2021). TS2Vec: Towards universal representation of time series. *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD)*.

5. Franceschi, J.-Y., et al. (2019). Unsupervised representation learning for time series with triplet loss. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

6. Eldele, E., et al. (2021). TS-TCC: A framework for unsupervised representation learning of time series. *Proceedings of the AAAI Conference on Artificial Intelligence*.

7. Yang, Z., & Hong, B. (2022). Unsupervised time-series representation learning via iterative bilinear temporal-spectral fusion (BTSF). *IEEE Transactions on Neural Networks and Learning Systems*.

8. Choi, J., Chiu, C., & Sontag, D. (2016). Learning low-dimensional representations of medical time series data using autoencoder-based models. *Proceedings of the Machine Learning for Healthcare Conference*.

9. Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2017). Long Short Term Memory networks for anomaly detection in time series. *Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*.

10. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for fast adaptation of deep networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.

---

## Appendix: Preliminary Results

### Table 1: Few-Shot Classification Accuracy on Simulated Dataset

| Method | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot |
|--------|--------|--------|--------|---------|---------|
| **Prototypical Networks** | 0.566 | 0.699 | 0.718 | 0.733 | 0.765 |
| **k-NN Baseline** | 0.357 | 0.621 | 0.659 | 0.704 | 0.725 |
| **Linear Baseline** | 0.555 | 0.691 | 0.721 | 0.751 | 0.778 |
| **Linear-Prototypical** | 0.578 | 0.727 | 0.724 | 0.743 | 0.752 |
| **Metric-Prototypical** | **0.587** | **0.729** | **0.774** | **0.769** | 0.772 |
| **Hybrid-Prototypical** | 0.580 | 0.691 | 0.740 | 0.727 | 0.756 |
| **Adaptive-Prototypical** | 0.531 | 0.672 | 0.705 | 0.719 | 0.648 |

**Key Observations:**
- **Best overall performance:** Metric-Prototypical Networks (77.4% at 5-shot, 76.9% at 10-shot)
- **Most consistent baseline:** Linear Classifier (strong across all shot settings, 77.8% at 20-shot)
- **Extreme low-data:** Enhanced methods (Linear-Prototypical, Metric-Prototypical, Hybrid-Prototypical) outperform baselines in 1-shot scenarios
- **Potential overfitting:** Adaptive-Prototypical shows degradation at 20-shot (64.8%), suggesting excessive architectural complexity

These preliminary results validate our research hypothesis that TNC representations support effective few-shot classification, and that learnable distance metrics can capture task-specific relationships more effectively than fixed Euclidean distances.

---

**Submission Instructions:**
1.  Complete all sections above
2.  Commit your changes to the repository
3.  Create an issue with the label "milestone" and "research-proposal"
4.  Tag your supervisors in the issue for review