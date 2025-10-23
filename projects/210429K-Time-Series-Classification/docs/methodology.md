# Methodology: Time Series Classification

**Student:** 210429K
**Research Area:** Time Series Classification
**Date:** 2025-10-19

## 1. Overview

This research extends the Temporal Neighborhood Coding (TNC) framework by incorporating few-shot classification strategies to address the challenge of time series classification under extreme data scarcity. The methodology comprises two principal phases: (1) self-supervised representation learning through pre-training on unlabeled temporal data, and (2) few-shot classification using both baseline and enhanced methods. This two-stage pipeline enables the evaluation of learned representations under realistic low-data scenarios, bridging the gap between powerful self-supervised encoders and sample-efficient classification techniques. The approach demonstrates how robust temporal representations learned without supervision can be effectively leveraged for classification tasks with minimal labeled examples.

## 2. Research Design

This research follows a two-phase experimental design that separates representation learning from classification:

**Phase 1: Self-Supervised Pre-training**
- Objective: Learn robust temporal representations without manual annotations
- Approach: Temporal Neighborhood Coding (TNC) with contrastive learning
- Output: Frozen encoder producing discriminative embeddings

**Phase 2: Few-Shot Classification**
- Objective: Evaluate representation quality under low-label conditions
- Approach: Episodic meta-learning with multiple classification strategies
- Evaluation: N-way, K-shot protocol across varying data availability scenarios

This separation allows for rigorous assessment of whether self-supervised representations transfer effectively to downstream tasks with minimal labeled data. The design ensures that performance differences between classification methods are attributable to the methods themselves rather than variations in feature extraction, as all methods utilize identical frozen embeddings from the TNC encoder.

## 3. Data Collection

### 3.1 Data Sources

The dataset was synthetically generated to emulate long, non-stationary multivariate time series commonly encountered in real-world applications. Synthetic generation provides several advantages:
- **Controlled complexity:** Known ground truth for underlying states and dynamics
- **Reproducibility:** Consistent experimental conditions across trials
- **Scalability:** Ability to generate arbitrary amounts of unlabeled data for pre-training
- **Challenge realism:** Mimics real-world non-stationarity and temporal dependencies

### 3.2 Data Description

**Signal Characteristics:**
- **Sequence length:** 2000 time steps per sequence
- **Dimensionality:** 3 features (multivariate time series)
- **Latent structure:** 4 distinct hidden states governing data generation
- **State transitions:** Sampled from a Hidden Markov Model (HMM)

**Generation Process:**
Within each latent state, data were generated using two distinct methods:
1. **Gaussian Process (GP):** With different kernel functions to create varied temporal patterns
   - Periodic Gaussian Process
   - Squared Exponential Gaussian Process
2. **Nonlinear Auto-Regressive Moving Average (NARMA):** Models for complex nonlinear dynamics
   - NARMA-5 (higher-order nonlinear dependencies)
   - NARMA-3 (moderate nonlinear complexity)

**Feature Dependencies:**
Two of the three features were maintained as correlated throughout each sequence, introducing realistic inter-feature dependencies that challenge representation learning algorithms to capture multivariate relationships.

**Signal Types (Classes):**
1. Periodic Gaussian Process
2. NARMA-5
3. Squared Exponential Gaussian Process
4. NARMA-3

### 3.3 Data Preprocessing

**Temporal Segmentation:**
- Time series were divided into windows of size 50 time steps
- Sliding window approach to generate multiple samples per sequence
- Window size balances capturing sufficient temporal context with maintaining local stationarity

**Temporal Neighborhood Definition:**
- Neighborhoods dynamically defined using the Augmented Dickey-Fuller (ADF) test
- Statistical stationarity testing determines which segments share similar dynamics
- Adaptive approach accommodates non-stationary signal characteristics

**Feature Extraction:**
- Each window encoded into 10-dimensional representation vectors
- Dimensionality reduction balances information retention with computational efficiency
- Fixed embedding dimension ensures consistency across all classification methods

**Data Splits:**
- **Pre-training set:** Large corpus of unlabeled windows for self-supervised learning
- **Support sets:** K examples per class for few-shot training (K ∈ {1, 3, 5, 10, 20})
- **Query sets:** 15 examples per class for few-shot evaluation
- Strict separation between support and query samples within each episode

## 4. Model Architecture

The proposed framework consists of two main components: the TNC encoder for representation learning and various classification heads for few-shot learning.

### 4.1 TNC Encoder Architecture

**Encoder Type:** Bidirectional single-layer Recurrent Neural Network (RNN)

**Architecture Details:**
- **Input:** Windows of 50 time steps × 3 features
- **Processing:** Bidirectional RNN captures temporal dependencies in both forward and backward directions
- **Output:** 10-dimensional representation vectors per window
- **Rationale:** Bidirectional processing enables the model to leverage both past and future context when encoding each temporal segment

**Training Objective:**
The encoder is optimized using the TNC contrastive loss:
- **Positive pairs:** Temporally neighboring segments (locally stationary regions)
- **Negative pairs:** Distant segments with different dynamics
- **Neighborhood detection:** Augmented Dickey-Fuller (ADF) test for stationarity
- **Loss formulation:** Debiased contrastive objective treating non-neighbors as unlabeled rather than strictly negative

**Pre-training Outcome:**
- Class separation ratio: 1.648 (indicates strong discriminative power)
- Embeddings clearly separate underlying signal types
- Frozen weights used for all downstream tasks (no fine-tuning during few-shot learning)

### 4.2 Classification Architectures

#### 4.2.1 Baseline Methods

**a) Linear Classifier**
- **Type:** Logistic Regression
- **Input:** 10-dimensional frozen TNC features
- **Regularization:** L2 penalty with strength C=1.0
- **Optimizer:** Limited-memory BFGS (L-BFGS)
- **Max iterations:** 1000
- **Purpose:** Assess linear separability of learned embeddings

**b) k-Nearest Neighbors (k-NN)**
- **Type:** Non-parametric classifier
- **Distance metric:** Euclidean distance in 10-dimensional embedding space
- **Hyperparameter:** k determined via cross-validation
- **Decision rule:** Majority vote among k nearest neighbors
- **Purpose:** Evaluate natural clustering without model training

**c) Standard Prototypical Networks**
- **Prototype computation:** Mean of support set embeddings per class
- **Distance metric:** Euclidean distance to nearest prototype
- **Classification:** Argmin distance to class prototypes
- **Purpose:** Baseline few-shot method without task-specific adaptation

#### 4.2.2 Enhanced Prototypical Methods

**a) Linear Prototypical Networks**
- **Feature transformation:** Two-layer neural network
  - Layer 1: Linear → ReLU → Dropout
  - Layer 2: Linear transformation
  - Initialization: Xavier normal
- **Prototype space:** Transformed feature space optimized for discrimination
- **Distance metric:** Euclidean distance with fixed temperature parameter
- **Advantage:** Learns optimal projection for prototype-based classification

**b) Metric Prototypical Networks**
- **Distance function:** Learnable neural network (3 fully connected layers)
  - Architecture: FC → ReLU → Dropout → FC → ReLU → Dropout → FC → Sigmoid
  - Input: Concatenation of query and prototype features
  - Output: Distance score ∈ [0, 1]
- **Initialization:** Xavier normal
- **Advantage:** Captures non-linear relationships in embedding space

**c) Hybrid Few-Shot Classifier**
- **Fusion mechanism:** Weighted combination of two branches
  - **Prototypical branch:** Feature transformation → distance-based logits with learnable temperature
  - **Linear branch:** Two-layer neural network with ReLU and dropout
- **Weighting:** Single sigmoid-activated parameter for static fusion
- **Advantage:** Leverages complementary strengths of both approaches

**d) Adaptive Prototypical Networks**
- **Shot-specific adaptation:** Multiple learnable linear transformations
- **Architecture:** 
  - Different transformation pathways for varying K (number of shots)
  - Linear → ReLU → Dropout for each pathway
  - Xavier normal initialization
- **Adaptation strategy:** Architectural complexity adjusts based on shot availability
- **Advantage:** Optimizes representation for specific data availability scenarios

## 5. Experimental Setup

### 5.1 Evaluation Metrics

**Primary Metric:**
- **Classification Accuracy:** Proportion of correctly classified query samples
  - Formula: Accuracy = (Correct Predictions) / (Total Query Samples)
  - Computed per episode and averaged across all episodes
  - Reports mean and standard deviation across 50 independent trials

**Secondary Metrics:**
- **Confusion Matrix Analysis:** Visualizes classification patterns across signal types
  - Identifies systematic misclassification patterns
  - Reveals class-specific performance characteristics
- **Per-Class Accuracy:** Breakdown of accuracy for each signal type
  - Detects potential biases toward specific classes
  - Assesses balanced performance across all categories
- **Class Separation Ratio:** Measures discriminative power of learned embeddings (pre-training evaluation)

### 5.2 Baseline Models

The experimental design includes three levels of baselines:

**Level 1: Simple Baselines**
1. **Logistic Regression (Linear Classifier):** Tests linear separability of TNC embeddings
2. **k-Nearest Neighbors:** Evaluates natural clustering without learning

**Level 2: Standard Few-Shot Baseline**
3. **Standard Prototypical Networks:** Canonical few-shot method without adaptation

**Level 3: Enhanced Methods (Proposed)**
4. **Linear Prototypical Networks:** Learned feature transformation
5. **Metric Prototypical Networks:** Learned distance metric
6. **Hybrid Few-Shot Classifier:** Fusion of prototypical and linear approaches
7. **Adaptive Prototypical Networks:** Shot-specific architectural adaptation

**Comparison Strategy:**
- All methods use identical frozen TNC features (fair comparison)
- Progressive complexity from simple to sophisticated
- Isolates impact of different architectural choices

### 5.3 Hardware/Software Requirements

**Hardware:**
- **GPU:** NVIDIA GPU with CUDA support (recommended for training)
- **RAM:** Minimum 16GB (32GB recommended for larger batch processing)
- **Storage:** ~5GB for datasets, models, and results

**Software Environment:**
- **Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch 1.10+
- **Key Libraries:**
  - NumPy, SciPy (numerical computation)
  - scikit-learn (baseline models, metrics)
  - pandas (data management)
  - matplotlib, seaborn (visualization)
  - statsmodels (ADF test for stationarity)

**Reproducibility:**
- Random seeds fixed for all experiments
- Episode sampling controlled for consistent evaluation
- Code version controlled (Git repository)

## 6. Implementation Plan

### 6.1 Detailed Timeline

| Phase | Tasks | Duration | Deliverables | Status |
|-------|-------|----------|--------------|--------|
| **Phase 1: Data Generation** | - Design synthetic data generator<br>- Implement HMM state transitions<br>- Generate GP and NARMA signals<br>- Validate data characteristics | 1.5 weeks | - Synthetic dataset<br>- Data generation scripts<br>- Validation report | ✓ |
| **Phase 2: TNC Pre-training** | - Implement TNC encoder<br>- Implement ADF-based neighborhood detection<br>- Train contrastive model<br>- Validate embeddings quality | 2 weeks | - Trained TNC encoder<br>- Frozen weights<br>- Embedding visualizations | ✓ |
| **Phase 3: Baseline Implementation** | - Implement Linear Classifier<br>- Implement k-NN<br>- Implement Standard Prototypical Networks<br>- Setup episodic evaluation | 1.5 weeks | - Working baseline models<br>- Evaluation framework | In Progress |
| **Phase 4: Enhanced Methods** | - Implement Linear Prototypical Networks<br>- Implement Metric Prototypical Networks<br>- Implement Hybrid Classifier<br>- Implement Adaptive Prototypical Networks | 2 weeks | - All enhanced methods<br>- Unified training pipeline | Planned |
| **Phase 5: Experiments** | - Run all methods across shot settings<br>- Collect accuracy metrics<br>- Generate confusion matrices<br>- Statistical significance testing | 1.5 weeks | - Complete experimental results<br>- Performance comparisons | Planned |
| **Phase 6: Analysis** | - Analyze per-class performance<br>- Compare method behaviors<br>- Investigate failure cases<br>- Generate visualizations | 1 week | - Analysis report<br>- Figures and tables | Planned |
| **Phase 7: Documentation** | - Write final report<br>- Prepare presentation<br>- Code documentation<br>- Results interpretation | 1.5 weeks | - Final report<br>- Presentation slides<br>- Clean codebase | Planned |

**Total Duration:** ~11 weeks

### 6.2 Experimental Protocol

**Task Configuration:**
- **Classification type:** 4-way classification
- **Classes:** Periodic GP, NARMA-5, Squared Exponential GP, NARMA-3
- **Difficulty:** Multi-class scenario testing discriminative power

**Episode Structure:**
- **Class sampling:** Random selection of 4 classes per episode
- **Support set:** K examples per class (training data)
- **Query set:** 15 examples per class (testing data)
- **Independence:** Strict separation between support and query samples

**Shot Settings (Data Availability Scenarios):**
1. **1-shot:** Extreme low-data condition (4 total support examples)
2. **3-shot:** Very limited data (12 total support examples)
3. **5-shot:** Standard few-shot setting (20 total support examples)
4. **10-shot:** Moderate data availability (40 total support examples)
5. **20-shot:** Increased data availability (80 total support examples)

**Statistical Validation:**
- **Episodes per configuration:** 50 independent trials
- **Randomization:** Different class and sample selections per episode
- **Reproducibility:** Fixed random seeds with documented variations
- **Reporting:** Mean accuracy ± standard deviation

**Feature Consistency:**
- All methods use identical 10-dimensional frozen TNC embeddings
- Performance differences attributable solely to classification strategies
- No fine-tuning of encoder during few-shot evaluation

## 7. Risk Analysis

### Risk 1: Overfitting in Enhanced Methods
**Description:** Complex learnable components in enhanced prototypical methods may overfit to limited support examples, especially in 1-shot and 3-shot scenarios.

**Impact:** High - Could negate benefits of sophisticated architectures

**Mitigation Strategies:**
- Employ dropout regularization in all learnable layers
- Use Xavier normal initialization for stable gradient flow
- Limit model complexity relative to available data
- Cross-validate hyperparameters on separate validation episodes
- Monitor training/validation performance divergence

### Risk 2: Poor Generalization from Synthetic Data
**Description:** Synthetic data may not capture full complexity of real-world time series, limiting applicability of findings.

**Impact:** Medium - Affects real-world relevance

**Mitigation Strategies:**
- Design synthetic generation to mimic realistic non-stationarity
- Include multiple generation processes (GP, NARMA)
- Introduce feature correlations similar to real applications
- Plan future evaluation on benchmark datasets (UCR/UEA)
- Document limitations and generalization assumptions

### Risk 3: Inadequate Statistical Power
**Description:** 50 episodes may be insufficient to detect small performance differences between methods, especially in high-variance scenarios.

**Impact:** Medium - Affects reliability of conclusions

**Mitigation Strategies:**
- Conduct power analysis to validate episode count
- Report confidence intervals alongside mean accuracy
- Perform statistical significance testing (t-tests, ANOVA)
- Increase episode count for close comparisons if needed
- Use multiple random seeds for robustness

### Risk 4: Encoder Quality Bottleneck
**Description:** If TNC pre-training produces poor embeddings, all downstream methods will underperform regardless of classification strategy quality.

**Impact:** High - Affects entire experimental framework

**Mitigation Strategies:**
- Validate embedding quality through class separation ratio
- Visualize embeddings using t-SNE/UMAP
- Compare against alternative self-supervised methods
- Ensure sufficient pre-training convergence
- Monitor contrastive loss during training

### Risk 5: Hyperparameter Sensitivity
**Description:** Enhanced methods may be sensitive to hyperparameters (learning rate, dropout rate, temperature), making fair comparison difficult.

**Impact:** Medium - Affects method comparison validity

**Mitigation Strategies:**
- Systematic hyperparameter search for each method
- Use consistent search protocol across all methods
- Report sensitivity analysis for key hyperparameters
- Document all hyperparameter choices
- Employ early stopping to prevent overtraining

### Risk 6: Computational Resource Constraints
**Description:** Training multiple methods across multiple shot settings and episodes may exceed available computational resources or time.

**Impact:** Low-Medium - Affects timeline

**Mitigation Strategies:**
- Prioritize most promising methods if resources limited
- Parallelize independent episode evaluations
- Optimize code for efficiency
- Use efficient batch processing
- Access cloud computing resources if needed

## 8. Expected Outcomes

### 8.1 Primary Outcomes

**1. Performance Validation of TNC for Few-Shot Learning**
- Demonstrate that TNC embeddings are effective for few-shot classification
- Achieve competitive or superior accuracy compared to supervised baselines
- Show consistent performance across different shot settings (1-shot to 20-shot)

**2. Method Performance Ranking**
- Identify which classification strategies work best with TNC representations
- Compare simple baselines (Linear, k-NN) against sophisticated few-shot methods
- Determine whether enhanced prototypical methods outperform standard approaches

**3. Data Efficiency Analysis**
- Quantify performance gains as labeled examples increase
- Identify diminishing returns point where additional shots provide minimal benefit
- Characterize extreme low-data performance (1-shot, 3-shot scenarios)

### 8.2 Research Contributions

**Methodological Contributions:**
- Novel integration of TNC self-supervised learning with prototypical few-shot classification
- Four enhanced prototypical architectures specifically designed for temporal embeddings
- Comprehensive evaluation framework for few-shot time series classification

**Empirical Contributions:**
- Systematic comparison of seven classification methods on identical feature representations
- Performance characterization across five data availability scenarios
- Analysis of method-specific strengths and failure modes

**Practical Contributions:**
- Demonstration of self-supervised learning value for low-label scenarios
- Guidelines for selecting classification methods based on available labeled data
- Framework applicable to real-world domains with annotation scarcity (healthcare, industrial monitoring)

### 8.3 Expected Results

**Hypothesis 1:** Enhanced prototypical methods will outperform simple baselines, particularly in extreme low-data regimes (1-shot, 3-shot), due to their ability to learn task-specific feature transformations and distance metrics.

**Hypothesis 2:** The Hybrid Few-Shot Classifier will demonstrate robust performance across all shot settings by leveraging complementary strengths of prototypical and linear approaches.

**Hypothesis 3:** Adaptive Prototypical Networks will show varying performance advantages depending on shot availability, performing best when architectural complexity matches data availability.

**Hypothesis 4:** All methods will benefit from TNC's strong pre-trained representations, achieving reasonable accuracy even in 1-shot scenarios where traditional supervised methods would fail.

### 8.4 Success Criteria

**Minimum Success:**
- TNC embeddings enable >70% accuracy in 5-shot scenarios
- At least one enhanced method outperforms all baselines
- Clear performance trends as shot number increases

**Target Success:**
- >60% accuracy in 1-shot, >80% accuracy in 5-shot, >90% accuracy in 20-shot
- Enhanced methods provide statistically significant improvements over baselines
- Consistent ranking of methods across multiple shot settings

**Exceptional Success:**
- Performance approaches supervised upper bound with minimal labeled data
- Clear understanding of which architectural components drive performance
- Findings generalize to benchmark datasets (UCR/UEA archives)

### 8.5 Future Directions

**Short-term Extensions:**
- Evaluation on real-world benchmark datasets (UCR, UEA)
- Comparison with other self-supervised methods (TS2Vec, BTSF)
- Extension to cross-domain transfer scenarios

**Long-term Research:**
- Application to specialized domains (medical time series, industrial sensors)
- Integration with active learning for intelligent sample selection
- Development of unified end-to-end training frameworks
- Exploration of multi-modal time series representation learning

### 8.6 Deliverables

1. **Code Repository:** Clean, documented implementation of all methods
2. **Experimental Results:** Comprehensive performance tables and visualizations
3. **Final Report:** Detailed analysis of findings with theoretical interpretation
4. **Presentation:** Summary of methodology, results, and contributions
5. **Documentation:** User guide for applying framework to new datasets

---