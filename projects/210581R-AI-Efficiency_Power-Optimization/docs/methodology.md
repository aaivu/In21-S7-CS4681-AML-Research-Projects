# Methodology: AI Efficiency:Power Optimization

**Student:** 210581R
**Research Area:** AI Efficiency:Power Optimization
**Date:** 2025-09-01

## 1. Overview

This research aims to optimize the fine-tuning of Transformer-based language models for reduced energy consumption without compromising performance accuracy. The study emphasizes Green AI principles, where the primary goal is to achieve a balance between computational efficiency and model accuracy during model adaptation. Using DistilBERT, this methodology investigates strategies like mixed precision training, gradient checkpointing, optimizer tuning, and learning rate scheduling to reduce power consumption.

## 2. Research Design

The research follows an experimental and quantitative design. The workflow involves implementing various optimization techniques on a baseline model (DistilBERT) and evaluating their impact on both performance metrics (accuracy, validation loss) and energy metrics (carbon footprint, training time).

Key Phases:

    1. Baseline fine-tuning of DistilBERT on the IMDB sentiment classification dataset.

    2. Incremental application of optimization strategies (e.g., FP16 precision, checkpointing, dropout adjustment).

    3. Measurement of energy consumption using CodeCarbon and system metrics.

    4. Comparative analysis of results against baseline and related methods.

## 3. Data Collection

### 3.1 Data Sources

Dataset: IMDB Movie Review Dataset (50,000 labeled samples)
Source: Kaggle / TensorFlow Datasets

### 3.2 Data Description

Training samples: 25,000
Testing samples: 25,000
Labels: Positive (1), Negative (0)
Average length: ~230 words per review

### 3.3 Data Preprocessing
Tokenization using the DistilBERT tokenizer (WordPiece).

Padding/truncation to a maximum sequence length of 256 tokens.

Text cleaning: removal of HTML tags, extra spaces, and special symbols.

Dataset splitting into training (80%), validation (10%), and test (10%).

## 4. Model Architecture

* Base Model: DistilBERT (pre-trained Transformer model with 6 layers, 66M parameters).

* Classification Head: A dense layer with a softmax activation for binary sentiment classification.

* Optimization Techniques Applied:

    -Mixed precision training (FP16)

    -Gradient checkpointing

    -Label smoothing

    -Increased dropout regularization

    -Optimizer comparisons (Adam, AdamW, SGD + Momentum)

    -Cosine learning rate scheduler

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Performance Metrics: Accuracy, Validation Loss, F1-score

- Energy Metrics: Carbon emission (kgCO₂), Power consumption (kWh), Training time (minutes)

### 5.2 Baseline Models
-DistilBERT (without optimization)

-BERT-base fine-tuned (for comparison of parameter efficiency)

### 5.3 Hardware/Software Requirements
*Hardware: NVIDIA T4 GPU (Colab environment)

*Software Stack:

    -Python 3.10

    -PyTorch / TensorFlow

    -HuggingFace Transformers

    -CodeCarbon (for energy tracking)

    -NumPy, Pandas, Matplotlib for analysis

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean and tokenized IMDB dataset|
| Phase 2 | Baseline and optimized model implementation | 3 weeks | Implemented models with and without optimization |
| Phase 3 | Experiments and energy tracking | 2 weeks | Collected performance and energy metrics |
| Phase 4 | Results analysis and visualization | 1 week | Comparative charts, final analysis report |

## 7. Risk Analysis

| Risk                   | Description                                  | Mitigation Strategy                                      |
| ---------------------- | -------------------------------------------- | -------------------------------------------------------- |
| Computational Limits   | Colab GPU timeout or limited runtime         | Use checkpointing and smaller subsets for iterative runs |
| Data Imbalance         | Class distribution bias                      | Ensure balanced batches during training                  |
| Overfitting            | Model may memorize data due to small dataset | Apply dropout and early stopping                         |
| Energy Tracking Errors | CodeCarbon inconsistencies on virtual GPUs   | Cross-verify with runtime logs and GPU utilization stats |


## 8. Expected Outcomes

-Demonstration that energy-efficient fine-tuning can significantly reduce carbon emissions while maintaining model accuracy.

-Empirical evidence on trade-offs between accuracy and energy for various optimization techniques.

-A set of reproducible Green AI guidelines for training and fine-tuning Transformer models.

-A potential benchmark for energy-aware NLP model training.

---

**Note:** Update this document as your methodology evolves during implementation.