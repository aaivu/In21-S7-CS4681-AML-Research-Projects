# Methodology: Exploring Fine-Tuning Enhancements for MolCLR

**Student:** 210306G
**Research Area:** Medical Domain:Drug Discovery
**Date:** 2025-10-23

## 1. Overview

In this work, I systematically investigated the impact of common enhancements on the MolCLR fine-tuning process. Explore advanced optimisation strategies (e.g., AdamW optimiser, cosine annealing schedules) and architectural modifications (e.g., Transformer-based readout, GIN/GCN ensembles). Through systematic evaluation, found these modifications do not yield significant performance gains over the well-tuned baseline, highlighting the robustness of the pre-trained representations.

## 2. Research Design

This project presents a systematic investigation into the practical impact of common deep learning enhancements on the MolCLR framework. The objective is to quantify the impact of these strategies and determine their efficacy in the context of fine-tuning pre-trained molecular models. The approach involves applying various optimisation and architectural changes and then conducting extensive evaluations on MoleculeNet benchmarks to compare performance against the robust MolCLR baseline.

## 3. Data Collection

### 3.1 Data Sources
The models are fine-tuned and evaluated on established **MoleculeNet benchmarks**. Experiments ware conducted specifically on the **BBBP dataset**. The pre-training phase leveraged large-scale unlabelled molecular databases.

### 3.2 Data Description
The data consists of **molecular graphs**, with atoms as nodes and bonds as edges. The downstream tasks include **classification** (e.g., chemical toxicity like BBBP, Tox21) and **regression** (e.g., solubility or free energy prediction).

### 3.3 Data Preprocessing
The MolCLR framework's pre-training (which is being fine-tuned) employs graph-level augmentations to generate multiple views of molecules. These augmentations include **atom masking**, **bond deletion**, and **subgraph removal**.

## 4. Model Architecture

The modifications investigated include:

1.  **Optimiser Strategy (AdamW):** Replacing the conventional Adam optimiser with **AdamW**. This version decouples weight decay from the gradient update, providing a different method of regularisation.
2.  **Learning Rate Schedulers:** Implementing dynamic learning rate adjustment using **Cosine Annealing LR** and **Cosine Annealing with Warm Restarts** schedules.
3.  **Hyperparameter Tuning & Differential Learning:** Performing a 32-experiment grid search on key hyperparameters (batch size, learning rates, weight decay). This also includes testing **differential learning rates** (a lower rate for the pre-trained GNN base, a higher rate for the prediction head) and decoupled base weight decay.
4.  **Transformer-Based Readout:** Replacing the standard global pooling layer with a Transformer-based readout mechanism. This module appends a learnable `[CLS]` token to the node embeddings and processes the entire sequence through a Transformer encoder. The resulting embedding for the `[CLS]` token serves as the final graph-level representation.
5.  **Ensemble Model:** Utilising two distinct, pre-trained backbones: a **Graph Isomorphism Network (GIN)** and a **Graph Convolutional Network (GCN)**. The graph-level representations from both models are concatenated and passed to a final, dedicated ensemble prediction head.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
Performance is evaluated using metrics appropriate for the downstream tasks, primarily:
* **ROC-AUC** (for classification tasks)
* **MAE** (Mean Absolute Error, for regression tasks)
* **Validation Loss** and **Test Loss**]

### 5.2 Baseline Models
The baseline for all comparisons is the **well-tuned baseline MolCLR framework** itself. The experiments are designed to test if enhancements can improve upon this existing robust baseline, not to compare against entirely different model families.

### 5.3 Hardware/Software Requirements
Python 3.7 with torch==1.7.1+cu110

## 6. Implementation Plan

| Phase | Tasks | Duration |
|-------|-------|----------|
| Phase 1 | Literature Review (Starts: 8/11/2025) | 12 days |
| Phase 2 | Progress Evaluation (Starts: 8/20/2025) | 5 days |
| Phase 3 | Baseline Setup (Starts: 8/22/2025) | 12 days |
| Phase 4 | Enhancement Implementation (Starts: 9/1/2025) | 30 days |
| Phase 5 | Mid Evaluation - Short Paper (Starts: 9/15/2025) | 4 days |
| Phase 6 | Final Paper (Starts: 10/1/2025) | 6 days |

## 7. Risk Analysis

* **Risk: Marginal Performance Gains.** The enhancements implemented in this repository (e.g., AdamW, Transformer readouts, ensembles) may not yield significant or consistent performance improvements over the well-tuned baseline MolCLR framework.
    * **Mitigation:** The goal is to rigorously benchmark these changes. All results will be thoroughly documented. If performance gains are marginal, this will be clearly reported, as this is a valuable finding about the baseline's robustness.

* **Risk: Reproducibility Issues.** As with many complex deep learning models, there is a risk of difficulty in reproducing the baseline or enhancement results. This can be due to differences in software environments (PyTorch, RDKit versions), hardware (GPU), or minor implementation details.
    * **Mitigation:** Mitigation for this is provided by adhering as strictly as possible to the original MolCLR implementation for the baseline. Clear environment files (`environment.yml` or `requirements.txt`) are provided, along with pre-trained models and detailed scripts, to help users replicate the setup and results.

## 8. Expected Outcomes

* A fully functional and reproducible implementation of the baseline MolCLR model.
* Working, modular code for the tested enhancements, including:
    * AdamW optimizer integration.
    * A Transformer-based readout module.
    * A GIN/GCN ensemble model architecture.
* A clear set of benchmark results, presented in tables or logs, comparing the performance of each enhancement against the baseline on relevant MoleculeNet datasets.
* Pre-trained model weights for the reproduced baseline and all tested enhancement configurations, allowing for direct use and verification of results.

---
