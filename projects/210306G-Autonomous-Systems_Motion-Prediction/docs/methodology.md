# Methodology: Exploring Fine-Tuning Enhancements for MolCLR

**Student:** 210306G
**Research Area:** Medical Domain:Drug Discovery
**Date:** 2025-10-23

## 1. Overview

[In this work, we systematically investigate the impact of common enhancements on the MolCLR fine-tuning process. We explore advanced optimisation strategies (e.g., AdamW optimiser, cosine annealing schedules) and architectural modifications (e.g., Transformer-based readout, GIN/GCN ensembles). Through systematic evaluation, we find these modifications do not yield significant performance gains over the well-tuned baseline, highlighting the robustness of the pre-trained representations.]

## 2. Research Design

[This paper presents a systematic investigation into the practical impact of common deep learning enhancements on the MolCLR framework. The objective is to quantify the impact of these strategies and determine their efficacy in the context of fine-tuning pre-trained molecular models. The approach involves applying various optimisation and architectural changes and then conducting extensive evaluations on MoleculeNet benchmarks to compare performance against the robust MolCLR baseline.]

## 3. Data Collection

### 3.1 Data Sources
[The models are fine-tuned and evaluated on established **MoleculeNet benchmarks**. A 32-experiment hyperparameter search was conducted specifically on the **BBBP dataset**. The pre-training phase leveraged large-scale unlabelled molecular databases.]

### 3.2 Data Description
[The data consists of **molecular graphs**, with atoms as nodes and bonds as edges. The downstream tasks include **classification** (e.g., chemical toxicity like BBBP, Tox21) and **regression** (e.g., solubility or free energy prediction).]

### 3.3 Data Preprocessing
[The MolCLR framework's pre-training (which is being fine-tuned) employs graph-level augmentations to generate multiple views of molecules. These augmentations include **atom masking**, **bond deletion**, and **subgraph removal**.]

## 4. Model Architecture

[The baseline model is the **MolCLR framework**, which uses a GIN-based Graph Neural Network (GNN) encoder and contrastive learning, with a standard global pooling layer (e.g., global mean/add) for graph representation.
The architectural modifications investigated include:
1.  **Transformer-Based Readout:** Replacing the standard global pooling layer with a Transformer-based readout mechanism that uses a `[CLS]` token to capture global interactions.
2.  **Ensemble Model:** Utilising two distinct, pre-trained backbones (GIN and GCN), concatenating their representations, and passing them to a final ensemble prediction head.]

## 5. Experimental Setup

### 5.1 Evaluation Metrics
[Performance is evaluated using metrics appropriate for the downstream tasks, primarily:
* **ROC-AUC** (for classification tasks)
* **MAE** (Mean Absolute Error, for regression tasks)
* **Validation Loss** and **Test Loss**]

### 5.2 Baseline Models
[The baseline for all comparisons is the **well-tuned baseline MolCLR framework** itself. The experiments are designed to test if enhancements can improve upon this existing robust baseline, not to compare against entirely different model families.]

### 5.3 Hardware/Software Requirements
[No information available in the provided document.]

## 6. Implementation Plan

[This document describes completed research, not a future implementation plan. The "plan" (grid search, model modifications) has already been executed.]

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean dataset |
| Phase 2 | Model implementation | 3 weeks | Working model |
| Phase 3 | Experiments | 2 weeks | Results |
| Phase 4 | Analysis | 1 week | Final report |

## 7. Risk Analysis

[This document presents a retrospective analysis of completed work. Risks identified through experimentation (e.g., potential for overfitting with the Transformer readout or ensemble models on smaller datasets) are discussed as results, not as pre-implementation risks.]

## 8. Expected Outcomes

[The actual outcomes (rather than expected) demonstrated that none of the tested modifications yielded significant or consistent performance improvements over the baseline. Key findings include:
1.  The pre-trained MolCLR framework is **exceptionally robust** and its representations are stable.
2.  Substantial performance gains are not easily achieved through standard optimisation or architectural adjustments at the fine-tuning stage.
3.  Future work should focus on more fundamental modifications, such as novel data augmentations, adjustments to the pre-training objective, or using larger pre-training datasets.]

---
