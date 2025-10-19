
# Methodology: Biocomputing:DNA Computing

  

**Student:** 210483T

**Research Area:** Biocomputing:DNA Computing

**Date:** 2025-09-01

  

## 1. Overview

  

This research investigates efficient adaptation strategies for DNA foundation models to improve genomic disease prediction. Specifically, it compares parameter-efficient LoRA fine-tuning and linear probing using frozen embeddings of the 500M-parameter Nucleotide Transformer (NT-500M) model. The study leverages a KEGG-derived biological reasoning dataset, aiming to identify an approach that achieves high classification performance with lower computational cost.

  
  

## 2. Research Design
The research follows an experimental comparative design:
1. Extract embeddings from NT-500M for reference and variant DNA sequences.
2. Apply LoRA fine-tuning on the transformer model for classification.
3. Train an MLP classifier using frozen embeddings (linear probing).
4. Compare both methods on multiple evaluation metrics.
5. Analyze performance, computational efficiency, and robustness.

  

## 3. Data Collection

  

### 3.1 Data Sources

KEGG-derived variant–disease dataset (from Kyoto Encyclopedia of Genes and Genomes).

### 3.2 Data Description

- 1,159 training, 144 validation, and 144 test instances.
- 37 disease classes (e.g., Alzheimer’s, Parkinson’s, melanoma).
- Each instance includes:
	- Reference sequence (canonical DNA)
	- Variant sequence (mutation-bearing DNA)
	- Disease

  
  
  

## 4. Model Architecture

Two adaptation strategies are employed:

1. LoRA Fine-Tuning
	- Low-Rank Adaptation applied to attention layers of NT-500M.
	- Fine-tuning performed on variant and reference sequences separately.

  2. Linear Probing with MLP
		- NT-500M kept frozen; embeddings extracted.
		- Two-layer MLP classifier with ReLU activation, batch normalization, and dropout.
		- Concatenation of reference and variant embeddings improves performance.

  

## 5. Experimental Setup

  

### 5.1 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

### 5.2 Baseline Models

 1. LoRA fine-tuned NT-500M on:
	 - List item
	- Variant sequences
	- Reference sequences


2. Embedding-based MLP on:
	- Variant embeddings
	- Reference embedding
	- Concatenated embeddings

### 5.3 Hardware/Software Requirements

- Hardware: GPU (≥12GB VRAM) for fine-tuning; CPU for embedding extraction
- Software: Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), NumPy, Pandas, Scikit-learn

  

Platform: Linux or Windows, GitHub for version control

## 6. Implementation Plan

  

| Phase | Tasks | Duration | Deliverables |

|-------|-------|----------|--------------|

| Phase 1 | Data preprocessing | 2 weeks | Clean dataset |

| Phase 2 | Model implementation | 3 weeks | Working model |

| Phase 3 | Experiments | 2 weeks | Results |

| Phase 4 | Analysis | 1 week | Final report |

  

## 7. Risk Analysis

  
| Risk  |  Mitigation |
|--|--|
| GPU resource limitation  | Find GPU providers |

  

## 8. Expected Outcomes

  

 - Empirical comparison between LoRA fine-tuning and linear probing for
   genomic disease prediction.
   
-  A reproducible and computationally efficient methodology for DNA
   foundation model adaptation.

---

  
