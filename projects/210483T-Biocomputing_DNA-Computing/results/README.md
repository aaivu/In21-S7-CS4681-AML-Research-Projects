#  Experimental Results: Adapting Nucleotide Transformers for Genomic Disease Prediction

## Overview

This study evaluates **two parameter adaptation strategies** for the 500M-parameter Nucleotide Transformer model (NT-500M):

1. **LoRA Fine-Tuning** — Low-rank adaptation of attention layers  
2. **Linear Probing** — MLP classifier on top of frozen embeddings

We use a **variant–disease reasoning dataset** derived from KEGG pathways with 37 disease classes.

---

## 📊 Dataset Summary

| Split        | Samples | Classes | 
|--------------|---------|---------|
| Train        | 1,159   | 37      | 
| Validation   | 144     | 27      | 
| Test         | 146     | 29      | 

---

## Experimental Setup

- **Base Model:** Nucleotide Transformer (NT-500M)
- **LoRA Rank:** 8  
- **Optimizer:** AdamW (LR=5e-5)  
- **Linear Probing:** 2-layer MLP with ReLU, dropout, and batch normalization
- **Metrics:** Accuracy, Precision, Recall, F1-score

---

##  Key Results

| Model                              | Input                     | Accuracy | F1-score | Precision | Recall |
|-------------------------------------|----------------------------|----------|----------|-----------|--------|
| LoRA Fine-Tuning                   | Variant                    | 84.03%   | 70.39%   | 73.37%    | 69.55% |
| LoRA Fine-Tuning                   | Reference                  | 82.64%   | 68.71%   | 70.25%    | 67.59% |
| Linear Probing (MLP)               | Variant embeddings         | 87.50%   | 72.99%   | 75.45%    | 72.43% |
| Linear Probing (MLP)               | Reference embeddings       | 88.89%   | 76.18%   | 78.22%    | 75.67% |
| **Linear Probing (MLP)**         | Variant + Reference concat | **91.78%** | **78.68%** | **80.33%** | **78.23%** |

 **Linear probing with concatenated embeddings outperformed LoRA fine-tuning in all metrics**, while being more computationally efficient.

---

##  Performance Insights

- LoRA fine-tuning provided competitive but lower accuracy compared to linear probing.  
- Concatenating variant and reference embeddings captures richer biological context.  
- Linear probing requires significantly less compute, making it ideal for scalable applications.


