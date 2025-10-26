
# Methodology: Small LLMs: Edge Computing

**Student:** 210554M  
**Research Area:** Small LLMs: Edge Computing  
**Date:** 2025-10-16

## 1. Overview

The methodology focuses on optimizing small language models (SLMs) for edge computing environments using quantization techniques. The workflow consists of benchmarking small-scale transformer models, applying post-training quantization (PTQ) and quantization-aware training (QAT), and fine-tuning with domain-specific disaster management data. The approach aims to balance latency, accuracy, and memory efficiency for on-device deployment.

## 2. Research Design

This research follows an experimental design combining comparative analysis and optimization. The process begins with establishing baseline performance for several small LLMs on general NLP tasks, followed by applying PTQ and QAT to evaluate their impact on performance. Finally, the quantized models are adapted and fine-tuned using a domain-specific disaster corpus to improve contextual understanding in real-world scenarios.

## 3. Data Collection

### 3.1 Data Sources

- **General Corpus:** WikiText-2 dataset for language modeling and quantization training.  
- **Domain-Specific Corpus:** Custom disaster-management corpus derived from:  
  - UN-OCHA Standard Operating Procedures (SOPs)  
  - National Disaster Management Centre (NDMC) Sri Lanka reports  
  - Red Cross recovery and compensation guidelines  
  - Verified humanitarian organization directives and field reports  

### 3.2 Data Description

The general corpus (WikiText-2) was used to evaluate linguistic performance, while the domain corpus focused on flood-related operational procedures and humanitarian response tasks. The disaster corpus contained approximately 550 text chunks categorized into six functional domains: Rescue & Evacuation, Medical & Health Care, Shelter & Housing, Food & Nutrition, Water Sanitation & Hygiene (WASH), and Waste Management.

### 3.3 Data Preprocessing

- Text normalization: lowercasing, Unicode normalization, and punctuation standardization.  
- Tokenization using the Qwen tokenizer to ensure compatibility.  
- Semantic chunking into 100–150 token units for coherence within model sequence limits.  
- Metadata annotation (document type, hazard type, agency, language).  

## 4. Model Architecture

The study employed **Qwen2.5–0.5B**, a 0.49B-parameter transformer-based model with 24 layers and a 32K token context window. This architecture offers a strong balance between efficiency and capability, making it suitable for quantization and edge deployment. Both PTQ and QAT were applied to this model:

- **PTQ (Post-Training Quantization):** Converted all linear layers to 8-bit precision using the BitsAndBytes library.  
- **QAT (Quantization-Aware Training):** Simulated quantization during forward passes, allowing gradients to adapt to quantized weight distributions.  

The quantization simulation followed Jacob et al. (2017), using affine quantization parameters (scale and zero-point) for both weights and activations.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Perplexity:** Evaluates language modeling capability.  
- **Latency (ms/token):** Measures inference efficiency.  
- **Accuracy (BoolQ):** Binary question-answering task accuracy.  
- **SQuAD F1 and EM Scores:** Assess extractive question-answering performance.  
- **Model Size (GB):** Measures storage efficiency before and after quantization.  

### 5.2 Baseline Models

The following models were benchmarked for comparison:  
- **Qwen2.5–0.5B**  
- **TinyLlama-1.1B**  
- **Phi-1.5**  
- **Gemma-3-1B-IT**  

Each was evaluated under consistent runtime settings using general benchmarks such as WikiText-2, BoolQ, and SQuAD.

### 5.3 Hardware/Software Requirements

- **Hardware:** GPU-enabled environment for training and benchmarking (CUDA-compatible).  
- **Software:**  
  - Hugging Face Transformers  
  - Evaluate library  
  - BitsAndBytes for quantization  
  - PyTorch for model fine-tuning and QAT implementation  

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Dataset collection and preprocessing | 2 weeks | Cleaned and tokenized datasets |
| Phase 2 | Baseline model benchmarking | 2 weeks | Baseline performance metrics |
| Phase 3 | PTQ and QAT implementation | 3 weeks | Quantized and fine-tuned models |
| Phase 4 | Domain adaptation and fine-tuning | 3 weeks | Domain-specific quantized models |
| Phase 5 | Evaluation and analysis | 2 weeks | Comparative performance report |

## 7. Risk Analysis

- **Data Scarcity:** Limited domain-specific datasets may affect model generalization.  
  *Mitigation:* Use data augmentation and incorporate multilingual sources.  
- **Quantization Error:** Precision loss during PTQ may degrade model performance.  
  *Mitigation:* Apply QAT to recover lost accuracy.  
- **Hardware Limitations:** Edge devices may underperform with large models.  
  *Mitigation:* Optimize further with mixed-precision quantization or pruning.  

## 8. Expected Outcomes

- Development of edge-compatible small language models capable of on-device inference.  
- Demonstrated reduction in model size by over 70% with minimal performance degradation.  
- Improved domain-specific understanding (e.g., disaster response) through fine-tuned quantized models.  
- Establishment of a reproducible benchmarking pipeline for evaluating quantized SLMs in constrained environments.

---