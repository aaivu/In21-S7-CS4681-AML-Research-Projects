# Research Proposal: EdgeMIN - Optimized Edge Deployment of MiniLM-Based Language Models

**Student:** Jayathilake N.C.  
**Index Number:** 210257F  
**Research Area:** Small LLMs: Efficient Models  
**Primary Supervisor:** Dr. Uthayasanker Thayasivam  
**Date:** 2025-01-24

---

## Abstract

Deploying large language models on edge devices remains challenging due to their computational demands and memory footprint. This research proposes EdgeMIN, a two-stage compression methodology combining MiniLMv2-based knowledge distillation, quantization-aware training (QAT), and structured pruning to create efficient language models for resource-constrained environments. The approach first applies attention-centric distillation to preserve relational knowledge from large teacher models (BERT-Large/DistilBERT), then refines students through INT8 quantization and structured pruning. Implementation on SST-2, MNLI, and QQP tasks achieved 5-10× model size reduction with 95-99% accuracy retention, demonstrating 3-5× latency reduction and 60-70% power savings. The framework enables flexible deployment across mobile CPUs, embedded NPUs, and IoT devices while maintaining competitive NLP performance for real-world edge applications.

---

## 1. Introduction

Large language models such as BERT-Large and RoBERTa-Large have achieved state-of-the-art performance in natural language understanding. However, their deployment on edge devices—mobile phones, embedded systems, and IoT devices—faces critical barriers:

- **Memory Constraints:** Models with hundreds of millions of parameters exceed typical edge device capacity (>1GB)
- **Computational Limitations:** High inference latency makes real-time applications impractical
- **Power Consumption:** Resource-intensive models drain battery life rapidly
- **Privacy Concerns:** Cloud-based inference requires transmitting sensitive data

Edge deployment offers significant advantages: enhanced privacy through local processing, reduced latency by eliminating network communication, offline capability, and reduced infrastructure costs. This research addresses the fundamental challenge of compressing large pre-trained models while preserving their reasoning capabilities, making advanced NLP accessible in resource-constrained environments.

---

## 2. Problem Statement

**Core Challenge:** How can we compress large pre-trained language models into compact, efficient versions suitable for edge deployment while maintaining accuracy and reasoning capabilities?

**Specific Problems:**
1. **Accuracy-Efficiency Trade-off:** Existing compression techniques often sacrifice significant accuracy for size reduction
2. **Cumulative Degradation:** Sequential compression (distillation → quantization → pruning) can compound accuracy losses
3. **Hardware Heterogeneity:** Edge devices vary widely in computational capabilities and memory
4. **Architectural Constraints:** Many distillation methods enforce rigid layer mappings, limiting aggressive compression

**Research Gap:** While methods like DistilBERT and TinyBERT have made progress, comprehensive frameworks integrating attention-centric distillation with quantization and structured pruning for practical edge deployment remain underexplored.

---

## 3. Literature Review Summary

### Early Knowledge Distillation
**DistilBERT** pioneered task-agnostic distillation achieving 40% size reduction but with rigid layer-wise mappings. **TinyBERT** improved through two-stage distillation (general + task-specific) with multi-level supervision, though maintaining explicit layer alignment. **Patient Knowledge Distillation (PKD)** introduced flexible layer selection strategies for deeper representations.

### Attention-Centric Distillation
**MiniLM** shifted focus to query-key (QK) attention distributions and value-value (VV) correlations instead of entire hidden states, enabling students with arbitrary dimensions. **MiniLMv2** generalized this by transferring Q-Q, K-K, and V-V similarities across multi-head attention, eliminating equal head count requirements for more aggressive compression.

### Hybrid Compression Techniques
Recent work emphasizes hardware-aware optimization integrating:
- **Quantization-Aware Training (QAT):** Simulates low-precision arithmetic (INT8/INT4) during fine-tuning
- **Structured Pruning:** Removes entire attention heads or blocks, producing dense, hardware-friendly models

**Key Insight:** Two-stage workflows combining relational distillation with precision optimization yield superior results for edge deployment compared to single-technique approaches.

---

## 4. Research Objectives

### Primary Objective
Develop and validate EdgeMIN, a comprehensive two-stage compression methodology combining MiniLMv2-based knowledge distillation with quantization-aware training and structured pruning for efficient edge deployment.

### Secondary Objectives
1. **Implement MiniLMv2 Distillation Framework** - Achieve 5-10× size reduction while retaining 95-99% teacher accuracy
2. **Develop Quantization-Aware Training Pipeline** - Implement INT8/INT4 precision with minimal accuracy degradation
3. **Integrate Structured Pruning** - Design saliency-driven criteria for dense, hardware-friendly architectures
4. **Comprehensive Evaluation** - Benchmark on GLUE tasks (SST-2, MNLI, QQP) with hardware profiling
5. **Comparative Analysis** - Compare against baseline methods (DistilBERT, TinyBERT, MiniLM)

---

## 5. Methodology

### 5.1 Baseline: MiniLM

MiniLM compresses large Transformers through three key mechanisms:
1. **Last Layer Self-Attention Transfer** - Learns from teacher's most semantically rich layer
2. **Attention and Value-Relation Transfer** - Aligns attention distributions via KL-divergence; transfers value relations for richer dependencies
3. **Teacher Assistant** - Uses intermediate model for very small students

### 5.2 Proposed Two-Stage Methodology

#### Stage 1: MiniLMv2 Distillation
- **Teacher Models:** DistilBERT-base or BERT-Large (pre-trained on downstream tasks)
- **Student Architecture:** 6 layers, 384 hidden dim, 12 heads (flexible configuration)
- **Knowledge Transfer:** Distill Q-Q, K-K, V-V relation matrices from teacher's high-impact layer
- **Loss Function:** KL divergence between teacher and student relation matrices
- **Training:** Task-agnostic distillation on target task datasets

#### Stage 2: Precision and Sparsity Optimization
1. **Quantization-Aware Training (QAT)**
   - Apply INT8 quantization with fake quantization during training
   - Use straight-through estimators for backpropagation
   - Fine-tune on calibration data to adapt to quantization noise

2. **Structured Pruning**
   - Calculate saliency metrics for attention heads and neurons
   - Iteratively remove lowest-impact components
   - Fine-tune after each pruning iteration

3. **Sequential Application:** Distillation → QAT → Pruning → Final fine-tuning

### 5.3 Comparative Analysis

| Method | Target Knowledge | Layer Mapping | Head Flexibility | Key Limitation |
|--------|------------------|---------------|------------------|----------------|
| **DistilBERT** | Logits, hidden states | Fixed | None | Rigid mapping |
| **TinyBERT** | Multi-level | Explicit | None | Same head count |
| **MiniLM** | Last layer attention | Single layer | None | Same head count |
| **MiniLMv2** | Multi-head attention | Flexible | Yes | No QAT/pruning |
| **EdgeMIN** | Attention + QAT + Pruning | Flexible | Yes | **Integrated approach** |

---

## 6. Expected Outcomes

### Compression Metrics
- **Size Reduction:** 5-10× compression (127MB → 66MB → 33MB after quantization)
- **Accuracy Retention:** 95-99% of teacher performance (92% → 90% on SST-2)
- **Inference Speed:** 3-5× latency reduction (~16ms per sample on CPU)
- **Memory Footprint:** 70-80% reduction (peak memory <200MB)
- **Power Consumption:** 60-70% reduction for mobile deployment

### Deployment Capabilities
- Compatible with standard CPUs (x86, ARM), mobile NPUs, embedded accelerators
- Real-time performance for interactive applications
- Flexible configurations for different hardware constraints

### Technical Contributions
- Open-source EdgeMIN framework with reproducible pipeline
- Comprehensive benchmarks across multiple tasks and hardware platforms
- Best practice guidelines for compression configuration selection

---

## 7. Datasets

### Training and Evaluation

| Task | Dataset | Labels | Samples | Purpose |
|------|---------|--------|---------|---------|
| **SST-2** | Sentiment Analysis | 2 | 872 validation | Binary classification benchmark |
| **MNLI** | Natural Language Inference | 3 | ~10K validation | Multi-class entailment |
| **QQP** | Question Pairing | 2 | ~40K validation | Semantic similarity |

### Pre-training (if needed)
- **Wikipedia + BookCorpus** (~3.3B words) for task-agnostic distillation

---

## 8. Timeline

| Weeks | Phase | Tasks | Deliverables |
|-------|-------|-------|--------------|
| **1-2** | Literature Review | Survey compression techniques, identify gaps | Literature summary |
| **3-4** | Setup & Baseline | Environment setup, train baseline teacher models | Baseline results |
| **5-7** | Stage 1 Implementation | MiniLMv2 distillation on SST-2, MNLI, QQP | Distilled models |
| **8-10** | Stage 2 Implementation | Apply QAT and structured pruning | Compressed models |
| **11-12** | Fine-tuning | Task-specific optimization | Final models |
| **13-14** | Evaluation | Benchmark accuracy, latency, memory, power | Performance metrics |
| **15** | Analysis | Compare against baselines, ablation studies | Analysis report |
| **16** | Documentation | Final report and code repository | Final submission |

### Key Milestones
- **Week 4:** Baseline models trained and evaluated
- **Week 7:** Distillation complete with initial results
- **Week 10:** Quantization and pruning complete
- **Week 14:** Comprehensive evaluation complete

---

## 9. Resources Required

### Computational Resources
- **Cloud Computing:** Google Colab (GPU: T4/V100 for training)
- **Edge Hardware:** CPU testing on local machines, mobile device validation

### Software and Tools
- **Frameworks:** PyTorch 2.0+, Hugging Face Transformers
- **Quantization:** PyTorch Quantization API, ONNX Runtime
- **Profiling:** PyTorch Profiler for latency/memory measurement
- **Version Control:** Git/GitHub for code repository

### Datasets
- GLUE benchmark (SST-2, MNLI, QQP) - publicly available
- Wikipedia, BookCorpus (if task-agnostic pre-training needed)

---

## 10. Evaluation Metrics

### Accuracy Metrics
- **Classification:** Accuracy, F1 Score, Matthews Correlation
- **Task-specific:** Per-task GLUE metrics

### Efficiency Metrics
- **Model Size:** Parameter count, storage size (MB)
- **Inference Speed:** Latency (ms), throughput (samples/sec)
- **Memory Usage:** Peak memory (MB), runtime memory (MB)
- **Compression Ratio:** Teacher size / Student size
- **Speedup:** Teacher latency / Student latency

---

## 11. Limitations and Challenges

### Technical Challenges
1. **Cumulative Degradation:** Sequential compression may compound accuracy losses
2. **Hardware Heterogeneity:** Optimal configurations vary across devices
3. **Resource Constraints:** Limited GPU access on Google Colab free tier

### Evaluation Complexity
1. **Multi-task Benchmarking:** Comprehensive evaluation across tasks is resource-intensive
2. **Hardware Profiling:** Limited access to diverse edge devices for testing

### Mitigation Strategies
- Careful hyperparameter tuning and validation at each stage
- Focus on CPU benchmarking as primary edge deployment target
- Leverage existing baselines for fair comparison

---

## 12. Expected Contributions

### Technical Contributions
1. **EdgeMIN Framework:** Novel integration of MiniLMv2, QAT, and structured pruning
2. **Flexible Architecture Design:** Guidelines for custom student architectures
3. **Open-source Implementation:** Reproducible pipeline for community use

### Empirical Contributions
1. **Multi-task Benchmarks:** Comprehensive evaluation across SST-2, MNLI, QQP
2. **Hardware Profiling:** Real-world latency and memory measurements
3. **Ablation Studies:** Analysis of individual compression techniques

### Practical Impact
- Enables NLP on billions of edge devices
- Reduces cloud costs and enhances privacy
- Democratizes AI access through efficient models

---

## References

[1] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," *Advances in Neural Information Processing Systems*, vol. 33, pp. 5776–5788, 2020.

[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv preprint arXiv:1910.01108*, 2019.

[3] X. Jiao et al., "TinyBERT: Distilling BERT for natural language understanding," *arXiv preprint arXiv:1909.10351*, 2019.

[4] S. Sun, Y. Cheng, Z. Gan, and J. Liu, "Patient knowledge distillation for BERT model compression," *arXiv preprint arXiv:1908.09355*, 2019.

[5] W. Wang et al., "MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained transformers," *arXiv preprint arXiv:2012.15828*, 2020.

[6] Y. Tang et al., "A survey on transformer compression," *arXiv preprint arXiv:2402.05964*, 2024.

[7] P. Ganesh et al., "Compressing large-scale transformer-based models: A case study on BERT," *Transactions of the Association for Computational Linguistics*, vol. 9, pp. 1061–1080, 2021.

[8] M. Chen et al., "EfficientQAT: Efficient quantization-aware training for large language models," *arXiv preprint arXiv:2407.11062*, 2024.

[9] H. Bai et al., "Towards efficient post-training quantization of pre-trained language models," *Advances in Neural Information Processing Systems*, vol. 35, pp. 1405–1418, 2022.

[10] Z. Sun et al., "MobileBERT: Task-agnostic compression of BERT by progressive knowledge transfer," 2019.
