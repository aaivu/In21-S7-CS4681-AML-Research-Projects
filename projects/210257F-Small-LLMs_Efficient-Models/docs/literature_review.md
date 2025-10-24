# Literature Review: EdgeMIN - Efficient Transformer Compression for Edge Deployment

**Student:** Jayathilake N.C.  
**Index Number:** 210257F  
**Research Area:** Small LLMs: Efficient Models  
**Date:** 2025-08-15

---

## Abstract

This literature review examines state-of-the-art model compression techniques for transformer-based language models, focusing on knowledge distillation, structured pruning, and quantization methods. The review covers foundational approaches (DistilBERT, TinyBERT) through recent advances (MiniLMv2, MLKD-BERT), analyzing their contributions to edge deployment feasibility. Key findings indicate that attention-centric distillation combined with hardware-aware compression techniques (structured pruning, INT8 quantization) offers the most promising path for aggressive model compression while maintaining acceptable accuracy. The review identifies a critical gap: while individual techniques are well-studied, systematic integration of multiple compression methods with validated hardware profiling remains underexplored, motivating the EdgeMIN pipeline approach.

---

## 1. Introduction

The deployment of large pre-trained language models (PLMs) such as BERT [1] and RoBERTa [2] on resource-constrained edge devices presents a fundamental challenge in natural language processing. These models, containing hundreds of millions of parameters and requiring gigabytes of memory, exceed the computational and storage capabilities of mobile phones, embedded systems, IoT sensors, and edge processors. This "deployment gap" prevents real-time inference in privacy-sensitive applications (on-device virtual assistants), latency-critical scenarios (industrial monitoring), and offline environments (limited connectivity regions).

Model compression techniques—primarily knowledge distillation, pruning, and quantization—have emerged as critical enablers for edge deployment. This review synthesizes research across these domains, focusing on transformer architectures and their suitability for resource-constrained environments. The scope encompasses distillation methods (output-level, feature-level, relation-level), pruning strategies (unstructured vs. structured), and quantization approaches (post-training vs. quantization-aware training).

---

## 2. Search Methodology

### Search Terms Used
- **Primary:** "transformer compression", "knowledge distillation BERT", "model quantization", "structured pruning", "edge deployment NLP"
- **Secondary:** "MiniLM", "DistilBERT", "TinyBERT", "attention head pruning", "INT8 quantization", "mobile NLP"
- **Combined:** "multi-stage compression", "edge inference", "efficient transformers"

### Databases Searched
- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] NeurIPS/ICLR/ACL Proceedings

### Time Period
Primary focus: 2019-2025 (recent developments)  
Foundational work: 2015-2018 (original distillation, early BERT compression)

---

## 3. Key Areas of Research

### 3.1 Knowledge Distillation for Transformers

Knowledge distillation (KD) trains a compact "student" model to mimic a larger "teacher" model [3], transferring knowledge through various supervision signals.

#### Output-Level Distillation

**DistilBERT** [4] pioneered task-agnostic distillation for transformers, combining soft-target matching (KL divergence between teacher/student logits), masked language modeling loss, and cosine distance between intermediate hidden states. It achieved 40% size reduction while retaining 97% of BERT-base performance, demonstrating feasibility of aggressive compression. However, fixed layer-to-layer mapping constrains architectural flexibility.

**Key Papers:**
- Sanh et al. (2019) - Introduced DistilBERT, establishing triple-loss objective for transformer distillation.

#### Feature-Level Distillation

**TinyBERT** [5] introduced multi-layer supervision across embeddings, attention matrices, hidden states, and logits through a two-stage framework: general distillation (pre-training) and task-specific distillation (fine-tuning with augmented data). It achieved 7.5× compression with minimal accuracy loss, demonstrating that intermediate layer supervision improves compression effectiveness. However, it requires explicit layer-to-layer mapping and identical attention head counts.

**Patient Knowledge Distillation (PKD)** [6] addressed rigidity by introducing flexible layer selection strategies (PKD-Last: supervise from teacher's last k layers; PKD-Skip: skip early layers for deeper representations). It improved training efficiency by avoiding overfitting to early-stage patterns, but still relies on hidden-state mimicry.

**Key Papers:**
- Jiao et al. (2019) - Developed TinyBERT with two-stage multi-level distillation.
- Sun et al. (2019) - Proposed PKD with patient layer selection strategies.

#### Relation-Level Distillation (Attention-Centric)

**MiniLM** [7] introduced a paradigm shift by distilling self-attention relations rather than absolute hidden states. It transfers query-key (Q-K) attention distributions via KL divergence and value-value (V-V) correlations, focusing on the last transformer layer (most semantically rich). The key advantage is architectural flexibility—students can have arbitrary hidden dimensions and head configurations.

**MiniLMv2** [8] generalized this approach by distilling Q-Q, K-K, V-V pairwise similarity matrices (not just Q-K and V-V), removing the constraint of equal head counts between teacher and student. This enables transfer from multi-head attention across different architectures, offering the greatest architectural freedom among distillation methods.

**MLKD-BERT** [9] combines feature-level and relation-level supervision, transferring both localized representations (hidden states) and global interaction patterns (attention relations). It explores multi-head mapping strategies (many teacher heads → one student head), achieving robustness under severe parameter reduction.

**Preference-Based Distillation (PLaD)** [10] preserves reasoning capabilities using pseudo-preference pairs to align student with teacher's instruction-following behavior, particularly relevant for conversational AI.

**Key Papers:**
- Wang et al. (2020) - Introduced MiniLM with attention relation distillation.
- Wang et al. (2020) - Extended to MiniLMv2 with multi-head self-attention relations.
- Zhang et al. (2024) - Developed MLKD-BERT combining multiple supervision levels.

### 3.2 Structured Pruning

Pruning removes less important parameters to reduce model size and computation. Structured pruning removes entire groups (attention heads, FFN neurons, layers), producing dense models compatible with standard CPUs and NPUs, unlike unstructured pruning which requires specialized sparse kernels.

**"Are Sixteen Heads Really Better Than One?"** [11] demonstrated that many attention heads are redundant and can be pruned with minimal impact. Magnitude-based importance scoring (`I_h = ||∇_h L_task||_2`, gradient magnitude w.r.t. head output) effectively identifies prunable heads. This validated that transformers exhibit significant redundancy.

**ROSITA** [12] refined BERT compression through integrated techniques, combining knowledge distillation with structured pruning and augmenting training data (up to 8M+ samples). It produces hardware-friendly dense models exploiting SIMD parallelism.

**Key Papers:**
- Michel et al. (2019) - Questioned necessity of sixteen heads, enabling structured pruning.
- Liu et al. (2021) - Developed ROSITA with integrated compression techniques.

### 3.3 Quantization

Quantization reduces numerical precision of weights and activations, drastically reducing memory footprint (up to 4× for INT8) and potentially accelerating computation on hardware with native low-precision support [13].

#### Post-Training Quantization (PTQ)

PTQ applies quantization after training without retraining. **Dynamic PTQ** quantizes weights offline while activations are quantized on-the-fly during inference. **Static PTQ** uses calibration datasets to determine activation statistics, processing both weights and activations using integer arithmetic.

**"Towards Efficient Post-Training Quantization"** [14] demonstrated that dynamic PTQ achieves significant memory savings with minimal accuracy impact for transformers, though transformer embeddings and layer norms are sensitive to quantization. Static PTQ offers greater speedups but requires calibration data.

#### Quantization-Aware Training (QAT)

QAT simulates quantization during fine-tuning by inserting "fake quantization" nodes (forward: `w_quant = round(w/scale) * scale`; backward: straight-through estimator). **EfficientQAT** [15] showed QAT typically achieves higher accuracy than PTQ, especially at INT4 or mixed-precision, but requires full retraining.

**Key Papers:**
- Jacob et al. (2018) - Established integer-arithmetic-only inference via quantization.
- Bai et al. (2022) - Analyzed efficient post-training quantization for PLMs.
- Chen et al. (2024) - Developed EfficientQAT for large language models.

### 3.4 Combined Approaches

Combining techniques targets different aspects of model redundancy for maximal compression.

**MobileBERT** [16] employed progressive knowledge transfer with multi-stage distillation using intermediate teacher assistants, combined with inverted bottleneck structures for mobile deployment, achieving task-agnostic compression.

**CompressBERT** explored various combinations, finding that sequential application (KD → Pruning → Quantization) yields complementary benefits, with order mattering significantly.

**Key Papers:**
- Sun et al. (2019) - Developed MobileBERT with progressive transfer for edge devices.

---

## 4. Research Gaps and Opportunities

### Gap 1: Systematic Integration of Multiple Compression Techniques

**Why it matters:** While individual techniques are well-studied, systematic integration—particularly the order of application and stage-wise contributions—remains underexplored.

**How EdgeMIN addresses it:** Provides a clear sequential pipeline (MiniLMv2 Distillation → Structured Head Pruning → Aggressive PTQ) with detailed ablation studies.

### Gap 2: Hardware-Agnostic Evaluation

**Why it matters:** Many studies evaluate on specialized hardware or report only theoretical metrics, limiting reproducibility.

**How EdgeMIN addresses it:** Focuses on metrics measurable without specialized hardware—actual file size, parameter count, CPU latency, estimated FLOPs.

### Gap 3: Practical Deployment Constraints

**Why it matters:** Academic studies often assume unlimited resources during training and inference.

**How EdgeMIN addresses it:** Demonstrates effective compression using DistilBERT teacher, 10% training data subsets, and post-training quantization.

---

## 5. Theoretical Framework

EdgeMIN's foundation rests on three complementary compression principles:

**Knowledge Transfer via Relational Distillation:** Self-attention relations capture more essential transformer behavior than absolute hidden-state values. Formalized as minimizing KL divergence between teacher-student relation matrices: `L_distill = Σ KL(R^(i)_T || R^(i)_S)`.

**Structured Redundancy Removal:** Transformers exhibit significant redundancy; removing low-importance structures reduces computation without critical loss. Importance score: `I_h = ||∇_(a_h) L_task||_2`.

**Precision-Computation Trade-off:** Reducing numerical precision (FP32 → INT8) drastically reduces memory with minimal accuracy impact. Per-tensor quantization: `w_quant = clamp(round(w/scale + zero_point), q_min, q_max)`.

**Sequential Order Justification:** Distill → Prune → Quantize ensures best small baseline, reduces complexity before quantization, and applies precision reduction to compacted model.

---

## 6. Methodology Insights

Common experimental practices include fine-tuning pretrained models on GLUE benchmarks (SST-2, MNLI, QQP), evaluating with accuracy/F1 metrics, using AdamW optimization with warmup, and measuring size, parameters, FLOPs, and latency.

Most promising approaches for maximum compression include MiniLMv2 distillation, structured head pruning (20-30% removal), and aggressive INT8 quantization. For minimal accuracy loss: multi-level distillation, QAT, and conservative pruning ratios. For edge deployment: post-training quantization (simplicity), structured pruning (hardware compatibility), and task-agnostic distillation (generalization).

---

## 7. Conclusion

This literature review synthesized research across knowledge distillation, structured pruning, and quantization for transformer compression. Key findings: (1) Attention-centric distillation (MiniLMv2) offers superior flexibility, (2) Structured pruning is essential for hardware compatibility, (3) Post-training quantization achieves significant memory savings with minimal overhead, (4) Sequential integration yields complementary benefits, (5) Hardware-agnostic evaluation provides reproducible validation.

The identified gaps—particularly lack of systematic multi-stage pipelines with validated CPU measurements—motivate the EdgeMIN methodology. Future research should prioritize on-device profiling on actual edge hardware (ARM CPUs, mobile NPUs, microcontrollers).

---

## References

[1] J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers for language understanding," *NAACL*, 2019.

[2] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.

[3] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv:1503.02531*, 2015.

[4] V. Sanh et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," *arXiv:1910.01108*, 2019.

[5] X. Jiao et al., "TinyBERT: Distilling BERT for natural language understanding," *arXiv:1909.10351*, 2019.

[6] S. Sun et al., "Patient knowledge distillation for BERT model compression," *arXiv:1908.09355*, 2019.

[7] W. Wang et al., "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," *NeurIPS*, vol. 33, pp. 5776–5788, 2020.

[8] W. Wang et al., "MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained transformers," *arXiv:2012.15828*, 2020.

[9] Y. Zhang, Z. Yang, and S. Ji, "MLKD-BERT: Multi-level knowledge distillation for pre-trained language models," *arXiv:2407.02775*, 2024.

[10] R. Zhang et al., "PLaD: Preference-based large language model distillation with pseudo-preference pairs," *arXiv:2406.02886*, 2024.

[11] P. Michel, O. Levy, and G. Neubig, "Are sixteen heads really better than one?" *NeurIPS*, vol. 32, 2019.

[12] Y. Liu, Z. Lin, and F. Yuan, "ROSITA: Refined BERT compression with integrated techniques," *AAAI*, vol. 35, no. 10, pp. 8715–8722, 2021.

[13] B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," *CVPR*, 2018.

[14] H. Bai et al., "Towards efficient post-training quantization of pre-trained language models," *NeurIPS*, vol. 35, pp. 1405–1418, 2022.

[15] M. Chen et al., "EfficientQAT: Efficient quantization-aware training for large language models," *arXiv:2407.11062*, 2024.

[16] Z. Sun et al., "MobileBERT: Task-agnostic compression of BERT by progressive knowledge transfer," 2019.

[17] A. Wang et al., "GLUE: A multi-task benchmark and analysis platform for natural language understanding," *arXiv:1804.07461*, 2018.
