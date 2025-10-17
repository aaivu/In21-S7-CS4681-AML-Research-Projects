# Research Proposal: Small LLMs:Mixture of Experts

**Student:** 210017V
**Research Area:** Small LLMs:Mixture of Experts
**Date:** 2025-09-01

## Abstract

Scaling deep learning models has demonstrated substantial performance gains in natural language processing tasks; however, it incurs high computational and memory costs. Mixture-of-Experts (MoE) architectures mitigate this by activating only a subset of experts per input. While Transformer-based MoEs have benefited from parameter-efficient adaptations such as LoRA and low-rank factorization, recurrent MoEs have been comparatively underexplored. This work investigates parameter-efficient techniques in LSTM + MoE models. We reproduce the original sparsely-gated MoE model and introduce modern efficiency strategies, including Switch-style gating, low-rank factorization, shared expert layers, and LoRA-based experts. A modular implementation is presented, with a framework for systematic evaluation of performance, stability, and efficiency.


## 1. Introduction

### 1.1 Background

Deep learning models have achieved remarkable performance gains by scaling the number of parameters. Architectures such as **LSTMs** and **Transformers** have set state-of-the-art results in tasks including language modeling, machine translation, and multimodal learning.  

However, this scaling comes with significant **computational and memory costs**. The **Mixture-of-Experts (MoE)** framework provides a promising solution by activating only a subset of experts for each input. This enables models to reach massive parameter counts **without a proportional increase in computation**.  

Early work by *Shazeer et al. (2017)* demonstrated the effectiveness of MoE layers in LSTM-based language models, achieving unprecedented model capacity while maintaining manageable computational budgets.


## 2. Problem Statement

Despite its success, the original **LSTM + MoE** framework suffers from **parameter redundancy and inefficiency**. Experts are large and largely independent, which leads to high memory usage and potential training instability.  

Recent advances in **parameter-efficient MoE approaches**—including Switch routing, low-rank decomposition, and LoRA-based experts—have significantly improved efficiency in Transformer architectures. However, these innovations **have not been systematically applied or evaluated in recurrent (LSTM) MoE models**, leaving an important research gap that this study aims to address.

## 2. Literature Review Summary

### 2.1 Scaling Challenges in AI Models

The rapid growth of deep learning models, such as BERT and GPT-4, has highlighted challenges in **training efficiency, memory usage, and deployment feasibility**. Dense architectures activate all parameters per input, leading to high computational costs. Sparse and modular designs, particularly the **Mixture-of-Experts (MoE)** paradigm, address this by activating only a fraction of parameters, enabling models to scale to **billions or trillions of parameters** without proportional compute overhead.


### 2.2 Early MoE Foundations

- **Jacobs et al.** introduced adaptive mixtures of local experts with a **gating mechanism**, enabling modular specialization.  
- **Deep Mixtures of Experts (DMoE, Eigen et al.)** stacked multiple expert and gating layers hierarchically, showing that lower layers capture local features while higher layers encode abstract semantics.  
- These early works established the potential for **expert specialization and modular sparse computation**.


### 2.3 Sparsely-Gated MoE Breakthrough

- **Shazeer et al. (2017)** introduced **Noisy Top-k gating**, activating only a few experts per input.  
- Achieved sparsity >99.99%, allowing neural networks with over 100 billion parameters while maintaining efficiency.  
- Experts naturally specialized in syntactic and semantic features, improving **language modeling and translation** performance.  
- Key challenges addressed: expert imbalance, shrinking batch problem, and distributed communication overhead.


### 2.4 Routing, Stability, and Balanced Assignment

- Ensuring **balanced expert utilization** is critical for efficiency and performance.  
- Auxiliary balancing losses help prevent expert collapse, but require careful tuning.  
- **BASE layers (Lewis et al.)** provide algorithmic guarantees for balanced assignment, improving stability and reducing communication overhead.


### 2.5 Scaling to Trillion-Parameter Models

- **Switch Transformer (Fedus et al.)** selects a single expert per input (k=1), reducing computation and communication while enabling trillion-parameter models.  
- **GShard (Lepikhin et al.)** uses distributed training and SPMD compilation to train 600-billion parameter models efficiently.  
- **Scaling laws (Ludziejewski et al.)** indicate efficiency depends on expert granularity and internal structure, emphasizing careful design at extreme scales.


### 2.6 Parameter-Efficient MoE and Fine-Tuning

- **LoRA (Low-Rank Adaptation)** and **TT-LoRA** allow fine-tuning of large pre-trained models with minimal trainable parameters.  
- MoE architectures incorporate similar **parameter-sharing techniques** (e.g., MPO decomposition, low-dimensional adapters) to reduce redundancy while preserving specialization.  
- These **PEFT-enhanced MoEs** achieve strong performance while updating only a fraction of parameters, ideal for resource-constrained environments.


### 2.7 Research Gaps

Despite extensive work in Transformer-based MoEs, several gaps remain for LSTM-based MoE systems:

1. **Expert Utilization and Balancing:** Efficient low-rank experts may suffer from imbalance, reducing their effectiveness.  
2. **Training Stability:** Sparse gating combined with parameter-efficient adaptations can worsen convergence issues in recurrent models.  
3. **Parameter Redundancy vs. Efficiency:** The optimal trade-off between compression and expressiveness in LSTM-MoEs is unexplored.  
4. **Generalization and Transferability:** The ability of efficient MoEs to transfer to downstream NLP tasks in recurrent architectures is unclear.  
5. **Extension Beyond Transformers:** Most modern efficiency innovations have been applied to Transformers, leaving LSTM + MoE underexplored.


### 2.8 Summary

Overall, the literature establishes **MoE as a scalable and efficient framework** for extreme model scaling. This project aims to bridge **modern efficiency techniques** (Switch routing, low-rank factorization, LoRA-based experts) with **LSTM-based MoEs**, addressing efficiency and stability challenges in recurrent architectures that have not yet been systematically studied.


## 4. Research Objectives

### Primary Objective

The primary goal of this research is to explore and evaluate **parameter-efficient techniques in LSTM + Mixture-of-Experts (MoE) architectures**, aiming to improve efficiency and stability while retaining predictive performance.

### Secondary Objectives

- Re-implement the **baseline LSTM + MoE model** as described by Shazeer et al. (2017).  
- Integrate **modern parameter-efficient strategies**, including low-rank factorization, shared expert layers, and Switch-style gating.  
- Compare the **performance, efficiency, and stability trade-offs** of these approaches.  
- Identify **best practices for designing lightweight MoE layers** in recurrent models.


## 5. Methodology

The methodology for this project is organized into four stages: **Baseline Reproduction, Parameter-Efficient Extensions, Evaluation, and Analysis**. Each stage is designed to systematically investigate parameter efficiency in LSTM + Mixture-of-Experts (MoE) architectures.


### 5.1 Baseline Reproduction: LSTM + MoE (Shazeer et al., 2017)

The first step is to faithfully reproduce the **Sparsely-Gated MoE layer** integrated with LSTM language models.

**Model Components:**

- **Experts:** Each expert is a feed-forward network consisting of two fully connected layers with ReLU activation.  
- **Gating Network:** A softmax-based router assigns each token to the top-k experts, with added noise to encourage balanced expert utilization.  
- **Load Balancing Loss:** An auxiliary term mitigates expert collapse, preventing a small subset of experts from dominating.  
- **LSTM Backbone:** MoE layers are inserted between LSTM layers to enable conditional computation.

**Dataset:**

- A benchmark language modeling dataset (e.g., **WikiText-103** or **One Billion Word Benchmark**) will be used to replicate conditions similar to the original work.

**Implementation:**

- The baseline model will be implemented in **PyTorch**, maintaining modularity to facilitate later modifications.


### 5.2 Parameter-Efficient MoE Extensions

To improve efficiency over the baseline, modern parameter-efficient strategies will be integrated and evaluated:

**(a) Switch-Style Gating (Fedus et al., 2021)**

- Replace noisy top-2 gating with **top-1 deterministic routing**, simplifying expert selection.  
- Reduces communication overhead and stabilizes training.

**(b) Low-Rank Factorization (Gao et al., 2022)**

- Factorize expert weight matrices into **low-rank components**.  
- Significantly reduces per-expert parameter counts (e.g., ×27.2 reduction with MPO decomposition).

**(c) Shared and Lightweight Experts (Bai et al., 2022)**

- Share parameters across multiple experts (partial weight tying).  
- Investigate “tiny experts” where only a fraction of full parameters are unique.

**(d) LoRA-Based MoE (Wu et al., 2024; Sun et al., 2025)**

- Replace full expert layers with **low-rank adapters (LoRA modules)**.  
- Multiple LoRA experts can be dynamically fused, allowing modular and efficient adaptation.  
- Enables updating **<1% of parameters** during fine-tuning.

**(e) Hybrid Designs**

- Combine multiple strategies (e.g., **Switch gating + LoRA experts**) to explore performance-efficiency trade-offs.  
- Test **TT-LoRA MoE** (Kunwar et al., 2025) for extreme efficiency using tensorized adapters.


This methodology ensures a **progressive and controlled exploration** from the baseline LSTM + MoE implementation to modern parameter-efficient variants, allowing systematic evaluation of trade-offs between **model performance, efficiency, and training stability**.


## 6. Expected Outcomes

By completing this research, we expect to achieve the following outcomes:

- **Hybrid LSTM + MoE Model:** Development of a parameter-efficient LSTM-based MoE model combining Switch-style gating and LoRA-based low-rank experts.  
- **Reduced Parameter Footprint:** Significant reduction in trainable parameters compared to the baseline, while maintaining comparable predictive performance.  
- **Improved Training Efficiency:** Lower epoch time and memory usage due to sparse routing and low-rank adaptations.  
- **Insights into Expert Utilization:** Analysis of how modern efficiency techniques affect expert specialization, load balancing, and stability in recurrent architectures.  
- **Guidelines for Lightweight MoEs:** Identification of best practices for designing small, efficient MoE layers in LSTM and other recurrent models.  
- **Reproducible Implementation:** A modular PyTorch implementation that can be adapted for further research or downstream NLP tasks.


## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-8  | Implementation |
| 9-12 | Experimentation |
| 13-15| Analysis and Writing |
| 16   | Final Submission |

## 8. Resources Required

To successfully implement and evaluate the proposed LSTM + MoE models, the following resources will be required:

### 8.1 Hardware

- **GPU:** NVIDIA GeForce RTX 4070 Ti SUPER (or equivalent) for efficient training of LSTM + MoE models.  
- **CPU:** 13th Gen Intel i7-13700 (24 cores) or equivalent for data preprocessing and model orchestration.  
- **Memory:** At least 32 GB RAM to handle batchified datasets and multiple experts.  
- **Storage:** Sufficient disk space (~100 GB) for datasets, checkpoints, and logs.

### 8.2 Software

- **Programming Language:** Python 3.10+  
- **Deep Learning Framework:** PyTorch (for model implementation, training, and evaluation)  
- **Supporting Libraries:**  
  - NumPy, pandas, PyArrow (data handling)  
  - Matplotlib / Seaborn (visualizations)  
  - Scikit-learn (evaluation metrics, optional)  
- **Version Control:** Git for code management and reproducibility

### 8.3 Datasets

- **Primary Dataset:** WikiText-2 for language modeling experiments  
- **Optional / Future Datasets:**  
  - WikiText-103 or One Billion Word Benchmark for large-scale replication  

### 8.4 Tools & Utilities

- **Experiment Tracking:** TensorBoard or Weights & Biases for logging loss, perplexity, and expert utilization  
- **Code Repository:** GitHub for version control and collaborative development  
- **Batch Processing Utilities:** Scripts for tokenization, vocabulary building, numericalization, and batching (as per prior data collection code)


These resources collectively enable **efficient training, evaluation, and analysis** of both baseline and hybrid parameter-efficient MoE models while ensuring reproducibility and scalability.


## References

[1] C. Jacobs, M. Jordan, S. Nowlan, and G. Hinton, “Adaptive mixtures of local experts,” Neural 
Computation, vol. 3, no. 1, pp. 79–87, 1991. 

[2] D. Eigen, M. Ranzato, and I. Sutskever, “Learning factored representations in a deep mixture of 
experts,” arXiv preprint arXiv:1312.4314, 2014. 

[3] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large 
neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017. 

[4] M. Lepikhin et al., “GShard: Scaling giant models with conditional computation and automatic 
sharding,” in Proc. ICLR, 2021. 

[5] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to trillion parameter models with 
simple and efficient sparsity,” arXiv preprint arXiv:2101.03961, 2021. 

[6] M. Lewis et al., “BASE Layers: Simplifying training of large, sparse models,” in Proc. ICML, 2021. 

[7] B. Ludziejewski, C. Dietrich, and W. Samek, “Scaling laws for mixture-of-experts,” arXiv preprint 
arXiv:2302.04676, 2023. 

[8] T. Mu and J. Lin, “A comprehensive survey of mixture-of-experts: Algorithms, theory and 
applications,” arXiv preprint arXiv:2302.00676, 2023. 

[9] A. Zadouri, R. Strudel, H. Zhang, and M. Elhoseiny, “Ultra-lean mixture of experts,” arXiv preprint 
arXiv:2310.13420, 2023. 

[10] M. Gao et al., “Parameter-efficient mixture-of-experts via matrix product operators,” in Advances in 
Neural Information Processing Systems (NeurIPS), 2022. 

[11] R. Kunwar et al., “Tensor-train low-rank adaptation for parameter-efficient fine-tuning,” arXiv 
preprint arXiv:2302.09095, 2023. 

[12] Z. Meng, S. Sun, and A. P. Parikh, “Parameter-efficient multi-task fine-tuning of pre-trained 
transformers,” in Proc. EMNLP, 2021. 

[13] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. Hinton, and J. Dean, “Outrageously 
Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” arXiv preprint 
arXiv:1701.06538, 2017.

---

**Submission Instructions:**
1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review
