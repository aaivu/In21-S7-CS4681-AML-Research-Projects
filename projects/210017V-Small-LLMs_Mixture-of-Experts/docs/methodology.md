# Methodology: Small LLMs:Mixture of Experts

**Student:** 210017V
**Research Area:** Small LLMs:Mixture of Experts
**Date:** 2025-09-01

## 1. Overview

This research extends the Mixture-of-Experts (MoE) framework for small Large Language Models (LLMs) by integrating two key innovations:  
1. **LoRA-based experts** for parameter-efficient fine-tuning, and  
2. **Switch-style deterministic routing** for reduced computational and communication overhead.  

The methodology builds upon the sparsely gated MoE architecture proposed by Shazeer et al. (2017), which activates only a subset of experts per input, enabling scalable capacity with efficient computation. Our proposed hybrid model preserves modularity while enhancing parameter efficiency and routing stability, enabling effective deployment of MoE structures in resource-constrained LLMs.

## 2. Research Design

The research follows an **experimental, comparative design**.  
Four configurations of the MoE architecture are implemented and evaluated:  

- **Baseline:** Standard top-2 noisy gating (Shazeer et al., 2017)  
- **Switch:** Deterministic top-1 routing (Fedus et al., 2022)  
- **LoRA:** Low-rank adaptation experts (Hu et al., 2021)  
- **Hybrid (Proposed):** Switch gating + LoRA experts  

Comparative experiments will evaluate model efficiency, performance, and scalability across these configurations.

## 3. Data Collection

### 3.1 Data Sources
The experiments utilize the **WikiText-2 Parquet dataset**, a widely used benchmark for language modeling. It provides a medium-scale, high-quality English text corpus suitable for evaluating small LLM architectures. The dataset includes separate training, validation, and test splits stored in Parquet format for efficient I/O performance.

### 3.2 Data Description
- **Training Data:** Sentences from Wikipedia articles used for model fitting.  
- **Validation Data:** Intermediate evaluation to tune hyperparameters.  
- **Test Data:** Final evaluation for generalization performance.  

Each record in the dataset consists of a text field representing a line of raw natural language, which is tokenized and numericalized for model training.

### 3.3 Data Preprocessing
The data preprocessing and loading pipeline was implemented in **PyTorch** with additional utilities in **Pandas** and **PyArrow**. The main components are:

- **Tokenization:**  
  Text lines are tokenized using a whitespace-based tokenizer, segmenting input into word-level tokens.

- **Vocabulary Construction:**  
  A vocabulary is built from the training corpus using a minimum frequency threshold. Special tokens such as `<pad>`, `<unk>`, and `<eos>` are reserved for padding, unknown words, and sentence termination.

- **Numericalization:**  
  Each tokenized line is converted into integer IDs based on the vocabulary mapping. Unknown tokens are replaced by the `<unk>` index.

- **Batchification:**  
  The flattened numeric token stream is reshaped into tensors of size `[batch_size, sequence_length]` for efficient mini-batch training. This allows sequential data processing compatible with the LSTM and MoE architectures.

- **Batch Sampling:**  
  A `get_batch()` function retrieves slices of token sequences and their next-token targets for next-word prediction tasks.

- **Dataset Loading:**  
  The loader function `load_wikitext2_parquet()` automatically reads the Parquet files, builds the vocabulary, performs numericalization, and returns ready-to-train tensors (`train_data`, `val_data`, `test_data`).

This modular design ensures reproducibility, efficient memory usage, and compatibility with distributed training setups.


## 4. Model Architecture

The proposed model builds upon the **sparsely gated Mixture-of-Experts (MoE)** framework introduced by *Shazeer et al. (2017)*, which scales model capacity by activating only a subset of experts for each input. The traditional MoE architecture comprises three main components:

1. **Gating Network** – selects which experts to activate for a given input.  
2. **Feedforward Experts** – independent neural modules that process their assigned tokens.  
3. **Sparse Dispatch–Combine Mechanism** – routes inputs to selected experts and merges their outputs efficiently.

Our work retains this modular structure but extends it in two significant directions to enhance efficiency and scalability:

1. **LoRA-based Experts:**  
   Each expert integrates *Low-Rank Adaptation (LoRA)* modules that freeze the main weight matrices while introducing lightweight, trainable low-rank matrices. This reduces redundant parameters while preserving expressive capacity.

2. **Switch-style Deterministic Routing:**  
   Instead of the standard noisy top-2 gating, we employ *top-1 deterministic routing* as introduced in the *Switch Transformer (Fedus et al., 2022)*, minimizing inter-expert communication and simplifying the routing process.

These two innovations create a **Hybrid MoE** architecture that combines the computational efficiency of sparse routing with the parameter savings of low-rank adaptation, enabling effective scaling for small LLMs.


### 4.1 Architectural Overview

The hybrid MoE framework is composed of the following interconnected modules:

- **Input Layer:**  
  The model receives token embeddings or LSTM hidden states as input. These representations form the basis for expert selection and processing.

- **Gating Network:**  
  Each input vector is projected through a learned matrix to produce logits representing affinities to experts.  
  - *Baseline Mode:* Noisy top-2 gating to encourage balanced usage.  
  - *Switch Mode:* Deterministic top-1 routing for efficiency.  
  - *Hybrid Mode (Proposed):* Combines deterministic top-1 routing with LoRA-based experts for both adaptability and speed.

- **Sparse Dispatcher:**  
  This component routes the selected tokens to their designated experts based on the gating outputs. Only the chosen experts are active per input, maintaining computational sparsity.

- **Expert Networks:**  
  Each expert independently processes its inputs:
  - **Standard MLP Experts (Baseline):** Two-layer feedforward networks with ReLU activation.  
  - **LoRA-based Experts (Proposed):** Experts that freeze the base projection matrix and add low-rank matrices.  

  This design significantly reduces the number of trainable parameters while maintaining high representational power.

- **Aggregation and Combination:**  
  Outputs from the active experts are recombined through a gate-weighted sum. This ensures differentiable computation while allowing gradients to flow efficiently through both gating and expert parameters.

- **Output and Loss Functions:**  
  The combined outputs are passed through an output projection layer to produce task-specific predictions.  
  To prevent expert imbalance, we include an **auxiliary load-balancing loss** that penalizes uneven routing using the coefficient of variation across experts.


### 4.2 Forward Pass and Training Flow

1. **Input Encoding:** Tokens or hidden states are passed into the gating network.  
2. **Expert Selection:** Gating network computes logits and selects top experts (1 or 2 depending on mode).  
3. **Sparse Dispatching:** Selected tokens are routed to the corresponding experts.  
4. **Expert Processing:** Each active expert performs independent feedforward computation.  
5. **Output Combination:** Expert outputs are merged via gate-weighted aggregation.  
6. **Loss Computation:** Task loss and auxiliary balancing loss are computed.  
7. **Backpropagation:** Gradients flow through both gating and expert layers, maintaining end-to-end differentiability.

This stepwise process allows experts to specialize in different linguistic or contextual patterns while maintaining efficient parallel computation.


### 4.3 Integration with LSTM Backbone

The hybrid MoE can be integrated into sequential architectures such as **LSTMs**. In this setup, the MoE layer replaces or augments the LSTM’s projection layer. The LSTM handles temporal dependencies, while the MoE introduces conditional computation—enabling experts to specialize in specific contexts or structures within sequences. This design effectively combines the temporal modeling strength of LSTMs with the scalability of MoE systems.


### 4.4 Architectural Illustration

The architecture is visually summarized below:

![Hybrid Mixture-of-Experts Architecture](hybrid_moe_architecture.png)

*Figure 1: The Hybrid Mixture-of-Experts (MoE) architecture integrates deterministic Switch gating and LoRA-based low-rank experts. Blue modules represent components adapted from Shazeer et al. (2017), while orange modules denote newly introduced innovations.*


### 4.5 Key Advantages

- **Parameter Efficiency:** LoRA experts drastically reduce trainable parameters.  
- **Routing Efficiency:** Deterministic Switch gating minimizes communication overhead.  
- **Scalability:** Sparse dispatching supports scaling without linear cost increase.  
- **Training Stability:** Load-balancing regularization prevents expert collapse.  
- **Flexibility:** Architecture supports Baseline, Switch, LoRA, and Hybrid operational modes for comprehensive analysis.


In summary, the **Hybrid Mixture-of-Experts (MoE)** architecture retains the modular, sparse design of the original framework while incorporating deterministic routing and low-rank expert adaptation. These enhancements deliver superior parameter efficiency and reduced computational cost, making the approach ideal for small LLMs that aim to achieve large-model performance under limited resources.
 


## 5. Experimental Setup

For evaluation, we implemented a **stacked LSTM language model** inspired by Shazeer et al. (2017), which serves as the base sequential model. The LSTM optionally integrates **Mixture-of-Experts (MoE) layers** between its hidden states and the output projection to evaluate the effect of conditional computation.

### 5.1 MoE Configurations Evaluated

Two MoE variants were tested:

1. **Baseline MoE:**  
   - Uses *top-2 noisy gating*, where each token is routed to the two most relevant experts.  
   - Matches the original design of Shazeer et al. (2017).

2. **Hybrid MoE (Proposed):**  
   - Combines *Switch routing* (top-1 deterministic selection) with **LoRA-based low-rank adapters** (rank = 8).  
   - Reduces parameter redundancy while retaining model capacity.  

These two configurations allow direct comparison of traditional noisy gating versus efficient low-rank adaptation with deterministic routing.


### 5.2 Dataset and Preprocessing

- **Dataset:** WikiText-2  
- **Vocabulary Size:** 76,619 tokens  
- **Training Setup:**  
  - Learning rate: 0.0001  
  - Auxiliary MoE loss coefficient: 0.01, to encourage balanced expert utilization  
- **Number of Epochs:** 8, selected to maintain comparable compute budgets across configurations  

WikiText-2 was chosen because it is a widely adopted benchmark for recurrent and Transformer-based language models, allowing comparison with prior MoE studies (Shazeer et al., 2017; Fedus et al., 2022).


### 5.3 Evaluation Metrics

Evaluation focuses on **language modeling quality** and **training efficiency**:

- **Perplexity (PPL):** Measures the predictive accuracy of the model on validation and test data.  
- **Training and Validation Loss:** Average cross-entropy per token.  
- **Epoch Time / Efficiency Metrics:** Normalized training time per million tokens to compare computational cost across configurations.  

These metrics jointly capture the trade-off between **model performance** and **computational efficiency**.


### 5.4 Hardware and Software Setup

Experiments were conducted on the following hardware:

- **CPU:** Intel 13th Gen i7-13700 (24 cores) @ 5.1 GHz  
- **GPU:** NVIDIA GeForce RTX 4070 Ti SUPER  
- **RAM:** 32 GB (5.3 GB used during training)  
- **OS:** Ubuntu 24.04.3 LTS x86_64  
- **Shell:** Bash 5.2.21  
- **Resolution:** 1920x1080  

Software environment:

- **Python** (Anaconda base environment)  
- **PyTorch** (with CUDA support)  
- **Additional Libraries:** Pandas, PyArrow, and other standard scientific Python packages

This configuration ensures efficient training of small LLMs with MoE layers while supporting GPU-accelerated computation and sparse routing operations.



## 6. Implementation Plan

| Phase   | Tasks                                                                                  | Duration | Deliverables                           |
|---------|----------------------------------------------------------------------------------------|----------|----------------------------------------|
| Phase 1 | Data preprocessing: tokenization, vocabulary building, numericalization, batchification | 2 weeks  | Clean, ready-to-train datasets         |
| Phase 2 | Model implementation: LSTM backbone, MoE layers, LoRA integration, gating mechanisms   | 3 weeks  | Fully functional Hybrid MoE model      |
| Phase 3 | Experiments: training Baseline and Hybrid MoE configurations, evaluation on WikiText-2 | 2 weeks  | Comparative results, metrics reports   |
| Phase 4 | Analysis: interpret results, visualize expert utilization, evaluate efficiency vs. performance | 1 week   | Final analysis report with figures     |


## 7. Risk Analysis

**Potential Risks and Mitigation Strategies:**

1. **Expert Imbalance / Collapse**  
   - *Risk:* Certain experts dominate routing while others remain underutilized.  
   - *Mitigation:* Use auxiliary load-balancing loss (CV^2 of importance and load), monitor expert utilization during training.

2. **Overfitting**  
   - *Risk:* Low-rank experts may overfit small datasets or specific contexts.  
   - *Mitigation:* Apply dropout, weight decay, and early stopping based on validation loss.

3. **Compute Constraints**  
   - *Risk:* Sparse dispatching and multiple experts may exceed GPU memory or increase epoch time.  
   - *Mitigation:* Optimize batch sizes, use mixed-precision training, and leverage gradient accumulation.

4. **Training Instability**  
   - *Risk:* Sparse routing and LoRA adaptation can create unstable gradient flow.  
   - *Mitigation:* Implement gradient clipping, monitor loss trends, and validate on small subsets before full-scale runs.


## 8. Expected Outcomes

By completing this study, we expect to achieve the following:

- A **Hybrid Mixture-of-Experts model** that combines **Switch routing** with **LoRA-based low-rank experts**, optimized for small LLMs.  
- **Reduced parameter count** without significant loss in predictive performance compared to Baseline MoE.  
- **Improved training efficiency**, measured as lower epoch time per million tokens, while maintaining comparable perplexity.  
- **Demonstration of modular scalability**, enabling easy switching between Baseline, Switch, LoRA, and Hybrid configurations for research and experimentation.  
- **Visualizations of expert specialization** and utilization patterns, supporting insights into how sparse routing and low-rank adaptation interact in small LLMs.  
- A reproducible **PyTorch implementation** that can serve as a baseline for future research in parameter-efficient MoE architectures.

---

**Note:** Update this document as your methodology evolves during implementation, especially after initial experiments or adjustments to LoRA rank, gating strategy, or hyperparameters.
