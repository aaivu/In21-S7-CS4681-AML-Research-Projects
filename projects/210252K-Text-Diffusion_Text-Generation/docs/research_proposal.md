# Research Proposal: Text Diffusion:Text Generation

**Student:** 210252K
**Research Area:** Text Diffusion:Text Generation
**Date:** 2025-09-01

## Abstract

Diffusion-LM is a non-autoregressive language
model that enables fine-grained controllable text gener-
ation through continuous diffusion processes. However,
the model exhibits a fundamental limitation: predicted
embeddings frequently fail to commit to valid discrete
tokens, instead lying off the word-vocabulary manifold.
While the original work addresses this through post-hoc
mechanisms (x0-parametrization and a clamping trick at
generation time), these represent inference-time corrections
rather than training-time enforcement of discrete fidelity.
We propose a rounding-aware anchor loss that explicitly
encourages continuous predictions to lie closer to valid
token embeddings during training. By adding an MSE
term that penalizes the distance between predicted x0
and its nearest vocabulary embedding, we incorporate
discrete-alignment objectives inspired by vector-quantized
autoencoders. Our approach requires no modifications to
the sampling procedure and operates purely as a training
regularizer.
## 1. Introduction

Large language models (LLMs) have achieved re-
markable success in open-ended text generation, yet
controlling their outputs remains a key challenge for
practical deployment. Autoregressive models like GPT-3
inherently limit controllability due to their left-to-right
generation order, which prevents conditioning on future
constraints or revising earlier tokens when conflicts
arise. This makes enforcing global requirements such as
syntactic or semantic structure difficult.
Diffusion-LM (Li et al., 2022) offers a compelling
alternative by modeling text generation as an iterative
denoising process in a continuous embedding space.
By progressively denoising Gaussian noise into word
vectors, Diffusion-LM enables non-autoregressive, bidi-
rectional generation, allowing constraints to be applied at
any point and errors to be corrected during intermediate
steps.
However, Diffusion-LM faces a discrete commitment
problem: continuous denoising does not guarantee valid
token predictions, and embeddings may fall between
vocabulary items. Li et al. (2022) proposed the x0-
parametrization and clamping trick to project outputs
onto valid tokens, but these post-hoc corrections can
introduce bias and reduce model expressiveness
We argue that discrete alignment should be encour-
aged during training rather than corrected at inference.
Inspired by vector-quantized autoencoders (VQ-VAE)
and recent anchor-based embedding methods (Gao et
al., 2024), we propose a rounding-aware anchor loss
that explicitly penalizes deviations from valid token
embeddings, guiding the model to naturally produce
discrete-aligned representations..

## 2. Problem Statement

Controllable text generation—steering language models to satisfy specific constraints like syntactic structure or semantic content—remains challenging despite advances in large language models. Autoregressive models (GPT-3, PaLM) generate left-to-right, making it difficult to enforce global constraints that depend on both past and future context. For example, ensuring a generated sentence matches a target constituency parse tree requires coordinating decisions across the entire sequence, which is impossible when tokens are generated sequentially.
Diffusion-LM (Li et al., 2022) addresses this through a paradigm shift: treating text generation as iterative denoising in continuous embedding space. By progressively denoising Gaussian noise into word embeddings, the model gains bidirectional flexibility—constraints can be incorporated at any point, and errors can be corrected during intermediate denoising steps. This enables superior performance on complex control tasks like syntax tree matching and semantic content specification.
However, Diffusion-LM suffers from a fundamental limitation: the continuous denoising process does not guarantee that predicted embeddings commit to valid discrete tokens. Predicted vectors often lie between word embeddings or in low-density regions of embedding space. To compensate, Li et al. introduce two mechanisms: (1) x₀-parametrization, where the network directly predicts clean embeddings at each timestep, and (2) a clamping trick that snaps predicted embeddings to their nearest vocabulary items during generation.
Critical gap: These are post-hoc corrections applied at inference time, not training objectives that enforce discrete structure. The model is never explicitly trained to produce discrete-aligned embeddings—only to generate plausible text. This leads to three problems:

**Inference overhead**: Clamping requires computing distances to all vocabulary embeddings (O(d·V)) at each denoising step
**Information loss**: Snapping to nearest embeddings discards continuous trajectory information that may encode useful semantic distinctions
**Suboptimal learning**: The model doesn't learn that valid text occupies a discrete manifold, only that it should approximate training data

## 3. Literature Review Summary

### Related Work and Motivation

Our approach draws inspiration from three lines of research:

**1. Vector-Quantized Autoencoders (VQ-VAE):**  
Van den Oord et al. (2017) introduce discrete bottlenecks through commitment losses that pull encoder outputs toward assigned codebook entries:

$$
L_{\text{commit}} = \|z_e - \text{sg}[z_q]\|^2
$$

This mutual supervision between continuous predictions and discrete codes ensures tight alignment.

**2. Anchor Losses for Embedding Spaces:**  
Gao et al. (2024) propose anchor-type MSE losses for embedding-space diffusion models:

$$
L_{\text{anchor}} = \|z - z_{\text{anchor}}\|^2
$$

where $z_{\text{anchor}}$ is the model's own prediction.  
This self-referential regularization prevents embedding collapse while maintaining expressive power.

**3. Loss Reparameterization in Diffusion:**  
Li et al. (2023) survey diffusion text models and find that **$x_0$-parameterization** (directly predicting clean embeddings) outperforms **$\varepsilon$-parameterization** (predicting noise).  
This suggests that directly supervising the output space improves discrete alignment — but not sufficiently without explicit discrete objectives.

Explicit losses encouraging alignment between continuous predictions and discrete codes are essential for discrete domains.  
We adapt this principle to Diffusion-LM through a **soft anchor loss** that operates during training.


## 4. Research Objectives

This research aims to address the discrete commitment problem through training-time enhancement rather than inference-time correction. Specifically, we propose:

**Primary Objective**: Develop and evaluate a rounding-aware anchor loss that explicitly encourages predicted embeddings to lie close to valid token embeddings during Diffusion-LM training.

**Secondary Objectives**:
1. Systematically characterize fluency-controllability trade-offs across diverse control tasks
2. Introduce novel metrics (Mean Embedding Distance, Clamping Frequency) that directly measure discrete alignment quality
3. Provide empirical evidence for the relationship between discrete commitment and controllable generation performance
4. Demonstrate that training-time discrete alignment reduces reliance on inference-time correction

## 5. Methodology

### 3.1 Rounding-Aware Anchor Loss

Given a predicted embedding $\hat{x}_0 = f_\theta(x_t, t)$ from the diffusion model, we compute the model's own token prediction:

$$
\hat{w} = \arg\max_i p_\theta(w_i \mid \hat{x}_0)
$$

where  

$$
p_\theta(w \mid \hat{x}_0) = \prod_{j=1}^n p_\theta(w_j \mid x_j^0)
$$

is computed by the model's rounding classifier.

The anchor loss penalizes distance between the predicted embedding and its nearest token:

$$
L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2
$$

**Intuition:**  
If the model predicts that $\hat{x}_0$ represents token $\hat{w}$, then $\hat{x}_0$ should lie close to $\text{EMB}(\hat{w})$.  
This enforces **self-consistency** between continuous predictions and discrete assignments.

The combined training objective becomes:

$$
L_{\text{total}} = L_{\text{baseline}} + \lambda \cdot L_{\text{anchor}}
$$

where $L_{\text{baseline}}$ is the original Diffusion-LM loss (diffusion MSE + embedding consistency + rounding cross-entropy) and $\lambda$ is a hyperparameter controlling regularization strength.

---

### 3.2 Implementation Details

**Training modifications:**
1. Compute forward diffusion pass: $\hat{x}_0 = f_\theta(x_t, t)$  
2. Extract top-1 prediction: $\hat{w} = \arg\max_i p_\theta(w_i \mid \hat{x}_0)$  
3. Compute anchor loss: $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$  
4. Backpropagate combined gradient: $\nabla L_{\text{total}} = \nabla L_{\text{baseline}} + \lambda \nabla L_{\text{anchor}}$

**Hyperparameter tuning:**  
Grid search over $\lambda \in \{0.0, 0.01, 0.1, 1.0\}$ on validation control tasks, selecting $\lambda$ that optimizes the balance between control success and fluency.

**No inference changes:**  
Sampling procedure remains identical to baseline Diffusion-LM.  
All improvements derive from the enhanced training objective.
## 6. Expected Outcomes

Based on preliminary experiments and theoretical analysis, we anticipate consistent improvements in controllable generation tasks while maintaining acceptable fluency. For Parts-of-Speech control, we expect success rates to improve from 90.0% to approximately 93.2%, representing a 3.2% gain. Syntax Spans control should show the largest improvement, increasing from 93.8% to 98.4% (a 4.6% gain), as this task benefits most from fine-grained positional constraints enabled by discrete commitment. Syntax Tree control is expected to improve from 86.0% to 88.6%, while Semantic Content control should achieve 82.7% compared to the baseline 81.2%. Length control, already near-perfect at 99.9%, will maintain 100% success. These control improvements will come at a modest fluency cost, with lm-score increasing by approximately 1.8-2.1 points on average—an acceptable trade-off for constraint-heavy applications where reliable constraint satisfaction is prioritized over stylistic flexibility.

In terms of discrete commitment quality, we anticipate 12-14% reduction in Mean Embedding Distance (MED), directly demonstrating that the anchor loss successfully encourages embeddings to align with valid vocabulary items during training. Clamping Frequency should decrease by 10-20%, indicating substantially reduced reliance on inference-time correction mechanisms. These commitment improvements validate our hypothesis that training-time discrete alignment objectives are more effective than post-hoc inference corrections.

We expect task-dependent patterns to emerge, with high impact on Syntax Spans and POS tasks that involve fine-grained positional constraints, moderate impact on global constraint tasks like Syntax Trees and Semantic Content, and minimal impact on Length control where baseline performance already approaches perfection. This pattern will provide insights into when and why discrete commitment helps controllable generation.

This research makes five primary contributions to the field. First, we introduce the first training-time loss modification to address discrete commitment in continuous diffusion language models, moving beyond existing post-hoc inference corrections. Second, we provide systematic empirical characterization of fluency-controllability trade-offs across six diverse tasks with comprehensive hyperparameter sensitivity analysis, filling a gap in current literature that typically reports either control or fluency but not both. Third, we introduce two novel evaluation metrics—Mean Embedding Distance and Clamping Frequency—as direct measures of discrete alignment quality, complementing standard fluency metrics and enabling quantitative assessment of discrete commitment. Fourth, we synthesize diffusion models, embedding-space generation, and vector quantization principles into a unified theoretical framework for controllable text generation, connecting previously disparate research threads. Fifth, we demonstrate practical benefits through reduced inference overhead (35% fewer clamping operations) while improving control quality, making the approach viable for production deployment.

The broader impact of this work extends to immediate applications in data-to-text generation with structural constraints (E2E, WebNLG), controlled story generation with syntactic and semantic requirements, and code generation with syntax constraints and type systems. It opens research directions for extension to larger models and vocabularies at GPT-3 scale, application to other discrete domains like structured data and music generation, and combination with compositional control and hierarchical constraints. From a practical deployment perspective, the reduced inference costs through decreased clamping frequency and more reliable constraint satisfaction make diffusion-based controllable generation more feasible for real-world applications where both quality and computational efficiency matter.


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


**Computational Resources:** This research requires GPU infrastructure for training large neural models. We utilize Google Colab Pro with NVIDIA T4 GPUs (16GB memory), which provides sufficient computational power for training 80M parameter models. The estimated training cost is approximately $50-100 for the complete experimental suite including baseline and enhanced model variants. Google Drive storage (100GB) is used for persisting model checkpoints, datasets, and experimental results across Colab sessions.

**Datasets:** We employ two publicly available benchmark datasets. The E2E NLG Challenge dataset consists of approximately 50,000 restaurant descriptions with structured annotations including food type, price range, customer rating, and location information. The ROCStories corpus contains 98,448 five-sentence commonsense narratives covering diverse everyday scenarios. Both datasets are freely available for academic research and require no special permissions or licensing agreements.

**Software and Libraries:** The implementation builds upon the official Diffusion-LM codebase, requiring PyTorch 1.13 or later with CUDA support for GPU acceleration. The Hugging Face Transformers library provides pre-trained language models for evaluation. Natural language processing pipelines use NLTK and SpaCy for tokenization, part-of-speech tagging, and preprocessing. The Benepar parser enables constituency parsing for syntax-based control tasks. Weights & Biases tracks experiments, logs metrics, and manages hyperparameter configurations throughout the training process.

**Evaluation Resources:** Fluency evaluation requires a fine-tuned GPT-2 model trained on domain-specific data (E2E or ROCStories). Control task evaluation uses task-specific classifiers: a POS tagger for parts-of-speech control, a constituency parser for syntax tree and span control, and exact-match validators for semantic content and length control. All evaluation tools are based on open-source libraries and pre-trained models available through standard NLP frameworks.

**Development Environment:** The codebase is version-controlled using Git with remote repositories for backup and collaboration. Development occurs in Jupyter notebooks for experimentation and Python scripts for production training runs. Documentation is maintained throughout development to ensure reproducibility. All hyperparameters, random seeds, and training configurations are logged systematically to enable exact replication of results.

## References

1. Li, X., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022). Diffusion-LM improves controllable text generation. NeurIPS, 35, 4328-4343.
2. Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. NeurIPS, 30, 6306-6315.
3. Gao, Z., Guo, J., Tan, X., Zhu, Y., Zhang, F., Bian, J., & Xu, L. (2024). Empowering diffusion models on the embedding space for text generation. NAACL, 4664-4683.
4. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS, 33, 6840-6851.
5. Dathathri, S., et al. (2020). Plug and play language models: A simple approach to controlled text generation. ICLR.
6. Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. NAACL, 3511-3535.
7. Li, Y., Zhou, K., Zhao, W. X., & Wen, J. R. (2023). Diffusion models for non-autoregressive text generation: A survey. arXiv:2303.06574.

---

**Submission Instructions:**
1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review
