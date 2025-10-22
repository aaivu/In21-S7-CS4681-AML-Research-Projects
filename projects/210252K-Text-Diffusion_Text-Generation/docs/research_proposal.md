# Research Proposal: Text Diffusion:Text Generation

**Student:** 210252K
**Research Area:** Text Diffusion:Text Generation
**Date:** 2025-09-01

## Abstract

This research proposes a novel training enhancement for Diffusion-LM that addresses the discrete commitment problem—the tendency of continuous diffusion language models to produce embeddings that lie off the word-vocabulary manifold. We introduce a rounding-aware anchor loss inspired by vector-quantized autoencoders that explicitly encourages predicted embeddings to align with discrete tokens during training, rather than relying on post-hoc inference corrections. Preliminary results demonstrate 2-5% improvements in controllable generation tasks (Parts-of-Speech: +3.2%, Syntax Spans: +4.6%) with 12-14% reduction in embedding-to-vocabulary distance, at a modest fluency cost of 1.8-2.1 lm-score points. This work provides a principled approach to improving discrete fidelity in continuous diffusion text models, with applications to controllable generation in structured domains.


## 1. Introduction
Large language models (LLMs) have achieved re-
markable success in open-ended text generation, yet
controlling their outputs remains a key challenge for
practical deployment. Autoregressive models like GPT-3
Code available at :
https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects/
tree/main/projects/210252K-Text-Diffusion Text-Generation
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
WWe argue that discrete alignment should be encour-
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

[Provide a brief summary of relevant literature and identify gaps]

## 4. Research Objectives

This research aims to address the discrete commitment problem through training-time enhancement rather than inference-time correction. Specifically, we propose:
**Primary Objective**: Develop and evaluate a rounding-aware anchor loss that explicitly encourages predicted embeddings to lie close to valid token embeddings during Diffusion-LM training.
**Secondary Objectives**:
Systematically characterize fluency-controllability trade-offs across diverse control tasks
Introduce novel metrics (Mean Embedding Distance, Clamping Frequency) that directly measure discrete alignment quality
Provide empirical evidence for the relationship between discrete commitment and controllable generation performance
Demonstrate that training-time discrete alignment reduces reliance on inference-time correction

## 5. Methodology

## 3.1 Rounding-Aware Anchor Loss

Given a predicted embedding  

\[
\hat{x}_0 = f_\theta(x_t, t)
\]

from the diffusion model, we compute the model's own token prediction:

\[
\hat{w} = \arg\max_i p_\theta(w_i | \hat{x}_0)
\]

where  

\[
p_\theta(w | \hat{x}_0) = \prod_{j=1}^n p_\theta(w_j | x_j^0)
\]

is computed by the model's rounding classifier.

The **anchor loss** penalizes the distance between the predicted embedding and its nearest token:

\[
L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2
\]

**Intuition:**  
If the model predicts that \(\hat{x}_0\) represents token \(\hat{w}\), then \(\hat{x}_0\) should lie close to \(\text{EMB}(\hat{w})\).  
This enforces **self-consistency** between continuous predictions and discrete assignments.

The combined training objective becomes:

\[
L_{\text{total}} = L_{\text{baseline}} + \lambda \cdot L_{\text{anchor}}
\]

where \(L_{\text{baseline}}\) is the original Diffusion-LM loss (diffusion MSE + embedding consistency + rounding cross-entropy) and \(\lambda\) is a hyperparameter controlling regularization strength.

---

## 3.2 Implementation Details

### Training Modifications

1. **Compute forward diffusion pass:**  
   \[
   \hat{x}_0 = f_\theta(x_t, t)
   \]

2. **Extract top-1 prediction:**  
   \[
   \hat{w} = \arg\max_i p_\theta(w_i | \hat{x}_0)
   \]

3. **Compute anchor loss:**  
   \[
   L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2
   \]

4. **Backpropagate combined gradient:**  
   \[
   \nabla L_{\text{total}} = \nabla L_{\text{baseline}} + \lambda \nabla L_{\text{anchor}}
   \]

### Hyperparameter Tuning

Perform grid search over  

\[
\lambda \in \{0.0, 0.01, 0.1, 1.0\}
\]

on validation control tasks, selecting the \(\lambda\) that optimizes the balance between **control success** and **fluency**.

### Inference

No inference changes are needed — the sampling procedure remains identical to baseline Diffusion-LM.  
All improvements derive solely from the enhanced **training objective**.

## 6. Expected Outcomes

[What do you expect to achieve?]

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

[List required resources, datasets, tools, etc.]

## References

[Add references in academic format]

---

**Submission Instructions:**
1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review
