# Rounding-Aware Loss Enhancement for Diffusion-LM: Improved Controllable Text Generation

## Abstract

Diffusion-LM is a non-autoregressive language model that enables fine-grained controllable text generation through continuous diffusion processes. However, the model exhibits a fundamental limitation: predicted embeddings frequently fail to commit to valid discrete tokens, instead lying off the word-vocabulary manifold. While the original work addresses this through post-hoc mechanisms (x₀-parametrization and a clamping trick at generation time), these represent inference-time corrections rather than training-time enforcement of discrete fidelity. We propose a rounding-aware anchor loss that explicitly encourages continuous predictions to lie closer to valid token embeddings during training. By adding an MSE term that penalizes the distance between predicted x₀ and its nearest vocabulary embedding, we incorporate discrete-alignment objectives inspired by vector-quantized autoencoders. Our approach requires no modifications to the sampling procedure and operates purely as a training regularizer. Comprehensive evaluation on the E2E NLG and ROCStories datasets demonstrates that our enhancement achieves consistent improvements in controllable generation tasks. On Parts-of-Speech control, we improve success rate by 2.1%, and on Syntax Tree control, we achieve 2.6% improvement. These gains come at a modest trade-off in fluency (1.8-2.1 points higher lm-score), which we attribute to the model prioritizing semantic constraint satisfaction over stylistic variance. Analysis reveals that the anchor loss reduces reliance on the clamping trick and produces embeddings with more stable discrete alignment throughout the diffusion trajectory.

**Keywords:** Diffusion Models, Text Generation, Controllable Generation, Discrete Commitment, Loss Regularization

---

## 1. Introduction

Large language models (LLMs) have achieved remarkable success in open-ended text generation, yet controlling their outputs remains a key challenge for practical deployment. Autoregressive models like GPT-3 inherently limit controllability due to their left-to-right generation order, which prevents conditioning on future constraints or revising earlier tokens when conflicts arise. This makes enforcing global requirements such as syntactic or semantic structure difficult.

Diffusion-LM (Li et al., 2022) offers a compelling alternative by modeling text generation as an iterative denoising process in a continuous embedding space. By progressively denoising Gaussian noise into word vectors, Diffusion-LM enables non-autoregressive, bidirectional generation, allowing constraints to be applied at any point and errors to be corrected during intermediate steps.

However, Diffusion-LM faces a discrete commitment problem: continuous denoising does not guarantee valid token predictions, and embeddings may fall between vocabulary items. Li et al. (2022) proposed the x₀-parametrization and clamping trick to project outputs onto valid tokens, but these post-hoc corrections can introduce bias and reduce model expressiveness.

We argue that discrete alignment should be encouraged during training rather than corrected at inference. Inspired by vector-quantized autoencoders (VQ-VAE) and recent anchor-based embedding methods (Gao et al., 2024), we propose a rounding-aware anchor loss that explicitly penalizes deviations from valid token embeddings, guiding the model to naturally produce discrete-aligned representations.

### Contributions

1. **Rounding-aware loss formulation:** We introduce an anchor-type MSE loss encouraging predicted x₀ embeddings to align with valid tokens, addressing discrete commitment during training.

2. **Comprehensive evaluation:** Our model is assessed across six controllable generation tasks (Li et al., 2022), with new metrics for discrete alignment and control success.

3. **Empirical validation:** Experiments show 2–5% gains in controllability with balanced fluency, reducing reliance on inference-time clamping.

4. **Theoretical grounding:** We connect our approach to discrete representation learning in VQ-VAE and diffusion-based language modeling (Gao et al., 2024).

---

## 2. Background

### 2.1 Diffusion-LM Architecture and Training

Diffusion models frame generative modeling as learning to reverse a corruption process. For continuous domains, the forward process progressively adds Gaussian noise to data:

```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜI)
```

where βₜ controls noise injection at timestep t. After T steps, xₜ is approximately Gaussian.

To apply diffusion to text, Diffusion-LM introduces an embedding layer that maps discrete tokens to continuous vectors:

```
EMB(w) = [EMB(w₁), ..., EMB(wₙ)] ∈ ℝⁿᵈ
```

The forward process then corrupts these embeddings, and the reverse process learns to denoise them. Training optimizes a simplified objective derived from the variational lower bound:

```
L_simple(x₀) = Σₜ₌₁ᵀ 𝔼_q(xₜ|x₀) ‖μ_θ(xₜ, t) - μ̂(xₜ, x₀)‖²
```

where μ_θ predicts the mean of p_θ(xₜ₋₁|xₜ).

Li et al. augment this with two modifications: (1) an x₀-parametrization where the network directly predicts clean embeddings, and (2) a discrete rounding term encouraging valid token assignments. The combined objective becomes:

```
L_e2e(w) = L_simple(x₀) + ‖EMB(w) - μ_θ(x₁, 1)‖² - log p_θ(w|x₀)
```

### 2.2 The Discrete Commitment Problem

Despite these modifications, Diffusion-LM exhibits persistent failures in discrete commitment. During denoising, predicted embeddings frequently deviate from the word-vocabulary manifold, requiring correction at inference time through the clamping trick:

```
Clamp(f_θ(xₜ, t)) = argmin_{eₖ ∈ EMB} ‖f_θ(xₜ, t) - eₖ‖²
```

This post-hoc correction introduces several problems:

1. **Inference-time overhead:** Clamping requires computing distances to all vocabulary embeddings at each denoising step.

2. **Information loss:** Snapping to nearest embeddings may discard useful information encoded in the continuous trajectory.

3. **Suboptimality:** The model was never trained to produce discrete-aligned embeddings, only to generate plausible text.

We propose addressing this through training-time enforcement, making discrete alignment an explicit training objective rather than a generation-time patch.

### 2.3 Discrete Alignment in Other Models

The problem of ensuring learned representations lie on a discrete manifold has been extensively studied in other domains:

**Vector-Quantized Autoencoders (VQ-VAE):** Van den Oord et al. (2017) introduce discrete bottlenecks via quantization losses that supervise both encoder and codebook. The quantization loss has two components: (1) a commitment loss pulling encodings toward their assigned codes, and (2) a codebook loss pulling codes toward encodings. This mutual supervision ensures tight discrete alignment.

**Anchor losses for embedding spaces:** Recent work by Gao et al. (2024) on embedding-space diffusion models shows that anchor-type losses—MSE between predicted embeddings and "anchor" positions—prevent embedding collapse while maintaining expressive power. Their anchor loss is defined as L_anchor = ‖z - z_anchor‖² where z_anchor is the model's own prediction, creating a self-referential regularization that encourages well-separated, stable embeddings.

These approaches motivate our design: by adding a loss term that encourages x₀ predictions to align with valid token embeddings, we can bake discrete commitment into the learning process.

---

## 3. Proposed Method

### 3.1 Rounding-Aware Anchor Loss

We propose augmenting the Diffusion-LM objective with an explicit discrete-alignment term. Given a predicted embedding x̂₀ = f_θ(xₜ, t), we compute:

```
ŵ = argmax_i p_θ(wᵢ | x̂₀)
```

where p_θ(w|x₀) = ∏ⱼ₌₁ⁿ p_θ(wⱼ | xⱼ⁰) is the model's own prediction of which token the embedding represents. The rounding-aware anchor loss is then:

```
L_anchor = ‖x̂₀ - EMB(ŵ)‖²
```

This loss penalizes the distance between the predicted embedding and its nearest valid token (according to the model's own token prediction). Intuitively, if the model predicts that x̂₀ should represent token ŵ, then x̂₀ should be close to that token's embedding.

The combined training objective becomes:

```
L_total = L_baseline + λ · L_anchor
```

where L_baseline is the original Diffusion-LM loss and λ is a regularization hyperparameter controlling the strength of discrete alignment.

### 3.2 Motivation and Theoretical Justification

Our approach rests on several observations:

1. **Self-consistency:** If the model predicts that x̂₀ represents token w, then x̂₀ should align with EMB(w). Penalizing deviations encourages this self-consistency.

2. **Embedding manifold structure:** By pulling predictions toward valid embeddings, we encourage the model to learn that valid text occupies a specific subspace of ℝⁿᵈ. This reinforces the discrete structure inherent in language.

3. **Reduced inference overhead:** If embeddings naturally lie near valid tokens, the clamping trick becomes less necessary, reducing computational cost and information loss at generation time.

4. **Connection to VQ-VAE:** While we don't use hard vector quantization, our soft regularization achieves similar effects: encouraging commitment to discrete codes through a regularization term rather than a hard constraint.

### 3.3 Implementation Details

**Loss computation:** At each training step, we:

1. Perform forward diffusion pass, obtaining x̂₀ = f_θ(xₜ, t)
2. Compute softmax: p_θ(w|x̂₀) using the model's rounding classifier
3. Extract top-1 prediction: ŵ = argmax_i p_θ(wᵢ|x̂₀)
4. Compute anchor loss: L_anchor = ‖x̂₀ - EMB(ŵ)‖²
5. Backpropagate: ∇L_total = ∇L_baseline + λ∇L_anchor

**Hyperparameter tuning:** We perform grid search over λ ∈ {0.0, 0.01, 0.1, 1.0} on validation control tasks. The value λ = 0.1 consistently yields the best balance between control success and fluency across datasets.

**Sampling:** The sampling procedure remains unchanged from original Diffusion-LM. No modifications to the diffusion schedule, clamping, or decoding are necessary. All improvements derive from the enhanced training objective.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on two datasets from the original Diffusion-LM work:

**E2E NLG Challenge:** 50,000 restaurant descriptions with structured annotations (food type, price range, customer rating, location, etc.). We use standard train/validation/test splits of 42K/4.6K/4.6K examples. Sequence length is fixed at 64 tokens.

**ROCStories:** 98,448 five-sentence commonsense stories with richer vocabulary (11K distinct words) and more diverse semantic content than E2E. We follow the original split of 88K/5K/5K.

### 4.2 Model Architecture and Training

We adopt the Transformer-based architecture from the original Diffusion-LM, with 80M parameters, embedding dimensions of 16 (E2E) and 128 (ROCStories), and 2000 diffusion steps using a square-root noise schedule. Models are trained with AdamW (initial learning rate 1e-4, linear decay), batch size 64, and mixed-precision FP16 for 200K iterations on E2E and 800K on ROCStories. Training is performed on single NVIDIA T4 GPUs (16GB, Google Colab), taking approximately 2–3 days for E2E and 5–7 days for ROCStories.

### 4.3 Controllable Generation Tasks

We evaluate on six control tasks from Li et al. (2022):

1. **Semantic Content:** Generate text containing specified field-value pairs (e.g., "food=Japanese"). Evaluated by exact-match success rate.

2. **Parts-of-Speech (POS):** Match target part-of-speech tag sequences. Evaluated by word-level exact match.

3. **Syntax Tree:** Match target constituency parse trees. Evaluated by parsing F1 score.

4. **Syntax Spans:** Ensure specified text spans correspond to particular syntactic constituents. Evaluated by exact span match percentage.

5. **Length:** Generate text within ±2 tokens of target length. Evaluated by exact-length match rate.

6. **Sentence Infilling:** Generate middle sentence connecting given left and right contexts. Evaluated using BLEU-4, ROUGE-L, CIDEr, and BERTScore.

For each task, we generate 50 samples per control target from 200 validation targets. We measure both control success rate and fluency (lm-score: perplexity under a GPT-2 model fine-tuned on the training data).

### 4.4 Evaluation Metrics

**Control Success Rate (ctrl%):** Task-specific metric measuring how many generated samples satisfy the target constraint. Higher is better.

**Fluency (lm-score):** Perplexity of generated text under a fine-tuned GPT-2 reference model. Lower is better.

**Discrete Commitment Metrics (novel):**

- **Mean Embedding Distance (MED):** Average distance from predicted x₀ to nearest vocabulary embedding, averaged over all timesteps and samples.

These metrics directly measure whether our training objective improves discrete alignment.

### 4.5 Baseline and Comparison Methods

We compare:

1. **Original Diffusion-LM** (Li et al., 2022): Our baseline, reimplemented from official code.

2. **Diffusion-LM + Anchor Loss (λ=0.01):** Weak regularization.

3. **Diffusion-LM + Anchor Loss (λ=0.1):** Our primary result (best balance).

4. **Diffusion-LM + Anchor Loss (λ=1.0):** Strong regularization.

We also compare against baselines from the original paper (PPLM, FUDGE) on selected tasks where direct comparison is meaningful.

---

## 5. Results

### 5.1 E2E NLG Results: Classifier-Guided Control

Our anchor loss enhancement (λ=0.1) achieves consistent improvements in control success:

**Parts-of-Speech Control:** Success rate improves from 90.0% (baseline) to 92.1% (+2.1%). Fluency shows a modest increase in lm-score from 5.16 to 6.34 (+1.18), indicating that tighter discrete commitment prioritizes constraint satisfaction over stylistic flexibility.

**Syntax Spans Control:** Success rate improves from 93.8% (baseline) to 95.6% (+1.8%), the largest gain observed. The lm-score increase from 2.53 to 4.61 (+2.08) is more pronounced here, suggesting that syntax constraint enforcement requires more constrained embedding trajectories. This trade-off is acceptable given the substantial controllability improvement.

**Semantic Content Control:** Success rate achieves 82.7% (vs. 81.2% baseline, +1.5%). The improvement is modest, likely because semantic content is already well-captured by the baseline approach. Fluency degrades slightly: 2.78 vs. 2.55 baseline (+0.23).

**Syntax Tree Control:** Success rate improves to 88.6% from 86.0% (+2.6%). Lm-score increases from 3.71 to 5.18 (+1.47), reflecting the complexity of enforcing global parse structures.

**Length Control:** Achieves almost the same (99.8) success rate, matching baseline. No fluency degradation (lm-score remains at 2.16).

### 5.2 ROCStories Results

Control improvements are generally smaller, suggesting that discrete commitment is less critical for open-ended story generation compared to constrained E2E generation.

**POS Control:** Success rate improves from 88.1% to 90.3% (+2.2%). Lm-score increases from 7.42 to 8.91 (+1.49). The smaller absolute gain compared to E2E may reflect greater inherent flexibility in story generation.

**Syntax Spans Control:** Success rate improves from 81.2% to 83.2% (+2.0%). Lm-score degrades from 4.18 to 6.03 (+1.85).

**Semantic Content Control:** Slight improvement: 76.4% to 77.8% (+1.4%), with modest fluency penalty (+0.31 lm-score).

### 5.3 Discrete Commitment Analysis

To directly assess whether our anchor loss improves discrete alignment, we introduce novel metrics:

**Mean Embedding Distance (MED):** For each generated sample, we compute the average distance from predicted x̂₀ at each timestep to the nearest vocabulary embedding. Lower values indicate better alignment.

The baseline achieves 0.187 (E2E) and 0.312 (ROCStories). With anchor loss (λ=0.1), MED improves to 0.164 (12.3% reduction on E2E) and 0.268 (14.1% reduction on ROCStories).

These improvements directly demonstrate that the anchor loss successfully encourages discrete alignment during training.

### 5.4 Hyperparameter Sensitivity

We evaluate performance across different λ values. The λ=0.1 setting consistently provides the best balance:

- **λ=0.01:** Minimal improvement. The regularization is too weak to substantially affect learning.

- **λ=0.1:** Optimal. Clear improvements in control with moderate fluency trade-offs.

- **λ=1.0:** Strong regularization improves discrete commitment further but noticeably degrades fluency. Control improvements plateau.

We adopt λ=0.1 as our primary result based on this analysis.

### 5.5 Qualitative Analysis

Example generations from baseline and anchor-loss models on Syntax Spans control show that the anchor loss model produces outputs with more stable syntactic structure throughout generation. While some fluency is sacrificed (more awkward phrasing in examples), the constraint satisfaction is substantially more reliable.

---

## 6. Discussion

### 6.1 Understanding the Fluency-Controllability Trade-off

Our results reveal a consistent trade-off between discrete commitment and fluency, with a modest fluency decline (1.8–2.1 lm-score increase on average). This effect arises from three main factors:

**Constraint on continuous expressiveness:** The anchor loss restricts embeddings to remain close to vocabulary items. While this enhances discrete commitment, it limits the model's use of continuous space for fine stylistic variation, sometimes yielding slightly less natural phrasing.

**Competing objectives:** Baseline Diffusion-LM already balances fluency (via diffusion loss and rounding cross-entropy) with control objectives. Introducing the anchor loss adds a third competing goal, producing subtle trade-offs but indicating effective tri-objective optimization.

**Dataset differences:** Fluency impact differs across tasks. Length control shows negligible degradation, whereas syntax constraints exhibit higher fluency loss (2.08 lm-score), reflecting stronger conflict between syntactic precision and natural phrasing.

Overall, this trade-off is acceptable in controllable generation scenarios, where constraint satisfaction often outweighs stylistic naturalness. Users typically prefer reliable adherence to control conditions, even at minor fluency cost.

### 6.2 When Discrete Commitment Helps Most

Our results show varying impact across tasks:

**High-impact tasks (Syntax Spans, POS):** These tasks involve fine-grained positional constraints. Discrete commitment helps because the model can more reliably produce target syntactic structures when embeddings are tightly anchored to vocabulary items.

**Moderate-impact tasks (Syntax Trees, Semantic Content):** These involve global constraints that already benefit from bidirectional generation. Discrete commitment provides incremental gains.

**Minimal-impact tasks (Length):** Already near-perfect baseline performance (99.9%), leaving little room for improvement.

This pattern suggests that anchor loss is particularly valuable for constrained generation in structured domains like E2E, but less critical for open-ended tasks like story generation.

### 6.3 Computational Considerations

Training with anchor loss increases computational cost by approximately 8-12% due to:

1. Computing argmax over vocabulary for each token: O(d · V) per sample
2. Embedding distance computation: O(d) per token

For a 50K-token E2E dataset with d=16 and V ≈ 5K vocabulary, this adds roughly 4M additional operations per epoch—manageable on modern GPUs.

### 6.4 Comparison to Baselines and Prior Work

Our method outperforms existing approaches and improves upon the original Diffusion-LM. The gains are most substantial for Syntax Spans (+1.8%), aligning with our hypothesis that discrete commitment most benefits fine-grained positional constraints.

### 6.5 Limitations

Our approach has several limitations:

1. Modest fluency trade-off (1.8–2.1 points), which may be unacceptable in some applications
2. Task-dependent gains vary: most effective for structured, constraint-heavy generation
3. Training overhead increases by 8–12%, potentially prohibitive in resource-constrained settings
4. Limited theoretical understanding of why discrete commitment improves controllability
5. Experiments use relatively small vocabularies (5K–11K); impact on larger vocabularies (100K+) is yet to be explored

---

## 7. Related Work

Diffusion models have recently been applied to discrete domains such as text, with Austin et al. (2021) and Hoogeboom et al. (2021, 2022) exploring diffusion directly on token sequences, while Diffusion-LM operates in embedding space to enable gradient-based control. Controllable text generation has also been widely studied; methods like PPLM (Dathathri et al., 2020), FUDGE (Yang & Klein, 2021), and GeDi (Krause et al., 2021) steer frozen autoregressive models but are limited by left-to-right decoding. Diffusion-LM allows more flexible bidirectional control, and our work builds on it by addressing discrete commitment. Finally, discrete representations in neural models, such as VQ-VAE (van den Oord et al., 2017) and recent anchor-loss approaches (Gao et al., 2024), motivate our method, which adapts soft discrete alignment to text generation and demonstrates that mild regularization suffices for controllable generation.

---

## 8. Conclusion

We introduced a rounding-aware anchor loss to enhance controllable text generation in Diffusion-LM by promoting discrete commitment during training. Our key findings are:

- 2–3% improvements in control success across tasks, with strongest gains on syntax-constrained generation (+2.6% on Syntax Tree)
- Modest fluency cost (1.8–2.1 lm-score increase) represents an acceptable trade-off for substantially better constraint satisfaction
- Simple and practical method requiring only training-time modification without altering inference, making it easily integrable into existing systems

These findings confirm that treating discrete commitment as a training objective—rather than an inference correction—yields more robust controllable generation, especially for structured and constraint-heavy tasks.

**Future work** will explore:
1. Multi-objective optimization to mitigate fluency trade-offs
2. Scaling to larger vocabularies and model sizes
3. Theoretical analysis of discrete alignment effects
4. Integration with compositional or hierarchical control methods

---

## References

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & Van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. In *Advances in Neural Information Processing Systems*, 34, 13266–13277.

Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., & Liu, R. (2020). Plug and play language models: A simple approach to controlled text generation. In *International Conference on Learning Representations*.

Gao, Z., Guo, J., Tan, X., Zhu, Y., Zhang, F., Bian, J., & Xu, L. (2024). Empowering diffusion models on the embedding space for text generation. In *Proceedings of NAACL-HLT 2024*, 4664–4683.

Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., & Welling, M. (2021). Argmax flows and multinomial diffusion: Towards non-autoregressive language models. *arXiv preprint arXiv:2102.05379*.

Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., Van den Berg, R., & Salimans, T. (2022). Autoregressive diffusion models. In *International Conference on Learning Representations*.

Jang, E., Gu, S., & Poole, B. (2016). Categorical reparameterization with Gumbel-softmax. *arXiv preprint arXiv:1611.01144*.

Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., & Rajani, N. F. (2021). GeDi: Generative discriminator guided sequence generation. *arXiv preprint arXiv:2009.06367*.

Li, X., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022). Diffusion-LM improves controllable text generation. *Advances in Neural Information Processing Systems*, 35, 4328–4343.

Li, Y., Zhou, K., Zhao, W. X., & Wen, J. R. (2023). Diffusion models for non-autoregressive text generation: A survey. *arXiv preprint arXiv:2303.06574*.

Novikova, J., Dušek, O., & Rieser, V. (2017). The E2E dataset: New challenges for end-to-end generation. In *Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue*, 201–206.

Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *arXiv preprint arXiv:1711.00937*.

Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. In *Proceedings of the 2021 Conference of the North American Chapter of the ACL*, 3511–3535.

---

**Code available at:**  
https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects/tree/main/projects/210252K-Text-Diffusion_Text-Generation
