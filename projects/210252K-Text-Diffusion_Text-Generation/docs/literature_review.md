
# Literature Review: Text Diffusion and Controllable Text Generation

**Student:** 210252K  
**Research Area:** Text Diffusion: Text Generation  
**Date:** 2025-01-09

---

## Abstract

This literature review examines diffusion-based text generation and controllable generation techniques, focusing on the discrete commitment challenge in continuous diffusion frameworks. We survey 25 key papers (2017-2024) across diffusion models, controllable generation, and discrete representation learning. Our analysis reveals a critical gap: continuous diffusion models enable superior controllability but suffer from poor discrete alignment, requiring post-hoc inference corrections. This motivates our research on rounding-aware training objectives that enforce discrete commitment during learning.

---

## 1. Introduction and Search Methodology

Language models excel at text generation but struggle with controllable output. Autoregressive models (GPT-3) generate left-to-right, making global constraints (syntax, semantics) difficult to enforce. Diffusion-based approaches address this through iterative denoising in continuous embedding space, enabling bidirectional generation. However, predicted embeddings often fail to align with discrete vocabulary tokens.

**Search Strategy:** We searched Google Scholar, ArXiv, ACM Digital Library, and IEEE Xplore using terms: "diffusion models text generation," "controllable text generation," "vector quantization," "discrete commitment," and "anchor loss embedding." Time period: 2017-2024, focusing on post-DDPM era (2020+) with foundational work from 2017-2019.

---

## 2. Key Research Areas

### 2.1 Foundational Diffusion Models

**Ho et al. (2020)** introduced Denoising Diffusion Probabilistic Models (DDPM), establishing the core framework: forward noise addition $q(x_t|x_{t-1})$ and learned reverse denoising $p_\theta(x_{t-1}|x_t)$. **Song et al. (2021)** unified diffusion as continuous-time SDEs and introduced classifier-guided generation through score function manipulation. **Dhariwal & Nichol (2021)** demonstrated diffusion models surpass GANs on image synthesis and established classifier guidance for controllable generation. These papers provide the mathematical foundation for text diffusion.

### 2.2 Diffusion Models for Text

**Li et al. (2022)** introduced Diffusion-LM, the first successful continuous diffusion model for text. Key innovations: (1) operates in learned embedding space with end-to-end training, (2) x₀-parametrization predicting clean embeddings directly, (3) clamping trick snapping predictions to nearest embeddings at inference. Demonstrates superior controllability on syntax, semantics, and infilling tasks. **Critical limitation:** Poor discrete commitment requires inference-time correction.

**Austin et al. (2021)** and **Hoogeboom et al. (2021, 2022)** explored discrete diffusion directly on token sequences via corruption processes (masking, random replacement). While operating natively on discrete space, these lose continuous controllability advantages. **Gong et al. (2023)** applied embedding diffusion to seq2seq tasks, still requiring rounding mechanisms.

**Gap identified:** All continuous diffusion text models require post-hoc rounding/clamping, suggesting fundamental training-time deficiency.

### 2.3 Controllable Text Generation

**Dathathri et al. (2020)** proposed PPLM: gradient-based control on autoregressive LM hidden states to maximize fluency and attribute satisfaction. Limited to simple attributes (sentiment), fails on structural constraints. **Yang & Klein (2021)** introduced FUDGE: future discriminators predict constraint satisfaction from prefixes, reweighting LM predictions. Struggles with structured constraints (syntax trees, POS sequences) due to error propagation.

**Qin et al. (2020, 2022)** explored continuous relaxation approaches (DELOREAN, COLD) for gradient-based control with lexical constraints. High computational cost and limited to specific tasks. **Key insight:** Autoregressive models struggle with global constraints requiring bidirectional context—motivating diffusion-based approaches.

### 2.4 Discrete Representation Learning

**Van den Oord et al. (2017)** introduced VQ-VAE with vector quantization bottleneck. **Commitment loss** $\|z_e - \text{sg}[z_q]\|^2$ pulls encoder outputs toward codebook entries; **codebook loss** pulls codes toward encodings. Straight-through estimator enables backpropagation through discrete bottleneck. **Core principle:** Mutual supervision between continuous and discrete spaces ensures alignment.

**Jang et al. (2017)** proposed Gumbel-Softmax for differentiable categorical sampling, enabling end-to-end training of discrete VAEs. **Razavi et al. (2019)** scaled VQ-VAE to high-resolution images, showing quantization losses prevent codebook collapse.

**Gao et al. (2024)** introduced anchor loss for embedding-space diffusion: $L_{\text{anchor}} = \|z - \text{EMB}(\arg\max p(w|z))\|^2$. Prevents embedding collapse by pulling predictions toward discrete codes. Demonstrates improved discrete fidelity compared to naive rounding loss. **Direct motivation for our work.**

---

## 3. Research Gaps and Our Approach

### Gap 1: Discrete Commitment in Continuous Diffusion

**Problem:** Current continuous diffusion models (Diffusion-LM, DiffuSeq) fail to naturally produce vocabulary-aligned embeddings. Li et al. address this through x₀-parametrization and clamping—post-hoc corrections rather than training objectives enforcing discrete structure.

**Why it matters:** (1) Inference overhead computing distances to all vocabulary embeddings, (2) information loss from snapping to nearest embeddings, (3) models never learn text lies on discrete manifold.

**Our solution:** Rounding-aware anchor loss $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$ explicitly penalizes distance between predicted embeddings and nearest tokens during training. Combined objective: $L_{\text{total}} = L_{\text{baseline}} + \lambda L_{\text{anchor}}$ where λ controls regularization strength.

### Gap 2: Fluency-Controllability Trade-offs

**Problem:** Most work reports either control success or fluency, not both, making true utility assessment difficult.

**Our solution:** Systematic evaluation of both control success and fluency (lm-score) across six tasks. Hyperparameter λ explicitly controls trade-off. Novel commitment metrics (Mean Embedding Distance, Clamping Frequency) provide mechanistic insight.

### Gap 3: Lack of Discrete Alignment Metrics

**Problem:** Text diffusion papers report generation quality (perplexity, BLEU) but not discrete alignment quality.

**Our solution:** Two novel metrics: (1) **Mean Embedding Distance (MED):** average distance from predicted x₀ to nearest vocabulary embedding, (2) **Clamping Frequency:** percentage of steps where clamping changes predictions.

---

## 4. Theoretical Framework

Our work synthesizes three frameworks:

**Diffusion Models (Ho et al., 2020):** Forward corruption $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ and learned reverse denoising enabling constraint incorporation at any step.

**Embedding Space Diffusion (Li et al., 2022):** Text mapped to continuous embeddings $\text{EMB}(w) \in \mathbb{R}^{nd}$, diffusion operates on embeddings with rounding to recover tokens. Challenge: predicted x₀ lies off discrete manifold.

**Vector Quantization (Van den Oord et al., 2017; Gao et al., 2024):** Commitment losses ensure discrete alignment. We adapt soft anchor loss $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$ without hard quantization, allowing gradient flow through embeddings for joint optimization.

**Integrated approach:** Combines diffusion's flexible generation, embedding-space gradient-based control, and VQ-VAE's discrete alignment principles. Balances fluency (diffusion loss) and discrete alignment (anchor loss) via hyperparameter λ.

---

## 5. Methodology Insights

**Embedding dimensions:** Small d (16-32) forces discrete commitment; larger d (128-256) enables expressiveness. Trade-off between constraint satisfaction and generation diversity.

**Noise schedules:** Square-root (Li et al., 2022) works better for text than linear/cosine, with faster initial noise injection accounting for discrete nature.

**Parametrization:** x₀-prediction superior to ε-prediction for text, directly supervising output space at every timestep.

**Evaluation:** Most work uses perplexity and task-specific metrics. We add commitment metrics (MED, clamping frequency) directly measuring discrete alignment, complementing standard fluency evaluation.

---

## 6. Conclusion

This review establishes that diffusion models enable superior controllability over autoregressive approaches but suffer from poor discrete commitment. All existing continuous diffusion text models require inference-time correction (clamping), representing a fundamental gap. VQ-VAE literature demonstrates commitment losses effectively enforce discrete alignment in other domains. Gao et al. (2024) show anchor losses work for embedding-space diffusion.

**Our contribution:** We address the discrete commitment gap by adapting anchor loss framework to Diffusion-LM training, systematically evaluating fluency-controllability trade-offs, and introducing metrics directly measuring discrete alignment. This work synthesizes diffusion models, embedding-space generation, and vector quantization principles into a principled enhancement for controllable text generation.

---

## References

1. Ho, J., et al. (2020). Denoising diffusion probabilistic models. *NeurIPS*, 33, 6840-6851.
2. Song, Y., et al. (2021). Score-based generative modeling through SDEs. *ICLR*.
3. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat GANs. *NeurIPS*, 34, 8780-8794.
4. Li, X., et al. (2022). Diffusion-LM improves controllable text generation. *NeurIPS*, 35, 4328-4343.
5. Austin, J., et al. (2021). Structured denoising diffusion models in discrete state-spaces. *NeurIPS*, 34.
6. Hoogeboom, E., et al. (2021). Argmax flows and multinomial diffusion. *arXiv:2102.05379*.
7. Hoogeboom, E., et al. (2022). Autoregressive diffusion models. *ICLR*.
8. Gong, S., et al. (2023). Diffuseq: Sequence to sequence text generation with diffusion. *ICLR*.
9. Dathathri, S., et al. (2020). Plug and play language models. *ICLR*.
10. Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation. *NAACL*, 3511-3535.
11. Qin, L., et al. (2020). Back to the future: Unsupervised backprop-based decoding. *EMNLP*, 794-805.
12. Qin, L., et al. (2022). COLD decoding: Energy-based constrained text generation. *arXiv:2202.11705*.
13. Van den Oord, A., et al. (2017). Neural discrete representation learning. *NeurIPS*, 30, 6306-6315.
14. Jang, E., et al. (2017). Categorical reparameterization with Gumbel-softmax. *ICLR*.
15. Razavi, A., et al. (2019). Generating diverse high-fidelity images with VQ-VAE-2. *NeurIPS*, 32.
16. Gao, Z., et al. (2024). Empowering diffusion models on embedding space. *NAACL*, 4664-4683.
17. Krause, B., et al. (2021). GeDi: Generative discriminator guided generation. *arXiv:2009.06367*.
18. Liu, A., et al. (2021). DExperts: Decoding-time controlled text generation. *ACL*, 6691-6706.
19. Kaiser, L., et al. (2018). Fast decoding with discrete latent variables. *ICML*, 2390-2399.
20. Zhao, Q., et al. (2021). Disentangling generative factors with discrete VAEs. *EMNLP*, 3500-3516.
21. Li, Y., et al. (2023). Diffusion models for non-autoregressive text: A survey. *arXiv:2303.06574*.
22. Kingma, D. P., et al. (2021). Variational diffusion models. *NeurIPS*, 34, 21696-21707.
23. Feng, G., et al. (2025). Theoretical benefit and limitation of diffusion LM. *arXiv:2502.09622*.
24. Nosaka, R., & Matsuzaki, T. (2025). Timestep embeddings trigger collapse. *CoNLL*, 397-406.
25. Novikova, J., et al. (2017). The E2E dataset. *SIGdial*, 201-206.
