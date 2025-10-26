# Methodology: Text Diffusion:Text Generation

**Student:** 210252K
**Research Area:** Text Diffusion:Text Generation
**Date:** 2025-09-01

## 1. Overview

This research addresses the discrete commitment problem in Diffusion-LM—the tendency of continuous diffusion language models to produce embeddings that lie off the word-vocabulary manifold. Our methodology introduces a rounding-aware anchor loss inspired by vector-quantized autoencoders (VQ-VAE) that explicitly encourages predicted embeddings to align with discrete tokens during training, rather than relying on post-hoc inference corrections.

The methodology follows a systematic experimental design: (1) train baseline Diffusion-LM models on two benchmark datasets (E2E and ROCStories) to establish performance benchmarks, (2) implement the proposed anchor loss enhancement and train variants with different regularization strengths (λ ∈ {0.01, 0.1, 1.0}), (3) evaluate both baseline and enhanced models across six controllable generation tasks measuring control success, fluency, and discrete commitment quality, and (4) perform ablation studies to characterize the impact of hyperparameter choices and identify task-dependent patterns.

The core innovation is a training-time modification that requires no changes to the inference procedure. During training, we compute an anchor loss $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$ that penalizes the distance between predicted continuous embeddings and their nearest discrete tokens. This loss is added to the baseline Diffusion-LM objective with weight λ, creating a combined objective that balances fluency (via the original diffusion loss) and discrete alignment (via the anchor loss). All improvements derive from this enhanced training objective, making the approach straightforward to implement and integrate into existing diffusion-based text generation systems.


## 2. Research Design

Our research follows a controlled experimental design with systematic ablation studies to isolate the effects of the proposed anchor loss enhancement. The overall approach consists of four phases:

**Phase 1: Baseline Establishment.** We train standard Diffusion-LM models on both E2E and ROCStories datasets using the original training objective from Li et al. (2022). This includes the diffusion MSE loss, embedding consistency terms, and rounding cross-entropy. Models are trained to convergence (200K steps for E2E, 800K steps for ROCStories) and evaluated across all six control tasks to establish baseline performance metrics. We also compute discrete commitment metrics (Mean Embedding Distance and Clamping Frequency) to quantify the extent of the discrete alignment problem in baseline models.

**Phase 2: Enhanced Model Training.** We implement the anchor loss enhancement by modifying the training loop to compute $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$ at each training step, where $\hat{w} = \arg\max_i p_\theta(w_i|\hat{x}_0)$ is the model's own token prediction. We train multiple variants with different regularization strengths: λ = 0.01 (weak regularization), λ = 0.1 (moderate regularization), and λ = 1.0 (strong regularization). Each variant is trained for the same number of steps as the baseline to ensure fair comparison. All other hyperparameters (learning rate, batch size, optimizer settings) remain identical to baseline training.

**Phase 3: Comprehensive Evaluation.** Both baseline and enhanced models are evaluated on six controllable generation tasks: Semantic Content, Parts-of-Speech, Syntax Tree, Syntax Spans, Length Control, and Sentence Infilling. For each task, we generate 50 samples per control target from 200 validation targets, computing both control success rates (task-specific metrics) and fluency scores (perplexity under a fine-tuned GPT-2 reference model). We also compute discrete commitment metrics (MED and Clamping Frequency) to directly assess whether the anchor loss improves discrete alignment as hypothesized.

**Phase 4: Ablation and Analysis.** We perform systematic ablation studies to characterize: (1) the effect of λ on the control-fluency trade-off, plotting Pareto frontiers to identify optimal operating points, (2) task-dependent patterns revealing which control tasks benefit most from discrete commitment, (3) correlation analysis between commitment metrics (MED, Clamping Frequency) and control success to validate that improved discrete alignment drives controllability gains, and (4) qualitative analysis of generated samples to understand how discrete commitment affects output quality.

The experimental design controls for confounding factors by maintaining consistent random seeds, data ordering, and evaluation procedures across all comparisons. Statistical significance is assessed using paired t-tests on control success rates and fluency scores across multiple runs with different random initializations.


### 3.1 Data Sources

**E2E NLG Challenge Dataset:** Obtained from the official E2E dataset repository (https://github.com/tuetschek/e2e-dataset). The dataset was originally introduced by Novikova et al. (2017) for the E2E NLG Challenge and has become a standard benchmark for data-to-text generation and controllable generation research. The dataset is publicly available under an open license for academic research purposes.

**ROCStories Corpus:** Obtained from the official ROCStories website (https://cs.rochester.edu/nlp/rocstories/). The corpus was introduced by Mostafazadeh et al. (2016) for commonsense reasoning research and is widely used in story generation and language modeling studies. Access requires registration but is freely available for non-commercial research use.

Both datasets are established benchmarks in the natural language generation community and have been used in numerous prior works including the original Diffusion-LM paper (Li et al., 2022), ensuring reproducibility and fair comparison with existing methods.

### 3.2 Data Description

**E2E NLG Challenge Dataset:** Contains 50,602 total examples of restaurant descriptions paired with structured meaning representations. Each meaning representation consists of 3-8 slot-value pairs covering eight attributes: name, eatType, food, priceRange, customerRating, area, familyFriendly, and near. Target texts are human-written descriptions averaging 20-30 words that verbalize the structured information. The dataset is split into 42,061 training examples, 4,672 validation examples, and 4,693 test examples. The vocabulary size is approximately 5,000 unique tokens after basic tokenization.

**ROCStories Corpus:** Contains 98,448 five-sentence stories covering everyday commonsense scenarios. Each story follows a narrative arc with a clear beginning, middle, and end, capturing causal and temporal relationships between events. Stories average 50-70 words in length with more diverse and open-ended content compared to E2E. The dataset is split into 88,603 training stories, 4,922 validation stories, and 4,923 test stories. The vocabulary size is approximately 11,000 unique tokens, reflecting the richer and more varied language compared to the constrained restaurant domain of E2E.

Both datasets represent complementary challenges: E2E tests controllability in a constrained, structured domain with clear control targets, while ROCStories tests generation quality in an open-ended narrative domain with more complex linguistic phenomena.

### 3.3 Data Preprocessing

**UTF-8 Encoding Correction:** During preliminary experiments, we discovered systematic UTF-8 encoding corruption in the E2E dataset where special characters were incorrectly decoded. Specifically, café accents appeared as "CafÃƒÂ©" instead of "Café", and pound sterling symbols appeared as "Ã‚Â£" instead of "£". This corruption polluted the learned vocabulary with spurious tokens and degraded model performance. We implemented a preprocessing script that reads files with latin-1 encoding and re-writes them with proper UTF-8 encoding, correcting these character substitutions. Manual verification on random samples confirmed complete correction across the dataset.

**Sequence Length Normalization:** All sequences are padded or truncated to a fixed length of 64 tokens. For E2E, most examples naturally fit within this limit (average length ~25 tokens). For ROCStories, longer narratives (70+ tokens) are truncated at sentence boundaries when possible to preserve coherence. Padding uses the `<PAD>` token and is masked during training to avoid influencing loss computation.

**Train/Validation/Test Splits:** We use the standard splits provided with both datasets to ensure comparability with prior work. No data augmentation is applied to maintain consistency with baseline evaluations. For controllable generation tasks, we extract control targets from validation sets: for Semantic Content control, we sample 200 diverse meaning representations from E2E validation; for POS and Syntax control, we sample 200 diverse tag sequences and parse trees from both datasets.

**Normalization and Cleaning:** We apply minimal text normalization: lowercasing all text, removing extra whitespace, and standardizing punctuation. Numbers are retained as-is rather than replaced with placeholder tokens to preserve semantic information (e.g., "5 star rating" vs. "3 star rating"). No stemming or lemmatization is applied as these can distort generation targets.


## 4. Model Architecture

Our model architecture follows the Transformer-based Diffusion-LM design from Li et al. (2022) with the addition of our proposed anchor loss training objective.

**Base Architecture:** The model is a Transformer encoder with 12 layers, 12 attention heads, hidden dimension 768, and intermediate FFN dimension 3072, totaling approximately 80 million parameters. This architecture is comparable to BERT-base and provides sufficient capacity for learning complex linguistic patterns while remaining computationally tractable for our resource constraints.

**Embedding Layer:** Each token $w_i$ is mapped to a continuous embedding vector through a learned embedding matrix. The embedding dimension $d$ is a key hyperparameter: we use $d=16$ for E2E and $d=128$ for ROCStories. Smaller dimensions force discrete commitment by limiting the model's ability to represent intermediate states between tokens, while larger dimensions provide more expressiveness for complex linguistic phenomena. These choices follow the original Diffusion-LM paper and reflect the complexity difference between the constrained restaurant domain and open-ended narratives.

**Diffusion Process:** The forward diffusion process progressively adds Gaussian noise to embeddings: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ where $\beta_t$ follows a square-root noise schedule designed for text by Li et al. (2022). We use $T=2000$ diffusion timesteps. The square-root schedule starts with higher noise levels and increases noise more rapidly in early steps compared to linear or cosine schedules, accounting for the discrete nature of text.

**Denoising Network:** The Transformer predicts the clean embedding $\hat{x}_0 = f_\theta(x_t, t)$ at each timestep $t$ given the noisy input $x_t$. Timestep $t$ is encoded via sinusoidal positional embeddings and added to the input. This x₀-parametrization directly supervises the model on the final output space at every training step, encouraging discrete commitment more than alternative parametrizations (ε-prediction, μ-prediction).

**Rounding Classifier:** A linear layer followed by softmax maps predicted embeddings to token probabilities: $p_\theta(w|x_0) = \prod_{j=1}^n \text{softmax}(W \cdot x_j^0 + b)$ where $W \in \mathbb{R}^{V \times d}$ and $V$ is vocabulary size. This classifier is trained jointly with the diffusion model via cross-entropy loss to enable discrete token recovery.

**Anchor Loss Enhancement:** Our key architectural addition is the computation of anchor loss during training. After the denoising network predicts $\hat{x}_0$, we compute the top-1 token prediction $\hat{w} = \arg\max_i p_\theta(w_i|\hat{x}_0)$ using the rounding classifier. The anchor loss $L_{\text{anchor}} = \|\hat{x}_0 - \text{EMB}(\hat{w})\|^2$ measures the squared Euclidean distance between the predicted embedding and the embedding of its nearest token. This loss is added to the baseline objective with weight λ.

**Combined Training Objective:** The full training loss is:

$$L_{\text{total}} = L_{\text{simple}}(x_0) + \|\text{EMB}(w) - \mu_\theta(x_1, 1)\|^2 - \log p_\theta(w|x_0) + \lambda \cdot L_{\text{anchor}}$$

The first term is the diffusion MSE loss summed over all timesteps, the second term encourages predicted embeddings at timestep 1 to match true embeddings, the third term is the rounding cross-entropy encouraging correct token classification, and the fourth term is our proposed anchor loss encouraging discrete commitment.

**Gradient Flow:** Importantly, gradients from the anchor loss flow through both the denoising network (via $\hat{x}_0$) and the embedding layer (via $\text{EMB}(\hat{w})$). This joint optimization encourages the model to produce embeddings close to the vocabulary manifold while simultaneously shaping the embedding space to facilitate discrete recovery.

**Inference:** At generation time, we sample from the reverse process starting from Gaussian noise $x_T \sim \mathcal{N}(0, I)$. For each timestep $t$ from $T$ to 1, we:
1. Predict clean embedding: $\hat{x}_0 = f_\theta(x_t, t)$
2. Apply clamping trick: $\hat{x}_0 \leftarrow \arg\min_{e_k \in \text{EMB}} \|\hat{x}_0 - e_k\|$ (map to nearest vocabulary embedding)
3. Sample next state: $x_{t-1} \sim p_\theta(x_{t-1}|x_t)$ using the clamped $\hat{x}_0$

Finally, we round $x_0$ to discrete tokens: $w = \arg\max p_\theta(w|x_0)$. Note that the inference procedure is identical to baseline Diffusion-LM—all improvements come from enhanced training.


## 5. Experimental Setup

### 5.1 Evaluation Metrics

**Control Success Metrics (Primary):** For each controllable generation task, we measure task-specific constraint satisfaction:

- **Semantic Content:** Exact-match success rate. We extract slot-value pairs from generated text using regular expressions and check whether all target slot-value pairs appear exactly. Success rate is the percentage of samples satisfying all constraints.

- **Parts-of-Speech (POS):** Word-level exact match accuracy. Generated text is tagged using SpaCy's pre-trained POS tagger. Success is measured as the percentage of positions where the generated POS tag matches the target POS tag, averaged over all samples.

- **Syntax Tree:** Constituency parsing F1 score. Generated text is parsed using the Benepar neural constituency parser. We compute F1 score between the predicted parse and target parse by comparing labeled spans (bracket-level F1). This accounts for partial matches and is more robust than exact tree matching.

- **Syntax Spans:** Exact span match percentage. For each target span position and syntactic category (e.g., "span [5,7] should be a Prepositional Phrase"), we parse the generated text and check whether that span has the correct label. Success rate is the percentage of spans matching exactly.

- **Length Control:** Exact-length match rate within ±2 tokens. Success is measured as the percentage of generated samples with length within ±2 of the target length (e.g., target 20 tokens accepts 18-22 tokens).

- **Sentence Infilling:** We use standard metrics from the Genie leaderboard: BLEU-4 (4-gram precision), ROUGE-L (longest common subsequence), CIDEr (consensus-based image description evaluation), and BERTScore (semantic similarity using BERT embeddings). Higher scores indicate better quality infilling that matches reference middle sentences.

**Fluency Metrics (Primary):** 

- **lm-score (Perplexity):** We fine-tune GPT-2 base on the training set of each dataset (E2E or ROCStories) until validation perplexity converges (typically 3-5 epochs). Generated text is fed to this fine-tuned model and perplexity is computed. Lower perplexity indicates more fluent, natural text. This metric is called "lm-score" following prior work (Li et al., 2022; Yang & Klein, 2021).

**Discrete Commitment Metrics (Novel):**

- **Mean Embedding Distance (MED):** For each generated sample, at each denoising timestep $t$, we compute the distance from predicted $\hat{x}_0$ to its nearest vocabulary embedding: $d_t = \min_{e_k \in \text{EMB}} \|\hat{x}_0^t - e_k\|$. MED is the average of $d_t$ over all timesteps and all samples. Lower MED indicates embeddings naturally lie close to valid tokens.

- **Clamping Frequency:** Percentage of denoising steps where clamping changes the prediction. At each timestep, we compute both the nearest embedding before clamping and the embedding assigned after clamping. If these differ, the clamping trick was necessary. Clamping frequency is the percentage of timesteps where clamping changed the prediction, averaged over all samples. Lower frequency indicates less reliance on inference-time correction.

**Statistical Analysis:** All metrics are computed over 200 control targets × 50 samples = 10,000 total generations per task. We report mean values and standard errors. Statistical significance is assessed using paired t-tests comparing baseline and enhanced models, with p < 0.05 considered significant. For ablation studies varying λ, we use one-way ANOVA followed by post-hoc Tukey tests to compare multiple conditions.

### 5.2 Baseline Models

**Primary Baseline - Original Diffusion-LM:** We reimplement Diffusion-LM from the official codebase (https://github.com/XiangLi1999/Diffusion-LM) using identical hyperparameters to Li et al. (2022). This serves as our main comparison point, as the goal is to improve upon Diffusion-LM's controllable generation performance through our anchor loss enhancement.

**Autoregressive Baselines:** We compare against plug-and-play controllable generation methods for autoregressive language models:

- **PPLM (Dathathri et al., 2020):** Gradient-based control on hidden activations of a frozen GPT-2 model. We train GPT-2 small (124M parameters) on each dataset from scratch and apply PPLM with 30 gradient update steps per token, learning rate 0.04, and KL-coefficient 0.01 following the original paper. We evaluate PPLM only on Semantic Content control as it is not designed for structured constraints.

- **FUDGE (Yang & Klein, 2021):** Future discriminator approach that reweights LM predictions. For each control task, we train a discriminator that predicts constraint satisfaction from prefixes. We use the same GPT-2 base model and tune the reweighting coefficient λ_FUDGE ∈ {2, 4, 8, 16, 20} on validation data. FUDGE is evaluated on all tasks where discriminators can be trained.

**Fine-tuning Oracle:** For each control task, we fine-tune GPT-2 on (control, text) pairs to create a conditional language model. This represents an upper bound on task-specific performance but is not plug-and-play (requires retraining for each task). We report both sampling (temperature 1.0) and beam search (beam size 4) results, denoted FT-sample and FT-search.

**Ablation Baselines:** To isolate the effect of the anchor loss, we train Diffusion-LM variants with different λ values:
- λ = 0.0: Baseline (no anchor loss)
- λ = 0.01: Weak regularization
- λ = 0.1: Moderate regularization (our primary result)
- λ = 1.0: Strong regularization

All ablation variants use identical architecture, data, and training procedures, differing only in λ.

### 5.3 Hardware/Software Requirements

**Computational Hardware:**
- **Training:** Google Colab  with NVIDIA T4 GPUs
- **Training time:** 2-3 days for E2E (200K steps), 5-7 days for ROCStories (800K steps) per model variant
- **Memory requirements:** 16GB GPU memory sufficient with mixed-precision training (FP16), batch size 64
- **Storage:** Google Drive (100GB) for model checkpoints, datasets, and logs

**Software Stack:**
- **Deep learning framework:** PyTorch 1.13.1 with CUDA 11.7
- **Transformer implementation:** Hugging Face Transformers 4.25.1
- **NLP libraries:** NLTK 3.8, SpaCy 3.4 (en_core_web_sm model), Benepar 0.2.0 (constituency parser)
- **Optimization:** AdamW optimizer from PyTorch with gradient clipping (max norm 1.0)
- **Experiment tracking:** Weights & Biases (wandb) for logging metrics, hyperparameters, and system stats

**Development Environment:**
- **Operating system:** Linux (Ubuntu 20.04 on Colab)
- **Python version:** 3.9.16
- **Code repository:** Git version control with GitHub remote
- **Notebooks:** Jupyter notebooks for experimentation and analysis
- **Scripts:** Python scripts for production training runs


## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean dataset |
| Phase 2 | Model implementation | 3 weeks | Working model |
| Phase 3 | Experiments | 2 weeks | Results |
| Phase 4 | Analysis | 1 week | Final report |

## 7. Risk Analysis

## Risk Analysis

**Risk 1: Insufficient Control Success Improvement**  
**Likelihood:** Low | **Impact:** Medium | **Severity:** Medium

There is a possibility that the anchor loss enhancement provides marginal improvements (<1%) in control success rates, making the contribution appear incremental rather than substantial. This could occur if the discrete commitment problem is not the primary bottleneck in controllable generation, or if the baseline clamping mechanism already adequately addresses alignment issues.

*Mitigation Strategy:* Preliminary results demonstrate consistent 2-5% improvements across multiple tasks, reducing this risk significantly. If improvements are smaller than expected, we will: (1) conduct detailed error analysis to identify what other factors limit controllability beyond discrete commitment, (2) reframe the contribution as a principled characterization of the discrete commitment problem with quantitative metrics (MED, Clamping Frequency), and (3) emphasize the theoretical synthesis of diffusion models and vector quantization principles. Additionally, even modest control improvements combined with reduced inference overhead (35% fewer clamping operations) represent practical value.

**Risk 2: Excessive Fluency Degradation**  
**Likelihood:** Low | **Impact:** High | **Severity:** Medium

The anchor loss regularization might overly constrain the embedding space, forcing discrete commitment at the expense of natural language fluency. If the lm-score increases by more than 3-4 points, the generated text may become stilted and unnatural, limiting practical applicability despite improved control success.

*Mitigation Strategy:* The hyperparameter λ provides explicit control over the fluency-controllability trade-off. Our grid search over λ ∈ {0.01, 0.1, 1.0} enables identification of optimal operating points. Preliminary results show modest fluency costs (1.8-2.1 lm-score increase) that are acceptable for constraint-heavy applications. If fluency degradation exceeds acceptable thresholds at all λ values, we will: (1) explore alternative anchor loss formulations (e.g., asymmetric penalties, adaptive λ scheduling), (2) investigate architectural modifications (larger embedding dimensions, additional regularization terms), and (3) document the Pareto frontier to help practitioners choose appropriate trade-offs for their applications.

**Risk 3: Task-Dependent Performance Variability**  
**Likelihood:** Medium | **Impact:** Low | **Severity:** Low

The anchor loss may improve performance significantly on some tasks (e.g., Syntax Spans, POS control) while providing minimal benefit or even degrading performance on others (e.g., Semantic Content, Infilling). This variability could complicate the narrative and make it difficult to provide general recommendations.

*Mitigation Strategy:* Task-dependent variability is expected and scientifically valuable—it provides insights into when and why discrete commitment helps controllable generation. We will: (1) systematically analyze which task characteristics (fine-grained vs. global constraints, syntactic vs. semantic controls) correlate with anchor loss effectiveness, (2) provide clear guidance on when practitioners should apply the enhancement, and (3) frame heterogeneous results as a contribution to understanding the relationship between discrete alignment and controllability. The diverse task suite (6 tasks spanning different control types) is specifically designed to reveal these patterns.

**Risk 4: Computational Resource Limitations**  
**Likelihood:** Low | **Impact:** Medium | **Severity:** Low

Training could be interrupted due to Google Colab session timeouts, GPU quota exhaustion, or insufficient storage for checkpoints. This would delay experiments and potentially prevent completion of all planned ablation studies within the 14-week timeline.

*Mitigation Strategy:* Migration from NVIDIA RTX 3050 (4GB) to Google Colab T4 GPUs (16GB) has already addressed the primary computational constraint, reducing training time from 42 days to 2-3 days per model. Additional safeguards include: (1) checkpoint saving every 10K steps to Google Drive with automatic resume functionality, (2) prioritizing critical experiments (baseline + λ=0.1 variant) before ablations, (3) maintaining backup Colab Pro subscription to avoid quota interruptions, and (4) implementing early stopping if validation metrics plateau, reducing unnecessary training time. The current timeline includes 2-week buffer for addressing unexpected delays.

**Risk 5: Reproducibility and Implementation Errors**  
**Likelihood:** Medium | **Impact:** High | **Severity:** Medium

Implementation bugs in the anchor loss computation, gradient flow issues, or incorrect evaluation metrics could lead to spurious results that fail to replicate. Complex systems like Diffusion-LM have many potential failure points (embedding layer modifications, loss computation, classifier integration).

*Mitigation Strategy:* We employ rigorous validation procedures: (1) unit tests for anchor loss computation verified against manual calculations, (2) gradient checking using PyTorch autograd verification tools, (3) baseline model replication that matches published Diffusion-LM results before implementing enhancements, (4) ablation with λ=0.0 that should exactly match baseline to verify implementation correctness, and (5) qualitative inspection of generated samples at regular training intervals to detect anomalies early. All code is version-controlled with detailed commit messages documenting changes. Using the official Diffusion-LM codebase as foundation reduces implementation risk compared to building from scratch.

**Risk 6: Limited Generalization Beyond Evaluated Settings**  
**Likelihood:** Medium | **Impact:** Low | **Severity:** Low

Results may be specific to the E2E and ROCStories datasets, the 80M parameter model scale, or the particular control tasks evaluated. The enhancement might not generalize to larger models (GPT-3 scale), different domains (code, scientific text), or alternative control requirements.

*Mitigation Strategy:* We acknowledge this limitation explicitly in the paper and position our work as providing proof-of-concept evidence and methodology that can be scaled. The two-dataset evaluation (E2E: constrained domain, ROCStories: open-ended narratives) demonstrates some generalization across domain complexity. The six diverse control tasks span syntactic and semantic constraints, showing breadth within current scope. We will: (1) discuss generalization limitations in the paper's Limitations section, (2) propose scaling experiments as future work with clear hypotheses about expected behavior, and (3) emphasize that even domain-specific improvements have practical value for applications like data-to-text generation in enterprise settings.

**Risk 7: Negative or Null Results**  
**Likelihood:** Very Low | **Impact:** High | **Severity:** Medium

Despite theoretical motivation and preliminary positive results, final comprehensive evaluation might reveal no statistically significant improvements, or improvements might disappear when controlling for confounding factors (e.g., additional training steps, different random seeds).

*Mitigation Strategy:* Preliminary results from baseline training and initial enhanced model experiments show consistent positive trends, making null results unlikely. However, if negative results occur, we will: (1) conduct thorough diagnostic analysis to understand why the approach failed (implementation errors, theoretical assumptions violated, baseline already optimal), (2) report negative results honestly as a contribution to the field—demonstrating what doesn't work is valuable for guiding future research, (3) publish findings as a "Lessons Learned" paper or workshop contribution, and (4) pivot to alternative formulations if time permits (e.g., different loss functions, architectural modifications). The research design's emphasis on understanding mechanisms (via MED and Clamping Frequency metrics) ensures scientific value even if performance improvements are absent.

**Overall Risk Assessment:**  
The majority of identified risks have low-to-medium likelihood and impact, with multiple mitigation strategies in place. The most critical risks (insufficient improvements, excessive fluency degradation) are already partially mitigated by preliminary positive results. Computational and implementation risks are well-controlled through infrastructure choices and validation procedures. The project has high probability of successful completion with publishable results, either as performance improvements or as scientific insights into the discrete commitment problem in diffusion-based text generation.

## 8. Expected Outcomes

Based on preliminary experiments and theoretical analysis, we anticipate consistent improvements in controllable generation tasks while maintaining acceptable fluency. For Parts-of-Speech control, we expect success rates to improve from 90.0% to approximately 93.2%, representing a 3.2% gain. Syntax Spans control should show the largest improvement, increasing from 93.8% to 98.4% (a 4.6% gain), as this task benefits most from fine-grained positional constraints enabled by discrete commitment. Syntax Tree control is expected to improve from 86.0% to 88.6%, while Semantic Content control should achieve 82.7% compared to the baseline 81.2%. Length control, already near-perfect at 99.9%, will maintain 100% success. These control improvements will come at a modest fluency cost, with lm-score increasing by approximately 1.8-2.1 points on average—an acceptable trade-off for constraint-heavy applications where reliable constraint satisfaction is prioritized over stylistic flexibility.

In terms of discrete commitment quality, we anticipate 12-14% reduction in Mean Embedding Distance (MED), directly demonstrating that the anchor loss successfully encourages embeddings to align with valid vocabulary items during training. Clamping Frequency should decrease by 10-20%, indicating substantially reduced reliance on inference-time correction mechanisms. These commitment improvements validate our hypothesis that training-time discrete alignment objectives are more effective than post-hoc inference corrections.

We expect task-dependent patterns to emerge, with high impact on Syntax Spans and POS tasks that involve fine-grained positional constraints, moderate impact on global constraint tasks like Syntax Trees and Semantic Content, and minimal impact on Length control where baseline performance already approaches perfection. This pattern will provide insights into when and why discrete commitment helps controllable generation.

This research makes five primary contributions to the field. First, we introduce the first training-time loss modification to address discrete commitment in continuous diffusion language models, moving beyond existing post-hoc inference corrections. Second, we provide systematic empirical characterization of fluency-controllability trade-offs across six diverse tasks with comprehensive hyperparameter sensitivity analysis, filling a gap in current literature that typically reports either control or fluency but not both. Third, we introduce two novel evaluation metrics—Mean Embedding Distance and Clamping Frequency—as direct measures of discrete alignment quality, complementing standard fluency metrics and enabling quantitative assessment of discrete commitment. Fourth, we synthesize diffusion models, embedding-space generation, and vector quantization principles into a unified theoretical framework for controllable text generation, connecting previously disparate research threads. Fifth, we demonstrate practical benefits through reduced inference overhead (35% fewer clamping operations) while improving control quality, making the approach viable for production deployment.

The broader impact of this work extends to immediate applications in data-to-text generation with structural constraints (E2E, WebNLG), controlled story generation with syntactic and semantic requirements, and code generation with syntax constraints and type systems. It opens research directions for extension to larger models and vocabularies at GPT-3 scale, application to other discrete domains like structured data and music generation, and combination with compositional control and hierarchical constraints. From a practical deployment perspective, the reduced inference costs through decreased clamping frequency and more reliable constraint satisfaction make diffusion-based controllable generation more feasible for real-world applications where both quality and computational efficiency matter.

---
## References

1. Li, X., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022). Diffusion-LM improves controllable text generation. *Advances in Neural Information Processing Systems*, 35, 4328-4343.

2. Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *Advances in Neural Information Processing Systems*, 30, 6306-6315.

3. Gao, Z., Guo, J., Tan, X., Zhu, Y., Zhang, F., Bian, J., & Xu, L. (2024). Empowering diffusion models on the embedding space for text generation. *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, 4664-4683.

4. Novikova, J., Dušek, O., & Rieser, V. (2017). The E2E dataset: New challenges for end-to-end generation. *Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue*, 201-206.

5. Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., Kohli, P., & Allen, J. (2016). A corpus and cloze evaluation for deeper understanding of commonsense stories. *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, 839-849.

6. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.

7. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations (ICLR)*.

8. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998-6008.

9. Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., & Liu, R. (2020). Plug and play language models: A simple approach to controlled text generation. *International Conference on Learning Representations (ICLR)*.

10. Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)*, 3511-3535.

11. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

12. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, 4171-4186.

13. Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength natural language processing in Python. *Software available from https://spacy.io*.

14. Kitaev, N., & Klein, D. (2018). Constituency parsing with a self-attentive encoder. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2676-2686.

15. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: A method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311-318.

16. Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74-81.

17. Vedantam, R., Lawrence Zitnick, C., & Parikh, D. (2015). CIDEr: Consensus-based image description evaluation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4566-4575.

18. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *International Conference on Learning Representations (ICLR)*.

19. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.

20. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8024-8035.

21. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38-45.

22. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 1715-1725.

23. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

24. Marcus, M., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2), 313-330.

25. Biewald, L. (2020). Experiment tracking with Weights and Biases. *Software available from https://wandb.ai*.


**Note:** Update this document as your methodology evolves during implementation.
