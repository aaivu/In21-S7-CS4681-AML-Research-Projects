# Literature Review: Time Series – Multivariate Forecasting

**Student:** 210173T  
**Research Area:** Time Series Forecasting  
**Date:** 2025-09-01

## Abstract

This literature review explores recent advances in multivariate long-sequence time-series forecasting, emphasizing the evolution of Transformer-based architectures and their efficiency improvements. It surveys canonical models such as the Vanilla Transformer, Informer, FEDformer, and PatchTST, comparing their design choices and computational trade-offs. The review identifies a key research gap, the absence of hybrid models that combine efficient attention mechanisms with PatchTST’s representational advantages and positions the current project to bridge that gap through integrated Sparse and Fourier attention frameworks.

## 1. Introduction

Time-series forecasting is fundamental to decision-making in domains such as finance, energy systems, logistics, and meteorology. Traditional statistical models (e.g., ARIMA) capture short-term dependencies but falter on highly nonlinear, long-range temporal dynamics.

The introduction of the Transformer revolutionized sequence modeling by enabling global dependency capture via self-attention. However, the quadratic complexity O(L²) in sequence length L limits its scalability for long-horizon forecasting.

Recent research therefore focuses on redesigning either (a) the tokenization and data representation strategy (e.g., PatchTST) or (b) the attention mechanism itself (e.g., Informer, FEDformer). This review analyzes these directions and their implications for efficient multivariate forecasting.

## 2. Search Methodology

### Search Terms Used
- Transformer time series forecasting
- Efficient attention mechanism
- Informer, FEDformer, PatchTST
- Multivariate forecasting transformer
- Channel-independent modeling

### Databases Searched
- IEEE Xplore
- Google Scholar
- ArXiv

### Time Period
2018 – 2025, focusing on modern Transformer-based approaches to long-sequence multivariate forecasting

## 3. Key Areas of Research

### 3.1 The Transformer in Time Series

Transformers model contextual dependencies through self-attention:

$$\text{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

While effective for NLP, computing an L × L attention matrix makes complexity O(L²). Early adaptations to time series thus suffered from limited context windows and high cost, motivating research into efficient variants.

**Key Papers:**
- Vaswani et al., 2017 – Introduced the Transformer architecture; foundation for all subsequent models.
- Zeng et al., 2022 – Showed simple linear baselines rivaled vanilla Transformers, motivating re-examination of their design.


### 3.2 PatchTST – A New Paradigm for Tokenization

PatchTST (Nie et al., 2023) introduced a patch-based and channel-independent tokenization scheme.
- **Patching:** Converts sequential samples into overlapping subseries patches, drastically reducing sequence length (N ≈ L/S) and preserving local trends.
- **Channel Independence:** Each variable is modeled separately with shared weights, avoiding overfitting and enabling distinct temporal dynamics learning.

PatchTST achieved ~21% lower MSE than previous SOTA models across Weather, Traffic, and Electricity datasets.

Despite its success, PatchTST still relies on vanilla self-attention, limiting scalability.

**Key Papers:**
- Nie et al., 2023 – "A Time Series is Worth 64 Words." Introduced PatchTST, setting new SOTA.
- Dosovitskiy et al., 2021 – ViT's patching inspired PatchTST's representation strategy.


### 3.3 Informer – Efficiency-Driven Transformer Design

Informer (Zhou et al., 2021) addresses the O(L²) bottleneck through ProbSparse attention, exploiting the empirical "long-tail" distribution of attention scores.

Only the top-u queries (u = c · ln L) are retained, reducing complexity to O(L log L).

A distilling encoder further halves sequence lengths after each layer, balancing efficiency and accuracy.

**Key Papers:**
- Zhou et al., 2021 – Proposed Informer with ProbSparse attention and self-attention distilling, achieving large-scale sequence forecasting feasibility.


### 3.4 FEDformer – A Frequency-Domain Approach

FEDformer (Zhou et al., 2022) projects time series into the frequency domain via FFT, performs frequency mode selection, and returns to time domain through IFFT. By operating on a fixed number of frequencies, complexity drops to O(L). This makes FEDformer particularly strong for periodic data such as energy and climate series.

**Key Papers:**
- Zhou et al., 2022 – Introduced frequency-enhanced decomposition with linear complexity and competitive accuracy.


### 3.5 Research Gap

| Model | Innovation | Limitation |
|-------|-----------|------------|
| PatchTST | Robust patch-based tokenization and channel independence | Quadratic self-attention |
| Informer | ProbSparse attention (O(L log L)) | Weaker representation of local patterns |
| FEDformer | Frequency-domain linear attention | Less spatial/temporal structure awareness |

No existing work integrates PatchTST's architectural strength with efficient attention mechanisms.

This project aims to bridge that gap through two hybrid variants:
- **PatchTST + Sparse** (ProbSparse attention)
- **PatchTST + Fourier** (Frequency-domain block)


## 4. Research Gaps and Opportunities

### Gap 1 – Lack of Efficiency in PatchTST

**Why it matters:** Quadratic attention limits scalability for long horizons.

**Project response:** Integrate ProbSparse attention to reduce complexity to O(L log L).

### Gap 2 – Limited Frequency-Domain Awareness

**Why it matters:** Temporal periodicity is crucial for domains like energy and weather.

**Project response:** Embed Fourier blocks from FEDformer within PatchTST to capture global frequency dependencies.


## 5. Theoretical Framework

The study is grounded in sequence modeling theory based on self-attention and representation learning. It combines three core principles:
1. **Self-Attention Mechanisms** (Vaswani et al., 2017) for long-range dependency capture.
2. **Hierarchical Tokenization** (Nie et al., 2023) to encode local context via patches.
3. **Efficient Transform Domain Computation** (Zhou et al., 2021 & 2022) to reduce computational load while preserving expressiveness.


## 6. Methodology Insights

Common methodological trends across the literature include:
- **Deep learning frameworks:** PyTorch implementations of modular attention blocks.
- **Datasets:** Weather, Traffic, Electricity, ILI, and ETT series.
- **Metrics:** MSE and MAE as primary performance indicators.
- **Evaluation Strategy:** Fixed forecast horizons T ∈ {96, 192, 336, 720}.

Most recent studies demonstrate that efficient attention variants yield similar or better accuracy at fraction of the training cost, highlighting efficiency as the key frontier for innovation.


## 7. Conclusion

The literature shows rapid progress from standard Transformers to domain-specific efficient variants. PatchTST revolutionized data representation while Informer and FEDformer redefined attention efficiency. However, a synergistic model that fuses these advantages remains unexplored. This project addresses that gap by proposing hybrid architectures that integrate efficient attention mechanisms within PatchTST's framework — aiming for state-of-the-art accuracy with superior scalability.


## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. *NeurIPS* (pp. 5998–6008).

2. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers. *ICLR*.

3. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022). Are Transformers Effective for Time Series Forecasting? *arXiv:2205.13504*.

4. Zhou, H., Zhang, S., Peng, J., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long-Sequence Time-Series Forecasting. *AAAI-35* (12), 11106–11115.

5. Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency Enhanced Decomposed Transformer for Long-Term Series Forecasting. *ICML-39* (PMLR 162).

6. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR*.
