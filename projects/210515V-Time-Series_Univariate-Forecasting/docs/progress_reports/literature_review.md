# Literature Review: Time Series Univariate Forecasting

**Student:** 210515V
**Research Area:** Time Series Univariate Forecasting
**Date:** 2025-09-01
**Last Updated:** 2025-10-20

## Abstract

This literature review examines recent advances in time series forecasting, with particular focus on linear models, decomposition-based approaches, and deep learning architectures. The review covers key developments from 2019 to 2024, analyzing the evolution from complex transformer-based models to simpler yet effective linear approaches. Key findings indicate that simpler linear models often outperform complex architectures for long-term time series forecasting, and that series decomposition remains a powerful technique for capturing trend and seasonal patterns. This review identifies gaps in adaptive decomposition methods and multi-scale temporal feature extraction, which inform the development of the DLinear-Improved model with adaptive moving averages and multi-scale decomposition capabilities.

## 1. Introduction

Time series forecasting is a fundamental problem in various domains including finance, weather prediction, energy management, and healthcare. Traditional statistical methods like ARIMA and exponential smoothing have long been the standard, but recent years have seen an explosion of deep learning approaches promising superior performance. However, recent work has challenged the necessity of complex models, showing that well-designed linear models can achieve competitive or superior results.

This literature review focuses on univariate time series forecasting, examining three main research directions: (1) transformer-based deep learning models, (2) linear and decomposition-based approaches, and (3) interpretable forecasting methods. The scope includes both theoretical foundations and practical applications, with emphasis on recent developments that challenge conventional wisdom about model complexity.

## 2. Search Methodology

### Search Terms Used

- Time series forecasting, univariate forecasting
- DLinear, NLinear, linear time series models
- Series decomposition, trend-seasonal decomposition
- Transformer time series, temporal models
- Long-term forecasting, LTSF (Long-Term Time Series Forecasting)
- Moving average, adaptive decomposition
- Feature attribution, interpretable forecasting
- Multi-scale decomposition, temporal feature extraction

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv
- [x] NeurIPS, ICML, ICLR proceedings
- [x] Other: Papers With Code, Semantic Scholar

### Time Period

2019-2024, with focus on developments from 2021-2024. Seminal papers from earlier periods included for foundational concepts (e.g., STL decomposition from 1990, moving averages from classical statistics).

## 3. Key Areas of Research

### 3.1 Linear Models for Time Series Forecasting

The paradigm shift toward simpler linear models represents one of the most significant recent developments in time series forecasting.

**Key Papers:**

- **Zeng et al., 2023** - "Are Transformers Effective for Time Series Forecasting?" introduces DLinear and NLinear models, demonstrating that simple one-layer linear models can outperform complex transformer-based architectures. DLinear uses series decomposition to separately model trend and seasonal components, achieving state-of-the-art results on multiple benchmarks with significantly fewer parameters.

- **Challu et al., 2023** - "NHITS: Neural Hierarchical Interpolation for Time Series Forecasting" presents a hierarchical approach using multi-rate data sampling and interpolation, showing that exploiting temporal hierarchies improves long-term forecasting accuracy.

- **Das et al., 2023** - "Long-term Forecasting with TiDE: Time-series Dense Encoder" proposes a simple MLP-based encoder-decoder architecture that captures long-term dependencies without attention mechanisms, achieving competitive performance with lower computational cost.

**Research Direction:** These works collectively challenge the assumption that complexity equals accuracy, showing that with proper inductive biases (like decomposition), simple architectures can be highly effective.

### 3.2 Series Decomposition Methods

Decomposition-based approaches separate time series into interpretable components (trend, seasonal, residual), enabling more focused modeling of each component.

**Key Papers:**

- **Cleveland et al., 1990** - "STL: A Seasonal-Trend Decomposition Procedure Based on Loess" presents the foundational STL decomposition algorithm using locally weighted regression, which remains widely used for its robustness and flexibility.

- **Wu et al., 2021** - "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" introduces decomposition into the transformer architecture, using moving averages for trend extraction and auto-correlation for seasonal pattern discovery.

- **Zhou et al., 2022** - "FEDformer: Frequency Enhanced Decomposition Transformer" performs decomposition in the frequency domain using Fourier and wavelet transforms, capturing periodic patterns more effectively than time-domain methods.

- **Liu et al., 2022** - "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling" uses multi-scale pyramidal attention to capture patterns at different temporal resolutions.

**Research Direction:** Decomposition methods continue to evolve from fixed moving averages to adaptive, learnable decomposition strategies that can adjust to data characteristics.

### 3.3 Transformer-Based Time Series Models

Despite recent challenges to their dominance, transformers have driven significant innovation in time series forecasting.

**Key Papers:**

- **Zhou et al., 2021** - "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" addresses the quadratic complexity of standard transformers through ProbSparse self-attention and a distilling operation, enabling direct multi-horizon forecasting.

- **Wu et al., 2021** - "Autoformer" (mentioned above) introduces auto-correlation mechanism replacing standard attention, showing that series-wise connections are more appropriate for time series than point-wise attention.

- **Zhang & Yan, 2023** - "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting" proposes cross-dimension attention to capture inter-variable dependencies while maintaining temporal structure.

- **Liu et al., 2023** - "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" inverts the traditional transformer architecture by treating time steps as features and variables as tokens, achieving improved performance on multivariate forecasting.

**Research Direction:** While transformers show promise for capturing complex temporal patterns, recent work questions whether their complexity is justified for many forecasting tasks.

### 3.4 Multi-Scale and Hierarchical Temporal Modeling

Capturing patterns at multiple temporal scales is crucial for accurate long-term forecasting.

**Key Papers:**

- **Oreshkin et al., 2020** - "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" uses stacked blocks with trend and seasonality basis functions, achieving strong interpretability and accuracy through hierarchical architecture.

- **Challu et al., 2023** - "NHITS" (mentioned above) explicitly models multiple temporal scales through hierarchical interpolation and pooling strategies.

- **Wang et al., 2023** - "Micn: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting" introduces isometric convolutions at multiple scales to capture both local and global temporal patterns efficiently.

**Research Direction:** Multi-scale approaches recognize that time series contain patterns operating at different frequencies, from short-term fluctuations to long-term trends.

### 3.5 Interpretability and Explainability in Time Series Forecasting

Understanding why models make certain predictions is crucial for trust and practical deployment.

**Key Papers:**

- **Sundararajan et al., 2017** - "Axiomatic Attribution for Deep Networks" introduces Integrated Gradients, a gradient-based attribution method that satisfies key axioms of feature importance, widely applicable to time series models.

- **Lim et al., 2021** - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" incorporates interpretability mechanisms including variable selection networks and temporal attention weights to explain predictions.

- **Tonekaboni et al., 2021** - "What Went Wrong and When? Instance-wise Feature Importance for Time-Series Models" develops methods specifically for time series to identify when and which features contribute to predictions or errors.

**Research Direction:** As deep learning models become more prevalent in critical forecasting applications, interpretability has evolved from a nice-to-have to a necessary feature.

### 3.6 Adaptive and Learnable Components

Moving from fixed hyperparameters to learnable, data-adaptive components represents a promising research direction.

**Key Papers:**

- **Ha et al., 2017** - "HyperNetworks" introduces the concept of using one network to generate weights for another, enabling adaptive behavior based on input characteristics.

- **Jia et al., 2021** - "Adopt: Automatic Parameter Optimization for Deep Time-series Forecasting" proposes automatic hyperparameter tuning specifically designed for time series models.

- **Kim et al., 2022** - "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift" introduces learnable normalization that adapts to distribution shifts in time series data.

**Research Direction:** Adaptive components allow models to adjust their behavior based on data characteristics rather than relying on fixed configurations.

## 4. Research Gaps and Opportunities

### Gap 1: Limited Adaptability in Decomposition Methods

**Why it matters:** Traditional decomposition methods use fixed moving average kernels (e.g., kernel size of 25) that may not be optimal for all datasets or time series with varying characteristics. Different time series exhibit different seasonal periods and trend characteristics that require different decomposition parameters.

**How your project addresses it:** The DLinear-Improved model implements adaptive moving averages with learnable kernel weights that can adjust to the specific characteristics of the data. This allows the model to learn optimal smoothing patterns rather than relying on predetermined window sizes.

### Gap 2: Single-Scale Decomposition Limitations

**Why it matters:** Time series often contain patterns at multiple temporal scales—short-term variations, medium-term cycles, and long-term trends. Single-scale decomposition with one kernel size cannot effectively capture all these patterns simultaneously, potentially losing important information.

**How your project addresses it:** Implementation of multi-scale decomposition with multiple kernel sizes (9, 25, 49) and learnable weighting allows the model to capture and combine patterns at different temporal resolutions. This hierarchical approach provides richer representations of temporal structure.

### Gap 3: Lack of Interpretability in Linear Models

**Why it matters:** While linear models like DLinear achieve strong performance, they often lack mechanisms to explain which input features or time steps are most important for predictions. This limits their applicability in domains requiring explainable decisions (finance, healthcare, energy management).

**How your project addresses it:** Integration of feature attribution methods including Integrated Gradients and Permutation Importance provides insight into model decisions, allowing users to understand which features and temporal patterns drive predictions.

### Gap 4: Trade-off Between Model Complexity and Performance

**Why it matters:** There is ongoing debate about the necessity of complex architectures. While recent work shows linear models can be effective, there remains uncertainty about when and why simple models outperform complex ones, and how to design minimal yet sufficient architectures.

**How your project addresses it:** By starting with the simple DLinear architecture and adding only targeted improvements (adaptive decomposition, multi-scale analysis), the project systematically evaluates which enhancements provide genuine value while maintaining model simplicity and efficiency.

## 5. Theoretical Framework

### Decomposition Principle

The theoretical foundation of this work rests on the classical decomposition of time series:

$$X_t = T_t + S_t + R_t$$

where $X_t$ is the observed series, $T_t$ is the trend component, $S_t$ is the seasonal component, and $R_t$ is the residual. DLinear models these components separately, recognizing that they have different characteristics and may benefit from different modeling approaches.

### Moving Average for Trend Extraction

The moving average is defined as:

$$MA_t = \frac{1}{k} \sum_{i=-(k-1)/2}^{(k-1)/2} X_{t+i}$$

where $k$ is the kernel size. In adaptive moving averages, we replace uniform weights $\frac{1}{k}$ with learnable weights $w_i$:

$$AMA_t = \sum_{i=-(k-1)/2}^{(k-1)/2} w_i \cdot X_{t+i}$$

where $\sum w_i = 1$ (enforced through softmax normalization).

### Multi-Scale Temporal Analysis

Different temporal scales capture different aspects of time series behavior. The multi-scale approach combines information from multiple kernel sizes:

$$T_t = \sum_{s=1}^{S} \alpha_s \cdot MA_t^{(k_s)}$$

where $MA_t^{(k_s)}$ is the moving average with kernel size $k_s$, and $\alpha_s$ are learnable scale weights satisfying $\sum \alpha_s = 1$.

### Linear Projection for Forecasting

Following decomposition, DLinear applies separate linear projections to seasonal and trend components:

$$\hat{Y}_{seasonal} = W_{seasonal} \cdot S + b_{seasonal}$$
$$\hat{Y}_{trend} = W_{trend} \cdot T + b_{trend}$$
$$\hat{Y} = \hat{Y}_{seasonal} + \hat{Y}_{trend}$$

This design respects the different characteristics of seasonal (high-frequency) and trend (low-frequency) components.

## 6. Methodology Insights

### Common Evaluation Practices

Based on the literature, standard evaluation practices include:

1. **Metrics**: MSE (Mean Squared Error) and MAE (Mean Absolute Error) are universal; some papers also report RMSE, MAPE, or domain-specific metrics.

2. **Benchmarks**: Common datasets include ETT (Electricity Transformer Temperature), Weather, Exchange Rate, Traffic, and ILI (Influenza-Like Illness).

3. **Multiple Horizons**: Evaluating performance at different prediction lengths (96, 192, 336, 720 steps) to assess both short and long-term forecasting capability.

4. **Look-back Window**: Typical input sequence lengths of 96, 336, or 512 time steps.

### Promising Methodological Approaches

1. **Ensemble Methods**: Multiple papers show that ensembling multiple simple models can outperform single complex models.

2. **Decomposition-First Approach**: Preprocessing with decomposition before applying any model architecture consistently improves results.

3. **Channel Independence**: The "individual" setting where each variable is modeled separately often outperforms shared-weight approaches for multivariate data.

4. **Proper Normalization**: Instance normalization or reversible normalization helps models handle distribution shifts between training and test data.

5. **Ablation Studies**: Systematic ablation of model components is essential to understanding which architectural choices provide genuine value.

### Implementation Best Practices

- **Reproducibility**: Fixed random seeds, detailed hyperparameter reporting, and code release are increasingly expected.
- **Fair Comparison**: Using identical preprocessing, evaluation protocols, and computational budgets when comparing methods.
- **Computational Efficiency**: Reporting training time, inference latency, and parameter count alongside accuracy metrics.
- **Statistical Significance**: Multiple runs with different seeds and statistical tests to ensure results are not due to random variation.

## 7. Conclusion

This literature review reveals several key insights that inform the DLinear-Improved project:

1. **Simplicity Can Be Superior**: Recent work conclusively demonstrates that simple linear models with appropriate inductive biases (like decomposition) can match or exceed complex transformer architectures for time series forecasting. This validates the choice of DLinear as a foundation.

2. **Decomposition Remains Powerful**: Series decomposition consistently improves forecasting performance across methods and datasets. However, traditional fixed-kernel approaches leave room for improvement through adaptive, learnable decomposition.

3. **Multi-Scale Analysis Is Underexplored**: While hierarchical and multi-scale approaches show promise, few works systematically explore multi-scale decomposition in the context of simple linear models.

4. **Interpretability Gap**: Most high-performing models lack interpretability mechanisms, despite growing demand for explainable forecasting in practical applications.

5. **Evaluation Rigor Is Critical**: The field has learned that careful evaluation with proper baselines is essential, as many complex methods failed to outperform simple baselines when evaluated fairly.

The DLinear-Improved project addresses identified gaps by:

- Introducing adaptive moving averages with learnable kernels
- Implementing multi-scale decomposition with learnable scale weights
- Integrating feature attribution methods for interpretability
- Maintaining the simplicity and efficiency of the DLinear architecture
- Following rigorous evaluation practices established in recent literature

Future work should continue exploring the balance between model complexity and performance, developing adaptive components that adjust to data characteristics, and ensuring that forecasting models are both accurate and interpretable for practical deployment.

## References

1. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? _AAAI Conference on Artificial Intelligence_.

2. Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. _Journal of Official Statistics_, 6(1), 3-73.

3. Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. _Advances in Neural Information Processing Systems (NeurIPS)_.

4. Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. _AAAI Conference on Artificial Intelligence_.

5. Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. _International Conference on Learning Representations (ICLR)_.

6. Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency Enhanced Decomposition Transformer for Long-term Series Forecasting. _International Conference on Machine Learning (ICML)_.

7. Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2023). NHITS: Neural Hierarchical Interpolation for Time Series Forecasting. _AAAI Conference on Artificial Intelligence_.

8. Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023). Long-term Forecasting with TiDE: Time-series Dense Encoder. _International Conference on Machine Learning (ICML)_.

9. Liu, S., Yu, H., Liao, C., Li, J., Lin, W., Liu, A. X., & Dustdar, S. (2022). Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling. _International Conference on Learning Representations (ICLR)_.

10. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. _International Journal of Forecasting_, 37(4), 1748-1764.

11. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. _International Conference on Machine Learning (ICML)_.

12. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). What Went Wrong and When? Instance-wise Feature Importance for Time-Series Models. _KDD Workshop on Mining and Learning from Time Series_.

13. Zhang, Y., & Yan, J. (2023). Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting. _International Conference on Learning Representations (ICLR)_.

14. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2023). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. _International Conference on Learning Representations (ICLR)_.

15. Wang, H., Peng, J., Huang, F., Wang, J., Chen, J., & Xiao, Y. (2023). MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting. _International Conference on Learning Representations (ICLR)_.

16. Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. _International Conference on Learning Representations (ICLR)_.

17. Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. _International Conference on Learning Representations (ICLR)_.

18. Jia, M., Zhao, X., Li, Y., & Zhang, J. (2021). ADOPT: Automatic Hyperparameter Tuning for Deep Time Series Forecasting. _IEEE International Conference on Data Mining (ICDM)_.

19. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.

20. Hyndman, R. J., & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts.

---

**Notes:**

- This review covers 20 high-quality references spanning foundational work to cutting-edge research
- Primary focus on 2021-2024 developments with seminal earlier papers for theoretical foundation
- Mix of top-tier conference papers (NeurIPS, ICML, ICLR, AAAI, ICDM) and journal articles
- References directly inform the architectural choices in DLinear-Improved
- Document should be updated as new relevant work emerges, particularly around adaptive decomposition and interpretable forecasting
