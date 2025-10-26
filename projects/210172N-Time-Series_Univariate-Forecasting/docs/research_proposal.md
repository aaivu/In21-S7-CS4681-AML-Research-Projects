# Research Proposal: Adapting PatchTST for Real-Time, Multi-Horizon Forecasting on the M4 Competition Benchmark

**Student:** Galappaththi A. S. (210172N)
**Research Area:** Time Series Univariate Forecasting
**Course:** CS4681 - Advanced Machine Learning Research
**Institution:** University of Moratuwa
**Supervisor:** Dr. Uthayasanker Thayasivam

## Abstract

Time series forecasting is critical for decision-making across industries, from finance to healthcare. While Transformer-based models like PatchTST have achieved state-of-the-art accuracy on long-term forecasting benchmarks, their deployment in production environments is limited by computational constraints. This research investigates the adaptation of PatchTST for real-time, multi-horizon forecasting on the M4 Competition benchmark—a diverse collection of 100,000 univariate time series across multiple frequencies.

We propose a comprehensive framework that combines model architecture optimization, ONNX conversion for platform-agnostic deployment, and post-training INT8 quantization for model compression. Our approach aims to achieve 2× model size reduction while maintaining accuracy degradation below 5%, with inference latency suitable for real-time applications (<10ms). The baseline comparison will be N-BEATS, the first deep learning model to outperform the M4 Competition winner. This research bridges the gap between state-of-the-art forecasting accuracy and practical deployment requirements, demonstrating that Transformer-based models can be efficiently deployed on resource-constrained devices.

## 1. Introduction

### 1.1 Background

Time series forecasting has evolved significantly with the advent of deep learning. Traditional statistical methods like ARIMA and exponential smoothing have been augmented—and in many cases surpassed—by neural approaches including RNNs, LSTMs, and more recently, Transformer-based architectures. The M4 Competition (2018) established a rigorous benchmark for evaluating forecasting methods across diverse domains, frequencies, and series lengths.

PatchTST (Nie et al., 2023), introduced at ICLR 2023, revolutionized time series forecasting by treating time series as sequences of patches rather than individual time steps. This approach reduces computational complexity from O(L²) to O((L/S)²), where L is sequence length and S is stride, while improving representation learning. However, PatchTST was primarily evaluated on Long-Term Time Series Forecasting (LTSF) benchmarks with fixed-length series, not on the heterogeneous M4 Competition dataset.

### 1.2 Motivation

Despite impressive accuracy, Transformer-based forecasting models face significant deployment challenges:

1. **Model Size:** Multi-megabyte models unsuitable for edge devices
2. **Inference Latency:** GPU-dependent operations limiting real-time applications
3. **Memory Footprint:** High RAM requirements during inference
4. **Platform Dependency:** Framework-specific implementations hindering cross-platform deployment

These limitations prevent the adoption of state-of-the-art models in production environments where resource constraints are paramount—IoT devices, mobile applications, and cloud cost optimization scenarios.

### 1.3 Research Gap

While extensive research exists on Transformer optimization for NLP (quantization, pruning, distillation), time series forecasting models remain under-explored in this domain. Specifically:

- PatchTST's applicability to diverse, variable-length series (like M4) is unexplored
- Trade-offs between model compression and forecasting accuracy are not well-characterized
- Real-time deployment strategies for Transformer-based forecasting models lack comprehensive benchmarking

This research addresses these gaps by systematically adapting PatchTST for the M4 benchmark with aggressive optimization while rigorously evaluating accuracy-efficiency trade-offs.

## 2. Problem Statement

**How can we adapt PatchTST, a state-of-the-art Transformer-based time series forecasting model, for real-time deployment on the M4 Competition benchmark while maintaining competitive accuracy?**

### Specific Challenges

1. **Variable-Length Series:** M4 contains series ranging from 42 to 2,794 observations across 6 frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)

2. **Multi-Horizon Forecasting:** Different frequencies require different forecast horizons (6-48 steps), necessitating adaptive architecture

3. **Distribution Shift:** M4 series span diverse domains (demographics, finance, industry) with varying scales and patterns

4. **Real-Time Constraints:** Target inference latency <10ms for practical deployment

5. **Model Compression:** Achieve 50%+ size reduction without exceeding 5% accuracy degradation

6. **Cross-Platform Deployment:** Enable deployment across PyTorch, ONNX Runtime, and quantized backends

## 3. Literature Review Summary

### 3.1 Time Series Forecasting with Transformers

**Transformer Architecture (Vaswani et al., 2017):** Introduced self-attention mechanism enabling parallel processing and long-range dependency modeling, originally for NLP.

**Informer (Zhou et al., 2021):** First Transformer for LTSF, introduced ProbSparse self-attention to reduce complexity from O(L²) to O(L log L).

**Autoformer (Wu et al., 2021):** Decomposition-based Transformer separating trend and seasonal components with auto-correlation mechanism.

**FEDformer (Zhou et al., 2022):** Frequency domain Transformer using Fourier/wavelet transforms for efficient long-term modeling.

**PatchTST (Nie et al., 2023):** Current state-of-the-art using patching to reduce complexity and channel-independence to improve generalization. Achieves best performance on 8 LTSF benchmarks.

### 3.2 M4 Competition and N-BEATS

**M4 Competition (Makridakis et al., 2018):** 100,000 time series across 6 frequencies, evaluated using sMAPE, MASE, and OWA. Winner: Hybrid statistical-ML approach.

**N-BEATS (Oreshkin et al., 2020):** First pure deep learning model to outperform M4 winner. Architecture: Doubly residual stacking with interpretable basis functions. Strengths: No need for domain knowledge, strong performance on short series.

### 3.3 Model Optimization Techniques

**Quantization (Jacob et al., 2018):** Post-training and quantization-aware training for FP32→INT8 conversion. Achieves 4× size reduction in CNNs with <2% accuracy loss.

**ONNX and ONNX Runtime (Microsoft, 2019):** Open format for ML models enabling cross-framework interoperability. ONNX Runtime provides hardware-accelerated inference with graph-level optimizations.

**RevIN (Kim et al., 2022):** Reversible Instance Normalization for time series, normalizing each series independently to handle distribution shift. Critical for transfer learning and diverse datasets.

### 3.4 Research Gaps

1. **PatchTST on M4:** No published evaluation of PatchTST on M4 benchmark
2. **Quantization for Time Series:** Limited research on INT8 quantization impact on forecasting accuracy
3. **Real-Time Benchmarking:** Lack of comprehensive latency analysis for Transformer-based forecasters
4. **Edge Deployment:** No studies on deploying PatchTST on resource-constrained devices

## 4. Research Objectives

### Primary Objective

**Adapt PatchTST for real-time, multi-horizon forecasting on the M4 Competition benchmark, achieving competitive accuracy with optimized model size and inference latency suitable for production deployment.**

### Secondary Objectives

1. **Baseline Establishment:** Implement and evaluate PatchTST on M4 Competition using official metrics (sMAPE, MASE, OWA)

2. **Architecture Adaptation:** Optimize PatchTST hyperparameters (patch length, model dimension, layers) for M4's variable-length series

3. **Distribution Handling:** Integrate RevIN to mitigate distribution shift across heterogeneous M4 series

4. **ONNX Conversion:** Export optimized models to ONNX format with graph-level optimizations for cross-platform deployment

5. **Model Quantization:** Apply post-training INT8 quantization to achieve 2× compression with <5% accuracy degradation

6. **Real-Time Benchmarking:** Measure inference latency across PyTorch (GPU), ONNX FP32 (GPU), and ONNX INT8 (CPU) backends

7. **Comparative Analysis:** Benchmark against N-BEATS to validate competitive performance

8. **Production Implementation:** Develop production-ready codebase with modular architecture, comprehensive testing, and documentation

## 5. Methodology

### 5.1 Dataset Preparation

**Primary Dataset:** M4 Competition
- 100,000 univariate time series
- 6 frequencies: Yearly (23,000), Quarterly (24,000), Monthly (48,000), Weekly (359), Daily (4,227), Hourly (414)
- Official train/test splits provided
- Focus: Monthly (largest subset, 48,000 series) as primary evaluation

**Secondary Datasets:** LTSF Benchmarks for ablation studies
- Weather (21 features, 52,696 timesteps)
- Traffic (862 sensors, 17,544 timesteps)
- Electricity (321 customers, 26,304 timesteps)
- ETT (Electricity Transformer Temperature) variants

### 5.2 Model Architecture

**Base Model:** PatchTST
- Standard Configuration (LTSF benchmarks):
  - Model Dimension (d_model): 128
  - Encoder Layers: 3
  - Attention Heads: 16
  - Feed-Forward Dimension: 256
  - Patch Length: 16, Stride: 8

- M4-Optimized Configuration:
  - Model Dimension: 64 (50% reduction)
  - Encoder Layers: 2 (33% reduction)
  - Attention Heads: 8 (50% reduction)
  - Feed-Forward Dimension: 128 (50% reduction)
  - Adaptive Patching: patch_len ≈ forecast_horizon

**Key Modifications:**
1. **ONNX Compatibility:** Manual patching implementation (no unfold operation)
2. **Inline RevIN:** Normalization integrated into model for single-pass inference
3. **Variable-Length Handling:** Adaptive padding strategy for M4's heterogeneous series

### 5.3 Training Pipeline

**Framework:** PyTorch 1.12+

**Optimization:**
- Optimizer: AdamW
- Learning Rate: 1×10⁻⁴ with cosine annealing
- Batch Size: 128
- Early Stopping: Patience of 5 epochs on validation loss
- Loss Function: MSE (Mean Squared Error)

**Data Split:**
- M4: Official train/test (no validation split, use last 20% of training for validation)
- Secondary: 70% train, 10% validation, 20% test

**Hardware:** Google Colab
- GPU: NVIDIA T4 (16 GB)
- CUDA 11.8
- Batch inference for latency measurement

### 5.4 ONNX Conversion

**Export Process:**
1. Load trained PyTorch checkpoint
2. Set model to evaluation mode
3. Define dynamic axes for variable batch size
4. Export using torch.onnx.export with opset_version=14
5. Verify model correctness with sample inputs

**Optimizations:**
- Operator Fusion: Combine consecutive operations
- Constant Folding: Pre-compute static operations
- Layout Optimization: Memory-efficient tensor formats

### 5.5 Quantization

**Method:** Post-Training Dynamic Quantization (ONNXRuntime)

**Configuration:**
- Precision: FP32 → INT8
- Quantization Type: Dynamic (weights INT8, activations FP32→INT8 at runtime)
- No calibration dataset required

**Rationale:** Dynamic quantization provides good compression without requiring representative data for calibration, suitable for diverse M4 series.

### 5.6 Evaluation Metrics

**M4 Competition Metrics:**
- **sMAPE:** Symmetric Mean Absolute Percentage Error (primary metric)
- **MASE:** Mean Absolute Scaled Error (baseline: naive forecast)
- **OWA:** Overall Weighted Average (official M4 ranking metric)

**Standard Metrics (LTSF benchmarks):**
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error

**Efficiency Metrics:**
- **Model Size:** MB for PyTorch (.pth), ONNX FP32 (.onnx), ONNX INT8 (.onnx)
- **Compression Ratio:** (FP32 size / INT8 size)
- **Inference Latency:** ms/batch (average over 100 runs, 10 warmup runs)
- **Accuracy Impact:** % degradation from PyTorch baseline

### 5.7 Baseline Comparison

**Primary Baseline:** N-BEATS
- Architecture: Doubly residual stacking with trend/seasonality decomposition
- Implementation: Official NeuralForecast library
- Configuration: As per original paper (Oreshkin et al., 2020)

**Comparison Dimensions:**
1. Forecasting accuracy (sMAPE, MASE, OWA)
2. Model complexity (parameter count)
3. Inference speed (ms/batch)
4. Training time (hours to convergence)

### 5.8 Experimental Design

**Phase 1: Baseline Evaluation (Weeks 1-3)**
- Implement PatchTST on secondary datasets (Weather, Traffic, etc.)
- Validate ONNX conversion pipeline
- Establish quantization methodology

**Phase 2: M4 Adaptation (Weeks 4-6)**
- Develop M4 data loader with variable-length handling
- Optimize architecture for M4 frequencies
- Integrate RevIN for distribution shift

**Phase 3: Optimization Pipeline (Weeks 7-9)**
- ONNX export and graph optimization
- Post-training INT8 quantization
- Latency benchmarking across backends

**Phase 4: Evaluation & Comparison (Weeks 10-12)**
- Comprehensive M4 evaluation across all frequencies
- N-BEATS baseline comparison
- Statistical significance testing

**Phase 5: Documentation & Reporting (Weeks 13-15)**
- Production code documentation
- Results analysis and visualization
- Final report preparation

## 6. Expected Outcomes

### 6.1 Quantitative Targets

1. **Model Compression:** Achieve 2× size reduction (FP32 → INT8) with <5% accuracy degradation
2. **Inference Latency:** <10ms per batch for real-time deployment
3. **M4 Performance:** Competitive OWA score (within 10% of N-BEATS baseline)
4. **Frequency Coverage:** Successful evaluation on 4+ M4 frequencies

### 6.2 Deliverables

1. **Production Codebase:**
   - Modular architecture (config, models, data, training, optimization, inference)
   - Comprehensive documentation and usage examples
   - Unit tests for critical components

2. **Trained Models:**
   - PyTorch checkpoints for all M4 frequencies
   - ONNX FP32 and INT8 variants
   - Reproducible training configurations

3. **Experimental Results:**
   - Weather dataset results (4 prediction horizons)
   - M4 Competition results (4+ frequencies)
   - Comparative analysis with N-BEATS
   - Ablation studies on architecture choices

4. **Research Documentation:**
   - Literature review (15+ references)
   - Methodology documentation
   - Final report with comprehensive analysis
   - Presentation slides

5. **Short Paper:**
   - 4-page conference-style paper
   - Title: "Efficient Real-Time Forecasting: Accelerating PatchTST on the M4 Benchmark"
   - Submission to workshop/conference (optional)

### 6.3 Expected Contributions

1. **Empirical Contribution:** First comprehensive evaluation of PatchTST on M4 Competition

2. **Methodological Contribution:** Framework for adapting LTSF models to variable-length, multi-horizon forecasting

3. **Practical Contribution:** Production-ready implementation of optimized PatchTST suitable for edge deployment

4. **Benchmarking Contribution:** Rigorous analysis of accuracy-efficiency trade-offs in Transformer-based forecasting

## 7. Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1-2 | Literature Review | Survey PatchTST, M4, N-BEATS, quantization papers | Literature review document |
| 3 | Baseline Setup | Implement PatchTST on Weather dataset | Working baseline code |
| 4 | Progress Report | Document methodology and initial results | Progress report submission |
| 5 | M4 Data Loader | Develop variable-length handling, padding strategy | M4 dataset module |
| 6 | Architecture Optimization | M4-specific hyperparameter tuning | Optimized configs |
| 7-8 | ONNX Pipeline | Export, verification, graph optimization | ONNX conversion module |
| 9 | Quantization | INT8 quantization, accuracy evaluation | Quantization module |
| 10 | M4 Experiments | Train and evaluate on Monthly, Quarterly, Weekly, Daily | M4 results |
| 11 | N-BEATS Baseline | Implement and compare with N-BEATS | Comparative analysis |
| 12 | Secondary Datasets | Complete Weather, Traffic experiments | Secondary results |
| 13 | Short Paper | Write 4-page conference paper | Short paper submission |
| 14-15 | Final Report | Comprehensive analysis and visualization | Final report draft |
| 16 | Presentation | Prepare slides and present findings | Final presentation |

### Milestone Checkpoints

- **Week 4:** Progress Evaluation Report (10%)
- **Week 6:** Literature Review (15%)
- **Week 9:** Methodology Implementation (20%)
- **Week 12:** Experimental Results (25%)
- **Week 15:** Final Report (25%)
- **Week 16:** Presentation (5%)

## 8. Resources Required

### 8.1 Hardware

**Primary Platform:** Google Colab
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: 12+ GB
- Storage: 50 GB for datasets and checkpoints

**Alternative:** Local machine with CUDA-capable GPU (optional)

### 8.2 Software & Frameworks

**Core Dependencies:**
- Python 3.8+
- PyTorch 1.12+ (with CUDA 11.8)
- ONNX 1.14+
- ONNXRuntime 1.15+ (GPU support)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)

**Development Tools:**
- Git for version control
- Jupyter Notebook for experimentation
- VS Code / PyCharm for development

### 8.3 Datasets

**Primary:**
- M4 Competition dataset (publicly available)
  - Source: https://github.com/Mcompetitions/M4-methods
  - Size: ~50 MB compressed

**Secondary:**
- Weather, Traffic, Electricity, ETT datasets
  - Source: https://github.com/yuqinie98/PatchTST
  - Size: ~500 MB total

### 8.4 Baseline Models

- PatchTST implementation: https://github.com/yuqinie98/PatchTST
- N-BEATS implementation: NeuralForecast library

### 8.5 Computational Budget

- Training time per model: 2-4 hours (M4 Monthly)
- Total experiments: ~20 model variants
- Estimated compute: 80-100 GPU hours (feasible within Colab limits)

## 9. Risk Analysis & Mitigation

### 9.1 Technical Risks

**Risk 1: ONNX Conversion Failures**
- Mitigation: Implement ONNX-compatible operations from the start (manual patching, no unfold)

**Risk 2: Quantization Accuracy Degradation >5%**
- Mitigation: Explore static quantization with calibration data if dynamic quantization insufficient

**Risk 3: Variable-Length Handling Complexity**
- Mitigation: Comprehensive padding strategy with attention masking

### 9.2 Resource Risks

**Risk 1: Colab Compute Limits**
- Mitigation: Optimize training efficiency, use checkpointing, consider Colab Pro if needed

**Risk 2: M4 Dataset Size (100,000 series)**
- Mitigation: Focus on Monthly (48,000) as primary, sample other frequencies if needed

### 9.3 Timeline Risks

**Risk 1: Implementation Delays**
- Mitigation: Prioritize core functionality, defer optional features (Yearly, Hourly frequencies)

**Risk 2: N-BEATS Baseline Unavailable**
- Mitigation: Use published results if implementation infeasible

## 10. Evaluation Criteria

### 10.1 Success Metrics

**Minimum Viable Success:**
1. PatchTST successfully trained on M4 Monthly with sMAPE < 15.0
2. ONNX conversion with <1% accuracy loss
3. INT8 quantization with 1.5×+ compression and <5% accuracy loss
4. Complete production codebase with documentation

**Target Success:**
1. M4 Monthly sMAPE competitive with N-BEATS (within 10%)
2. 2× compression with <3% accuracy degradation
3. Inference latency <10ms per batch
4. 4+ M4 frequencies evaluated

**Stretch Goals:**
1. M4 Monthly sMAPE better than N-BEATS
2. INT8 quantization improves accuracy (observed in preliminary results)
3. Sub-5ms inference on CPU
4. All 6 M4 frequencies evaluated

### 10.2 Quality Standards

- Code: PEP 8 compliant, >80% test coverage for core modules
- Documentation: Comprehensive README, docstrings, usage examples
- Reproducibility: Fixed seeds, documented configurations, version pinning
- Experimentation: Ablation studies, statistical significance testing

## References

1. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

2. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. *International Conference on Learning Representations (ICLR)*.

3. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. *International Journal of Forecasting*, 34(4), 802-808.

4. Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *International Conference on Learning Representations (ICLR)*.

5. Kim, T., Kim, J., Tae, Y., et al. (2022). Reversible instance normalization for accurate time-series forecasting against distribution shift. *International Conference on Learning Representations (ICLR)*.

6. Jacob, B., Kligys, S., Chen, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2704-2713.

7. Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

8. Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34, 22419-22430.

9. Zhou, T., Ma, Z., Wen, Q., et al. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. *International Conference on Machine Learning (ICML)*.

10. Microsoft. (2019). ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator. Retrieved from https://onnxruntime.ai

11. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

12. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: Forecasting and control* (5th ed.). John Wiley & Sons.

13. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts.

14. Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: A survey. *Philosophical Transactions of the Royal Society A*, 379(2194), 20200209.

15. Cheng, Y., Wang, D., Zhou, P., & Zhang, T. (2020). A survey of model compression and acceleration for deep neural networks. *IEEE Signal Processing Magazine*, 37(5), 35-49.

---

**Submission Date:** Week 3
**Approved By:** [Supervisor Signature]
**Status:** Approved for Implementation
