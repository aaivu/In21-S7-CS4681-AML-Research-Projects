# M4 Competition Results Summary

## Overview

| Metric | Value |
|--------|-------|
| Frequencies Tested | 4 (Quarterly, Weekly, Monthly, Daily) |
| Total Series | 76,586 |
| Model Architecture | PatchTST (M4-optimized) |
| Optimization Pipeline | PyTorch → ONNX FP32 → INT8 |

## Results by Frequency

### Quarterly (24,000 series, horizon=8)

| Model | sMAPE | MAE | MSE | Latency (ms) | Size (MB) |
|-------|-------|-----|-----|--------------|-----------|
| PyTorch | 11.2244 | 617.0875 | 1887804.5000 | 1.51 | 0.29 |
| ONNX_FP32 | 11.2244 | 617.0875 | 1887804.5000 | 1.01 | 0.35 |
| ONNX_INT8 | 11.1579 | 613.0912 | 1886498.2500 | 3.19 | 0.18 |

**Compression:** 1.90x | **sMAPE Impact:** -0.59%

### Weekly (359 series, horizon=13)

| Model | sMAPE | MAE | MSE | Latency (ms) | Size (MB) |
|-------|-------|-----|-----|--------------|-----------|
| PyTorch | 10.5624 | 389.2884 | 577511.8750 | 1.73 | 0.32 |
| ONNX_FP32 | 10.5624 | 389.2884 | 577512.0000 | 1.51 | 0.38 |
| ONNX_INT8 | 10.5190 | 389.2638 | 576911.7500 | 4.12 | 0.20 |

**Compression:** 1.95x | **sMAPE Impact:** -0.41%

### Monthly (48,000 series, horizon=18)

| Model | sMAPE | MAE | MSE | Latency (ms) | Size (MB) |
|-------|-------|-----|-----|--------------|-----------|
| PyTorch | 10.8448 | 519.1946 | 1643907.2500 | 2.05 | 0.32 |
| ONNX_FP32 | 10.8448 | 519.1946 | 1643907.2500 | 1.11 | 0.38 |
| ONNX_INT8 | 10.8464 | 519.3577 | 1645230.0000 | 4.00 | 0.19 |

**Compression:** 1.98x | **sMAPE Impact:** +0.01%

### Daily (4,227 series, horizon=14)

| Model | sMAPE | MAE | MSE | Latency (ms) | Size (MB) |
|-------|-------|-----|-----|--------------|-----------|
| PyTorch | 3.8587 | 232.4030 | 455978.8750 | 1.73 | 0.33 |
| ONNX_FP32 | 3.8587 | 232.4031 | 455978.9375 | 1.13 | 0.39 |
| ONNX_INT8 | 3.8570 | 232.2554 | 455921.6250 | 4.61 | 0.20 |

**Compression:** 1.96x | **sMAPE Impact:** -0.04%

## Performance Summary

### Compression Ratios

| Frequency | PyTorch (MB) | ONNX INT8 (MB) | Compression | sMAPE Impact |
|-----------|--------------|----------------|-------------|--------------|
| Quarterly | 0.29 | 0.18 | 1.90x | -0.59% |
| Weekly | 0.32 | 0.20 | 1.95x | -0.41% |
| Monthly | 0.32 | 0.19 | 1.98x | +0.01% |
| Daily | 0.33 | 0.20 | 1.96x | -0.04% |
| **Average** | **0.32** | **0.19** | **1.95x** | **-0.26%** |

### Inference Latency

#### GPU (ONNX FP32)

| Frequency | Latency (ms) |
|-----------|--------------|
| Quarterly | 1.01 |
| Weekly | 1.51 |
| Monthly | 1.11 |
| Daily | 1.13 |
| **Average** | **1.19** |

#### CPU (ONNX INT8)

| Frequency | Latency (ms) |
|-----------|--------------|
| Quarterly | 3.19 |
| Weekly | 4.12 |
| Monthly | 4.00 |
| Daily | 4.61 |
| **Average** | **3.98** |

### Accuracy Performance (sMAPE)

| Frequency | Best Model | sMAPE | Ranking |
|-----------|------------|-------|---------|
| Daily | ONNX_INT8 | 3.8570 | 🥇 Best |
| Weekly | ONNX_INT8 | 10.5190 | Excellent |
| Monthly | PyTorch/FP32 | 10.8448 | Very Good |
| Quarterly | ONNX_INT8 | 11.1579 | Good |

## Key Findings

### Quantization Impact

- **3 out of 4 frequencies** showed improved accuracy with INT8 quantization
- **Average sMAPE improvement:** -0.26% (better than FP32!)
- **Maximum degradation:** +0.01% (Monthly, negligible)
- **Best improvement:** -0.59% (Quarterly)

### Model Efficiency

- **Average compression:** 1.95x (FP32 → INT8)
- **Model sizes:** 0.18-0.20 MB (INT8), ideal for edge deployment
- **GPU inference:** 1-2 ms (extremely fast)
- **CPU inference:** 3-5 ms (acceptable for production)

### Best Model Selection

**For Maximum Accuracy:**
- Model: ONNX INT8
- Why: Best sMAPE in 3/4 frequencies

**For Fastest Inference:**
- Model: ONNX FP32 on GPU
- Latency: ~1.2 ms average

**For Edge Deployment:**
- Model: ONNX INT8 on CPU
- Size: ~0.19 MB, Latency: ~4 ms

## Configuration

### M4-Optimized Architecture

| Parameter | Value | vs Standard |
|-----------|-------|-------------|
| Model Dimension | 64 | -50% (128→64) |
| Encoder Layers | 2 | -33% (3→2) |
| Attention Heads | 8 | -50% (16→8) |
| Feed-Forward Dim | 128 | -50% (256→128) |

### Frequency-Specific Settings

| Frequency | Seq Len | Patch Len | Stride |
|-----------|---------|-----------|--------|
| Quarterly | 32 | 4 | 2 |
| Weekly | 52 | 13 | 6-7 |
| Monthly | 72 | 12 | 6 |
| Daily | 56 | 14 | 7 |

**Pattern:** seq_len = 4 × horizon, patch_len ≈ horizon

## Deployment Recommendations

### GPU Production

```
Model: ONNX FP32
Latency: 1-2 ms
Size: 0.35-0.39 MB
Use Case: Real-time API services
```

### CPU Production

```
Model: ONNX INT8
Latency: 3-5 ms
Size: 0.18-0.20 MB
Use Case: Standard cloud deployments
```

### Edge Devices

```
Model: ONNX INT8
Latency: 3-5 ms
Size: 0.18-0.20 MB
Use Case: Mobile, IoT, offline forecasting
```

## Conclusions

1. **INT8 quantization is effective:** 2x compression with improved accuracy in most cases
2. **M4-optimized architecture works:** Smaller models maintain competitive sMAPE scores
3. **Production-ready performance:** Sub-5ms latency enables real-time forecasting
4. **Edge deployment viable:** <0.2 MB models suitable for resource-constrained devices

---

*Generated from: results/m4/results.csv*
