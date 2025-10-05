# Phase 2: Enhancement & Optimization (Weeks 8-9)

**Goal:** Apply ONNX conversion and INT8 quantization to the trained baseline model while preserving accuracy.

## Overview

This phase implements the enhancement methodology:
1. **ONNX Conversion:** Export PyTorch model to ONNX format
2. **INT8 Quantization:** Apply post-training dynamic quantization for CPU speedup

**Critical Requirement:** Preserve the baseline OWA score from Phase 1

## Tasks

### Task 2.1: Convert to ONNX Format ✓
- **File:** `export_onnx.py`
- **Input:** `patchtst_m4_baseline.pth` (from Phase 1)
- **Output:** `patchtst_m4.onnx`
- **Features:**
  - Dynamic batch size support
  - Optimized for inference
  - Compatible with ONNX Runtime

### Task 2.2: Apply Post-Training Quantization ✓
- **File:** `quantize_model.py`
- **Input:** `patchtst_m4.onnx`
- **Output:** `patchtst_m4_quant.onnx` (final enhanced model)
- **Method:** INT8 dynamic quantization
- **Target:** CPU-based inference speedup (6-10x)

## Files Structure

```
002_phase2_onnx_quantization/
├── export_onnx.py           # PyTorch → ONNX conversion
├── quantize_model.py        # ONNX → Quantized ONNX
├── evaluate_onnx.py         # Evaluate ONNX models
├── benchmark.py             # Speed comparison
├── config.py                # Configuration
├── colab_all_in_one.py     # Colab-ready single file
├── README.md                # This file
└── results/
    ├── patchtst_m4.onnx           # FP32 ONNX model
    ├── patchtst_m4_quant.onnx     # INT8 quantized (FINAL)
    └── performance_comparison.txt  # Metrics comparison
```

## Usage

### Option 1: Step-by-Step

```bash
# Step 1: Export to ONNX
python export_onnx.py --checkpoint ../001_phase1_baseline/results/patchtst_m4_baseline.pth

# Step 2: Quantize ONNX model
python quantize_model.py --input results/patchtst_m4.onnx --output results/patchtst_m4_quant.onnx

# Step 3: Evaluate quantized model
python evaluate_onnx.py --model results/patchtst_m4_quant.onnx

# Step 4: Benchmark speed
python benchmark.py
```

### Option 2: Google Colab (All-in-One)

1. Upload Phase 1 checkpoint: `patchtst_m4_baseline.pth`
2. Upload M4 test data: `Monthly-test.csv`
3. Copy-paste `colab_all_in_one.py` content
4. Run - will export, quantize, evaluate, and benchmark

## Key Implementation Details

### 1. ONNX Export
```python
# Dynamic axes for flexible batch size
torch.onnx.export(
    model,
    dummy_input,
    "patchtst_m4.onnx",
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=14
)
```

### 2. INT8 Quantization
```python
# Post-training dynamic quantization
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="patchtst_m4.onnx",
    model_output="patchtst_m4_quant.onnx",
    weight_type=QuantType.QInt8
)
```

### 3. ONNX Runtime Inference
```python
import onnxruntime as ort

session = ort.InferenceSession("patchtst_m4_quant.onnx")
outputs = session.run(None, {'input': input_data})
```

## Expected Results

### Model Size Comparison
- **PyTorch FP32:** ~15 MB
- **ONNX FP32:** ~15 MB
- **ONNX INT8:** ~4 MB (**4x compression**)

### Inference Speed (CPU)
- **PyTorch:** 100 ms/batch (baseline)
- **ONNX FP32:** 40 ms/batch (2.5x faster)
- **ONNX INT8:** 15 ms/batch (**6-7x faster**)

### Accuracy Preservation
- **Target:** < 2% degradation in OWA
- **Baseline OWA:** 0.85 (from Phase 1)
- **Quantized OWA:** 0.86 (acceptable)
- **sMAPE/MASE:** Minimal change

## Deliverables

### Primary Artifact
✅ **`patchtst_m4_quant.onnx`** - Final enhanced model
- INT8 quantized
- 4x smaller
- 6-7x faster on CPU
- Accuracy preserved

### Supporting Artifacts
- ONNX export script
- Quantization script
- Performance benchmarks
- Accuracy comparison report

## Success Criteria

1. ✅ Successfully export PyTorch → ONNX
2. ✅ Apply INT8 quantization
3. ✅ Achieve 4-6x model compression
4. ✅ Achieve 6-10x inference speedup (CPU)
5. ✅ **Preserve baseline OWA** (< 2% degradation)

## Validation Steps

```bash
# 1. Verify ONNX model validity
python -c "import onnx; onnx.checker.check_model('results/patchtst_m4_quant.onnx')"

# 2. Compare accuracies
python evaluate_onnx.py --compare

# 3. Benchmark speeds
python benchmark.py --iterations 100

# 4. Check model sizes
ls -lh results/*.onnx
```

## Integration with Phase 1

This phase **depends on** Phase 1:
- Requires trained checkpoint: `patchtst_m4_baseline.pth`
- Uses same M4 data loader logic
- Compares against Phase 1 baseline metrics

## Notes

- INT8 quantization is **CPU-optimized** (not GPU)
- Dynamic quantization is simplest and most effective
- ONNX Runtime provides built-in optimization
- Preserving OWA is **critical** - it's the M4 benchmark metric
