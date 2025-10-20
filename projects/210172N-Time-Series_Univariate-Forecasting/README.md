# Efficient Real-Time Forecasting: PatchTST on M4 Benchmark

## Student Information

- **Index Number:** 210172N
- **Name:** Galappaththi A. S.
- **Research Area:** Time Series Univariate Forecasting
- **GitHub Username:** @anush47
- **Email:** sharadag.21@cse.mrt.ac.lk
- **Course:** CS4681 - Advanced Machine Learning Research
- **Institution:** University of Moratuwa

## Project Overview

**Title:** Adapting PatchTST for Real-Time, Multi-Horizon Forecasting on the M4 Competition Benchmark

**Baseline Model:** PatchTST (A Time Series is Worth 64 Words - ICLR 2023)

**Research Focus:** This project investigates the application of PatchTST, a state-of-the-art Transformer-based model for Long-Term Time Series Forecasting (LTSF), to the M4 Competition benchmark with real-time optimization through ONNX conversion and INT8 quantization.

**Key Objectives:**

1. Baseline evaluation of PatchTST on M4 Competition
2. Model architecture adaptation for M4 benchmark
3. Real-time enhancement via ONNX Runtime and quantization
4. Rigorous benchmarking using M4 metrics (sMAPE, MASE, OWA)

**Comparison Baseline:** N-BEATS model

## Current Status

**Phase:** Evaluation & Analysis (Weeks 7-11) - IMPLEMENTATION COMPLETE

### Completed Milestones

- **Week 4:** Research Proposal & Progress Report
- **Week 5:** Literature Review
- **Week 6-7:** Short Paper & Production Implementation
- **Week 8:** Weather Dataset Experiments Complete
- **Week 9:** M4 Competition Experiments Complete

### Key Achievements

**Production Code:**

- Complete production-ready implementation in `src/`
- Modular architecture with 9 packages, 26 modules
- ONNX-compatible PatchTST model
- M4 and standard dataset loaders
- Training, evaluation, optimization pipeline
- Comprehensive configuration system

**Experimental Results:**

- Weather dataset: 4 prediction horizons (96, 192, 336, 720)
- M4 Competition: 4 frequencies (Quarterly, Weekly, Monthly, Daily)
- PyTorch → ONNX FP32 → INT8 quantization pipeline
- Compression: 1.95x-3.60x average
- Minimal accuracy degradation: -0.26% to +1.44%

## Project Structure

```
210172N-Time-Series_Univariate-Forecasting/
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
│
├── src/                                      # Production source code
│   ├── config/                               # Configuration management
│   │   ├── base_config.py                    # Base configuration
│   │   ├── m4_config.py                      # M4-specific config
│   │   └── standard_config.py                # Standard datasets config
│   ├── models/                               # Model implementations
│   │   ├── patchtst.py                       # ONNX-compatible PatchTST
│   │   └── revin.py                          # Reversible normalization
│   ├── data/                                 # Data loaders
│   │   ├── m4_dataset.py                     # M4 Competition loader
│   │   └── standard_dataset.py               # Standard benchmark loader
│   ├── training/                             # Training pipeline
│   │   └── trainer.py                        # Trainer with early stopping
│   ├── evaluation/                           # Evaluation modules
│   │   ├── evaluator.py                      # Model evaluator
│   │   └── metrics.py                        # M4 & standard metrics
│   ├── optimization/                         # Model optimization
│   │   ├── onnx_export.py                    # ONNX conversion
│   │   └── quantization.py                   # INT8 quantization
│   ├── inference/                            # Inference engine
│   │   └── predictor.py                      # Unified predictor
│   ├── utils/                                # Utilities
│   │   ├── helpers.py                        # Helper functions
│   │   ├── checkpoint.py                     # Checkpoint management
│   │   └── logger.py                         # Logging setup
│   └── examples/                             # Example scripts
│       ├── example_m4_training.py            # M4 pipeline example
│       └── example_standard_training.py      # Standard training example
│
├── docs/                                     # Research documentation
│   ├── research_proposal.md                  # Research proposal
│   ├── literature_review.md                  # Literature review
│   ├── methodology.md                        # Detailed methodology
│   ├── usage_instructions.md                 # Production code usage
│   └── progress_reports/                     # Progress reports
│       └── progress_report.md
│
├── data/                                     # Datasets (gitignored)
│   ├── m4/                                   # M4 Competition data
│   ├── secondary/                            # Secondary benchmarks
│   │   ├── weather/                          # Weather dataset
│   │   ├── traffic/                          # Traffic dataset
│   │   ├── electricity/                      # Electricity dataset
│   │   └── ...
│   └── PatchTST/                            # PatchTST baseline
│
├── experiments/                              # Experiment scripts
│   ├── 003_unified_full_pipeline.py          # Weather experiments
│   ├── 004_m4_baseline_with_optimization.py  # M4 baseline
│   └── 004_m4_full_pipeline.py              # M4 full pipeline
│
└── results/                                  # Experimental results
    ├── README.md                             # Results overview
    ├── weather/                              # Weather results
    │   ├── pred_96_results.txt
    │   ├── pred_192_results.txt
    │   ├── pred_336_results.txt
    │   ├── pred_720_results.txt
    │   ├── results.csv
    │   └── summary.txt
    └── m4/                                   # M4 Competition results
        ├── quarterly_results.txt
        ├── weekly_results.txt
        ├── monthly_results.txt
        ├── daily_results.txt
        ├── results.csv
        └── summary.md
```

## Quick Start

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**

- PyTorch 1.12+
- ONNX 1.14+
- ONNXRuntime (with CUDA support)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### Running Experiments

**M4 Competition Training:**

```python
from src.examples.example_m4_training import run_m4_pipeline

# Run complete M4 pipeline: train → ONNX → INT8 → evaluate
run_m4_pipeline(frequency='Monthly')
```

**Standard Dataset Training:**

```python
from src.examples.example_standard_training import run_standard_pipeline

# Run Weather dataset pipeline
run_standard_pipeline(dataset='weather', pred_len=96)
```

**For detailed usage, see:** `docs/usage_instructions.md`

## Key Results

### Weather Dataset

| Horizon | Compression | MAE Impact | Best Model |
| ------- | ----------- | ---------- | ---------- |
| 96      | 3.33x       | -0.36%     | ONNX INT8  |
| 192     | 3.54x       | +3.71%     | PyTorch    |
| 336     | 3.68x       | +2.28%     | PyTorch    |
| 720     | 3.83x       | +0.13%     | ONNX INT8  |
| **Avg** | **3.60x**   | **+1.44%** | -          |

### M4 Competition

| Frequency | Series | Horizon | Compression | sMAPE Impact | Best Model |
| --------- | ------ | ------- | ----------- | ------------ | ---------- |
| Quarterly | 24,000 | 8       | 1.90x       | -0.59%       | ONNX INT8  |
| Weekly    | 359    | 13      | 1.95x       | -0.41%       | ONNX INT8  |
| Monthly   | 48,000 | 18      | 1.98x       | +0.01%       | PyTorch    |
| Daily     | 4,227  | 14      | 1.96x       | -0.04%       | ONNX INT8  |
| **Avg**   | -      | -       | **1.95x**   | **-0.26%**   | -          |

**Key Findings:**

- INT8 quantization improved accuracy in 3 out of 4 M4 frequencies
- Sub-5ms inference latency enables real-time forecasting
- Model sizes <0.2 MB suitable for edge deployment
- Production-ready performance achieved

## Documentation

### Research Documents

- **Research Proposal:** `docs/research_proposal.md`
- **Literature Review:** `docs/literature_review.md`
- **Methodology:** `docs/methodology.md`
- **Usage Instructions:** `docs/usage_instructions.md`

### Example Scripts

- **M4 Training:** `src/examples/example_m4_training.py`
- **Standard Training:** `src/examples/example_standard_training.py`

## Technical Implementation

**PatchTST Architecture:**

- Model Dimension: 128 (standard) / 64 (M4-optimized)
- Encoder Layers: 3 (standard) / 2 (M4-optimized)
- Attention Heads: 16 (standard) / 8 (M4-optimized)
- Patch Length: 16 (standard) / frequency-adaptive (M4)
- Channel-Independent Processing

**Optimization Pipeline:**

1. PyTorch training with early stopping
2. ONNX FP32 export with graph optimizations
3. Post-training dynamic INT8 quantization
4. Evaluation across all model variants

**M4 Competition Metrics:**

- sMAPE (Symmetric Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)
- OWA (Overall Weighted Average)

## Next Steps

- [ ] Complete remaining M4 frequencies (Yearly, Hourly)
- [ ] N-BEATS baseline comparison
- [ ] Static quantization with calibration

## Academic Integrity

- All code is original work

## References

For detailed references, see:

- `docs/literature_review.md` - Comprehensive literature review
- `docs/research_proposal.md` - Research proposal and citations
- `docs/methodology.md` - Methodology and technical references

---
