# Enhanced nnFormer for Brain Tumor Segmentation - 210353V

## Student Information

- **Index Number:** 210353V
- **Name:** Lakshan Madusanka
- **Research Area:** Healthcare AI - Medical Image Segmentation
- **GitHub Username:** @lakshan1946
- **Email:** Lakshan.21@cse.mrt.ac.lk
- **Supervisor:** Dr. Uthayasanker Thayasivam
- **Institution:** University of Moratuwa, Department of Computer Science and Engineering

## 🎯 Project Overview

This research project focuses on **enhancing the nnFormer architecture with multi-scale cross-attention mechanisms** for improved **3D brain tumor segmentation** on the BraTS 2021 dataset. The project investigates novel attention mechanisms, adaptive feature fusion strategies, and progressive training techniques to achieve state-of-the-art performance in medical image segmentation.

### Key Research Contributions

1. **Multi-Scale Cross-Attention Framework**: Bidirectional feature interaction between encoder stages
2. **Adaptive Feature Fusion**: Channel-wise and spatial attention for optimal multi-scale integration
3. **Progressive Training Strategy**: Gradual activation of cross-attention mechanisms
4. **Comprehensive Evaluation**: Extensive experiments on BraTS 2021 dataset with ablation studies

## 📁 Project Structure

```
210353V-Healthcare-AI_Medical-Segmentation/
├── README.md                           # This file
├── requirements.txt                    # Complete project dependencies
│
├── docs/                              # Research documentation
│   ├── research_proposal.md           # ✅ Complete research proposal
│   ├── literature_review.md           # ✅ Comprehensive literature review
│   ├── methodology.md                 # ✅ Detailed methodology
│   ├── usage_instructions.md          # ✅ Complete usage guide
│   └── progress_reports/              # Weekly progress tracking
│
├── src/                               # Source code
│   └── nnformer/                      # Enhanced nnFormer implementation
│
├── data/                              # Dataset management
│   ├── README.md                      # Data documentation
│
├── experiments/                       # Experiment configurations
│   ├── README.md                      # Experiment documentation
│   ├── baseline/                      # Baseline experiments
│   ├── enhanced/                      # Enhanced model experiments
│   └── ablation/                      # Ablation studies
│
└── results/                           # Experimental results
    ├── README.md                      # Results documentation
    ├── baseline/                      # Baseline results
    ├── enhanced/                      # Enhanced model results
    └── visualizations/                # Result visualizations
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project
cd 210353V-Healthcare-AI_Medical-Segmentation

# Create conda environment
cd src/nnformer
conda env create -f environment.yml
conda activate nnFormer

# Install nnFormer package
pip install -e .
```

### 2. Dataset Preparation

```bash
# Set environment variables (Windows PowerShell)
$env:nnFormer_raw_data_base = "D:\DATASET\nnFormer_raw\nnFormer_raw_data"
$env:nnFormer_preprocessed = "D:\DATASET\nnFormer_preprocessed"
$env:RESULTS_FOLDER = "D:\DATASET\nnFormer_trained_models"

# Download BraTS 2021 dataset (requires registration)
# Follow instructions in data/README.md

# Preprocess dataset
nnFormer_convert_decathlon_task -i $env:nnFormer_raw_data_base/Task120_BraTS2021
nnFormer_plan_and_preprocess -t 120
```

### 3. Training

```bash
# Train baseline nnFormer
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_nnformer_tumor 120 0

# Train enhanced nnFormer
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_enhanced 120 0
```

### 4. Evaluation

```bash
# Run inference
python -m nnformer.inference_tumor -t 120 -f 0 -chk model_best

# Evaluate with BraTS metrics
python -m nnformer.evaluation.evaluate_brats \
    --pred_dir results/enhanced/predictions/ \
    --gt_dir $env:nnFormer_raw_data_base/Task120_BraTS2021/labelsTr/
```

## 📊 Research Milestones

- [x] **Week 1-2:** Literature review and research proposal
- [x] **Week 3-4:** Environment setup and baseline implementation
- [x] **Week 5-6:** Enhanced architecture implementation
- [x] **Week 7-8:** Initial experiments and debugging
- [x] **Week 9-10:** Comprehensive experiments and ablation studies

## 🔬 Research Objectives

### Primary Objective

Develop and validate an enhanced nnFormer architecture with multi-scale cross-attention mechanisms that achieves superior performance on BraTS 2021 brain tumor segmentation compared to baseline methods.

### Secondary Objectives

1. Design efficient multi-scale cross-attention modules for 3D medical imaging
2. Implement adaptive feature fusion with dual attention mechanisms
3. Develop progressive training strategy for stable convergence
4. Conduct comprehensive ablation studies to validate each component
5. Achieve state-of-the-art results on BraTS 2021 benchmark

## 📊 Expected Outcomes

| Method                       | Dice ET   | Dice TC   | Dice WT   | HD95 WT     |
| ---------------------------- | --------- | --------- | --------- | ----------- |
| nnU-Net                      | 0.682     | 0.741     | 0.847     | 18.5 mm     |
| nnFormer (baseline)          | 0.703     | 0.761     | 0.863     | 16.5 mm     |
| **Enhanced nnFormer (ours)** | **0.737** | **0.785** | **0.884** | **13.6 mm** |

## 💻 System Requirements

### Hardware

- GPU: NVIDIA RTX 3090 / A100 (16GB+ VRAM)
- CPU: 16+ cores recommended
- RAM: 64GB+ recommended
- Storage: 500GB+ SSD

### Software

- Windows 11 / Ubuntu 20.04
- Python 3.6-3.8
- PyTorch 1.8.1+
- CUDA 10.1+
