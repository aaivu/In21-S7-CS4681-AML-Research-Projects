# Data Management - BraTS 2021 Dataset

**Project:** 210353V - Enhanced nnFormer for Brain Tumor Segmentation  
**Dataset:** BraTS 2021 (Brain Tumor Segmentation Challenge 2021)  
**Last Updated:** October 22, 2025

---

## Dataset Overview

### BraTS 2021 Statistics

| Split      | Cases | Modalities          | Annotations |
| ---------- | ----- | ------------------- | ----------- |
| Training   | 1,251 | T1, T1ce, T2, FLAIR | ET, TC, WT  |
| Validation | 219   | T1, T1ce, T2, FLAIR | ET, TC, WT  |
| Test       | 530   | T1, T1ce, T2, FLAIR | Hidden      |

### MRI Modalities

1. **T1-weighted (T1)**: Provides anatomical context
2. **T1-weighted with Contrast Enhancement (T1ce)**: Highlights blood-brain barrier disruption
3. **T2-weighted (T2)**: Shows total tumor extent
4. **Fluid Attenuated Inversion Recovery (FLAIR)**: Reveals peritumoral edema

### Tumor Regions

- **ET (Enhancing Tumor)**: Label 3 - Active tumor with contrast enhancement
- **NCR/NET (Non-Enhancing Tumor/Necrosis)**: Label 2 - Non-enhancing components
- **ED (Peritumoral Edema)**: Label 1 - Surrounding edema
- **TC (Tumor Core)**: ET + NCR/NET (Labels 2+3)
- **WT (Whole Tumor)**: TC + ED (Labels 1+2+3)

---

## Preprocessing Pipeline

### Environment Setup (Windows PowerShell)

```powershell
# Set environment variables
$env:nnFormer_raw_data_base = "D:\DATASET\nnFormer_raw\nnFormer_raw_data"
$env:nnFormer_preprocessed = "D:\DATASET\nnFormer_preprocessed"
$env:RESULTS_FOLDER = "D:\DATASET\nnFormer_trained_models"
```

### Step 1: Convert to nnFormer Format

```bash
nnFormer_convert_decathlon_task -i $env:nnFormer_raw_data_base/Task120_BraTS2021
```

This command:

- Validates dataset structure
- Creates dataset.json
- Organizes files according to nnFormer convention

### Step 2: Plan and Preprocess

```bash
nnFormer_plan_and_preprocess -t 120 --verify_dataset_integrity
```

This command:

- Analyzes dataset statistics
- Determines optimal patch size
- Computes normalization parameters
- Creates preprocessing plan
- Executes preprocessing pipeline

**Expected Duration:** 2-4 hours (depending on hardware)

---

## Preprocessing Steps Details

### 1. Intensity Normalization

For each modality independently:

```python
normalized = (image - mean) / std
```

### 2. Resampling

- Target spacing: 1mm × 1mm × 1mm isotropic
- Interpolation: Linear for images, nearest-neighbor for labels

### 3. Cropping

- Remove background voxels
- Crop to foreground bounding box + margin
- Reduces memory and computation

### 4. Data Splits

**Training Split (80%):**

- ~1,001 cases for training
- Used for model optimization

**Validation Split (20%):**

- ~250 cases for validation
- Used for hyperparameter tuning and model selection

**Official Validation Set:**

- 219 cases
- Used for final performance evaluation

---

## Data Augmentation

Applied during training (online):

### Spatial Transformations

- **Elastic Deformation**: σ=30, α=900
- **Random Scaling**: 0.7-1.4×
- **Random Rotation**: ±15° on all axes
- **Random Flipping**: Along x, y, z axes

### Intensity Transformations

- **Brightness Adjustment**: 0.7-1.3×
- **Contrast Adjustment**: 0.65-1.5×
- **Gamma Correction**: γ ∈ [0.7, 1.5]
- **Gaussian Noise**: σ ∈ [0, 0.1]

### Patch Extraction

- **Patch Size**: [64, 128, 128] voxels
- **Sampling Strategy**: Foreground-biased
- **Overlap**: 50% during inference

---

## Data Statistics

### Image Dimensions

- **Original Size**: 240 × 240 × 155 voxels
- **Preprocessed Size**: Variable (after cropping)
- **Patch Size (Training)**: 64 × 128 × 128 voxels

### Intensity Ranges (after normalization)

- **Mean**: 0.0
- **Std**: 1.0
- **Typical Range**: [-3, +3]

### Class Distribution

| Region | Avg Volume (cm³) | % of WT | Frequency  |
| ------ | ---------------- | ------- | ---------- |
| WT     | ~100-150         | 100%    | All cases  |
| TC     | ~40-60           | ~50%    | All cases  |
| ET     | ~10-20           | ~15%    | ~95% cases |

**Note:** High class imbalance, especially for ET regions.

---

## Storage Requirements

### Disk Space

- **Raw Data**: ~50 GB
- **Preprocessed Data**: ~80 GB
- **Training Checkpoints**: ~20 GB
- **Results and Predictions**: ~10 GB
- **Total Recommended**: 200 GB minimum

---

## Data Loading

### During Training

```python
# Handled automatically by nnFormer
DataLoader:
    - Patch extraction: [64, 128, 128]
    - Batch size: 2
    - Augmentation: On-the-fly
    - Normalization: Pre-computed stats
    - Background sampling: 33%
```

### During Inference

```python
# Sliding window approach
Inference:
    - Full volume processing
    - Overlap: 50%
    - Batch size: 1 (due to memory)
    - Test-time augmentation: Optional
```

---
