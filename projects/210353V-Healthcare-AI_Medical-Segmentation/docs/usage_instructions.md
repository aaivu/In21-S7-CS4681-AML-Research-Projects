# Usage Instructions: Enhanced nnFormer for Brain Tumor Segmentation

**Project:** 210353V - Enhanced nnFormer  
**Student:** Lakshan Madusanka  
**Last Updated:** October 22, 2025

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Running Experiments](#4-running-experiments)

---

## 1. System Requirements

### Hardware Requirements

**Minimum:**

- GPU: NVIDIA GPU with 16GB VRAM (e.g., Tesla V100 16GB)
- CPU: 8 cores
- RAM: 32 GB
- Storage: 300 GB SSD

**Recommended:**

- GPU: NVIDIA A100 40GB or V100 32GB
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 500 GB NVMe SSD

**Estimated Training Time:**

- **Baseline (1 fold):** 6-7 days on V100
- **Enhanced (1 fold):** 7-8 days on V100
- **Full 5-fold CV:** ~35-40 days on single V100

### Software Requirements

- **OS:** Ubuntu 20.04 LTS (or compatible Linux)
- **CUDA:** 11.3 or higher
- **cuDNN:** 8.2.0 or higher
- **Python:** 3.8
- **Conda:** Anaconda or Miniconda

---

## 2. Environment Setup

### Step 1: Clone the Repository

```bash
# Navigate to project directory
cd /path/to/210353V-Healthcare-AI_Medical-Segmentation

# Ensure src/nnformer is present
ls src/nnformer
```

### Step 2: Create Conda Environment

```bash
# Create environment from YAML file
cd src/nnformer
conda env create -f environment.yml

# Activate environment
conda activate nnformer

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected Output:**

```
PyTorch: 1.11.0+cu113
CUDA: True
```

### Step 3: Install nnFormer Package

```bash
# Install in development mode
cd src/nnformer
pip install -e .

# Verify installation
python -c "import nnformer; print('nnFormer installed successfully')"
```

### Step 4: Set Environment Variables

Create a file `~/.bashrc` or `~/.zshrc` and add:

```bash
# nnFormer paths
export nnFormer_raw_data_base="/path/to/data/raw"
export nnFormer_preprocessed="/path/to/data/preprocessed"
export RESULTS_FOLDER="/path/to/results"

# Add nnFormer to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/210353V-Healthcare-AI_Medical-Segmentation/src"
```

**Example:**

```bash
export nnFormer_raw_data_base="/data/medical/nnFormer_raw"
export nnFormer_preprocessed="/data/medical/nnFormer_preprocessed"
export RESULTS_FOLDER="/data/medical/nnFormer_results"
```

**Apply changes:**

```bash
source ~/.bashrc  # or source ~/.zshrc
```

**Verify:**

```bash
echo $nnFormer_raw_data_base
echo $nnFormer_preprocessed
echo $RESULTS_FOLDER
```

---

## 3. Dataset Preparation

### Step 1: Download BraTS 2021 Dataset

1. **Register for BraTS 2021:**

   - Visit: https://www.synapse.org/#!Synapse:syn25829067
   - Create Synapse account (free)
   - Accept data usage terms

2. **Download Training Data:**

   ```bash
   # Using Synapse client (recommended)
   pip install synapseclient

   python << EOF
   import synapseclient
   syn = synapseclient.login('your_username', 'your_password')
   syn.get('syn25829067', downloadLocation='$nnFormer_raw_data_base/')
   EOF
   ```

   **Manual Download Alternative:**

   - Download from Synapse website
   - Extract to `$nnFormer_raw_data_base/`

3. **Expected Directory Structure:**
   ```
   $nnFormer_raw_data_base/
   └── BraTS2021_Training_Data/
       ├── BraTS2021_00000/
       │   ├── BraTS2021_00000_flair.nii.gz
       │   ├── BraTS2021_00000_t1.nii.gz
       │   ├── BraTS2021_00000_t1ce.nii.gz
       │   ├── BraTS2021_00000_t2.nii.gz
       │   └── BraTS2021_00000_seg.nii.gz
       ├── BraTS2021_00001/
       └── ...
   ```

### Step 2: Prepare Dataset for nnFormer

```bash
cd src/nnformer

# Create Task120_BraTS2021 dataset structure
python -c "
from nnformer.dataset_conversion.Task120_BraTS2021 import convert_brats2021
convert_brats2021()
"
```

**This script:**

- Organizes data into nnFormer format
- Creates `dataset.json` with metadata
- Separates training and validation sets
- Verifies data integrity

**Expected Output Structure:**

```
$nnFormer_raw_data_base/Task120_BraTS2021/
├── dataset.json
├── imagesTr/
│   ├── BraTS2021_00000_0000.nii.gz  # FLAIR
│   ├── BraTS2021_00000_0001.nii.gz  # T1
│   ├── BraTS2021_00000_0002.nii.gz  # T1ce
│   ├── BraTS2021_00000_0003.nii.gz  # T2
│   └── ...
├── labelsTr/
│   ├── BraTS2021_00000.nii.gz
│   └── ...
└── imagesTs/ (if validation data available)
```

### Step 3: Experiment Planning and Preprocessing

```bash
# Analyze dataset and create preprocessing plan
python -m nnformer.experiment_planning.nnFormer_plan_and_preprocess \
    -t 120 --verify_dataset_integrity

# Expected output:
# - Dataset fingerprint
# - Preprocessing plans saved to $nnFormer_preprocessed/Task120_BraTS2021/
# - Normalization statistics computed
```

**This generates:**

- `nnFormerPlansv2.1_plans_3D.pkl`: Architecture and training configuration
- Normalization schemes per modality
- Patch size recommendations
- Batch size recommendations

### Step 4: Preprocess Dataset

```bash
# Preprocess for 3D full-resolution training
python -m nnformer.experiment_planning.nnFormer_plan_and_preprocess \
    -t 120 -pl3d nnFormerPlansv2.1 -no_pp

# Explanation:
# -t 120: Task ID (BraTS2021)
# -pl3d: Use 3D plans version 2.1
# -no_pp: Skip planning, only preprocess

# Monitor progress (takes 4-6 hours for 1,251 cases)
# Watch for completion message: "Preprocessing done!"
```

**Preprocessed Output:**

```
$nnFormer_preprocessed/Task120_BraTS2021/
├── nnFormerPlansv2.1_plans_3D.pkl
├── nnFormerData_plans_v2.1_stage0/
│   ├── BraTS2021_00000.npz
│   ├── BraTS2021_00000.pkl
│   └── ...
└── gt_segmentations/  # Ground truth for validation
```

**Verify Preprocessing:**

```bash
# Check number of preprocessed files
ls $nnFormer_preprocessed/Task120_BraTS2021/nnFormerData_plans_v2.1_stage0/*.npz | wc -l
# Should output: 1251
```

---

## 4. Running Experiments

### Baseline nnFormer

#### Single Fold Training

```bash
cd src/nnformer

# Train baseline on fold 0
python -m nnformer.run.run_training \
    3d_fullres \
    nnFormerTrainerV2_nnformer_tumor \
    120 \
    0

# Arguments:
# 3d_fullres: Network configuration
# nnFormerTrainerV2_nnformer_tumor: Trainer class
# 120: Task ID
# 0: Fold number (0-4)
```

**Training Progress:**

```
Epoch 1/1000 | Train Loss: 0.8234 | Val Dice: 0.4521 | Time: 54s
Epoch 2/1000 | Train Loss: 0.7821 | Val Dice: 0.5234 | Time: 53s
...
Epoch 847/1000 | Train Loss: 0.1234 | Val Dice: 0.8654 | Time: 54s (BEST)
...
Training completed in 6.3 days
```

#### 5-Fold Cross-Validation

```bash
# Train all folds sequentially
for fold in 0 1 2 3 4; do
    python -m nnformer.run.run_training \
        3d_fullres \
        nnFormerTrainerV2_nnformer_tumor \
        120 \
        $fold
done

# Total time: ~30-35 days on single V100
```

**Parallel Training (if multiple GPUs):**

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m nnformer.run.run_training \
    3d_fullres nnFormerTrainerV2_nnformer_tumor 120 0

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m nnformer.run.run_training \
    3d_fullres nnFormerTrainerV2_nnformer_tumor 120 1

# Continue for folds 2, 3, 4 on additional GPUs
```

### Enhanced nnFormer

#### Single Fold Training

```bash
# Train enhanced model on fold 0
python -m nnformer.run.run_training \
    3d_fullres \
    nnFormerTrainerV2_enhanced \
    120 \
    0 \
    --config full

# --config full: Use full enhanced configuration
# (cross-attention + fusion + progressive training)
```

#### 5-Fold Cross-Validation

```bash
for fold in 0 1 2 3 4; do
    python -m nnformer.run.run_training \
        3d_fullres \
        nnFormerTrainerV2_enhanced \
        120 \
        $fold \
        --config full
done
```

### Ablation Studies

#### Ablation 1: Cross-Attention Only

```bash
python -m nnformer.run.run_training \
    3d_fullres \
    nnFormerTrainerV2_enhanced \
    120 \
    0 \
    --config cross_attn
```

#### Ablation 2: Fusion Only

```bash
python -m nnformer.run.run_training \
    3d_fullres \
    nnFormerTrainerV2_enhanced \
    120 \
    0 \
    --config fusion
```

#### Ablation 3: Progressive Training Only

```bash
python -m nnformer.run.run_training \
    3d_fullres \
    nnFormerTrainerV2_enhanced \
    120 \
    0 \
    --config progressive
```

#### Automated Ablation Study

```bash
# Run all ablations for all folds
python run_ablation_study.py \
    --task 120 \
    --cuda_device 0 \
    --fold 0 \
    --epochs 1000

# For quick testing (reduced epochs):
python run_ablation_study.py \
    --task 120 \
    --cuda_device 0 \
    --fold 0 \
    --epochs 200 \
    --configs baseline full
```

---
