# Experiments - Enhanced nnFormer for Brain Tumor Segmentation

**Project:** 210353V - Enhanced nnFormer  
**Student:** Lakshan Madusanka  
**Last Updated:** October 22, 2025

---

## Experiment Overview

This directory contains all experimental configurations, scripts, and logs for the Enhanced nnFormer research project. Experiments are organized into three main categories:

1. **Baseline**: Original nnFormer without enhancements
2. **Enhanced**: Full enhanced model with all components
3. **Ablation**: Systematic removal of components to isolate contributions

---

## Experimental Configurations

### Experiment 1: Baseline nnFormer

**Objective**: Establish baseline performance using original nnFormer architecture

**Configuration:**

```yaml
name: "baseline_nnformer"
network: "3d_fullres"
trainer: "nnFormerTrainerV2_nnformer_tumor"
task_id: 120
fold: 0

model:
  embedding_dim: 96
  input_channels: 4
  num_classes: 4
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]

training:
  epochs: 1000
  batch_size: 2
  patch_size: [64, 128, 128]
  lr: 0.01
  optimizer: "SGD"
  momentum: 0.99
  weight_decay: 3e-5

enhancements:
  cross_attention: false
  adaptive_fusion: false
  progressive_training: false
```

**Expected Results:**

- Dice ET: ~0.703
- Dice TC: ~0.761
- Dice WT: ~0.863
- HD95 WT: ~16.5mm

**Command:**

```bash
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_nnformer_tumor 120 0
```

---

### Experiment 2: Enhanced nnFormer (Full)

**Objective**: Evaluate full enhanced model with all components

**Configuration:**

```yaml
name: "enhanced_nnformer_full"
network: "3d_fullres"
trainer: "nnFormerTrainerV2_enhanced"
task_id: 120
fold: 0

model:
  embedding_dim: 192 # Increased from 96
  input_channels: 4
  num_classes: 4
  depths: [2, 2, 2, 2]
  num_heads: [6, 12, 24, 48] # Doubled

training:
  epochs: 1000
  batch_size: 2
  patch_size: [64, 128, 128]
  base_lr: 0.01
  cross_attn_lr: 0.001 # 10x lower
  optimizer: "SGD"
  momentum: 0.99
  weight_decay: 3e-5

enhancements:
  cross_attention: true
  num_heads_cross: 8
  adaptive_fusion: true
  fusion_reduced_dim: 64
  progressive_training: true
  warmup_epochs: 50
  progressive_ramp_epochs: 50
```

**Expected Results:**

- Dice ET: ~0.737 (+4.8%)
- Dice TC: ~0.785 (+3.2%)
- Dice WT: ~0.884 (+2.4%)
- HD95 WT: ~13.6mm (-17.6%)

**Command:**

```bash
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_enhanced 120 0 --config full
```

---

### Experiment 3: Ablation - Cross-Attention Only

Test multi-scale cross-attention impact. Config: `cross_attention=true`, others false. Expected: Dice ET ~0.721 (+2.6%).

---

### Experiment 4: Ablation - Adaptive Fusion Only

Test learnable fusion weights impact. Config: `adaptive_fusion=true`, others false. Expected: Dice ET ~0.715 (+1.7%).

---

### Experiment 5: Ablation - Progressive Training Only

Test α schedule (0→1) impact. Config: `progressive_training=true` with cross-attention, fusion disabled. Expected: Dice ET ~0.708 (+0.7%).

---

## Running Experiments

### Single Experiment

```bash
# Baseline
cd src/nnformer
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_nnformer_tumor 120 0

# Enhanced
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_enhanced 120 0 --config full

# Ablation
python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_enhanced 120 0 --config cross_attn
```

### Automated Ablation Study

```bash
# Run all ablations sequentially
python run_ablation_study.py --task 120 --cuda_device 0 --fold 0 --epochs 1000

# Quick test (reduced epochs)
python run_ablation_study.py --task 120 --cuda_device 0 --fold 0 --epochs 200 --configs baseline full
```

### 5-Fold Cross-Validation

```bash
# Run all folds for baseline
for fold in 0 1 2 3 4; do
    python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_nnformer_tumor 120 $fold
done

# Run all folds for enhanced
for fold in 0 1 2 3 4; do
    python -m nnformer.run.run_training 3d_fullres nnFormerTrainerV2_enhanced 120 $fold --config full
done
```

---

## Experiment Tracking

### Metrics Tracked

- **Training Loss**: Dice + CE combined
- **Validation Dice**: ET, TC, WT
- **Validation HD95**: ET, TC, WT
- **Learning Rate**: Current LR
- **Epoch Time**: Seconds per epoch
- **GPU Memory**: Peak VRAM usage

---

## Reproducibility

### Random Seeds

```python
# Set in all experiments
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False  # For speed
torch.backends.cudnn.benchmark = True       # For speed
```
