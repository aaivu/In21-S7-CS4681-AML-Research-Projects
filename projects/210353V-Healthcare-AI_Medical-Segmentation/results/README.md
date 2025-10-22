# Results - Enhanced nnFormer for Brain Tumor Segmentation

**Project:** 210353V - Enhanced nnFormer  
**Student:** Lakshan Madusanka  
**Institution**: University of Moratuwa  
**Last Updated:** October 22, 2025

---

## Results Overview

This directory contains all experimental results, metrics, visualizations, and analysis for the Enhanced nnFormer research project.

---

## Baseline Results (Original nnFormer)

### 5-Fold Cross-Validation Summary

| Metric      | Mean ± Std    | Median | Min   | Max   |
| ----------- | ------------- | ------ | ----- | ----- |
| **Dice ET** | 0.703 ± 0.024 | 0.705  | 0.671 | 0.729 |
| **Dice TC** | 0.761 ± 0.018 | 0.763  | 0.738 | 0.781 |
| **Dice WT** | 0.863 ± 0.012 | 0.865  | 0.847 | 0.876 |
| **HD95 ET** | 24.3 ± 4.2 mm | 23.8   | 18.9  | 30.1  |
| **HD95 TC** | 18.7 ± 3.1 mm | 18.2   | 14.6  | 23.4  |
| **HD95 WT** | 16.5 ± 2.8 mm | 16.1   | 12.9  | 20.7  |

### Per-Fold Results

**Fold 0:**

```json
{
  "dice_et": 0.705,
  "dice_tc": 0.763,
  "dice_wt": 0.865,
  "hd95_et": 23.8,
  "hd95_tc": 18.2,
  "hd95_wt": 16.1,
  "training_time": "6.2 days",
  "epochs": 1000,
  "best_epoch": 847
}
```

**Fold 1:**

```json
{
  "dice_et": 0.729,
  "dice_tc": 0.781,
  "dice_wt": 0.876,
  "hd95_et": 18.9,
  "hd95_tc": 14.6,
  "hd95_wt": 12.9,
  "training_time": "5.8 days",
  "epochs": 1000,
  "best_epoch": 921
}
```

**Fold 2:**

```json
{
  "dice_et": 0.698,
  "dice_tc": 0.756,
  "dice_wt": 0.861,
  "hd95_et": 26.4,
  "hd95_tc": 20.1,
  "hd95_wt": 17.8,
  "training_time": "6.5 days",
  "epochs": 1000,
  "best_epoch": 782
}
```

**Fold 3:**

```json
{
  "dice_et": 0.671,
  "dice_tc": 0.738,
  "dice_wt": 0.847,
  "hd95_et": 30.1,
  "hd95_tc": 23.4,
  "hd95_wt": 20.7,
  "training_time": "6.8 days",
  "epochs": 1000,
  "best_epoch": 695
}
```

**Fold 4:**

```json
{
  "dice_et": 0.712,
  "dice_tc": 0.768,
  "dice_wt": 0.869,
  "hd95_et": 22.3,
  "hd95_tc": 17.1,
  "hd95_wt": 15.4,
  "training_time": "6.1 days",
  "epochs": 1000,
  "best_epoch": 856
}
```

## Enhanced nnFormer Results

### 5-Fold Cross-Validation Summary

| Metric      | Mean ± Std    | Median | Min   | Max   | **Improvement** |
| ----------- | ------------- | ------ | ----- | ----- | --------------- |
| **Dice ET** | 0.737 ± 0.021 | 0.739  | 0.708 | 0.761 | **+4.8%** ✅    |
| **Dice TC** | 0.785 ± 0.016 | 0.787  | 0.763 | 0.804 | **+3.2%** ✅    |
| **Dice WT** | 0.884 ± 0.011 | 0.886  | 0.869 | 0.897 | **+2.4%** ✅    |
| **HD95 ET** | 19.8 ± 3.6 mm | 19.2   | 15.1  | 25.3  | **-18.5%** ✅   |
| **HD95 TC** | 15.2 ± 2.7 mm | 14.8   | 11.9  | 19.4  | **-18.7%** ✅   |
| **HD95 WT** | 13.6 ± 2.4 mm | 13.2   | 10.4  | 17.1  | **-17.6%** ✅   |

### Per-Fold Results

**Fold 0:**

```json
{
  "dice_et": 0.739,
  "dice_tc": 0.787,
  "dice_wt": 0.886,
  "hd95_et": 19.2,
  "hd95_tc": 14.8,
  "hd95_wt": 13.2,
  "improvement_et": "+3.4%",
  "improvement_tc": "+2.4%",
  "improvement_wt": "+2.1%",
  "training_time": "7.1 days",
  "epochs": 1000,
  "best_epoch": 892
}
```

**Fold 1:**

```json
{
  "dice_et": 0.761,
  "dice_tc": 0.804,
  "dice_wt": 0.897,
  "hd95_et": 15.1,
  "hd95_tc": 11.9,
  "hd95_wt": 10.4,
  "improvement_et": "+3.2%",
  "improvement_tc": "+2.3%",
  "improvement_wt": "+2.1%",
  "training_time": "6.7 days",
  "epochs": 1000,
  "best_epoch": 945
}
```

**Fold 2:**

```json
{
  "dice_et": 0.731,
  "dice_tc": 0.779,
  "dice_wt": 0.882,
  "hd95_et": 21.3,
  "hd95_tc": 16.4,
  "hd95_wt": 14.6,
  "improvement_et": "+3.3%",
  "improvement_tc": "+2.3%",
  "improvement_wt": "+2.1%",
  "training_time": "7.3 days",
  "epochs": 1000,
  "best_epoch": 814
}
```

**Fold 3:**

```json
{
  "dice_et": 0.708,
  "dice_tc": 0.763,
  "dice_wt": 0.869,
  "hd95_et": 25.3,
  "hd95_tc": 19.4,
  "hd95_wt": 17.1,
  "improvement_et": "+3.7%",
  "improvement_tc": "+2.5%",
  "improvement_wt": "+2.2%",
  "training_time": "7.6 days",
  "epochs": 1000,
  "best_epoch": 723
}
```

**Fold 4:**

```json
{
  "dice_et": 0.746,
  "dice_tc": 0.792,
  "dice_wt": 0.891,
  "hd95_et": 18.1,
  "hd95_tc": 13.9,
  "hd95_wt": 12.5,
  "improvement_et": "+3.4%",
  "improvement_tc": "+2.4%",
  "improvement_wt": "+2.2%",
  "training_time": "6.9 days",
  "epochs": 1000,
  "best_epoch": 881
}
```

## Ablation Study Results

### Component Contributions

| Configuration         | Dice ET   | Dice TC   | Dice WT   | Contribution |
| --------------------- | --------- | --------- | --------- | ------------ |
| Baseline              | 0.703     | 0.761     | 0.863     | -            |
| + Cross-Attn          | 0.721     | 0.772     | 0.875     | **+2.6%**    |
| + Fusion              | 0.715     | 0.768     | 0.871     | **+1.7%**    |
| + Progressive         | 0.708     | 0.764     | 0.866     | **+0.7%**    |
| + Cross-Attn + Fusion | 0.734     | 0.782     | 0.882     | **+4.4%**    |
| **Full Enhanced**     | **0.737** | **0.785** | **0.884** | **+4.8%**    |

### Key Findings

1. **Multi-Scale Cross-Attention**: Largest single contribution (+2.6%)
   - Enables better feature interaction across scales
   - Most beneficial for ET segmentation (small regions)
2. **Adaptive Feature Fusion**: Second-largest contribution (+1.7%)
   - Learns optimal feature combination weights
   - Reduces redundancy between scale features
3. **Progressive Training**: Smallest but important (+0.7%)
   - Stabilizes training of complex attention mechanisms
   - Prevents overfitting to auxiliary components
4. **Synergistic Effect**: Components work together
   - Combined: 4.8% improvement
   - Sum of individual: 5.0% (slight redundancy)

---
