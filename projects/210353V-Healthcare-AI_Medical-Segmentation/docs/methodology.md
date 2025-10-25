# Methodology: Enhanced nnFormer for Brain Tumor Segmentation

**Student:** 210353V - Lakshan Madusanka  
**Research Area:** Healthcare AI - Medical Image Segmentation  
**Supervisor:** Dr. Uthayasanker Thayasivam  
**Institution:** University of Moratuwa  
**Date:** October 22, 2025

---

## 1. Overview

This document details the complete methodology for developing and evaluating the Enhanced nnFormer architecture for brain tumor segmentation. Our approach builds upon the baseline nnFormer (Zhou et al., 2023) by introducing three key enhancements: (1) **multi-scale cross-attention mechanisms** for bidirectional feature interaction between encoder stages, (2) **adaptive feature fusion** using learned channel and spatial attention weights, and (3) **progressive training strategy** to stabilize the learning of complex multi-component architectures. The methodology follows a rigorous experimental design including baseline reproduction, incremental component addition via ablation studies, comprehensive evaluation on BraTS 2021 dataset with 5-fold cross-validation, and statistical significance testing to validate improvements.

---

## 2. Research Design

### 2.1 Research Type

**Experimental Research** with quantitative evaluation and comparative analysis.

### 2.2 Research Questions

**RQ1:** How does multi-scale cross-attention affect segmentation performance compared to standard skip connections?

**RQ2:** What is the contribution of adaptive feature fusion versus fixed fusion strategies?

**RQ3:** Does progressive training improve convergence and final performance for complex multi-component architectures?

**RQ4:** What is the combined effect of all enhancements on brain tumor segmentation accuracy?

**RQ5:** How do computational requirements (parameters, FLOPs, inference time) scale with architectural enhancements?

### 2.3 Hypothesis

**H1:** Multi-scale cross-attention will improve Dice coefficient for enhancing tumor (ET) by 2-3% by enabling fine-grained features to access semantic context.

**H2:** Adaptive feature fusion will improve tumor core (TC) segmentation by 1-2% through context-dependent weight learning.

**H3:** Progressive training will improve overall performance by 0.5-1% through better convergence.

**H4:** The combined approach will achieve 4-5% improvement in mean Dice ET while maintaining computational feasibility (inference time <5s per case).

### 2.4 Experimental Design

```
Baseline nnFormer
    ↓
+ Multi-Scale Cross-Attention → Ablation 1
    ↓
+ Adaptive Fusion → Ablation 2
    ↓
+ Progressive Training → Enhanced nnFormer (Full)
```

**Control Variables:**

- Dataset (BraTS 2021, same splits)
- Preprocessing (same pipeline)
- Training hyperparameters (except differentiated LR for new components)
- Data augmentation (identical configuration)
- Hardware (same GPU)

**Independent Variables:**

- Presence/absence of cross-attention
- Presence/absence of adaptive fusion
- Presence/absence of progressive training

**Dependent Variables:**

- Dice coefficient (ET, TC, WT)
- 95th percentile Hausdorff Distance (HD95)
- Training time
- Inference time
- GPU memory usage

---

## 3. Dataset

### 3.1 Data Source

**Dataset:** Brain Tumor Segmentation Challenge 2021 (BraTS 2021)  
**Source:** https://www.synapse.org/#!Synapse:syn25829067  
**License:** CC BY-NC 4.0 (Academic use only)  
**Registration:** Required via Synapse account

### 3.2 Dataset Description

**Training Set:**

- **Cases:** 1,251 patients with glioblastoma or lower-grade glioma
- **Imaging:** Multi-parametric MRI with 4 modalities
  - T1-weighted (T1)
  - T1-weighted with contrast enhancement (T1ce)
  - T2-weighted (T2)
  - T2 Fluid Attenuated Inversion Recovery (FLAIR)
- **Resolution:** 1mm³ isotropic (resampled)
- **Dimensions:** 240 × 240 × 155 voxels
- **Format:** NIfTI (.nii.gz)

**Validation Set:**

- **Cases:** 219 patients
- **Labels:** Not publicly available (online evaluation only)

**Annotation:**

- **Annotators:** Expert neuroradiologists
- **Labels:**
  - Label 0: Background
  - Label 1: Necrotic and non-enhancing tumor (NET/NCR)
  - Label 2: Peritumoral edema (ED)
  - Label 4: Enhancing tumor (ET)

**Regions of Interest:**

- **ET (Enhancing Tumor):** Label 4
- **TC (Tumor Core):** Labels 1 + 4
- **WT (Whole Tumor):** Labels 1 + 2 + 4

### 3.3 Data Preprocessing

Our preprocessing pipeline follows the nnFormer standard with additional quality control steps:

#### Stage 1: Format Conversion and Verification

```python
# Verify data integrity
- Check NIfTI header integrity
- Validate image dimensions (240×240×155)
- Verify 4 modalities present per case
- Check label value validity (0, 1, 2, 4 only)
```

#### Stage 2: Intensity Normalization

```python
# Per-modality, per-case normalization
for modality in [T1, T1ce, T2, FLAIR]:
    brain_mask = (volume > 0)  # Exclude background
    mean = np.mean(volume[brain_mask])
    std = np.std(volume[brain_mask])
    volume_normalized = (volume - mean) / (std + 1e-8)
    volume_normalized = np.clip(volume_normalized, -5, 5)
```

**Rationale:** Z-score normalization ensures consistent intensity distributions across patients while preserving relative tissue contrasts.

#### Stage 3: Cropping to Non-Zero Region

```python
# Remove excess background
bbox = compute_bounding_box(volume > 0)
cropped_volume = volume[bbox]  # Typical size: ~140×180×140
```

**Benefit:** Reduces memory usage by ~35%, focuses network on brain region.

#### Stage 4: Resampling (if needed)

```python
# Ensure 1mm isotropic spacing
current_spacing = nifti_header['pixdim'][1:4]
if current_spacing != [1.0, 1.0, 1.0]:
    resample_to_spacing(volume, target_spacing=[1.0, 1.0, 1.0])
```

#### Stage 5: nnFormer-Specific Preprocessing

- **Dataset Fingerprint Extraction:** Analyze intensity statistics, spacing, size distributions
- **Plan Generation:** Determine patch size, batch size, network architecture based on fingerprint
- **Preprocessed Data Storage:** Save normalized, cropped data in NPZ format for fast loading

**Output Structure:**

```
nnFormer_preprocessed/Task120_BraTS2021/
├── nnFormerData_plans_v2.1_stage0/
│   ├── BraTS2021_00001.npz
│   ├── BraTS2021_00002.npz
│   └── ...
└── nnFormerPlansv2.1_plans_3D.pkl
```

### 3.4 Data Augmentation

**Online augmentation** during training (using batchgenerators library):

#### Spatial Transformations

```python
# Elastic deformation
- Alpha range: [0, 900]
- Sigma range: [9, 13]
- Probability: 50%

# Random rotation
- X-axis: [-15°, +15°]
- Y-axis: [-15°, +15°]
- Z-axis: [-15°, +15°]
- Probability: 50%

# Random scaling
- Scale range: [0.7, 1.4]
- Probability: 30%

# Random mirroring
- Axes: [0, 1, 2] (all axes)
- Probability: 50% per axis
```

#### Intensity Transformations

```python
# Gamma transformation
- Gamma range: [0.7, 1.5]
- Retain statistics: True
- Probability: 30%
- Per-channel: True

# Gaussian noise
- Noise std: 0.1
- Probability: 15%

# Gaussian blur
- Sigma: [0.5, 1.0]
- Probability: 20%
```

**Rationale:** Extensive augmentation critical for medical imaging where data is limited and model must generalize across different scanners, protocols, and patient anatomies.

### 3.5 Cross-Validation Split

**5-Fold Cross-Validation:**

- **Fold 0:** Train on cases 0-1000, validate on cases 1001-1250 (250 cases)
- **Fold 1:** Train on 251-1250, validate on 0-250
- **Fold 2:** Train on 0-250 + 501-1250, validate on 251-500
- **Fold 3:** Train on 0-500 + 751-1250, validate on 501-750
- **Fold 4:** Train on 0-750 + 1001-1250, validate on 751-1000

**Stratification:** Folds are pre-defined by nnFormer to ensure balanced tumor size distribution.

---

## 4. Model Architecture

### 4.1 Baseline: nnFormer

#### Encoder (4 Stages)

```python
Stage 1: Input (4 channels) → 96 channels
  - Patch embedding: Conv3D(4, 96, kernel=4, stride=4)
  - 2 × Swin Transformer blocks
  - Window size: [3, 5, 5]
  - 3 attention heads
  - Output: 96 channels, spatial size /4

Stage 2: 96 → 192 channels
  - Patch merging: reduce spatial by 2×, double channels
  - 2 × Swin Transformer blocks
  - Window size: [3, 5, 5]
  - 6 attention heads
  - Output: 192 channels, spatial size /8

Stage 3: 192 → 384 channels
  - Patch merging
  - 2 × Swin Transformer blocks
  - Window size: [7, 10, 10]
  - 12 attention heads
  - Output: 384 channels, spatial size /16

Stage 4: 384 → 768 channels
  - Patch merging
  - 2 × Swin Transformer blocks
  - Window size: [3, 5, 5]
  - 24 attention heads
  - Output: 768 channels, spatial size /32
```

#### Decoder (3 Stages with Skip Connections)

```python
Decoder Stage 1: 768 → 384 channels
  - Upsample by 2×
  - Concatenate with Stage 3 encoder features
  - 2 × Swin Transformer blocks

Decoder Stage 2: 384 → 192 channels
  - Upsample by 2×
  - Concatenate with Stage 2 encoder features
  - 2 × Swin Transformer blocks

Decoder Stage 3: 192 → 96 channels
  - Upsample by 2×
  - Concatenate with Stage 1 encoder features
  - 2 × Swin Transformer blocks

Segmentation Head:
  - Upsample to original resolution
  - Conv3D(96, 4, kernel=1) → 4 class logits
```

#### Deep Supervision

```python
# Additional segmentation heads at each decoder stage
Deep_sup_1: From decoder stage 1 → upsample → 4 classes
Deep_sup_2: From decoder stage 2 → upsample → 4 classes
Deep_sup_3: From decoder stage 3 → upsample → 4 classes

# Combined loss
Loss = Loss_main + 0.5×Loss_deep1 + 0.25×Loss_deep2 + 0.125×Loss_deep3
```

**Parameters:** 62.4M  
**FLOPs:** 287G (for 128³ patch)

### 4.2 Enhanced nnFormer

Our enhancements modify the encoder and feature fusion:

#### Enhancement 1: Multi-Scale Cross-Attention

**Inserted after each encoder stage:**

```python
class MultiScaleCrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8):
        self.norm_q = LayerNorm(dim1)
        self.norm_kv = LayerNorm(dim2)
        self.cross_attn = MultiHeadAttention(dim1, num_heads)

    def forward(self, x_query, x_context):
        # x_query: [B, C1, D1, H1, W1] from stage i
        # x_context: [B, C2, D2, H2, W2] from stage i±1

        # Flatten spatial dimensions
        Q = rearrange(x_query, 'b c d h w -> b (d h w) c')
        KV = rearrange(x_context, 'b c d h w -> b (d h w) c')

        # Normalize
        Q = self.norm_q(Q)
        KV = self.norm_kv(KV)

        # Cross-attention: query=stage_i, key/value=stage_i±1
        out = self.cross_attn(Q, KV, KV)  # [B, N1, C1]

        # Reshape back
        out = rearrange(out, 'b (d h w) c -> b c d h w',
                       d=D1, h=H1, w=W1)

        # Residual connection
        return x_query + out
```

**Application:**

- **Stage 1 → Stage 2:** Stage 1 queries Stage 2 (high-res queries low-res context)
- **Stage 2 → Stage 1:** Stage 2 queries Stage 1 (low-res queries high-res details)
- **Stage 2 → Stage 3:** Similar bidirectional attention
- **Stage 3 → Stage 4:** Similar bidirectional attention

**Computational Cost:**

- Attention complexity: O(N₁ × N₂) where N₁, N₂ are spatial sizes
- Mitigated by downsampling: largest is Stage 1↔2 with N₁=64k, N₂=8k
- Additional parameters: ~15M
- Additional FLOPs: ~80G

#### Enhancement 2: Adaptive Feature Fusion

**Replaces fixed concatenation in skip connections:**

```python
class AdaptiveFusionModule(nn.Module):
    def __init__(self, channels, reduced_dim=64):
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_dim, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x_encoder, x_cross_attn):
        # Combine encoder and cross-attended features
        x_combined = x_encoder + x_cross_attn

        # Channel attention weights
        ca_weights = self.channel_attention(x_combined)
        x_channel = x_combined * ca_weights

        # Spatial attention weights
        sa_weights = self.spatial_attention(x_channel)
        x_fused = x_channel * sa_weights

        return x_fused
```

**Application:** Applied to each encoder stage output before skip connection to decoder.

**Computational Cost:**

- Parameters: ~12M
- FLOPs: ~45G

#### Enhancement 3: Progressive Training

**Controller for gradual enhancement activation:**

```python
class ProgressiveTrainingController:
    def __init__(self, warmup_epochs=50, ramp_epochs=50, target_alpha=1.0):
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.target_alpha = target_alpha

    def get_alpha(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            return 0.0  # No enhancement, train like baseline
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            # Linear ramp from 0 to target_alpha
            progress = (current_epoch - self.warmup_epochs) / self.ramp_epochs
            return self.target_alpha * progress
        else:
            return self.target_alpha  # Full enhancement
```

**Modified Forward Pass:**

```python
def forward(self, x, alpha):
    # Encoder stages with conditional cross-attention
    x1 = self.stage1(x)
    if alpha > 0:
        x1 = x1 + alpha * self.cross_attn_1_2(x1, x2_prev)
        x1 = self.adaptive_fusion_1(x1, x1_cross)

    # Similar for other stages
    ...
```

**No additional parameters**, only training schedule modification.

#### Enhanced Architecture Summary

**Total Parameters:** 89.7M (+44% vs baseline)  
**Total FLOPs:** 412G (+44% vs baseline)  
**Inference Time:** ~3.1s per case (vs 2.3s baseline, on V100 GPU)  
**GPU Memory:** ~18.7 GB (vs 15.2 GB baseline, batch size 2)

---

## 5. Experimental Setup

### 5.1 Baseline Reproduction

Train standard nnFormer on BraTS 2021 using 5-fold cross-validation. Expected results: Dice ET 0.703±0.024, TC 0.761±0.018, WT 0.863±0.012. Validate preprocessing pipeline and training stability before enhancement experiments.

### 5.2 Ablation Study Design

Five configurations tested to isolate component contributions:

1. **Cross-Attention Only** - Multi-scale cross-attention, standard fusion
2. **Adaptive Fusion Only** - Learnable fusion weights, standard attention
3. **Progressive Training Only** - α schedule (0→1) applied to cross-attention
4. **Cross-Attn + Fusion** - Combined without progressive training
5. **Full Enhanced Model** - All three components integrated

Each configuration: 5-fold CV, 1000 epochs/fold, identical hyperparameters except enhancement flags.

### 5.3 Statistical Validation

**Primary:** Paired t-test (enhanced vs baseline on same folds, α=0.05, expected p<0.01)  
**Secondary:** Wilcoxon signed-rank test (non-parametric robustness check)  
**Effect Size:** Cohen's d to measure improvement magnitude (target d>0.8 for clinical significance)

---

## 6. Implementation Plan

| Phase                                   | Duration   | Tasks                                                           | Deliverables                         |
| --------------------------------------- | ---------- | --------------------------------------------------------------- | ------------------------------------ |
| **Phase 1: Setup**                      | Week 3     | Environment setup, data download, preprocessing verification    | Preprocessed BraTS 2021 dataset      |
| **Phase 2: Baseline**                   | Week 4-5   | Train baseline nnFormer (5 folds), validate reproduction        | Baseline results matching literature |
| **Phase 3: Enhancement Implementation** | Week 5-7   | Implement cross-attention, fusion, progressive training modules | Working enhanced architecture code   |
| **Phase 4: Ablation Studies**           | Week 7-8   | Train all ablation configurations (5 folds each)                | Ablation study results               |
| **Phase 5: Full Enhanced Training**     | Week 8-9   | Train full enhanced model (5 folds)                             | Final enhanced model results         |
| **Phase 6: Analysis**                   | Week 9-10  | Statistical tests, visualizations, failure analysis             | Comprehensive results analysis       |
| **Phase 7: Documentation**              | Week 10-12 | Write final report, prepare publication materials               | Final thesis/paper                   |

**Total Duration:** 32 weeks (~8 months)

---

## 7. Risk Analysis and Mitigation

### Risk 1: Insufficient GPU Resources

**Impact:** High (cannot train models)  
**Probability:** Medium

**Mitigation:**

- Apply for university HPC cluster access
- Use cloud GPUs (AWS/GCP academic credits)
- Reduce batch size to 1 if needed (doubles training time but feasible)

### Risk 2: Training Instability

**Impact:** High (failed experiments waste weeks)  
**Probability:** Medium

**Mitigation:**

- Implement gradient clipping (max norm = 1.0)
- Use differentiated learning rates (already planned)
- Progressive training to stabilize (already planned)
- Extensive logging to detect divergence early

### Risk 3: Limited Improvement

**Impact:** Medium (research still valid but less impactful)  
**Probability:** Low

**Mitigation:**

- Baseline reproduction ensures fair comparison
- Ablation studies isolate component contributions
- Even negative results publishable if methodology rigorous
- Fallback: Focus on computational efficiency if accuracy similar

### Risk 4: Data Issues

**Impact:** High (invalid results)  
**Probability:** Low

**Mitigation:**

- Verify data integrity during preprocessing
- Compare preprocessed statistics with literature
- Visual inspection of random samples
- Cross-reference with official BraTS baseline results

### Risk 5: Reproducibility Issues

**Impact:** Medium (difficulty validating results)  
**Probability:** Medium

**Mitigation:**

- Fix random seeds (42 everywhere)
- Document exact library versions (requirements.txt)
- Save all hyperparameters in configuration files
- Version control all code (git)
- Save model checkpoints and training logs

---

## 8. Expected Outcomes

### Quantitative Outcomes

- **Mean Dice ET:** 0.737 ± 0.021 (+4.8% vs baseline)
- **Mean Dice TC:** 0.785 ± 0.016 (+3.2% vs baseline)
- **Mean Dice WT:** 0.884 ± 0.011 (+2.4% vs baseline)
- **Mean HD95 WT:** 13.6 ± 2.4 mm (-17.6% vs baseline)

### Qualitative Outcomes

- Better small tumor segmentation (visual inspection)
- Improved boundary delineation (HD95 reduction)
- Ablation study insights on component contributions

---
