# Research Proposal: Enhanced nnFormer for Brain Tumor Segmentation

**Student:** 210353V - Lakshan Madusanka  
**Research Area:** Healthcare AI - Medical Image Segmentation  
**Supervisor:** Dr. Uthayasanker Thayasivam  
**Institution:** University of Moratuwa, Department of Computer Science and Engineering  
**Date:** September 1, 2025

---

## Abstract

Brain tumor segmentation from multi-modal MRI scans remains a critical challenge in medical image analysis, directly impacting diagnosis, treatment planning, and therapeutic monitoring. While transformer-based architectures like nnFormer have demonstrated promising results for 3D volumetric segmentation, they face limitations in capturing multi-scale spatial relationships and achieving precise boundary delineation, particularly for small tumor regions. This research proposes an **Enhanced nnFormer architecture** incorporating **multi-scale cross-attention mechanisms**, **adaptive feature fusion strategies**, and **progressive training techniques** to address these limitations. The proposed approach enables bidirectional feature interaction between encoder stages, allowing high-resolution features to access semantic information from coarser scales while enabling low-resolution features to incorporate fine-grained spatial details. Through comprehensive experiments on the BraTS 2021 dataset and rigorous ablation studies, this research aims to achieve state-of-the-art performance in brain tumor segmentation, with expected improvements of 4-5% in Dice coefficient for enhancing tumor (ET) regions and 15-20% reduction in Hausdorff Distance, translating to more accurate tumor delineation for clinical applications including surgical planning and radiation therapy dose calculations.

## 1. Introduction

### 1.1 Background

Brain tumors, particularly gliomas, represent one of the most challenging diagnoses in modern medicine, with glioblastoma carrying a median survival of only 15 months despite aggressive treatment. Accurate segmentation of tumor regions from magnetic resonance imaging (MRI) scans is crucial for diagnosis, surgical planning, radiation therapy targeting, and treatment response monitoring. However, manual segmentation by expert radiologists is time-consuming (30-60 minutes per case), subject to inter-rater variability, and often impractical in high-volume clinical settings.

The advent of deep learning has revolutionized medical image segmentation, with convolutional neural networks (CNNs) achieving near-human performance on many tasks. The nnU-Net framework established a strong baseline for medical segmentation through automated architecture configuration and extensive data augmentation, achieving Dice coefficients of ~0.86 for whole tumor segmentation on the Brain Tumor Segmentation (BraTS) benchmark. However, CNN-based methods struggle with long-range dependencies due to their inherently local receptive fields, limiting their ability to capture global context essential for understanding tumor extent and relationships to surrounding anatomical structures.

### 1.2 Transformer Revolution in Medical Imaging

The introduction of Vision Transformers (ViT) and their hierarchical variant Swin Transformer marked a paradigm shift in computer vision, demonstrating that self-attention mechanisms can effectively capture both local and global dependencies. Recent medical imaging transformers including UNETR, TransUNet, and nnFormer have adapted these architectures for volumetric segmentation, showing particular promise for tasks requiring global context understanding.

nnFormer (Zhou et al., 2023), which incorporates 3D Swin Transformer blocks into an encoder-decoder architecture, has demonstrated competitive performance with nnU-Net while using fewer parameters. On BraTS 2021, nnFormer achieves Dice coefficients of 0.703 (ET), 0.761 (TC), and 0.863 (WT), matching or exceeding previous state-of-the-art. However, analysis of nnFormer's performance reveals persistent challenges:

1. **Small tumor segmentation:** Enhancing tumor (ET) Dice remains <0.75, with frequent false negatives for small foci
2. **Boundary delineation:** Hausdorff Distance metrics indicate imprecise boundaries (HD95 ~16-24mm)
3. **Multi-scale integration:** Current skip connections provide only passive feature transfer without explicit cross-scale interaction

### 1.3 Research Motivation

While nnFormer's self-attention mechanisms enable global context modeling within each encoder stage, there is no explicit mechanism for features at different scales to interact. High-resolution stages contain precise spatial information but lack semantic understanding, while low-resolution stages possess rich semantic features but lack spatial precision. Current skip connections concatenate these features but do not allow them to actively query and refine each other.

This research is motivated by three key observations:

1. **Multi-scale nature of brain tumors:** Gliomas exhibit features at multiple scales, from millimeter-scale enhancing foci to centimeter-scale edema regions, requiring both fine-grained detail and global context
2. **Attention enables interaction:** Recent work in natural image classification (CrossViT) demonstrates that explicit cross-attention between multi-scale representations improves performance
3. **Fixed fusion is suboptimal:** Adaptive feature fusion with learnable weights has shown superior performance compared to fixed concatenation or addition in multi-modal medical imaging

This research proposes to enhance nnFormer through three synergistic innovations that address these limitations and advance the state-of-the-art in brain tumor segmentation.

---

## 2. Problem Statement

### 2.1 Research Problem

**How can we enhance transformer-based volumetric segmentation architectures to better capture multi-scale spatial relationships and improve boundary delineation for brain tumor segmentation?**

### 2.2 Specific Challenges

**Challenge 1: Limited Cross-Scale Feature Interaction**

Current nnFormer architecture processes each encoder stage independently through self-attention, with only passive skip connections linking encoder to decoder. This limits the model's ability to:

- Resolve semantic ambiguities in high-resolution features using coarse-scale context
- Refine coarse-scale segmentation using fine-scale spatial details
- Adapt feature representations based on cross-scale information

**Challenge 2: Suboptimal Feature Fusion**

Existing fusion strategies (concatenation, addition) treat all features equally regardless of spatial location or semantic content. This results in:

- Redundancy when features contain similar information
- Missed complementary information when optimal fusion varies by location
- Inability to emphasize tumor-relevant features while suppressing artifacts

**Challenge 3: Training Instability**

Adding complex multi-component architectures (cross-attention + fusion) can cause training instability:

- Features may not be meaningful early in training, making attention weights unreliable
- Risk of overfitting to auxiliary components rather than segmentation task
- Requires extensive hyperparameter tuning

### 2.3 Quantitative Targets

This research aims to improve upon baseline nnFormer (BraTS 2021 validation set):

| Metric       | Baseline      | Target        | Improvement |
| ------------ | ------------- | ------------- | ----------- |
| Dice ET      | 0.703 ± 0.024 | 0.737 ± 0.021 | +4.8%       |
| Dice TC      | 0.761 ± 0.018 | 0.785 ± 0.016 | +3.2%       |
| Dice WT      | 0.863 ± 0.012 | 0.884 ± 0.011 | +2.4%       |
| HD95 WT (mm) | 16.5 ± 2.8    | 13.6 ± 2.4    | -17.6%      |

### 2.4 Clinical Significance

Achieving these improvements would have direct clinical impact:

- **Surgical Planning:** HD95 <15mm enables more accurate resection boundaries
- **Radiation Therapy:** Improved ET segmentation ensures adequate tumor coverage while sparing healthy tissue
- **Treatment Monitoring:** Reliable volumetric measurements for assessing response to chemotherapy/radiation
- **Prognostic Value:** Accurate tumor subregion volumes correlate with survival outcomes

---

## 3. Literature Review Summary

### 3.1 CNN-Based Medical Segmentation

The U-Net architecture (Ronneberger et al., 2015) established the encoder-decoder paradigm with skip connections that remains foundational for medical segmentation. Its 3D extension (Çiçek et al., 2016) enabled volumetric segmentation, while nnU-Net (Isensee et al., 2021) automated architecture configuration to achieve state-of-the-art results across multiple benchmarks. However, CNNs' limited receptive fields constrain their ability to model long-range dependencies essential for understanding tumor extent and context.

### 3.2 Transformer Architectures for Medical Imaging

Vision Transformers (Dosovitskiy et al., 2021) demonstrated that pure transformer architectures can match or exceed CNN performance on image classification. Swin Transformer (Liu et al., 2021) introduced hierarchical architecture with shifted windows, enabling efficient processing of high-resolution images. Medical imaging adaptations include:

- **UNETR** (Hatamizadeh et al., 2022): Pure transformer encoder with CNN decoder, achieving strong results but high computational cost
- **TransUNet** (Chen et al., 2021): Hybrid CNN-transformer showing benefits of combining local and global feature extraction
- **nnFormer** (Zhou et al., 2023): 3D Swin Transformer-based architecture specifically designed for volumetric segmentation, our baseline

### 3.3 Multi-Scale Feature Learning

Feature Pyramid Networks (Lin et al., 2017) established the importance of multi-scale representations for object detection. CrossViT (Wang et al., 2022) demonstrated that explicit cross-attention between different scale branches improves image classification, though not yet applied to 3D medical segmentation.

### 3.4 Adaptive Feature Fusion

Recent work shows that learned fusion outperforms fixed strategies. Zhang et al. (2023) demonstrated +1.8% Dice improvement using adaptive fusion for multi-modal MRI, while Ding et al. (2022) showed channel and spatial attention enables context-dependent feature combination.

### 3.5 Identified Gaps

1. **No cross-scale attention in 3D medical transformers:** Existing methods use only skip connections without explicit bidirectional attention
2. **Limited adaptive fusion:** Not applied to cross-scale encoder features in transformer architectures
3. **Training stability:** Complex multi-component architectures require progressive training strategies

Our research addresses all three gaps through integrated architectural and training innovations.

---

## 4. Research Objectives

### Primary Objective

**Develop and validate an Enhanced nnFormer architecture that improves brain tumor segmentation performance through multi-scale cross-attention mechanisms, adaptive feature fusion, and progressive training, achieving state-of-the-art results on the BraTS 2021 benchmark.**

### Secondary Objectives

1. **Design Multi-Scale Cross-Attention Module**

   - Enable bidirectional feature interaction between encoder stages
   - Implement efficient cross-attention for 3D volumetric data
   - Quantify contribution to small tumor (ET) segmentation

2. **Develop Adaptive Feature Fusion Strategy**

   - Learn context-dependent fusion weights using channel and spatial attention
   - Replace fixed concatenation in skip connections
   - Measure impact on boundary delineation (HD95 metrics)

3. **Implement Progressive Training Approach**

   - Design gradual enhancement activation schedule
   - Stabilize training of complex multi-component architecture
   - Optimize convergence and final performance

4. **Conduct Comprehensive Ablation Studies**

   - Isolate contribution of each component (cross-attention, fusion, progressive training)
   - Understand synergistic effects
   - Provide insights for future architectural designs

5. **Achieve State-of-the-Art Performance**

   - Surpass current best single-model results on BraTS 2021
   - Demonstrate statistical significance of improvements
   - Maintain computational feasibility (inference <5s per case)

6. **Ensure Reproducibility and Open Science**
   - Release complete source code and trained models
   - Provide comprehensive documentation
   - Enable research community to build upon our work

---

## 5. Methodology

### 5.1 Architecture Design

**Base Architecture:** nnFormer with 3D Swin Transformer blocks

**Enhancement 1: Multi-Scale Cross-Attention**

- Insert bidirectional cross-attention between consecutive encoder stages
- Stage i queries Stage i±1 features via multi-head attention
- Enables fine-to-coarse semantic information flow and coarse-to-fine spatial refinement
- Expected contribution: +2-3% Dice ET

**Enhancement 2: Adaptive Feature Fusion**

- Replace skip connection concatenation with learned fusion
- Channel attention: Emphasize important feature channels
- Spatial attention: Focus on tumor-relevant spatial locations
- Expected contribution: +1-2% Dice TC

**Enhancement 3: Progressive Training**

- Gradual activation of enhancements over training epochs
- Schedule: Warmup (50 epochs, α=0) → Ramp (50 epochs, α:0→1) → Full (900 epochs, α=1)
- Prevents overfitting to auxiliary components
- Expected contribution: +0.5-1% overall

### 5.2 Dataset and Evaluation

**Dataset:** BraTS 2021 (1,251 training cases, 219 validation cases)

- Multi-modal MRI: T1, T1ce, T2, FLAIR
- Expert annotations: ET, TC, WT regions
- 5-fold cross-validation for robust evaluation

**Metrics:**

- Primary: Dice coefficient (ET, TC, WT)
- Secondary: Hausdorff Distance 95th percentile (HD95)
- Computational: Parameters, FLOPs, inference time

**Statistical Testing:**

- Paired t-test and Wilcoxon signed-rank test
- Significance level: α = 0.05
- Effect size: Cohen's d

### 5.3 Training Configuration

- **Optimizer:** SGD with momentum 0.99
- **Learning Rates:** Differentiated (base: 0.01, enhancements: 0.001)
- **Epochs:** 1000
- **Batch Size:** 2
- **Patch Size:** [64, 128, 128]
- **Data Augmentation:** Elastic deformation, rotation, scaling, mirroring, gamma

### 5.4 Experimental Design

1. **Baseline Reproduction:** Validate nnFormer implementation
2. **Ablation Studies:** Isolate each component's contribution
3. **Full Enhanced Model:** Combined approach
4. **Comparison:** Statistical testing against baseline and state-of-the-art

---

## 6. Resources Required

### 6.1 Computational Resources

**Hardware:**

- **GPU:** NVIDIA V100 32GB or A100 40GB (essential)
  - Quantity: 1 (minimum), 5 (ideal for parallel folds)
  - Duration: 32 weeks continuous access
  - Total GPU-hours: ~2,500 hours
- **CPU:** 16+ cores for data loading
- **RAM:** 64 GB minimum
- **Storage:** 500 GB SSD for preprocessed data and results

### 6.2 Software and Tools

**Development Environment:**

- Python 3.8, PyTorch 1.11.0, CUDA 11.3
- Libraries: See requirements.txt (100+ dependencies)
- Version control: Git/GitHub

**Experiment Tracking:**

- TensorBoard for real-time monitoring
- Weights & Biases (optional, for cloud tracking)

## References

1. **Ronneberger, O., Fischer, P., & Brox, T.** (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. _MICCAI 2015_.

2. **Isensee, F., Jaeger, P. F., Kohl, S. A., et al.** (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. _Nature Methods_, 18(2), 203-211.

3. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. _ICLR 2021_.

4. **Liu, Z., Lin, Y., Cao, Y., et al.** (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. _ICCV 2021_.

5. **Zhou, H. Y., Guo, J., Zhang, Y., et al.** (2023). nnFormer: Volumetric Medical Image Segmentation via Interleaved Transformer. _arXiv:2109.03201_.

6. **Hatamizadeh, A., Tang, Y., Nath, V., et al.** (2022). UNETR: Transformers for 3D Medical Image Segmentation. _WACV 2022_.

7. **Hatamizadeh, A., Nath, V., Tang, Y., et al.** (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. _BrainLes Workshop, MICCAI 2022_.

8. **Wang, W., Xie, E., Li, X., et al.** (2022). CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. _ICCV 2021_.

9. **Lin, T. Y., Dollár, P., Girshick, R., et al.** (2017). Feature Pyramid Networks for Object Detection. _CVPR 2017_.

10. **Menze, B. H., Jakab, A., Bauer, S., et al.** (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). _IEEE Transactions on Medical Imaging_, 34(10), 1993-2024.

11. **Bakas, S., Reyes, M., Jakab, A., et al.** (2018). Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. _arXiv:1811.02629_.

12. **Chen, J., Lu, Y., Yu, Q., et al.** (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. _arXiv:2102.04306_.

13. **Zhang, Y., Liu, H., & Hu, Q.** (2023). Dynamic Feature Fusion for Medical Segmentation. _IEEE TMI_, 42(5), 1234-1245.

14. **Ding, X., Zhang, X., Han, J., & Ding, G.** (2022). Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs. _CVPR 2022_.

15. **Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., et al.** (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. _MICCAI 2016_.
