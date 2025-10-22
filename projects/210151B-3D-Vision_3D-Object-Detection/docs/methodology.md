# Methodology: 3D Vision: 3D Object Detection

**Student:** 210151B
**Research Area:** 3D Vision: 3D Object Detection
**Date:** 2025-09-01

## 1. Overview

The methodology centers on a comparative study to evaluate the effects of three complementary enhancements—**Enhanced Data Augmentation (EDA)**, **Multi-Scale Attention Fusion (MSAF)**, and **Dynamic Focal Loss (DFL)**—when integrated into the state-of-the-art **PV-RCNN++** baseline model for 3D object detection. The approach is primarily empirical, involving modifications to the existing OpenPCDet framework, systematic training, and rigorous evaluation on the KITTI benchmark.

## 2. Research Design

The research follows a **Single-Subject Comparative Design**, where the baseline model (PV-RCNN++) is progressively enhanced with new modules. This allows for the measurement of the isolated and combined impact of each enhancement on detection performance and inference efficiency.

1.  **Baseline:** Train and evaluate the original PV-RCNN++ model.
2.  **Enhancement 1 (EDA):** Baseline + Enhanced Data Augmentation.
3.  **Enhancement 2 (MSAF):** Baseline + Multi-Scale Attention Fusion (integrated into the voxel backbone).
4.  **Enhancement 3 (DFL):** Baseline + Dynamic Focal Loss (applied to the detection head).
5.  **Full Model:** Baseline + EDA + MSAF + DFL (The fully integrated model).

## 3. Data Collection

### 3.1 Data Sources
The primary data source is the **KITTI 3D Object Detection Benchmark**.

### 3.2 Data Description
The KITTI dataset contains LiDAR point clouds and corresponding RGB images for autonomous driving scenarios. The study focuses exclusively on the LiDAR data and the following three categories for detection: **Car**, **Pedestrian**, and **Cyclist**. The official KITTI training set is divided into the standard **3,712 training samples** and **3,769 validation samples**.

### 3.3 Data Preprocessing
Standard preprocessing steps are inherited from the PV-RCNN++ implementation within the OpenPCDet framework:
1.  **Point Cloud Filtering:** Points outside the defined region-of-interest (ROI) are removed.
2.  **Voxelization:** Point clouds are converted into sparse voxels for the Voxel-based CNN backbone.
3.  **Ground Truth Sampling:** Data augmentation is applied via the standard database sampling approach.
4.  **Enhanced Data Augmentation (EDA):** The proposed EDA module introduces improved rotation, scaling, flipping, and a novel **density-aware point dropout** mechanism.

## 4. Model Architecture

The base architecture is **PV-RCNN++**. The key modifications are:

* **Multi-Scale Attention Fusion (MSAF):** This module is inserted into the PV-RCNN++ Voxel Feature Extractor (VFE) backbone. It uses an attention mechanism to explicitly aggregate and combine features across different scales of the 3D sparse convolutional network output, providing richer context to the subsequent point-based refinement stages.
* **Dynamic Focal Loss (DFL):** This adaptive loss function replaces the standard Focal Loss in the prediction head. It dynamically adjusts the focusing parameter ($\gamma$) on a per-sample basis, based on the predicted difficulty or uncertainty of the bounding box regression and classification, to focus training on hard negative and hard positive examples.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
The primary metric is **3D Average Precision (AP)** for the **Moderate** difficulty setting, following the official KITTI protocol. This is reported across the three target classes: Car, Pedestrian, and Cyclist.

* **Car:** IoU threshold 0.7
* **Pedestrian & Cyclist:** IoU threshold 0.5
* **Secondary Metric:** Inference efficiency (Frames Per Second - FPS) on the validation set.

### 5.2 Baseline Models
The sole baseline is the official, pre-trained **PV-RCNN++** model (as implemented in OpenPCDet), allowing for direct, controlled comparison against the enhanced versions.

### 5.3 Hardware/Software Requirements
* **Software Framework:** OpenPCDet, Python, PyTorch.
* **Hardware:** Training was conducted on high-performance GPUs (e.g., NVIDIA RTX series or equivalent), emphasizing the need for significant computational resources due to the complexity of the PV-RCNN++ architecture and the large point cloud dataset.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|:-------|:-------|:----------|:--------------|
| Phase 1 | **Baseline Setup & Verification** | 1 week | PV-RCNN++ baseline performance verified on KITTI val set |
| Phase 2 | **Module Implementation (MSAF & DFL)** | 4 weeks | Working, integrated MSAF and DFL modules within OpenPCDet |
| Phase 3 | **Systematic Training & Evaluation** | 4 weeks | Training logs and 3D AP results for all four configurations (Baseline, EDA, MSAF, DFL, Full) |
| Phase 4 | **Analysis & Reporting** | 2 weeks | Comparative analysis of performance and inference time; Final Report |

## 7. Risk Analysis

| Risk | Mitigation Strategy |
|:---|:---|
| **Complexity of Integration:** New modules break the highly coupled PV-RCNN++ framework. | Implement MSAF and DFL initially as stand-alone replacements for existing components (e.g., swapping Focal Loss). Use extensive unit tests on feature shapes and tensors. |
| **Computational Cost:** Training complex models for extended periods is time-consuming. | Utilize pre-trained weights where possible and focus training time on the most promising configurations first. Use sparse convolutions for efficiency. |
| **No Performance Gain:** The enhancements fail to improve AP over the already strong baseline. | A comprehensive analysis will still be performed to explain *why* the enhancement was ineffective (e.g., baseline is already saturated, hyperparameter tuning is required). |

## 8. Expected Outcomes

The primary expected outcome was a statistically significant improvement in 3D Average Precision (AP) for the Car, Pedestrian, and Cyclist classes at the Moderate difficulty level compared to the PV-RCNN++ baseline. Specifically, MSAF was expected to improve detection of distant objects, and DFL was expected to boost overall AP by mitigating sample imbalance. The overall project contribution is the **feasibility study and systematic evaluation** of these enhancements on a cutting-edge 3D detection model.
