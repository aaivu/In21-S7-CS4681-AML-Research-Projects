# Research Proposal: 3D Vision: 3D Object Detection

**Student:** 210151B
**Research Area:** 3D Vision: 3D Object Detection
**Date:** 2025-09-01

## Abstract

This research proposes a systematic investigation into the effects of three complementary enhancements—**Enhanced Data Augmentation (EDA)**, **Multi-Scale Attention Fusion (MSAF)**, and **Dynamic Focal Loss (DFL)**—on the performance of the state-of-the-art **PV-RCNN++** model for LiDAR-based 3D object detection. The core problem is addressing sub-optimal feature integration and training imbalance within the complex Point-Voxel architecture. We will implement these modules within the OpenPCDet framework and rigorously evaluate their isolated and combined impact on the KITTI 3D benchmark across Car, Pedestrian, and Cyclist categories. The expected outcome is a detailed analysis of performance changes in 3D Average Precision (AP) and inference efficiency, providing insights for future development of robust and accurate 3D detectors.

---

## 1. Introduction

Accurate 3D object detection from **LiDAR point clouds** is a foundational technology for autonomous driving and robotics. While hybrid Point-Voxel models like PV-RCNN++ have achieved state-of-the-art results by balancing computational efficiency and geometric precision, they still face challenges related to **feature representation** (integrating context across different scales) and **training stability** (handling severe class and sample imbalance). This research is significant as it directly attempts to push the performance limits of a leading detection framework by integrating well-established computer vision techniques into the 3D domain.

---

## 2. Problem Statement

The research problem is two-fold:
1. **Feature Integration Limitation:** The PV-RCNN++ voxel backbone may not effectively fuse context and dependencies across its multi-scale feature maps, potentially limiting detection accuracy, especially for distant or small objects.
2. **Training Imbalance:** The inherent data imbalance in 3D detection causes the loss function to be dominated by easy negative samples, preventing the model from effectively learning from hard, informative samples.

This proposal aims to determine whether the strategic integration of **MSAF**, **DFL**, and **EDA** can successfully mitigate these limitations and yield a quantifiable improvement in the 3D detection accuracy of the PV-RCNN++ model on a real-world benchmark.

---

## 3. Literature Review Summary

The review established that 3D object detection has evolved from early **Voxel-based** (e.g., VoxelNet, SECOND) and **Point-based** (e.g., PointNet, PointRCNN) methods to sophisticated **Hybrid Point-Voxel** architectures, with **PV-RCNN++** representing the current peak.
**Gaps Identified:**
1. **Lack of explicit cross-scale feature integration** in the voxel backbone. (Addressed by MSAF)
2. **Sub-optimal loss functions** for handling the intrinsic imbalance of 3D data. (Addressed by DFL)

---

## 4. Research Objectives

### Primary Objective
To systematically implement and evaluate the effect of Enhanced Data Augmentation (EDA), Multi-Scale Attention Fusion (MSAF), and Dynamic Focal Loss (DFL) on the 3D Average Precision (AP) of the PV-RCNN++ model using the KITTI benchmark.

### Secondary Objectives
- **Objective 1:** To implement the MSAF module into the PV-RCNN++ voxel backbone and measure its impact on long-range object detection.
- **Objective 2:** To replace the standard loss with DFL and quantify its effectiveness in mitigating sample imbalance and improving AP for all three categories (Car, Pedestrian, Cyclist).
- **Objective 3:** To analyze the computational overhead (inference speed/FPS) of the fully enhanced model compared to the PV-RCNN++ baseline.

---

## 5. Methodology

The research design is a **Single-Subject Comparative Design** using the PV-RCNN++ model as the baseline. All implementations will be conducted within the **OpenPCDet** framework.

1.  **Baseline Establishment:** Train and evaluate the original PV-RCNN++ on the KITTI validation set.
2.  **Module Implementation:** Implement the MSAF and DFL modules, and the EDA pipeline modification.
3.  **Systematic Experimentation:** Conduct four main experiments: Baseline + EDA, Baseline + MSAF, Baseline + DFL, and Baseline + EDA + MSAF + DFL (Full Model).
4.  **Evaluation:** Use the **KITTI 3D Average Precision (AP)** for the **Moderate** difficulty setting (IoU 0.7 for Car, 0.5 for Pedestrian/Cyclist) and **Inference FPS** as evaluation metrics.

---

## 6. Expected Outcomes

The research is expected to yield the following outcomes:

1.  **Enhanced Model:** A functionally enhanced version of PV-RCNN++ integrated with MSAF, DFL, and EDA, publicly available on a GitHub repository.
2.  **Detailed Analysis:** A comprehensive technical report (Final Paper) providing a comparative analysis of the isolated and combined effects of the enhancements on 3D AP and efficiency.
3.  **Core Contribution:** A feasibility study demonstrating that while the additions are technically viable, the highly optimized PV-RCNN++ baseline is difficult to surpass without deeper architectural modifications, as the current experiments **do not present any significant improvements** over the original PV-RCNN++ in terms of 3D AP or inference efficiency.

---

## 7. Timeline

| Week | Task |
|:------|:------|
| 1-2 | Literature Review (Completed and Refined) |
| 3-4 | Methodology Development & Environment Setup (OpenPCDet, KITTI) |
| 5-8 | Implementation of EDA, MSAF, and DFL Modules |
| 9-12| Systematic Experimentation (Training all configurations and gathering results) |
| 13-15| Results Analysis, Interpretation, and Draft Writing (Short/Final Paper) |
| 16 | Final Submission & Presentation |

---

## 8. Resources Required

* **Dataset:** KITTI 3D Object Detection Benchmark.
* **Software:** OpenPCDet framework, PyTorch, Python 3.x.
* **Hardware:** High-performance GPU workstation (e.g., NVIDIA RTX series) for accelerated training.
* **Code:** GitHub repository for version control and sharing.

---

## References

1.  M.W.P. Dulmith, R.T. Uthayasanker, "Effects of Data Augmentation, Attention Fusion, and Dynamic Loss on 3D LiDAR Detection using PV-RCNN++," Final Paper, University of Moratuwa, 2025.
2.  S. Shi, L. Jiang, J. Deng, Z. Wang, C. Guo, J. Shi, X. Wang, and H. Li, "PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection," *Int. J. Comput. Vis.*, vol. 131, no. 3, pp. 531-551, Mar 2023.
3.  S. Shi, C. Guo, L. Jiang, Z. Wang, J. Shi, X. Wang, and H. Li, "PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020.
4.  T.-Y. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318-327, Feb 2020.
5.  Y. Zhou and O. Tuzel, "VoxelNet: End-to-end learning for point cloud based 3d object detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018.
