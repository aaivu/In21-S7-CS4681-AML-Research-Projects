# Research Proposal: Exploring Hyperparameters and Training Strategies in Plenoxels: Radiance Fields without Neural Networks

**Student:** 210302P <br>
**Research Area:** 3D Vision:Neural Radiance Fields <br>
**Date:** 2025-09-01 <br>

## Abstract

Neural Radiance Fields (NeRFs) have revolutionized novel view synthesis but are constrained by severe computational demands, limiting real-time use. Plenoxels address this by replacing the neural network with a directly optimized sparse voxel grid, reducing training times from days to minutes. However, this efficiency comes at the cost of sensitivity to hyperparameters, potential training instability, and reconstructions that can lack perceptual quality. This research proposes to enhance the Plenoxels framework by investigating targeted improvements in three key areas: hyperparameter optimization, loss function design, and training strategies. The methodology involves implementing adaptive learning rate schedules like cyclical learning rates, integrating perceptual losses such as LPIPS to improve visual fidelity, and exploring progressive regularization schemes to enhance robustness. The expected outcome is a more reliable and generalizable voxel-based rendering framework capable of producing higher-quality, perceptually convincing reconstructions across diverse scenes while preserving real-time performance, making it more feasible for practical applications in VR/AR and robotics.

## 1. Introduction

Recent advancements in neural scene representation, particularly Neural Radiance Fields (NeRFs), have significantly transformed the field of novel view synthesis. By learning a continuous volumetric representation of a scene, NeRFs can generate highly photorealistic images from arbitrary viewpoints, capturing fine geometric details and complex lighting effects. Despite this impressive visual fidelity, traditional NeRF models are computationally intensive, with training often taking hours or days, making real-time applications practically infeasible.

To overcome these limitations, research has shifted towards sparse and explicit voxel-based representations. Among these, Plenoxels has emerged as a state-of-the-art method. Plenoxels represent scenes using a sparse voxel grid containing opacity and spherical harmonic (SH) coefficients, completely eliminating the need for a neural network. This allows for direct optimization of voxel values, which dramatically accelerates training to just a few minutes while achieving real-time rendering.

This research is significant because while Plenoxels offer a major leap in efficiency, their stability and reconstruction quality are highly dependent on hyperparameter tuning and regularization. There remains a clear opportunity to improve the robustness and visual fidelity of this framework. This work aims to push the performance boundaries of voxel-based neural rendering by investigating targeted enhancements, making the technology more accurate, robust, and applicable to a wider variety of real-world scenes and applications.

## 2. Problem Statement

While Plenoxels provide a fast and effective alternative to NeRFs, the framework exhibits key limitations that hinder its widespread adoption and performance on challenging datasets. The core problems this research aims to solve are:

- Training Instability and Hyperparameter Sensitivity: The model's convergence and final reconstruction quality are highly sensitive to the manual selection of hyperparameters, such as learning rates and regularization weights. This makes the optimization process fragile and difficult to generalize across different scenes without extensive manual tuning.
- Suboptimal Perceptual Quality: The standard Mean Squared Error (MSE) loss function, while mathematically convenient, only measures pixel-level differences. This often leads to reconstructions that appear blurry and lack fine texture details, even when achieving a high quantitative score.
- Lack of Robustness: Challenges remain in achieving stable regularization, especially when training on noisy or incomplete real-world datasets where visual artifacts can easily appear.

This research aims to solve these issues by introducing more sophisticated and automated training methodologies to create a more robust and perceptually-driven Plenoxel framework.

## 3. Literature Review Summary

The field of neural scene representation has rapidly progressed from dense, computationally expensive models toward efficient, voxel-centric frameworks. This evolution has been driven by the need for real-time performance without sacrificing photorealistic quality.

### 3.1. The Evolution from NeRF to Hybrid Voxel Approaches

The introduction of Neural Radiance Fields (NeRF) marked a major advance in novel view synthesis, demonstrating that continuous volumetric scenes could be learned from 2D images using MLPs. However, NeRF's reliance on dense neural architectures results in computationally expensive training and slow inference, making it unsuitable for real-time applications. To address these limitations, research shifted towards voxel-based extensions. Hybrid methods like Neural Sparse Voxel Fields (NSVF) and PlenOctrees combined voxel grids with neural components to significantly reduce memory usage and accelerate rendering speeds. These models highlight a broader trend in the field: moving away from purely neural representations toward hybrid or voxel-centric ones that better balance efficiency and quality.

### 3.2. The Rise of Plenoxels and Subsequent Enhancements

Plenoxels took this shift to its logical extreme by removing the neural network entirely. Instead, the model directly optimizes voxel densities and spherical harmonics coefficients stored in a sparse grid. This design dramatically reduces training times from days to minutes while achieving results competitive with NeRF on synthetic benchmarks. Since its release, subsequent research has aimed to improve upon the Plenoxels baseline. For example, TensoRF improved memory efficiency and reconstruction quality using tensor decompositions, while FastNeRF investigated modifications for real-time rendering without sacrificing fidelity. These developments underscore that while Plenoxels provide a powerful and efficient foundation, their success is heavily dependent on the choice of hyperparameters and optimization schemes.

### 3.3. The Critical Role of Optimization and Perceptual Losses

The broader literature on radiance fields confirms that hyperparameter settings and loss function design are crucial for determining reconstruction quality and training stability. For voxel-based methods like Plenoxels, key factors include the order of spherical harmonics (SH), regularization weights, and adaptive learning rate schedules (e.g., cosine annealing, cyclical learning rates) that help avoid premature stagnation.

Beyond hyperparameter tuning, the choice of loss function is central. Most models rely on Mean Squared Error (MSE), which penalizes per-pixel intensity differences but often fails to capture perceptual similarity, leading to overly smooth or blurry reconstructions. Alternatives such as the Learned Perceptual Image Patch Similarity (LPIPS) metric and the Structural Similarity Index Measure (SSIM) offer more perceptually aligned supervision by emphasizing edges, textures, and structural coherence. Incorporating these hybrid losses has been shown to produce visually sharper results. Together, these findings highlight the dual importance of principled hyperparameter tuning and advanced loss function design in advancing voxel-based methods like Plenoxels.

## 4. Research Objectives

### 4.1. Primary Objective

The main objective of this project is to investigate and implement targeted enhancements to the Plenoxels framework to improve its stability, reconstruction accuracy, and generalization, making it more robust for real-world applications.

### 4.2. Secondary Objectives

- To mitigate training instability and reduce manual effort by implementing adaptive learning rate schedules like cyclical learning rates.
- To improve the perceptual quality of renderings beyond what pixel-wise MSE can achieve by incorporating structure-aware loss functions like LPIPS.
- To enhance model robustness and prevent over-smoothing by designing and evaluating a progressive regularization scheme and other advanced training strategies.
- To systematically evaluate the proposed enhancements against the baseline Plenoxel model to provide a clear measure of improvement in reconstruction quality.

## 5. Methodology

### 5.1. Baseline Model - Plenoxels

The baseline for this research is the Plenoxels framework. Unlike NeRF, which uses a continuous MLP representation, Plenoxels discretize a scene into a sparse voxel grid. Each occupied voxel stores two key properties:

- Opacity (Density): A scalar value that controls how much light is blocked by that point in space.
- Spherical Harmonics (SH) Coefficients: A set of coefficients (typically degree 2, requiring 27 values per voxel) that model the view-dependent color of the voxel.

Rendering is performed via differentiable volume rendering, similar to NeRF, but optimization is applied directly to the opacity and SH values in the grid instead of to neural network weights. This direct optimization, combined with a coarse-to-fine strategy where the grid resolution is progressively increased, enables extremely fast training and real-time inference. To ensure smoothness and prevent artifacts, the model relies on Total Variation (TV) regularization.

### 5.2. Experimental Setup

Experiments will be conducted using Python with the PyTorch framework. Training and evaluation will be performed on a system with a high-end GPU to handle the computational load. The baseline Plenoxels implementation will serve as the foundation for all modifications, ensuring a controlled comparison.

### 5.3. Planned Modifications

This project will explore targeted, incremental improvements to the baseline model across three main areas.

#### 5.3.1. Hyperparameter Optimization:

- Adaptive Learning Rates: The standard exponential decay learning rate schedule will be replaced with more flexible alternatives, including cosine annealing and cyclical learning rates, to improve convergence stability.
- Various Tuning sets: Variuos hyperparameter sets are searched for optimal values for critical hyperparameters like TV regularization weights, SH degree, and voxel pruning thresholds

#### 5.3.2. Loss Function Enhancement:

- The standard MSE loss will be augmented with perceptual and structure-aware losses to better align with human visual perception. We will implement a hybrid loss function incorporating,LPIPS (Learned Perceptual Image Patch Similarity) to measure similarity in a deep feature space, capturing semantic details like edges and textures.

#### 5.3.3. Training Strategy Improvements:

- Progressive Regularization: A scheme where strong TV and sparsity penalties are applied in early, coarse optimization stages and are gradually weakened in finer stages. This prevents over-smoothing while ensuring stability.
- Transfer Learning & Ensembling: We will also investigate using pretrained voxel grids to accelerate convergence on new scenes and explore ensembling models trained with different hyperparameters to enhance robustness.

### 5.4. Evaluation

The performance of the modified Plenoxel models will be rigorously compared against the original baseline using a combination of quantitative metrics and qualitative analysis.

#### 5.4.1. Quantitative Metrics:

- PSNR (Peak Signal-to-Noise Ratio): A standard metric for reconstruction accuracy.
- MSE (Mean Squared Error): To measure pixel-level differences.
- LPIPS (Learned Perceptual Image Patch Similarity): To evaluate perceptual quality.
  training time

#### 5.4.2. Qualitative Analysis:

Visual inspection of rendered images to assess sharpness, detail preservation, and the presence of artifacts.

### 5.5. Datasets

Experiments will be conducted on standard benchmarks for neural rendering to ensure comparability with existing work. The primary dataset will be the NeRF Synthetic Dataset, which includes complex scenes with challenging geometry and view-dependent effects. We will also test on real-world datasets to evaluate the robustness of our proposed enhancements.

## 6. Expected Outcomes

This project is expected to deliver several key outcomes:

- An Enhanced Plenoxels Framework: An improved, open-source implementation of Plenoxels that is more stable and produces reconstructions with higher visual fidelity.
- Empirical Insights: A comprehensive analysis of how adaptive learning rates, perceptual loss functions, and advanced training strategies impact the performance of voxel-based neural rendering.
- A More Robust Rendering Model: The proposed enhancements are expected to make the model more reliable and generalizable, particularly for noisy and diverse real-world datasets.
- Contribution to Real-Time Graphics: By improving the quality and reliability of a real-time capable rendering framework, this work will contribute towards making practical applications in VR/AR, robotics, and graphics production more feasible.

## 7. Timeline

| Week | Task                                                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------ |
| 1-2  | Background Study & Setup: In-depth literature review, setup dev environment, run baseline analysis.                      |
| 3-4  | Methodology Implementation: Implement adaptive learning rates and perceptual loss functions.                             |
| 5-7  | Experimentation: Train and test modified models on datasets, focusing on hyperparameter optimization and loss functions. |
| 8    | Advanced Testing: Implement and test training strategy improvements (e.g., progressive regularization).                  |
| 9-10 | Final Analysis & Writing: Analyze all results, create visualizations, write the final paper.                             |
| 11   | Finalization: Finalize paper and submit to a conference.                                                                 |

## 8. Resources Required

- Hardware: A high-performance workstation with a modern NVIDIA GPU (e.g., RTX 3090/4090 or equivalent cloud instance) for efficient training and experimentation.
- Datasets: Access to standard public datasets, including the NeRF Synthetic Dataset.
- Python 3.8+ and the PyTorch deep learning framework.
- Libraries: NumPy, OpenCV, scikit-image, LPIPS, scikit-optimize.
- Version Control: Git and GitHub for code management and collaboration.

## References

[1] A. Yu, S. Fridovich-Keil, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” arXiv preprint arXiv:2112.05131, 2021.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” arXiv preprint arXiv:2003.08934, 2020.

[3] L. Liu, J. Gu, K. Z. Lin, T. S. Chua, and C. Theobalt, “Neural sparse voxel fields,” in Advances in Neural Information Processing Systems, vol. 33, pp. 15651–15663, 2020.

[4] L. Liu, “Neural Sparse Voxel Fields,” Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), 2020.

[5] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa, “PlenOctrees for real-time rendering of neural radiance fields,” Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Montreal, Canada, Oct. 2021, pp. 5752–5761, doi: 10.1109/ICCV48922.2021.00570

[6] L. Wang, J. Zhang, X. Liu, F. Zhao, Y. Zhang, Y. Zhang, M. Wu, L. Xu, and J. Yu, “Fourier PlenOctrees for dynamic radiance field rendering in real-time,” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), New Orleans, LA, USA, Jun. 2022, pp. 13514–13524, doi: 10.1109/CVPR52688.2022.01316

[7] C. Yang, M. Xiao, H. Jia, Y. Xu, H. Duan, and X. Liu, “P‐3.4: Plenoxels‐based Parallax Map Generation for Flexible Scale Ultra‐High Resolution in 3D Imaging,” SID Symposium Digest of Technical Papers, vol. 55, no. S1, pp. 725–728, Apr. 2024, doi: 10.1002/sdtp.17187.

[8] K. Jun-Seong, K. Yu-Ji, M. Ye-Bin, and T.-H. Oh, “HDR-Plenoxels: Self-Calibrating high dynamic range radiance fields,” in Lecture notes in computer science, 2022, pp. 384–401. doi: 10.1007/978-3-031-19824-3_23.

[9] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “TeNSORF: Tensorial Radiance Fields,” in Lecture notes in computer science, 2022, pp. 333–350. doi: 10.1007/978-3-031-19824-3_20.

[10] H. Jang and D. Kim, “D-TensoRF: Tensorial radiance fields for dynamic scenes,” arXiv.org, Dec. 05, 2022. https://arxiv.org/abs/2212.02375

[11] A. Luthra, J. Shi, X. Song, Z. Lin, and H. Yu, “Generalized TensoRF: Efficient multi-scene radiance fields and view-consistent 3D editing,” Companion Proc. ACM SIGGRAPH Symp. Interact. 3D Graphics Games (I3D), May 2025, pp. 1–3, doi: 10.1145/3722564.3728385

[12] S. J. Garbin, M. Kowalski, M. Johnson, J. Shotton, and J. Valentin, “FastNeRF: High-fidelity neural rendering at 200FPS,” Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Montreal, Canada, Oct. 2021, pp. 14346–14355, doi: 10.1109/ICCV48922.2021.01408

[13] K. Wadhwani and T. Kojima, “SqueezeNeRF: Further factorized FastNeRF for memory-efficient inference,” Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) Workshops, New Orleans, LA, USA, Jun. 2022, pp. 2717–2725, doi: 10.1109/CVPRW56347.2022.00307

[14] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5470–5479, 2022.

[15] M. Debbagh, “Neural Radiance Fields (NeRFs): A review and some insights,” arXiv preprint arXiv:2305.00375, 2023.

[16] A. K. M. Rabby, “BeyondPixels: A comprehensive review of the evolution of NeRF,” arXiv preprint arXiv:2306.03000, 2023.

[17] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[18] R. Zhang, “Learned Perceptual Image Patch Similarity (LPIPS): A perceptual metric for evaluating image similarity,” TransferLab Blog, 2023. [Online]. Available: https://transferlab.ai/blog/perceptual-metrics/

[19] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004.

[20] T. Oanda, “A review of the image quality metrics used in image synthesis models,” Paperspace Blog, 2023. [Online]. Available: https://blog.paperspace.com/review-metrics-image-synthesis-models

---
