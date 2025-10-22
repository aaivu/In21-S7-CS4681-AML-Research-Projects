# Literature Review: Exploring Hyperparameters and Training Strategies in Plenoxels: Radiance Fields without Neural Networks

**Student:** 210302P <br>
**Research Area:** 3D Vision:Neural Radiance Fields <br>
**Date:** 2025-09-01 <br>

## Abstract

This literature review provides a comprehensive analysis of the evolution of neural scene representations for novel view synthesis, tracing the progression from the foundational Neural Radiance Fields (NeRF) to modern, efficient voxel-based methods like Plenoxels. The review covers three key areas: the initial shift from dense neural networks to hybrid and explicit voxel-centric representations, the specific architectural improvements made to the Plenoxels framework since its inception, and the critical role of hyperparameter optimization and perceptual loss functions in achieving high-fidelity 3D reconstruction. The key finding is that while the field has successfully addressed the computational bottlenecks of early NeRF models, the current research frontier lies in enhancing the training stability, robustness, and perceptual quality of these faster models. This review identifies a clear gap in the systematic exploration of advanced training strategies and perceptually-aware loss functions for network-free radiance fields, which directly informs the direction of this project. Through a critical analysis of over 20 key papers, this review establishes the context and motivation for research focused on refining the optimization process of explicit radiance fields to unlock their full potential for real-time, photorealistic rendering.

## 1. Introduction

The ability to synthesize photorealistic images of a 3D scene from novel viewpoints is a cornerstone of modern computer graphics and vision, with profound implications for applications ranging from virtual and augmented reality (VR/AR) to robotics, digital twins, and entertainment. For decades, this task was dominated by traditional graphics pipelines that required explicit 3D geometry and complex material properties. The introduction of Neural Radiance Fields (NeRF) by Mildenhall et al. (2020) revolutionized this area by demonstrating that a continuous, implicit volumetric scene representation could be learned directly from a sparse set of 2D images, achieving unprecedented photorealism.

However, the immense computational cost of traditional NeRF models, stemming from their reliance on large multilayer perceptrons (MLPs) queried hundreds of times per pixel, has severely limited their practical application. Training a single scene can take hours or even days on high-end GPUs, and rendering a single frame is far too slow for interactive use cases. This fundamental limitation has catalyzed a wave of research aimed at drastically improving the efficiency of radiance fields without compromising their remarkable quality.

This literature review explores the subsequent body of research that has emerged in response to this challenge. The scope of this review covers the pivotal and rapid shift from purely implicit, neural network-based representations to explicit, voxel-based frameworks that prioritize computational efficiency. We will critically analyze the state-of-the-art Plenoxels model—a paradigm-shifting approach that completely eliminates the neural network—as well as its subsequent improvements and the broader context of optimization techniques and loss functions that govern its performance. By charting this evolution, this review aims to provide a deep understanding of the current landscape, identify the most pressing unsolved challenges, and thereby contextualize the research gaps that this project is designed to address.

## 2. Search Methodology

### Search Terms Used

A combination of broad and specific search terms was used to capture the relevant literature. These included:

- Primary Terms: Neural Radiance Fields, NeRF, Plenoxels, Radiance Fields without Neural Networks, Novel View Synthesis.
- Efficiency-Focused Terms: Efficient NeRF, Fast NeRF, Real-Time Radiance Fields, Voxel-Based Rendering, Neural Voxel Fields.
- Optimization and Training Terms: Hyperparameter Tuning in NeRF, NeRF Optimization, Perceptual Loss for 3D Reconstruction, LPIPS for NeRF, Cosine Annealing, Cyclical Learning Rates.
- Architectural Variants: TensoRF, PlenOctrees, Neural Sparse Voxel Fields.

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv

### Time Period

The review focuses on work published from 2020 to 2025. This period was chosen to start with the publication of the original NeRF paper and cover the subsequent explosion of research into efficient alternatives. Seminal papers on foundational concepts, such as the SSIM and LPIPS metrics, from before this period were also included to provide necessary theoretical context.

## 3. Key Areas of Research

The literature on efficient radiance fields can be broadly categorized into three interconnected areas: the architectural evolution from implicit to explicit models, the specific enhancements made to the dominant explicit frameworks like Plenoxels, and the overarching role of the optimization process in achieving high-quality results.

### 3.1. The Foundational Paradigm: From NeRF to Hybrid Voxel Approaches

The introduction of Neural Radiance Fields (NeRF) marked a major advance in novel view synthesis, showing that continuous volumetric scene representations could be learned from 2D images using multilayer perceptrons (MLPs). The core idea of NeRF is to represent a scene as a 5D function that maps a 3D coordinate (x, y, z) and a 2D viewing direction (θ, φ) to a volume density (σ) and an emitted color (c). This function is approximated by an MLP. To render an image, rays are cast through each pixel, and the volume rendering equation is numerically integrated by sampling points along each ray and querying the MLP. This approach achieved unprecedented photorealism. However, its reliance on dense neural architectures makes training computationally expensive and inference slow, often requiring hours of optimization per scene.

To alleviate these challenges, a trend emerged towards voxel-based extensions, which sought to combine the benefits of explicit data structures with the representational power of neural networks. These hybrid models aimed to reduce the number of expensive MLP queries by storing information in a grid.

**Key Papers:**

- Mildenhall et al. (2020): Introduced the foundational NeRF model, demonstrating state-of-the-art photorealistic rendering quality. This paper set the benchmark for quality but also highlighted the severe computational challenges that the field would need to overcome.
- Liu et al. (2020): Proposed Neural Sparse Voxel Fields (NSVF), a prominent early hybrid approach. NSVF represents a scene with a sparse voxel octree where each non-empty voxel stores a learned feature embedding. A small MLP is then used to decode this feature into color and density. By localizing computation to relevant areas of the scene, NSVF significantly reduced both memory usage and training time compared to the original NeRF.
- Yu et al. (2021): Advanced this direction with PlenOctrees. This work focused on achieving real-time rendering speeds by baking a pre-trained NeRF into a sparse octree data structure that stored pre-computed spherical harmonic coefficients. While this enabled interactive frame rates, its primary drawback was the requirement of a fully trained NeRF as a prerequisite, making the overall process (training + baking) still very time-consuming.

These hybrid models were a crucial stepping stone, demonstrating that explicit data structures could drastically improve efficiency. However, their continued reliance on some form of neural network meant that a significant computational bottleneck remained.

### 3.2. The Network-Free Revolution: Plenoxels and Subsequent Enhancements

The next logical step in the pursuit of efficiency was to question whether the neural network was necessary at all. This led to a paradigm shift towards purely explicit representations, with Plenoxels at the forefront. Plenoxels took the shift towards explicit representations to its extreme by removing the neural network entirely. Instead of learning the weights of an MLP, the model directly optimizes voxel densities and spherical harmonics (SH) coefficients stored in a sparse grid. This direct optimization, combined with a coarse-to-fine training strategy where the grid resolution is progressively increased, allows for extremely rapid convergence.

This design dramatically reduced training times from days to minutes per scene while producing results competitive with NeRF on synthetic benchmarks. The success of Plenoxels spurred further innovation in network-free radiance fields, with researchers exploring more efficient data structures and regularization techniques.[Continue with other relevant areas]

**Key Papers:**

- Yu et al. (2021): Introduced Plenoxels, the first highly efficient radiance field model that works without a neural network. Its key innovation was demonstrating that high-quality view synthesis could be achieved through direct optimization of a simple voxel grid, a concept that fundamentally changed the direction of the field.
- Chen et al. (2022): Proposed TensoRF, a highly influential follow-up that argued a simple voxel grid was not the most efficient representation. TensoRF represents the 4D radiance field as a set of low-rank tensor components (specifically using CANDECOMP/PARAFAC decomposition). This tensor factorization leads to a much more compact model, significantly reducing memory footprint compared to Plenoxels while often achieving better reconstruction quality. Further works like D-TensoRF extended this to dynamic scenes.
- Garbin et al. (2021): Developed FastNeRF, which, while still using a neural network, focused on modifications for extreme real-time rendering at over 200 FPS. It achieved this by caching the output of a position-dependent MLP into a 3D grid and using a second, much smaller view-dependent MLP. This work, along with others like SqueezeNeRF, is part of the broader research effort to make radiance fields practical for interactive applications.

### 3.3 The Critical Role of Optimization and Perceptual Losses

The shift to explicit, network-free models like Plenoxels moved the core challenge from network architecture design to the direct optimization of a massive number of parameters (the voxel values). The broader literature on radiance fields consistently shows that the performance of these models is critically dependent on hyperparameter settings and the choice of loss function. This is even more true for explicit models, which lack the implicit regularization that deep neural networks often provide.

For voxel-based methods like Plenoxels, key factors include the order of spherical harmonics (SH), which controls the complexity of view-dependent effects, regularization weights that influence the trade-off between sharpness and smoothness, and learning rate schedules. As highlighted in review papers by Debbagh (2023) and Rabby (2023), adaptive strategies like cosine annealing and cyclical learning rates are crucial for navigating the complex loss landscapes of these models and avoiding premature stagnation.

Beyond hyperparameter tuning, the design of the loss function is central to the final quality of the reconstruction. Most models rely on Mean Squared Error (MSE), a pixel-wise metric that penalizes intensity differences. However, MSE is notoriously poor at capturing perceptual similarity. An image with a slight spatial shift or minor high-frequency noise can have a high MSE, even if it looks nearly identical to a human observer. This leads to MSE-optimized models often producing overly smooth textures and blurred edges.

**Key Papers:**

- Barron et al. (2022): While focused on NeRF, their work on Mip-NeRF 360 highlights the immense importance of proper regularization and scene parameterization for achieving high-quality results on unbounded, 360-degree scenes, a lesson that applies broadly to all radiance field models.
- Zhang et al. (2018): Introduced the Learned Perceptual Image Patch Similarity (LPIPS) metric. LPIPS revolutionized perceptual quality assessment by comparing images in the feature space of a pre-trained deep neural network (like VGG or AlexNet). Because these networks are trained on image classification, their features are more sensitive to object shapes, textures, and structures, making LPIPS a much better proxy for human perception than MSE.
- Wang et al. (2004): Developed the Structural Similarity Index Measure (SSIM), an influential and widely used metric that assesses image quality based on three components: luminance, contrast, and structure. It provides a more nuanced measure of similarity than simple pixel-wise errors and has been shown to correlate well with human judgment. The principles behind SSIM and LPIPS underscore the need for loss functions that go beyond simple color accuracy.

## 4. Research Gaps and Opportunities

The literature reveals a clear trajectory from slow but high-quality models to fast, efficient ones. However, this speed has introduced new challenges, particularly in the stability and robustness of the training process, leading to several research gaps.

### Gap 1: Lack of Systematic Hyperparameter Exploration

The performance of explicit radiance fields like Plenoxels is highly sensitive to the choice of hyperparameters, including learning rates, scheduler parameters, and regularization weights. Most works on Plenoxels and related models rely on fixed, manually-tuned configurations that were optimized for specific benchmark datasets. These settings may not generalize well to different scene types, camera setups, or lighting conditions. There is a significant lack of systematic investigation into how advanced, adaptive optimization strategies affect the stability, convergence speed, and final quality of these network-free models.

**Why it matters:** The brittle nature of current models and their reliance on expert tuning creates a barrier to entry for non-experts and makes them less reliable for real-world deployment. A more principled approach to optimization could unlock better performance and make the models more robust and easier to use.

**How this project addresses it:** This project directly tackles this gap by systematically implementing and evaluating the impact of different learning rate schedules (cosine annealing, cyclical rates) and a progressive regularization scheme. This provides empirical evidence for more robust and generalizable training strategies.

### Gap 2: Over-reliance on Pixel-Wise Loss Functions

The vast majority of radiance field models, including Plenoxels, are trained using Mean Squared Error (MSE) as the primary objective. As discussed, MSE is a poor proxy for human visual perception. This leads to a divergence between the quantitative metrics used for evaluation (like PSNR, which is based on MSE) and the true qualitative experience. A model can achieve a state-of-the-art PSNR score while producing a blurry image that a human would rate as low quality.

- **Why it matters:** For real-world applications in virtual reality, digital heritage, e-commerce, and entertainment, perceptual realism and the preservation of fine details are often more important than pixel-perfect color accuracy. The current training paradigm is not optimized for this goal.
- **How this project addresses it:** This research takes the first step by rigorously evaluating all reconstructions using the perceptual LPIPS metric, providing a more human-aligned assessment of quality. This analysis highlights the shortcomings of MSE and lays the groundwork for future work to directly incorporate perceptual losses into the training objective to guide the model towards generating visually sharper and more detailed results.

## 5. Theoretical Framework

The theoretical foundation of this research is built upon two core concepts that underpin modern radiance fields:

- Differentiable Volume Rendering: This is the rendering technique, popularized by NeRF, that connects the 3D scene representation to the 2D image. It works by numerically integrating the classic volume rendering equation along camera rays. The key insight is that this integration process is differentiable, which allows for the backpropagation of gradients from the 2D pixel values of a rendered image back to the parameters of the 3D scene representation. This project utilizes the same rendering model.
- Explicit Scene Representation: Unlike NeRF's implicit MLP, this research is based on an explicit representation—a sparse voxel grid. In this paradigm, the learnable parameters are the values stored directly in the grid (density and SH coefficients). Optimization is therefore performed directly on these parameters, in contrast to the indirect optimization of MLP weights in NeRF. This approach leverages classic optimization principles, most notably Total Variation (TV) regularization, which penalizes the magnitude of the gradient of the density field to enforce smoothness and prevent noisy, "floating" artifacts in empty space.

This project explores how modern optimization techniques, such as adaptive learning rate schedules and progressive regularization, can be applied to this direct, high-dimensional optimization problem to improve its convergence properties and the quality of the final solution.

## 6. Methodology Insights

The literature shows a clear and successful methodological shift from the training of large neural networks towards the direct optimization of explicit data structures. For improving upon the current state-of-the-art (Plenoxels), the most promising methodologies identified in the literature are those that focus on refining the optimization process itself, rather than proposing entirely new architectures. The consistent finding that Plenoxels' performance is highly sensitive to its hyperparameters suggests that a methodology focused on a controlled, systematic exploration of these settings is highly relevant and likely to yield valuable insights. Therefore, this project adopts a controlled experimental approach, which is well-suited to isolate and understand the effects of different learning rate schedules, initialization strategies, and regularization schemes on the final reconstruction quality.

## 7. Conclusion

The evolution from NeRF to Plenoxels has largely solved the problem of slow training and rendering in novel view synthesis, opening the door for real-time applications. The research frontier has consequently shifted to improving the robustness, stability, and perceptual quality of these highly efficient, network-free models. This literature review confirms that hyperparameter tuning and loss function design are critical yet underexplored areas for Plenoxels. The identified gaps, a lack of systematic optimization strategies and an over-reliance on pixel-wise loss functions directly motivate and validate the direction of this research. By systematically exploring these areas through controlled experimentation, this project aims to contribute practical, empirical insights that can lead to more reliable, easier-to-use, and visually compelling real-time neural rendering systems.

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
