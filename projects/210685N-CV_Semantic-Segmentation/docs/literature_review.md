# Literature Review: CV:Semantic Segmentation

**Student:** 210685N
**Research Area:** CV:Semantic Segmentation
**Date:** 2025-09-01

## Abstract

This review covers the evolution of image segmentation, focusing on the highly impactful **transformer-based architectures**. While these models have set the performance benchmark for semantic segmentation, the review highlights their **massive computational footprint** and reliance on computationally expensive architectural components, leading to slow and cumbersome inference. This remains a significant barrier for practical application, particularly on **resource-constrained edge devices**. The key area of research addresses this critical **efficiency-performance trade-off** by exploring methods for designing **lightweight, efficient architectures** for image segmentation and efficient attention mechanisms. The proposed work, **Efficient Mask2Former for Semantic Segmentation**, directly addresses this gap by replacing the expensive cross-attention with a novel **prototype-based cross-attention module** that uses computationally cheap element-wise operations.

---

## 1. Introduction

Semantic segmentation is a fundamental computer vision problem that requires the **dense prediction of a class label for every pixel in an image**. Its role is pivotal in applications ranging from **autonomous driving** to **medical image analysis**. Recent advancements in this domain are dominated by **transformer-based segmentation architectures**, which treat the task as an **end-to-end set prediction task**, inspired by DETR. These models typically employ an **encoder-decoder structure**: a visual backbone to generate high-resolution features and a transformer decoder that refines a set of learnable queries to produce the final segmentation masks and class predictions. However, the appealing properties and performance of these models come at a significant cost, as they rely on **computationally expensive architectural components**. This inefficiency carries two major consequences: **High Operational Cost** and a **Deployment Barrier** on resource-constrained edge devices. The proposed work addresses this critical efficiency-performance trade-off by proposing a novel architecture, the **Efficient Mask2Former** for semantic segmentation.

---

## 2. Search Methodology

### Search Terms Used
- **semantic segmentation, panoptic segmentation, transformer, Mask2Former, efficient attention, prototype selection, cross-attention bottleneck**
- **lightweight segmentation, real-time CV, edge device deployment**

### Databases Searched
- [ ] IEEE Xplore
- [ ] ACM Digital Library
- [ ] Google Scholar
- [X] ArXiv (Implied by the paper's references)
- [x] ADE20k
- [ ] Other: **[Fill in other databases used]**

### Time Period
2025 June to October
---

## 3. Key Areas of Research

### 3.1 Image Segmentation Architectures
A growing field of research aims to design architectures capable of operating in multiple image segmentation settings (semantic and panoptic) without requiring changes to the loss function or architectural components. The recent, highly impactful development is the emergence of transformer-based segmentation architectures.

**Key Papers:**
- **DETR [Carion et al., 2020]** - Demonstrated the possibility of achieving competitive results in object detection and panoptic segmentation using an end-to-end set prediction network based on mask classification.
- **MaskFormer [Cheng et al., 2021]** - Proposed an architecture for general image segmentation based on mask classification, achieving state-of-the-art performance in both semantic and panoptic segmentation.
- **Mask2Former [Cheng et al., 2022]** - Further improved upon MaskFormer, introducing architectural enhancements that led to faster convergence and results that surpassed both general-purpose and specialized, task-specific architectures.

### 3.2 Efficient Image Segmentation
This area aims to reduce the computational complexity of image segmentation models so they can be effectively deployed on devices and run in real-time. A key limitation in this area is that many prior works have focused on architectures tailored for a single segmentation task.

**Key Papers:**
- **PIDNet [Xu et al., 2023]** - Proposed a three-branch semantic segmentation architecture to enhance object boundaries and improve performance on small objects.
- **UPSNet [Xiong et al., 2019]** - Introduced a network for panoptic segmentation that incorporates a parameter-free panoptic head to efficiently merge predictions.
- **YOSO [Hu et al., 2023]** - Proposed an efficient transformer method that predicts masks for both 'things' and 'stuff' categories, towards real-time panoptic segmentation.

### 3.3 Efficient Attention Mechanisms
The standard Attention mechanism does not scale effectively for large input dimensions because it considers all pairwise relationships among input tokens. This has prompted exploration into methods for mitigating the computational complexity of this module.

**Key Papers:**
- **MobileViT V2 [Mehta and Rastegari, 2022]** - Replaced expensive dot products between input tokens with cheap **element-wise multiplications** of the features with a small context vector, focusing on efficient self-attention.
- **SwiftFormer [Shaker et al., 2023]** - Also replaced expensive dot products with cheap element-wise multiplications, demonstrating how pairwise interactions can be redundant.
- **kMaX-DeepLab [Yu et al., 2023]** - Attempted to improve the cross-attention mechanism by replacing the classic attention operation with a $k$-Means clustering step.

---

## 4. Research Gaps and Opportunities

The current research focuses on addressing the key limitations of high-performance, transformer-based segmentation models.

### Gap 1: Computationally Expensive Cross-Attention in Mask2Former
The transformer decoder in Mask2Former is a major bottleneck, consisting of a sequence of expensive cross-attention operations that use dot products between object queries and high-resolution image features. This operation is "inevitably inefficient when applied to large input features", contributing to high operational cost and deployment barriers on resource-constrained platforms.The project introduces **Efficient Masked Cross-Attention (EMCA)**, which capitalizes on the intrinsic redundancy of visual features to reduce input tokens via a **prototype selection mechanism** and redesigns the cross-attention using computationally cheap **element-wise operations**.

### Gap 2: Single-Task Focus in Efficient Architectures
Despite significant progress across various efficient segmentation settings, a key limitation remains: these architectures typically operate only on a single task. This task-specific focus can **duplicate research efforts**. The work aims to address this gap by presenting an efficient architecture that can be **seamlessly employed across multiple segmentation tasks**.

---

## 5. Theoretical Framework

The research builds on the **MaskFormer/Mask2Former** unified framework. The theoretical foundation for the efficiency gain is two-fold. First, it hypothesizes that using all the pixels belonging to an object for refinement is **redundant**, as pixels associated with a specific object query will naturally become close to each other during training. This justifies leveraging this redundancy to focus solely on the **prototype**. Second, inspired by recent advancements in efficient attention, it adopts the principle that global information can be condensed into lightweight context vectors, allowing the replacement of expensive dot products with computationally cheap **element-wise operations** for the cross-attention interaction.

---

## 6. Methodology Insights

The core methodological contribution is the **Efficient Masked Cross-Attention (EMCA)**, which is incorporated into the Mask2Former's transformer decoder.

The EMCA module consists of two main steps:
1.  **Prototype Selection Mechanism:**
    * High-resolution features $F_{bi} \in \mathbb{R}^{H_i \times W_i \times C}$ and object queries $Q_{in} \in \mathbb{R}^{N \times C}$ are projected to the same dimension to get $K \in \mathbb{R}^{H_iW_i \times D}$ and $Q \in \mathbb{R}^{N \times D}$.
    * The similarity matrix $S \in \mathbb{R}^{H_iW_i \times N}$ is computed as $S = K^T Q$.
    * The prototypes $K_p \in \mathbb{R}^{N \times D}$ are obtained as $K_p = K[G]$. This reduces the input from $K$ (HW pixels) to $K_p$ (N prototypes), significantly decreasing the complexity.
2.  **Prototype-based Cross-Attention:**
    * The attention matrix $A$ is computed using an element-wise product: $A = (Q \odot K_p) W_A$.
    * $A$ is then normalized and scaled by a learnable parameter $\alpha \in \mathbb{R}^{D}$  to dynamically reweight the prototypes $K_p$ in an additive manner: $B=\alpha \odot \frac{A}{\|A\|_2}+K_{p}$.
    * The output $Q_{out}$ is obtained via a final linear projection and a residual connection: $Q_{out} = B W_{out} + Q$.

This new approach demonstrates a notable advantage, being nearly **1.2x faster** than the masked cross-attention counterpart and achieving a **1.13x speedup in ADE20K**, underscoring the efficiency gains.

---

## 7. Conclusion

The enhancement of Mask2Former through the adoption of the **Efficient Masked Cross-Attention (EMCA)** architecture successfully addresses the critical performance-efficiency bottleneck in semantic image segmentation. By fundamentally substituting the computationally expensive masked cross-attention with a novel prototype-based, element-wise mechanism, the model significantly reduces computational complexity. The empirical results demonstrate a superior performance-speed trade-off. For instance, the **Mask2Former (EMCA)** model achieves an mIoU of **45.1** and a speed of **33.7 FPS**, surpassing the original **Mask2Former** ($\text{mIoU}=44.6$, $\text{FPS}=29.8$) on the ADE20K dataset. This sets a clear path toward deployable, sustainable, and high-performance semantic segmentation solutions for real-world scenarios, particularly on resource-constrained platforms.

---

## References
1. Ravindu Weerakoon and Uthayasanker Thayasivam. Rethinking Mask2former for Efficient Semantic Segmentation. October 2025.
2. Senay Cakir et al. Semantic Segmentation for Autonomous Driving: Model Evaluation, Dataset Generation, Perspective Comparison, and Real-Time Capability. 2022. arXiv: 2207.12939 [cs.CV]. URL: https://arxiv.org/abs/2207.12939.
3. Nicolas Carion et al. End-to-End Object Detection with Transformers. 2020. arXiv: 2005.12872 [cs.CV]. URL: https://arxiv.org/abs/2005.12872.
4. Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-Pixel Classification is Not All You Need for Semantic Segmentation. 2021. arXiv: 2107.06278 [cs.CV]. URL: https://arxiv.org/abs/2107.06278.
5. Bowen Cheng et al. Masked-attention Mask Transformer for Universal Image Segmentation. 2022. arXiv: 2112.01527 [cs.CV]. URL: https://arxiv.org/abs/2112.01527.
6. Enze Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. 2021. arXiv: 2105.15203 [cs.CV]. URL: https://arxiv.org/abs/2105.15203.
7. Qihang Yu et al. kMaX-DeepLab: k-means Mask Transformer. 2023. arXiv: 2207.04044 [cs.CV]. URL: https://arxiv.org/abs/2207.04044.
8. Yuwen Xiong et al. UPSNet: A Unified Panoptic Segmentation Network. 2019. arXiv: 1901.03784 [cs.CV]. URL: https://arxiv.org/abs/1901.03784.
9. Jiacong Xu, Zixiang Xiong, and Shankar P. Bhattacharyya. PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers. 2023. arXiv: 2206.02066 [cs.CV]. URL: https://arxiv.org/abs/2206.02066.
10. Jie Hu et al. You Only Segment Once: Towards Real-Time Panoptic Segmentation. 2023. arXiv: 2303.14651 [cs.CV]. URL: https://arxiv.org/abs/2303.14651.
11. Sachin Mehta and Mohammad Rastegari. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. 2022. arXiv: 2110.02178 [cs.CV]. URL: https://arxiv.org/abs/2110.02178.
12. Abdelrahman Shaker et al. SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications. 2023. arXiv: 2303.15446 [cs.CV]. URL: https://arxiv.org/abs/2303.15446.
13. Ashish Vaswani et al. Attention Is All You Need. 2023. arXiv: 1706.03762 [cs.CL]. URL: https://arxiv.org/abs/1706.03762.
14. Bolei Zhou et al. Semantic Understanding of Scenes through the ADE20K Dataset. 2018. arXiv: 1608.05442 [cs.CV]. URL: https://arxiv.org/abs/1608.05442.
15. Kaiming He et al. Deep Residual Learning for Image Recognition. 2015. arXiv: 1512.03385 [cs.CV]. URL: https://arxiv.org/abs/1512.03385.
16. Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. 2019. arXiv: 1711.05101 [cs.LG]. URL: https://arxiv.org/abs/1711.05101.
17. Jia Deng et al. "ImageNet: A large-scale hierarchical image database". In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. 2009, pp. 248-255. DOI: 10.1109/CVPR.2009.5206848.
18. Aoran Xiao et al. "FPS-Net: A convolutional fusion network for large-scale LiDAR point cloud segmentation". In: ISPRS Journal of Photogrammetry and Remote Sensing 176 (June 2021), pp. 237-249. ISSN: 0924-2716. DOI: 10.1016/j.isprsjprs.2021.04.011. URL: http://dx.doi.org/10.1016/j.isprsjprs.2021.04.011.
19. Sachin Mehta and Mohammad Rastegari. Separable Self-attention for Mobile Vision Transformers. 2022. arXiv: 2206.02680 [cs.CV]. URL: https://arxiv.org/abs/2206.02680.
20. M Krithika Alias AnbuDevi and K Suganthi. "Review of semantic segmentation of medical images using modified architectures of UNET". en. In: Diagnostics (Basel) 12.12 (Dec. 2022), p. 3064.
21. Ahmad Faiz et al. LLMCarbon: Modeling the end-to-end Carbon Footprint of Large Language Models. 2024. arXiv: 2309.14393 [cs.CL]. URL: https://arxiv.org/abs/2309.14393.
