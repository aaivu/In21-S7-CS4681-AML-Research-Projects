# Research Proposal: CV:Super Resolution

Student: 210572P

Research Area: CV:Super Resolution

Date: 2025-09-01

## Abstract

Blind Face Restoration (BFR) aims to recover high-quality facial images from degraded inputs, a highly ill-posed problem. The CodeFormer model represents a significant advancement in BFR, achieving remarkable robustness by casting the task as code prediction in a discrete latent space. However, this robustness is often achieved at the expense of identity fidelity, as the model's finite codebook can cause restored faces to regress towards a mean representation, losing unique subject characteristics. This research proposes ID-CodeFormer, a novel enhancement to directly address this limitation. We will integrate an explicit identity-preserving loss function, supervised by a pre-trained ArcFace network, into the CodeFormer training pipeline. This loss will compel the model to generate features that are not only perceptually realistic but also discriminative of the subject's identity. The expected outcome is a tangible improvement in identity similarity with a minimal and acceptable trade-off in reconstruction quality.

## 1\. Introduction

The restoration of facial images from low-quality inputs captured in unconstrained, real-world environments is a formidable challenge in computer vision. These images suffer from complex, unknown degradations like blur, noise, and low resolution. Blind Face Restoration (BFR) seeks to solve this ill-posed problem.

Recently, the CodeFormer model has set a new state-of-the-art by reframing BFR as a code prediction task in a discrete proxy space. It uses a vector-quantized autoencoder to learn a finite "codebook" of high-quality facial parts, or "visual atoms." A Transformer module then predicts the correct sequence of codes from a degraded input, allowing it to produce highly realistic outputs even from severely corrupted images \[1\]. This approach is robust but introduces a critical new problem.

## 2\. Problem Statement

The core problem this research addresses is the **"identity drift"** inherent in the CodeFormer model. The very mechanism that grants CodeFormer its robustness-reliance on a finite, pre-trained codebook-is also its primary limitation regarding identity.

The model struggles with features, accessories, or poses that are underrepresented in its codebook. This leads to a phenomenon where the restored face is perceptually high-quality but may belong to a different individual, as the model regresses towards a more common or "mean" facial representation. The original training objectives (L1, perceptual, and adversarial losses) do not explicitly optimize for identity similarity. Therefore, the model is trained to generate a face that _looks real_, not necessarily the face of the _correct person_.

## 3\. Literature Review Summary

Modern BFR heavily relies on generative priors, often from pre-trained GANs like StyleGAN (e.g., GFP-GAN \[3\]). CodeFormer \[1\] represents an alternative paradigm, using discrete codebook priors from a VQ-VAE \[4, 5\], which enhances robustness but introduces the identity preservation challenge.

To improve identity, many generative models incorporate an identity-preserving loss, typically derived from a pre-trained face recognition network. ArcFace \[2\] is a state-of-the-art choice, using an Additive Angular Margin Loss to create highly discriminative embeddings. The cosine distance between ArcFace embeddings serves as a robust metric for identity similarity.

**Research Gap:** While identity-preserving losses are not new, their **targeted application to mitigate the specific identity drift caused by the discrete-representation paradigm of CodeFormer has not been adequately explored.** This research aims to fill that gap by integrating ArcFace supervision directly into the CodeFormer training pipeline.

## 4\. Research Objectives

### Primary Objective

To significantly mitigate identity drift and enhance identity preservation in the state-of-the-art CodeFormer Blind Face Restoration model.

### Secondary Objectives

- To design and integrate an explicit identity-preserving loss (\$L_{ids}\$), supervised by a pre-trained ArcFace network, into the CodeFormer training pipeline.
- To quantitatively validate the proposed **ID-CodeFormer** model and demonstrate a tangible improvement in identity fidelity, measured by the Identity Similarity (IDS) score.
- To rigorously evaluate the trade-off between enhanced identity preservation and standard reconstruction quality metrics (e.g., PSNR, SSIM, LPIPS).

## 5\. Methodology

The proposed methodology introduces **ID-CodeFormer**, which modifies the training of the baseline CodeFormer model without altering its core architecture.

- **Core Component:** A pre-trained and **frozen ArcFace network** \[2\] will be used as an "expert" feature extractor.
- Loss Formulation: An identity-preserving loss, \$L_{ids}\$, will be defined as one minus the cosine similarity between the ArcFace embeddings of the restored output image (\$I_{res}\$) and the high-quality ground-truth image (\$I_{gt}\$):  
    \$L_{ids} = 1 - cos(ArcFace(I_{res}), ArcFace(I_{gt}))\$
- **Pipeline Integration:** This \$L_{ids}\$ term will be strategically integrated into **Stage II (Transformer Learning)** and **Stage III (CFT Tuning)** of the CodeFormer training pipeline.
- **Training:** The LQ Encoder (\$E_L\$) and Transformer (\$T\$) will be trained with a combined loss function (\$\\mathcal{L^{\\prime}}\_{tf} = \\mathcal{L}\_{tf} + \\lambda_{ids} \\cdot L_{ids}\$), compelling them to learn a mapping that is explicitly identity-aware.
- **Evaluation:** The proposed ID-CodeFormer will be trained on the FFHQ dataset \[6\] and evaluated against the baseline CodeFormer on standard benchmarks (CelebA-Test, LFW-Test) using a full suite of metrics (IDS, PSNR, SSIM, LPIPS, FID).

## 6\. Expected Outcomes

The primary expected outcome is that the ID-CodeFormer model will **demonstrate a tangible and quantitative improvement in the Identity Similarity (IDS) score** compared to the baseline CodeFormer.

This enhancement in identity fidelity is expected to come with **only a minimal and acceptable trade-off in pixel-level reconstruction metrics** (PSNR and SSIM). This would confirm that direct identity supervision can successfully navigate the quality-fidelity trade-off in discrete latent space models, resulting in restorations that are not only high-quality but also faithful to the subject's unique identity.

## 7\. Timeline

| **Week** | **Task** |
| --- | --- |
| 1-2 | Literature Review & Setup |
| --- | --- |
| 3-4 | Methodology Development (Loss integration) |
| --- | --- |
| 5-8 | Implementation & Stage I/II Training |
| --- | --- |
| 9-12 | Experimentation (Stage III Tuning & Baseline Training) |
| --- | --- |
| 13-15 | Analysis (Metric calculation, qualitative comparison) & Writing |
| --- | --- |
| 16  | Final Submission & Code Release |
| --- | --- |

## 8\. Resources Required

- **Datasets:** FFHQ \[6\], CelebA-HQ, LFW.
- **Models:**
  - Official CodeFormer public codebase \[1\].
  - Pre-trained ArcFace network (e.g., IR-SE50) \[2\].
- **Software:** Python, PyTorch, and standard CV/DL libraries (e.g., face-alignment, facenet-pytorch).
- **Hardware:** High-performance GPUs (e.g., NVIDIA Tesla V100 or A100) for training and evaluation.

## References

\[1\] S. Zhou, K. C. K. Chan, C. Li, and C. C. Loy, "Towards Robust Blind Face Restoration with Codebook Lookup Transformer," in _Advances in Neural Information Processing Systems (NeurIPS)_, 2022.

\[2\] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2019.

\[3\] X. Wang, Y. Li, H. Zhang, and Y. Shan, "Towards Real-World Blind Face Restoration with Generative Facial Prior," in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2021.

\[4\] A. Van Den Oord and O. Vinyals, "Neural Discrete Representation Learning," in _Advances in Neural Information Processing Systems (NeurIPS)_, 2017.

\[5\] P. Esser, R. Rombach, and B. Ommer, "Taming Transformers for High-Resolution Image Synthesis," in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2021.

\[6\] T. Karras, S. Laine, and T. Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks," in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2019.

\[7\] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," _IEEE Trans. Image Process._, vol. 13, no. 4, pp. 600-612, Apr. 2004.

\[9\] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2018.

**Submission Instructions:**

- Complete all sections above
- Commit your changes to the repository
- Create an issue with the label "milestone" and "research-proposal"
- Tag your supervisors in the issue for review