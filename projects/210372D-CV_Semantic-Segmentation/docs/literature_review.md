# Literature Review: CV: Semantic Segmentation

**Student:** 210372D  
**Research Area:** CV: Semantic Segmentation  
**Date:** 2025-10-20  

---

## Abstract

This literature review explores the progression of semantic segmentation techniques, emphasizing the transition from convolution-based architectures to transformer-based methods. The review covers key advances such as Fully Convolutional Networks (FCN), encoder–decoder structures, Vision Transformers (ViT), and hybrid frameworks like SegFormer. Special attention is given to architectural innovations, efficiency improvements, and robustness in real-world scenarios. The findings highlight how transformer-based models have improved accuracy and generalization while exposing areas such as computational efficiency, robustness under domain shifts, and model interpretability as ongoing research challenges.

---

## 1. Introduction

Semantic segmentation aims to assign a semantic label to each pixel in an image, enabling fine-grained scene understanding. This field has evolved from classical CNN architectures (VGG, ResNet) to fully convolutional networks (FCN), and later to attention-based and transformer-based architectures. With the success of transformers in NLP, their extension to vision tasks revolutionized the field, leading to architectures like ViT, Swin Transformer, and SegFormer. The review focuses on understanding these developments, the current state of the art, and possible directions for further research in efficient and robust segmentation models.

---

## 2. Search Methodology

### Search Terms Used
- "Semantic segmentation"
- "Vision Transformers for semantic segmentation"
- "Transformer backbones"
- "SegFormer architecture"
- "Lightweight semantic segmentation"
- "Efficient transformer segmentation"
- "Mix-FFN", "MLP decoder"

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2018–2025, focusing on recent advancements in transformer-based segmentation and efficiency-oriented architectures.

---

## 3. Key Areas of Research

### 3.1 Convolutional Architectures for Segmentation

Early approaches in semantic segmentation were dominated by convolutional neural networks. Models such as Fully Convolutional Networks (FCN) and U-Net laid the foundation for dense prediction by replacing fully connected layers with convolutional ones to produce pixel-level outputs. Extensions like DeepLab introduced "atrous convolutions" and conditional random fields to refine boundaries.

**Key Papers:**
- Long et al., 2015 – Introduced Fully Convolutional Networks (FCN), pioneering end-to-end segmentation.  
- Ronneberger et al., 2015 – Proposed U-Net, an encoder-decoder structure effective for biomedical segmentation.  
- Chen et al., 2018 – DeepLabv3+ refined feature maps with atrous convolution and ASPP module for multi-scale context.

---

### 3.2 Vision Transformers (ViT) and Hierarchical Architectures

The Vision Transformer (ViT) adapted the transformer mechanism from NLP to vision tasks by splitting images into patches and processing them with self-attention. While achieving high accuracy, ViT suffered from computational inefficiency and poor scalability on smaller datasets. Subsequent architectures introduced hierarchical and pyramid-like structures for dense prediction tasks.

**Key Papers:**
- Dosovitskiy et al., 2020 – Introduced ViT, demonstrating transformer viability for image classification.  
- Liu et al., 2021 – Proposed Swin Transformer, enabling hierarchical representation and shifted windows for efficiency.  
- Wang et al., 2021 – Introduced PVT (Pyramid Vision Transformer), providing multi-scale feature extraction crucial for dense prediction.

---

### 3.3 SegFormer: A Unified Transformer Framework for Segmentation

SegFormer (Xie et al., 2021) combined hierarchical transformer encoders with a lightweight MLP-based decoder to achieve high performance with fewer parameters. It removed positional encodings through Mix-FFN and achieved robustness and generalization across datasets like ADE20K and Cityscapes.

**Key Papers:**
- Xie et al., 2021 – Proposed SegFormer, achieving state-of-the-art segmentation with all-MLP decoder and Mix-FFN.  
- Chen et al., 2023 – Enhanced SegFormer for photovoltaic array segmentation using inception-enhanced attention and feature pyramids.  
- Kienzle et al., 2024 – Introduced SegFormer++ with token-merging strategies for improved efficiency.  
- Bai et al., 2022 – Proposed dynamically pruned SegFormer using gated layers and knowledge distillation to improve inference efficiency.

---

### 3.4 Robustness and Efficiency Improvements

Research has increasingly focused on improving model robustness under noise, weather conditions, and domain shifts. Efficiency-driven architectures target real-time deployment on edge devices while retaining segmentation accuracy.

**Key Papers:**
- Zhao et al., 2017 – PSPNet introduced pyramid pooling for capturing contextual information.  
- Strudel et al., 2021 – Segmenter demonstrated end-to-end transformer-based segmentation with fewer inductive biases.  
- Bai et al., 2022 – Dynamic pruning of SegFormer reduced redundant neurons, improving inference time without retraining.

---

## 4. Research Gaps and Opportunities

### Gap 1: Limited exploration of auxiliary and class-balanced losses in SegFormer
**Why it matters:** The original SegFormer does not employ techniques like Online Hard Example Mining (OHEM) or class-balanced loss, potentially reducing accuracy in class-imbalanced datasets.  
**How your project addresses it:** The proposed work aims to experiment with auxiliary losses and class-weighting strategies to improve minority class recognition.

### Gap 2: Decoder inefficiency in domain-specific adaptation
**Why it matters:** The MLP decoder, while lightweight, may not optimally capture spatial relationships in domain-specific tasks such as medical or satellite imagery.  
**How your project addresses it:** Modify decoder architecture to incorporate domain-sensitive attention mechanisms while maintaining computational efficiency.

### Gap 3: Computational limitations for large-scale training
**Why it matters:** Training large transformer models is resource-intensive, limiting practical usability.  
**How your project addresses it:** Focus on lightweight variants (e.g., SegFormer-B0/B1) and explore fine-tuning on domain-specific data.

---

## 5. Theoretical Framework

The theoretical foundation of this study lies in attention-based learning, specifically self-attention mechanisms enabling long-range dependency modeling. Transformers generalize CNN-like feature extraction by computing contextual relationships globally. The encoder-decoder paradigm remains central—transformer encoders extract multi-level hierarchical features, and lightweight decoders reconstruct segmentation masks efficiently. Concepts like Mix-FFN and token merging extend this theory to optimize positional representation and computational efficiency.

---

## 6. Methodology Insights

Common methodologies in recent segmentation research:
- Encoder-decoder structures with hierarchical features.  
- Data augmentation (random cropping, resizing, flipping).  
- Optimizers: AdamW with poly learning rate schedules.  
- Evaluation metrics: Mean Intersection over Union (mIoU), pixel accuracy.  
- Benchmark datasets: ADE20K, Cityscapes, COCO-Stuff.  

For this research, adopting SegFormer’s Mix-FFN and hierarchical transformer backbone provides a solid baseline. Enhancements will involve decoder modifications and loss function adjustments for improved efficiency and accuracy.

---

## 7. Conclusion

Semantic segmentation has evolved from convolutional architectures to hybrid and transformer-based methods emphasizing accuracy, scalability, and efficiency. SegFormer represents a major step toward lightweight, high-performing segmentation models. However, opportunities remain in improving decoder effectiveness, robustness, and loss formulation. This review provides a foundational understanding to guide the proposed enhancements to SegFormer within computational and practical constraints.

---

## References

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.  
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.  
3. Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV.  
4. Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.  
5. Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. arXiv.  
6. Wang, W. et al. (2021). Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction. arXiv.  
7. Xie, E. et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. arXiv.  
8. Zheng, S. et al. (2021). Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers. CVPR.  
9. Chen, W., Jin, S., Luo, Y., & Li, J. (2023). Enhanced Segmentation of PV Arrays in Infrared Images using an Improved SegFormer Approach. IEEE Big Data.  
10. Kienzle, D., Kantonis, M., Schön, R., & Lienhart, R. (2024). SegFormer++: Efficient Token-Merging Strategies for High-Resolution Semantic Segmentation. IEEE MIPR.  
11. Bai, H., Mao, H., & Nair, D. (2022). Dynamically Pruning SegFormer for Efficient Semantic Segmentation. ICASSP.  
12. Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid Scene Parsing Network. CVPR.  
13. Strudel, R., Garcia, R., Laptev, I., & Schmid, C. (2021). Segmenter: Transformer for Semantic Segmentation. ICCV.

