
---

# Research Proposal: Edge AI: Model Compression

**Student:** 210450P
**Research Area:** Edge AI: Model Compression
**Date:** 2025-10-20

---

## Abstract

Deploying deep convolutional neural networks (CNNs) on edge and embedded devices presents major challenges due to limited computational resources, memory, and energy constraints. This research proposes an in-depth exploration of **model compression using quantization** for efficient edge deployment of modern vision architectures. The study focuses on implementing and analyzing **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** for **EfficientNetV2**, using the **TensorFlow Lite** and **TensorFlow Model Optimization Toolkit** frameworks. By systematically evaluating compression ratios, accuracy retention, and inference latency, the research aims to identify the optimal balance between model compactness and predictive performance. The expected outcome is a reproducible quantization pipeline capable of achieving up to **3.6× reduction in model size** with negligible accuracy loss, enabling real-time, energy-efficient inference on mobile and embedded platforms. This work contributes to advancing **Edge AI** by providing practical quantization strategies for deploying high-performance neural networks in constrained environments.

---

## 1. Introduction

Modern deep learning models such as EfficientNetV2 achieve exceptional accuracy in computer vision tasks but are computationally expensive and memory-intensive. Deploying such models on **edge devices**—smartphones, IoT boards, and embedded systems—remains challenging due to constraints in hardware, power, and latency.
Quantization has emerged as a critical model compression technique that reduces numerical precision (e.g., from 32-bit floating point to 8-bit integer) without significantly compromising accuracy. With the rapid adoption of **Edge AI**, the demand for **efficient, deployable deep learning models** has never been greater. This research focuses on systematically studying quantization approaches to make advanced neural networks like EfficientNetV2 suitable for **on-device, real-time inference**.

---

## 2. Problem Statement

Despite the availability of efficient architectures, deploying deep CNNs on edge devices remains constrained by **large model sizes**, **high computational costs**, and **limited memory bandwidth**. Existing quantization methods often suffer from **accuracy degradation** when applied aggressively or without retraining. Therefore, a **comprehensive, reproducible quantization pipeline** is needed to evaluate and optimize these techniques for real-world edge inference, balancing model compression, accuracy, and latency.

---

## 3. Literature Review Summary

Existing literature highlights the effectiveness of **EfficientNetV2** for scalable, high-accuracy image classification with improved parameter utilization. Research on **model quantization**—including works by Jacob et al. (2018) and Zhou et al. (2018)—demonstrates that reducing precision to int8 or float16 significantly decreases memory usage and inference time. However, most studies focus on individual quantization methods rather than an **end-to-end comparison** of PTQ and QAT across configurations.
Recent surveys on **Edge AI optimization** stress the need for practical, hardware-aware compression pipelines. The current gap lies in integrating PTQ and QAT workflows with a **systematic evaluation framework** to understand trade-offs among compression ratio, latency, and accuracy in realistic deployment environments.

---

## 4. Research Objectives

### Primary Objective

To develop and evaluate a **quantization-based model compression pipeline** for EfficientNetV2, enabling efficient deployment on edge and embedded devices without significant loss in accuracy.

### Secondary Objectives

* To implement **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** using TensorFlow Lite.
* To measure and compare model **size, inference latency, and top-1 accuracy** across quantization configurations.
* To identify the **optimal trade-off** between accuracy and compression for edge deployment.
* To ensure **reproducibility** through open-source implementation and standardized benchmarking.

---

## 5. Methodology

The proposed approach follows an experimental workflow:

1. **Dataset Preparation:** Use the ImageNet-Mini dataset (1,000 classes) for efficient prototyping.
2. **Model Selection:** Utilize the pretrained **EfficientNetV2B0** model with the top classifier retained.
3. **Quantization Techniques:**

   * Apply PTQ methods: dynamic range, float16, and full int8 quantization.
   * Implement QAT using TensorFlow Model Optimization Toolkit with selective layer quantization.
4. **Training and Fine-Tuning:** Conduct 5 epochs of fine-tuning using the Adam optimizer (lr = 1×10⁻⁵).
5. **Evaluation Metrics:** Measure accuracy, model size, and inference latency on CPU runtime using TensorFlow Lite.
6. **Reproducibility:** Use fixed random seeds, consistent datasets, and open-source scripts.

This methodology ensures an empirical analysis of quantization’s effects and enables direct deployment on mobile and embedded systems for validation.

---

## 6. Expected Outcomes

* A **3–3.6× reduction in model size** with minimal accuracy degradation (≤0.5%).
* **Inference latency below 30 ms** per image on CPU runtime.
* A **validated quantization pipeline** for TensorFlow Lite with reproducible code.
* Insights into trade-offs among quantization strategies for edge deployment.
* Foundation for **future integration with Coral Edge TPU** or Jetson-based devices.

---

## 7. Timeline

| Week  | Task                                |
| ----- | ----------------------------------- |
| 1–2   | Literature Review                   |
| 3–4   | Methodology Development             |
| 5–8   | Model Implementation & Quantization |
| 9–12  | Experiments and Evaluation          |
| 13–15 | Analysis and Report Writing         |
| 16    | Final Submission                    |

---

## 8. Resources Required

* **Software:** TensorFlow 2.19.0, TensorFlow Lite, TensorFlow Model Optimization Toolkit
* **Hardware:** Google Colab (NVIDIA T4 GPU), Edge CPU (for latency testing)
* **Dataset:** ImageNet-Mini (from Kaggle)
* **Libraries:** NumPy, Matplotlib, Keras
* **Version Control:** GitHub repository for open-source code and reproducibility

---

## References

1. M. Tan and Q. Le, “EfficientNetV2: Smaller Models and Faster Training,” *arXiv preprint arXiv:2104.00298*, 2021.
2. B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” *CVPR*, 2018.
3. TensorFlow Model Optimization Toolkit. Available: [https://www.tensorflow.org/model_optimization](https://www.tensorflow.org/model_optimization)
4. TensorFlow Lite. Available: [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)
5. J. Deng et al., “ImageNet: A Large-Scale Hierarchical Image Database,” *CVPR*, 2009.
6. A. Zhou et al., “Value-Aware Quantization for Training and Inference of Neural Networks,” *ECCV*, 2018.
7. Z. Yang and H. Lee, “Hardware-Aware Mixed Precision Quantization via Differentiable Search,” *PMLR*, 2024.
8. T. Pathirana, “Extreme Quantization of EfficientNetV2,” *GitHub Repository*, 2025.

---
