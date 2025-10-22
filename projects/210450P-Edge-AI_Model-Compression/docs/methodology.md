
---

# Methodology: Edge AI: Model Compression

**Student:** 210450P
**Research Area:** Edge AI: Model Compression
**Date:** 2025-10-20

---

## 1. Overview

This study investigates **extreme quantization** techniques for compressing the **EfficientNetV2B0** architecture to enable efficient edge deployment using **TensorFlow Lite (TFLite)**. The methodology focuses on implementing and evaluating both **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** using the **TensorFlow Model Optimization Toolkit (TF-MOT)**. The workflow includes dataset preparation, model quantization, and empirical evaluation in terms of model size, accuracy, and inference latency to identify optimal configurations for on-device performance.

---

## 2. Research Design

A **quantitative experimental design** was adopted to systematically evaluate different quantization strategies. The study compares multiple quantization methods—dynamic range, float16, full int8 PTQ, and QAT—applied to EfficientNetV2B0 pretrained on ImageNet. Each model variant was benchmarked using identical datasets and evaluation protocols to ensure comparability. Results were analyzed to determine trade-offs among compression ratio, inference latency, and top-1 accuracy for real-world edge AI deployment.

---

## 3. Data Collection

### 3.1 Data Sources

The dataset used was **ImageNet-Mini (1,000 classes)**, sourced from **Kaggle**, providing a smaller but balanced subset of the full ImageNet-1k benchmark.

### 3.2 Data Description

* **Training samples:** ~34,745 images
* **Validation samples:** ~3,923 images
* **Image size:** 224×224×3 RGB
  The dataset retains ImageNet’s class diversity while being computationally feasible for experimentation.

### 3.3 Data Preprocessing

* Images were resized to 224×224 pixels.
* Input normalization followed `tf.keras.applications.efficientnet_v2.preprocess_input()`.
* The **`image_dataset_from_directory()`** utility was used to load and label data automatically.
* Performance was optimized with the **`tf.data`** API using prefetching and parallel calls.
* For **integer quantization calibration**, a representative subset of 500 images was randomly selected and reused for inference testing.

---

## 4. Model Architecture

The base model was **EfficientNetV2B0**, pretrained on ImageNet (~7.2M parameters) with the classifier head retained.

* **Quantization-Aware Training (QAT):** Implemented via `tfmot.quantization.keras.quantize_apply()`.
* **Selective quantization:** Only `Conv2D` and `Dense` layers were quantized using a custom `NoOpQuantizeConfig` to bypass non-quantizable operations (`Add`, `Multiply`, `Dropout`, etc.).
* **Training parameters:**

  * Optimizer: Adam
  * Learning rate: 1×10⁻⁵
  * Epochs: 5
  * Loss: `sparse_categorical_crossentropy`
    Fine-tuning preserved numerical stability within residual and squeeze-and-excitation blocks, producing QAT models compatible with integer TFLite deployment.

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics

* **Top-1 Accuracy** – for classification performance.
* **Model Size (MB)** – for compression efficiency.
* **Inference Latency (ms/sample)** – for deployment performance.

### 5.2 Baseline Models

* **Baseline:** FP32 EfficientNetV2B0
* **PTQ Variants:** Dynamic range, Float16, Full Int8
* **QAT Variant:** Full Int8 (fine-tuned)

### 5.3 Hardware/Software Requirements

* **Frameworks:** TensorFlow 2.19.0, TF-MOT, TFLite
* **Environment:** Google Colab with NVIDIA T4 GPU
* **Hardware Target:** Edge CPUs/NPUs (e.g., ARM, Jetson, Coral TPU)
* **Reproducibility:** Fixed random seeds for TensorFlow and NumPy

---

## 6. Implementation Plan

| Phase   | Tasks                              | Duration | Deliverables                             |
| ------- | ---------------------------------- | -------- | ---------------------------------------- |
| Phase 1 | Data preprocessing                 | 2 weeks  | Clean ImageNet-Mini dataset              |
| Phase 2 | Model implementation and QAT setup | 3 weeks  | Baseline and QAT EfficientNetV2 models   |
| Phase 3 | Quantization and evaluation        | 2 weeks  | PTQ & QAT TFLite models with results     |
| Phase 4 | Analysis and reporting             | 1 week   | Final performance and trade-off analysis |

---

## 7. Risk Analysis

| Potential Risk                                      | Impact | Mitigation Strategy                                       |
| --------------------------------------------------- | ------ | --------------------------------------------------------- |
| Accuracy degradation after quantization             | High   | Apply QAT for fine-tuning to recover accuracy             |
| Incompatible layers during quantization             | Medium | Use custom `NoOpQuantizeConfig` to bypass unsupported ops |
| Limited calibration data affecting int8 performance | Medium | Use a representative subset for accurate scaling          |
| Hardware-specific latency variation                 | Low    | Benchmark across multiple devices for consistency         |

---

## 8. Expected Outcomes

* **3×–3.6× model size reduction** (27.9 MB → ~8 MB)
* **Minimal accuracy loss** (≤0.5% from FP32 baseline)
* **Average inference latency:** 27–30 ms per sample on CPU
* **Validated quantization workflow** combining PTQ and QAT for edge AI deployment
* **Practical deployment blueprint** for compressing large CNNs using TensorFlow Lite

---

**Summary:**
This methodology establishes a reproducible pipeline for **quantization-based model compression** in Edge AI, demonstrating that extreme quantization—especially with QAT—enables substantial efficiency gains with negligible accuracy loss, paving the way for deploying state-of-the-art image classifiers on mobile and embedded devices.

---
