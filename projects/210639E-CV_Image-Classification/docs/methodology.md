# Methodology: CV:Image Classification

**Student:** 210639E
**Research Area:** CV:Image Classification
**Date:** 2025-09-01

## 1. Overview

This study adopts a transfer learning approach to fine-tune the AutoFormerV2 transformer architecture on the CIFAR-100 dataset. The goal is to evaluate how efficiently a vision transformer pre-trained on large-scale datasets like ImageNet can adapt to smaller-scale image classification tasks. The methodology focuses on model configuration, training strategies, and experimental setup to ensure reproducibility and robustness.

## 2. Research Design

The research follows an experimental design approach involving model fine-tuning and performance evaluation. Initially, pre-trained baseline checkpoints of AutoFormerV2 are obtained from the official model zoo. These weights, originally trained on ImageNet, are then fine-tuned on CIFAR-100. The performance of the fine-tuned model is compared with the baseline to assess the improvement in task-specific accuracy.

## 3. Data Collection

### 3.1 Data Sources
The dataset used in this research is CIFAR-100, a publicly available benchmark dataset for image classification tasks. It contains 60,000 color images, each of size 32×32 pixels, divided into 100 distinct classes.

### 3.2 Data Description
Training set: 50,000 images

Testing set: 10,000 images

Number of classes: 100
Each image belongs to one of 100 fine-grained categories, such as animals, vehicles, and everyday objects. The dataset was normalized and augmented to enhance generalization during model training.

## 4. Model Architecture

The AutoFormerV2 model serves as the core architecture. It is a vision transformer designed to automatically search for optimal network configurations using neural architecture search (NAS). The model utilizes patch embedding, self-attention layers, and multi-layer perceptrons (MLPs) to capture spatial and contextual information efficiently.
For fine-tuning, the pre-trained classifier head (trained for ImageNet’s 1,000 classes) is replaced with a new head corresponding to CIFAR-100’s 100 classes. The rest of the model parameters are initialized from the pre-trained checkpoint to leverage learned visual features.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
To assess model performance, the following metrics are used:
  Top-1 Accuracy: Measures the percentage of images where the top predicted class matches the ground truth.
  Top-5 Accuracy: Measures the percentage where the correct label is among the top five predictions.
  Cross-Entropy Loss: Used as the objective function during training.

### 5.2 Baseline Models
The baseline model used for comparison is the pre-trained AutoFormerV2 trained on ImageNet. Fine-tuned results are evaluated against this baseline to quantify the improvement achieved through transfer learning.

### 5.3 Hardware/Software Requirements
Hardware: NVIDIA GPU (12 GB VRAM or higher)

Software:
Python 3.12
PyTorch 2.3
timm library for optimization and augmentation
CUDA-enabled environment for GPU acceleration
Ubuntu 20.04 / Google Colab environment

Hyperparameters:
Optimizer: AdamW
Learning Rate: 5e-4 (cosine schedule)
Weight Decay: 0.05
Label Smoothing: 0.1
Batch Size: 64
Epochs: 50
